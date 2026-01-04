"""CuTe DSL fused MoE GEMM kernel (H100 / SM90-focused).

This implements the core routed expert GEMM used by `kestrel.fused_moe`:
  out[sorted_token_ids] = x @ W_expert.T  (optionally scaled by routing weights)

The reference implementation lives in `kestrel/fused_moe/kernels.py` (Triton).
This file provides an equivalent kernel using NVIDIA's CuTe DSL.
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch import ProxyKind, SharedSpace
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic

from kestrel_kernels.flash_attn.cute import ampere_helpers
from kestrel_kernels.flash_attn.cute import hopper_helpers as sm90_utils
from kestrel_kernels.flash_attn.cute import utils as fa_utils

_CUTLASS_INITIALIZED = False
_CUTE_KERNEL_ATTRS_SET: set[str] = set()
_DEVICE_CACHE_CONFIG_SET = False

# Precompiled kernel registry
_precompiled_cache: Dict[Tuple[str, "CuteMoeConfig"], Any] = {}
_precompiled_cache_fp8: Dict[Tuple[str, "CuteMoeConfig"], Any] = {}
_precompiled_dir = Path(__file__).parent / "precompiled"
_cuda_arch: str | None = None


def _get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper, 'sm100' for Blackwell)."""
    global _cuda_arch
    if _cuda_arch is None:
        major, minor = torch.cuda.get_device_capability()
        _cuda_arch = f"sm{major}{minor}"
    return _cuda_arch


def _get_precompiled_kernel_path(kind: str, config: "CuteMoeConfig") -> Path | None:
    """Get path to precompiled kernel if it exists."""
    arch = _get_cuda_arch()
    filename = (
        f"cute_moe_{kind}_m{config.block_m}_n{config.block_n}_k{config.block_k}"
        f"_w{config.num_warps}_s{config.num_stages}_{arch}.so"
    )
    path = _precompiled_dir / filename
    return path if path.exists() else None


def _load_precompiled_kernel(kind: str, config: "CuteMoeConfig"):
    """Load a precompiled kernel if available, return None otherwise."""
    compile_key = (kind, config)

    # Check if already loaded
    if compile_key in _precompiled_cache:
        return _precompiled_cache[compile_key]

    # Check if precompiled file exists
    so_path = _get_precompiled_kernel_path(kind, config)
    if so_path is None:
        return None

    # Load the module
    mod = cute.runtime.load_module(str(so_path))

    # Get the function by its exported name
    arch = _get_cuda_arch()
    function_name = (
        f"cute_moe_{kind}_m{config.block_m}_n{config.block_n}_k{config.block_k}"
        f"_w{config.num_warps}_s{config.num_stages}_{arch}"
    )

    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[compile_key] = kernel_fn
    return kernel_fn


def _ensure_cutlass_initialized() -> None:
    global _CUTLASS_INITIALIZED
    if _CUTLASS_INITIALIZED:
        return
    # The upstream helper `cutlass.cuda.initialize_cuda_context()` uses `cuCtxCreate`
    # (new context), which is incompatible with PyTorch/Triton tensors already
    # allocated in the process. We must use the device *primary* context instead.
    res, = cuda.cuInit(0)
    if res != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed with {res}")
    res, cur_ctx = cuda.cuCtxGetCurrent()
    if res != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed with {res}")
    if int(cur_ctx) == 0:
        res, dev = cuda.cuDeviceGet(int(torch.cuda.current_device()))
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed with {res}")
        res, primary_ctx = cuda.cuDevicePrimaryCtxRetain(dev)
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain failed with {res}")
        res, = cuda.cuCtxSetCurrent(primary_ctx)
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed with {res}")
    _CUTLASS_INITIALIZED = True


def _set_compiled_kernel_shared_carveout(compiled: Any, *, carveout_pct: int = 100) -> None:
    """Best-effort: prefer shared memory over L1 for the compiled kernel."""
    try:
        import cuda.bindings.runtime as cuda_rt

        jit_module = getattr(compiled, "jit_module", None)
        libs = getattr(jit_module, "cuda_library", None)
        if not libs:
            return
        lib = libs[0]

        kernel_info = getattr(compiled, "kernel_info", None)
        if not kernel_info:
            return

        dev = int(torch.cuda.current_device())
        for name in kernel_info.keys():
            if not isinstance(name, str):
                continue
            if name in _CUTE_KERNEL_ATTRS_SET:
                continue
            err, kernel = cuda_rt.cudaLibraryGetKernel(lib, name.encode())
            if int(err) != 0:
                continue
            # Match Triton's cache preference (and increase effective shared memory capacity).
            # This is separate from the carveout attribute and affects NCU's "Function Cache Configuration".
            try:
                cuda_rt.cudaFuncSetCacheConfig(
                    kernel, cuda_rt.cudaFuncCache.cudaFuncCachePreferShared
                )
            except Exception:
                pass
            cuda_rt.cudaKernelSetAttributeForDevice(
                kernel,
                cuda_rt.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout,
                int(carveout_pct),
                dev,
            )
            _CUTE_KERNEL_ATTRS_SET.add(name)
    except Exception:
        # Never fail the op because of optional tuning.
        return


def _maybe_set_device_cache_config() -> None:
    """Best-effort: prefer shared memory over L1 on SM90."""
    global _DEVICE_CACHE_CONFIG_SET
    if _DEVICE_CACHE_CONFIG_SET:
        return
    try:
        import cuda.bindings.runtime as cuda_rt

        cuda_rt.cudaDeviceSetCacheConfig(cuda_rt.cudaFuncCache.cudaFuncCachePreferShared)
        _DEVICE_CACHE_CONFIG_SET = True
    except Exception:
        return


@dataclass(frozen=True)
class CuteMoeConfig:
    # Tile sizes (tuned for Moondream MoE shapes; adjust via benchmarking).
    block_m: int = 16
    block_n: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        """Validate config constraints at construction time."""
        # CRITICAL: block_m must not exceed num_threads!
        # The kernel's metadata loading loop uses `if tx < block_m` to load per-row
        # metadata into shared memory. With only num_threads threads, rows beyond
        # num_threads-1 read uninitialized shared memory -> illegal memory access.
        if self.block_m > self.num_threads:
            raise ValueError(
                f"block_m ({self.block_m}) must not exceed num_threads ({self.num_threads}). "
                f"With {self.num_warps} warps, max block_m is {self.num_threads}."
            )

        # block_n must be divisible by 8 * num_warps (MMA layout constraint)
        # The MMA atom is (16, 8, 16) and warps are tiled (1, num_warps, 1) in N
        mma_n_coverage = 8 * self.num_warps
        if self.block_n % mma_n_coverage != 0:
            raise ValueError(
                f"block_n ({self.block_n}) must be divisible by 8 * num_warps ({mma_n_coverage})."
            )

        # block_n=32 with num_warps=8 fails LDSM alignment verification
        if self.block_n == 32 and self.num_warps == 8:
            raise ValueError(
                "block_n=32 with num_warps=8 causes LDSM alignment verification failure."
            )

        # Shared memory constraint: sA + sB must fit in ~200KB (H100 has 228KB, need room for sC + metadata)
        # sA = num_stages * block_m * block_k * 2 bytes
        # sB = num_stages * block_n * block_k * 2 bytes
        sA_elements = self.num_stages * self.block_m * self.block_k
        sB_elements = self.num_stages * self.block_n * self.block_k
        total_elements = sA_elements + sB_elements
        max_elements = 100000  # ~200KB for BF16, leaving room for sC and metadata
        if total_elements > max_elements:
            sA_kb = sA_elements * 2 // 1024
            sB_kb = sB_elements * 2 // 1024
            raise ValueError(
                f"Shared memory exceeded: sA={sA_kb}KB + sB={sB_kb}KB = {sA_kb + sB_kb}KB "
                f"(stages={self.num_stages}, m={self.block_m}, n={self.block_n}, k={self.block_k}). "
                f"Max ~200KB."
            )

    @property
    def num_threads(self) -> int:
        return 32 * self.num_warps


# Config auto-loading from JSON files
_CONFIGS_DIR = Path(__file__).parent / "configs"


@lru_cache(maxsize=None)
def _load_cute_moe_configs(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str,
    arch: str,
) -> dict | None:
    """Load configs for given model shape and hardware. Return None if not found."""
    filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
    config_file = _CONFIGS_DIR / filename
    if not config_file.exists():
        return None
    with open(config_file) as f:
        return json.load(f)


def get_cute_moe_block_m(
    num_tokens: int,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "bf16",
) -> int:
    """Get the block_m value for routing alignment.

    Both UP and DOWN kernels share the same block_m for a given token count,
    so this returns the common value used by moe_align_block_size.

    Args:
        num_tokens: Number of tokens (batch size)
        num_experts: Number of experts (E)
        hidden_size: Hidden/model dimension (H)
        intermediate_size: Expert intermediate dimension (I)
        dtype: Data type ("bf16" or "fp8")

    Returns:
        block_m value for routing alignment.

    Raises:
        ValueError: If no config file exists for the model shape + GPU arch.
    """
    arch = _get_cuda_arch()
    configs = _load_cute_moe_configs(num_experts, hidden_size, intermediate_size, dtype, arch)

    if configs is None:
        filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
        raise ValueError(
            f"No CuTe MoE configs for this model shape. "
            f"Expected file: {_CONFIGS_DIR / filename}"
        )

    # Use "up" config to get block_m (UP and DOWN have matching block_m)
    up_configs = configs.get("up", {})
    if not up_configs:
        raise ValueError("No 'up' configs in config file")

    # Find nearest token count
    token_keys = [int(k) for k in up_configs.keys()]
    nearest = min(token_keys, key=lambda t: abs(t - num_tokens))
    cfg = up_configs[str(nearest)]

    return cfg["block_m"]


def get_cute_moe_config(
    kind: Literal["up", "down"],
    num_tokens: int,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "bf16",
) -> CuteMoeConfig:
    """Get optimal config for given parameters. Raises if not available.

    Args:
        kind: "up" or "down" kernel type
        num_tokens: Number of tokens (batch size)
        num_experts: Number of experts (E)
        hidden_size: Hidden/model dimension (H)
        intermediate_size: Expert intermediate dimension (I)
        dtype: Data type ("bf16" or "fp8")

    Returns:
        CuteMoeConfig with optimal tile sizes for the given parameters.

    Raises:
        ValueError: If no config file exists for the model shape + GPU arch.
    """
    arch = _get_cuda_arch()
    configs = _load_cute_moe_configs(num_experts, hidden_size, intermediate_size, dtype, arch)

    if configs is None:
        filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
        raise ValueError(
            f"No CuTe MoE configs for this model shape. "
            f"Expected file: {_CONFIGS_DIR / filename}"
        )

    kind_configs = configs.get(kind, {})
    if not kind_configs:
        raise ValueError(f"No '{kind}' configs in config file")

    # Find nearest token count
    token_keys = [int(k) for k in kind_configs.keys()]
    nearest = min(token_keys, key=lambda t: abs(t - num_tokens))
    cfg = kind_configs[str(nearest)]

    return CuteMoeConfig(
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        block_k=cfg["block_k"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _to_cute_tensor_1d_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 1D tensor")
    return from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_1d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous 1D tensor")
    return from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_scalar_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or t.numel() != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 tensor with shape (1,)")
    return from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_2d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 2 or not t.is_contiguous():
        raise ValueError("Expected row-major contiguous 2D tensor")
    # The MoE kernel relies on 128-bit vectorized accesses (vec_size=8 for BF16/FP16).
    # Add compactness/divisibility hints so the compiler can prove alignment for cp.async.
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=1, alignment=assumed_align, divisibility=8, enable_tvm_ffi=True
    )


def _to_cute_tensor_2d_contig_u8(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor (got {t.dtype})")
    if t.ndim != 2 or not t.is_contiguous():
        raise ValueError("Expected row-major contiguous 2D uint8 tensor")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=1, alignment=assumed_align, divisibility=16, enable_tvm_ffi=True
    )


def _to_cute_tensor_3d_last_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 3 or t.stride(-1) != 1:
        raise ValueError("Expected 3D tensor contiguous in the last dim")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=2, alignment=assumed_align, divisibility=8, enable_tvm_ffi=True
    )


def _to_cute_tensor_3d_last_contig_u8(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor (got {t.dtype})")
    if t.ndim != 3 or t.stride(-1) != 1:
        raise ValueError("Expected 3D uint8 tensor contiguous in the last dim")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=2, alignment=assumed_align, divisibility=16, enable_tvm_ffi=True
    )


class _FusedMoeMatmulCuTe:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        config: CuteMoeConfig,
        *,
        mul_routed_weight: bool,
        top_k: int,
    ) -> None:
        self.dtype = dtype
        self.config = config
        self.mul_routed_weight = mul_routed_weight
        self.top_k = int(top_k)

    def _shared_storage_cls(self):
        sA_elems = self.config.num_stages * self.config.block_m * self.config.block_k
        sB_elems = self.config.num_stages * self.config.block_n * self.config.block_k
        sC_elems = self.config.block_m * self.config.block_n
        sMeta_elems = self.config.block_m
        # Match flash-attn's practice of aligning swizzled SMEM buffers to large boundaries.
        # This helps the compiler prove alignment for vectorized LDGSTS and reduces any
        # bank-conflict risk for LDSM (ldmatrix) loads.
        sA_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sA_elems], 1024]
        sB_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sB_elems], 1024]
        sC_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sC_elems], 16]
        sAid_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        # Store routing weights as fp32 in shared memory so the epilogue doesn't need
        # per-element dtype->fp32 conversion.
        sW_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]
        sArowBase_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, sMeta_elems], 16
        ]
        sCrowBase_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, sMeta_elems], 16
        ]

        @cute.struct
        class SharedStorage:
            sA: sA_struct
            sB: sB_struct
            sC: sC_struct
            sAid: sAid_struct
            sW: sW_struct
            sArowBase: sArowBase_struct
            sCrowBase: sCrowBase_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M_tokens_or_assignments, K)
        mB: cute.Tensor,  # (E, N, K) row-major in K
        mC: cute.Tensor,  # (M_assignments, N) row-major in N
        mTopkWeights: cute.Tensor,  # (M_assignments,) or empty
        mSortedTokenIds: cute.Tensor,  # (EM,)
        mExpertIds: cute.Tensor,  # (EM / block_m,)
        mNumTokensPostPadded: cute.Tensor,  # (1,)
        stream: cuda.CUstream,
    ):
        # Warp-level MMA (works on SM90; we prioritize small-M tiles here).
        # We tile warps along N so block_m can stay small.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (1, self.config.num_warps, 1),  # (M, N, K) warp tiling
            permutation_mnk=(self.config.block_m, self.config.block_n, 16),
        )

        block_m = self.config.block_m
        block_n = self.config.block_n
        # Launch for the *allocated* routing workspace (sorted ids length) and
        # mask out unused blocks using `mNumTokensPostPadded` inside the kernel.
        grid_m = cute.ceil_div(Int32(mSortedTokenIds.shape[0]), block_m)
        grid_n = cute.ceil_div(Int32(mB.shape[1]), block_n)
        SharedStorage = self._shared_storage_cls()

        self.kernel(
            tiled_mma,
            mA,
            mB,
            mC,
            mTopkWeights,
            mSortedTokenIds,
            mExpertIds,
            mNumTokensPostPadded,
            SharedStorage,
        ).launch(
            grid=(grid_m, grid_n, 1),
            block=(self.config.num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mTopkWeights: cute.Tensor,
        mSortedTokenIds: cute.Tensor,
        mExpertIds: cute.Tensor,
        mNumTokensPostPadded: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tx, _, _ = cute.arch.thread_idx()
        pid_m, pid_n, _ = cute.arch.block_idx()

        block_m = self.config.block_m
        block_n = self.config.block_n
        block_k = self.config.block_k
        num_threads = self.config.num_threads
        num_stages = self.config.num_stages

        n_start = pid_n * block_n
        row_start = pid_m * block_m

        num_tokens_post_padded = Int32(mNumTokensPostPadded[0])
        block_active = row_start < num_tokens_post_padded

        # NOTE: For this kernel, we assume K and N are static across the op invocation.
        N = Int32(mB.shape[1])
        K = Int32(mA.shape[1])

        # `num_valid_tokens` matches the Triton kernel semantics:
        # - When top_k > 1 (up projection): A is [M, K], valid assignments = M * top_k.
        # - When top_k == 1 (down projection): A is [M_assignments, K], valid assignments = M_assignments.
        num_valid_tokens = Int32(mA.shape[0]) * Int32(self.top_k)

        expert_id = Int32(mExpertIds[pid_m])

        # Shared memory.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Load assignment metadata once per CTA into shared memory:
        # - Avoid per-thread redundant loads from mSortedTokenIds.
        # - Avoid dynamic-indexed register arrays that frequently spill to local memory.
        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        sW = storage.sW.get_tensor(s_meta_layout)
        sArowBase = storage.sArowBase.get_tensor(s_meta_layout)
        sCrowBase = storage.sCrowBase.get_tensor(s_meta_layout)
        # Precompute per-row base pointers in bytes to avoid repeated 64-bit multiplies
        # in the hot inner loops.
        element_bytes = cutlass.Int64(self.dtype.width // 8)
        stride_am_elems = cutlass.Int64(mA.stride[0])
        stride_cm_elems = cutlass.Int64(mC.stride[0])
        if tx < Int32(block_m):
            idx = row_start + tx
            aid = Int32(num_valid_tokens)  # padded sentinel
            if idx < num_tokens_post_padded:
                aid = Int32(mSortedTokenIds[idx])
            sAid[tx] = aid
            tok = Int32(0)
            if aid < num_valid_tokens:
                tok = aid // Int32(self.top_k)
            arow_base = cutlass.Int64(0)
            crow_base = cutlass.Int64(0)
            if aid < num_valid_tokens:
                arow_base = cutlass.Int64(tok) * stride_am_elems * element_bytes
                crow_base = cutlass.Int64(aid) * stride_cm_elems * element_bytes
            sArowBase[tx] = arow_base
            sCrowBase[tx] = crow_base
            if const_expr(self.mul_routed_weight):
                w32 = Float32(0.0)
                if aid < num_valid_tokens:
                    w32 = Float32(mTopkWeights[aid])
                sW[tx] = w32
        cute.arch.barrier()

        # Flash-attn-style swizzled SMEM layout to reduce bank conflicts on ldmatrix loads.
        sB_layout_atom = ampere_helpers.get_smem_layout_atom(self.dtype, block_k)
        sA_tile_layout = cute.tile_to_shape(sB_layout_atom, (block_m, block_k), (0, 1))
        sA_elems = num_stages * block_m * block_k
        sA = storage.sA.get_tensor(cute.make_layout((sA_elems,), stride=(1,)))
        sB_tile_layout = cute.tile_to_shape(sB_layout_atom, (block_n, block_k), (0, 1))
        sB_elems = num_stages * block_n * block_k
        sB = storage.sB.get_tensor(cute.make_layout((sB_elems,), stride=(1,)))
        # Flash-attn-style swizzled SMEM layout for C. This makes `stmatrix` stores conflict-free
        # and preserves 128-bit vector contiguity for the subsequent SMEM->GMEM scatter stores.
        sC_layout_atom = ampere_helpers.get_smem_layout_atom(self.dtype, block_n)
        sC_layout = cute.tile_to_shape(sC_layout_atom, (block_m, block_n), (0, 1))
        sC_linear = storage.sC.get_tensor(cute.make_layout((block_m * block_n,), stride=(1,)))
        sC = cute.make_tensor(sC_linear.iterator, sC_layout)

        # Thread MMA setup.
        thr_mma = tiled_mma.get_slice(tx)
        acc_shape = thr_mma.partition_shape_C((block_m, block_n))
        acc = cute.make_rmem_tensor(acc_shape, Float32)
        acc.fill(0.0)

        # Smem->rmem copy atoms (ldmatrix).
        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma).get_slice(tx)
        smem_thr_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma).get_slice(tx)

        # If this block is inactive or expert is invalid, just write zeros (matches Triton).
        # NOTE: Branch is uniform across the CTA, so barriers remain safe.
        if (not block_active) or (expert_id == -1):
            vec_size = 8
            copy_bits = vec_size * self.dtype.width
            gmem_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=copy_bits,
            )
            element_bytes = cutlass.Int64(self.dtype.width // 8)
            align_bytes = vec_size * int(self.dtype.width // 8)
            mC_base_i64 = mC.iterator.toint()

            zero_vec = cute.make_rmem_tensor((vec_size,), self.dtype)
            zero_vec.fill(self.dtype(0.0))
            vec_n = block_n // vec_size
            total_vec_c = block_m * vec_n
            for vec_linear_c in range(tx, total_vec_c, num_threads):
                r_c = vec_linear_c // vec_n
                nvec_c = vec_linear_c - r_c * vec_n
                n0 = Int32(nvec_c * vec_size)
                aid_c = sAid[r_c]
                col0 = n_start + n0
                if (aid_c < num_valid_tokens) and (col0 < N):
                    g_off_bytes_c = sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes
                    g_ptr_c = cute.make_ptr(
                        self.dtype,
                        mC_base_i64 + g_off_bytes_c,
                        cute.AddressSpace.gmem,
                        assumed_align=align_bytes,
                    )
                    dst_c = cute.make_tensor(g_ptr_c, (vec_size,))
                    cute.copy(gmem_store_atom, zero_vec, dst_c)
        else:
            # Copy atoms (cp.async for gmem->smem).
            vec_size = 8
            copy_bits = vec_size * self.dtype.width
            # For Hopper, routing makes A gather-heavy while B is large/streaming.
            # Using a lower cache level for B can reduce L1/TEX pressure.
            atom_async_copy_a = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.dtype,
                num_bits_per_copy=copy_bits,
            )
            b_cache_mode = cpasync.LoadCacheMode.GLOBAL
            atom_async_copy_b = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=b_cache_mode),
                self.dtype,
                num_bits_per_copy=copy_bits,
            )

            pred_false = cute.make_rmem_tensor((vec_size,), Boolean)
            pred_false.fill(False)

            element_bytes = cutlass.Int64(self.dtype.width // 8)
            align_bytes = vec_size * int(self.dtype.width // 8)

            mA_base_i64 = mA.iterator.toint()
            mC_base_i64 = mC.iterator.toint()
            sA_base_i64 = sA.iterator.toint()
            sB_base_i64 = sB.iterator.toint()
            sC_base_i64 = sC.iterator.toint()

            block_vec_k = block_k // vec_size
            total_vec_a = block_m * block_vec_k

            stage_stride_a = Int32(block_m * block_k)
            stage_stride_b = Int32(block_n * block_k)

            k_tiles = cute.ceil_div(K, block_k)
            full_n_tile = (n_start + Int32(block_n)) <= N

            vB_layout = cute.make_layout((1, vec_size))
            tB_shape_dim_1 = sB_layout_atom.outer.shape[1] // vec_size
            tB_layout = cute.make_ordered_layout(
                (num_threads // tB_shape_dim_1, tB_shape_dim_1),
                order=(1, 0),
            )
            gmem_tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_b, tB_layout, vB_layout)
            gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(tx)

            cB = cute.make_identity_tensor((block_n, block_k))
            tBcB = gmem_thr_copy_B.partition_S(cB)
            t0BcB = gmem_thr_copy_B.get_slice(0).partition_S(cB)
            mB_expert = mB[expert_id, None, None]

            # Prologue: prefetch the first `num_stages` K tiles.
            for stage_prefetch in cutlass.range_constexpr(num_stages):
                tile_idx_prefetch = Int32(stage_prefetch)
                k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                tile_in_range = tile_idx_prefetch < k_tiles

                # A tile: [block_m, block_k]
                for vec_linear_a in range(tx, total_vec_a, num_threads):
                    r_a = vec_linear_a // block_vec_k
                    kvec_a = vec_linear_a - r_a * block_vec_k
                    k_a = Int32(kvec_a * vec_size)
                    kg_a = k_start_prefetch + k_a

                    aid_a = sAid[r_a]
                    valid_row_a = aid_a < num_valid_tokens
                    arow_base = sArowBase[r_a]

                    valid_a = tile_in_range and valid_row_a and (kg_a < K)
                    g_off_bytes_a = arow_base + cutlass.Int64(kg_a) * element_bytes
                    g_ptr_a = cute.make_ptr(
                        self.dtype,
                        mA_base_i64 + g_off_bytes_a,
                        cute.AddressSpace.gmem,
                        assumed_align=align_bytes,
                    )
                    src_a = cute.make_tensor(g_ptr_a, (vec_size,))

                    s_linear_a = Int32(stage_prefetch) * stage_stride_a + Int32(
                        sA_tile_layout((Int32(r_a), k_a))
                    )
                    s_off_bytes_a = cutlass.Int64(s_linear_a) * element_bytes
                    s_ptr_a = cute.make_ptr(
                        self.dtype,
                        sA_base_i64 + s_off_bytes_a,
                        cute.AddressSpace.smem,
                        assumed_align=align_bytes,
                    )
                    dst_a = cute.make_tensor(s_ptr_a, (vec_size,))
                    if valid_a:
                        cute.copy(atom_async_copy_a, src_a, dst_a)
                    else:
                        cute.copy(atom_async_copy_a, src_a, dst_a, pred=pred_false)

                # B tile: [block_n, block_k] (swizzled SMEM).
                if tile_in_range:
                    full_k_tile = (k_start_prefetch + Int32(block_k)) <= K
                    # Use `local_tile` to preserve compactness/divisibility metadata (flash-attn style).
                    gB_tile = cute.local_tile(
                        mB_expert,
                        (block_n, block_k),
                        (pid_n, tile_idx_prefetch),
                    )
                    tBgB = gmem_thr_copy_B.partition_S(gB_tile)
                    stage_off_elems_b = Int32(stage_prefetch) * stage_stride_b
                    stage_off_bytes_b = cutlass.Int64(stage_off_elems_b) * element_bytes
                    sB_stage_ptr = cute.make_ptr(
                        self.dtype,
                        sB_base_i64 + stage_off_bytes_b,
                        cute.AddressSpace.smem,
                        assumed_align=align_bytes,
                    )
                    sB_stage = cute.make_tensor(sB_stage_ptr, sB_tile_layout)
                    tBsB = gmem_thr_copy_B.partition_D(sB_stage)
                    if full_k_tile:
                        if full_n_tile:
                            for n in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                                cute.copy(
                                    gmem_tiled_copy_B,
                                    tBgB[None, n, None],
                                    tBsB[None, n, None],
                                )
                        else:
                            for n in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                                if n_start + t0BcB[0, n, 0][0] < N:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                    )
                    else:
                        tBpB = fa_utils.predicate_k(tBcB, limit=K - k_start_prefetch)
                        if full_n_tile:
                            for n in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                                cute.copy(
                                    gmem_tiled_copy_B,
                                    tBgB[None, n, None],
                                    tBsB[None, n, None],
                                    pred=tBpB[None, n, None],
                                )
                        else:
                            for n in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                                if n_start + t0BcB[0, n, 0][0] < N:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                        pred=tBpB[None, n, None],
                                    )

                cute.arch.cp_async_commit_group()

            stage_idx = Int32(0)

            # Main loop: keep `num_stages` groups in flight, so `wait_group(num_stages - 1)`
            # correctly waits for the oldest stage. In the tail, fewer groups remain, so we
            # must drain with progressively smaller wait values (see below).
            main_tiles = k_tiles - Int32(num_stages - 1)
            for tile_idx in cutlass.range(main_tiles, unroll=1):
                cute.arch.cp_async_wait_group(num_stages - 1)
                cute.arch.barrier()

                stage_off_bytes_a = (
                    cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes
                )
                sA_stage_ptr = cute.make_ptr(
                    self.dtype,
                    sA_base_i64 + stage_off_bytes_a,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                stage_off_elems_b = stage_idx * stage_stride_b
                stage_off_bytes_b = cutlass.Int64(stage_off_elems_b) * element_bytes
                sB_stage_ptr = cute.make_ptr(
                    self.dtype,
                    sB_base_i64 + stage_off_bytes_b,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                sA_stage = cute.make_tensor(sA_stage_ptr, sA_tile_layout)
                sB_stage = cute.make_tensor(sB_stage_ptr, sB_tile_layout)
                tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_stage))
                tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB_stage))
                tCsA = smem_thr_copy_A.partition_S(sA_stage)
                tCsB = smem_thr_copy_B.partition_S(sB_stage)
                tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
                tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

                # Prefetch first fragment.
                cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

                k_frags = cute.size(tCsA.shape[2])
                for kf in cutlass.range_constexpr(k_frags):
                    if kf < k_frags - 1:
                        cute.copy(
                            smem_thr_copy_A,
                            tCsA[None, None, kf + 1],
                            tCrA_copy_view[None, None, kf + 1],
                        )
                        cute.copy(
                            smem_thr_copy_B,
                            tCsB[None, None, kf + 1],
                            tCrB_copy_view[None, None, kf + 1],
                        )
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf], tCrB[None, None, kf], acc)
                cute.arch.barrier()

                # Prefetch the next tile into the stage we just consumed.
                next_tile = tile_idx + Int32(num_stages)
                if next_tile < k_tiles:
                    k_start_next = next_tile * Int32(block_k)

                    for vec_linear_a2 in range(tx, total_vec_a, num_threads):
                        r_a2 = vec_linear_a2 // block_vec_k
                        kvec_a2 = vec_linear_a2 - r_a2 * block_vec_k
                        k_a2 = Int32(kvec_a2 * vec_size)
                        kg_a2 = k_start_next + k_a2

                        aid_a2 = sAid[r_a2]
                        valid_row_a2 = aid_a2 < num_valid_tokens
                        arow_base2 = sArowBase[r_a2]

                        valid_a2 = valid_row_a2 and (kg_a2 < K)
                        g_off_bytes_a2 = arow_base2 + cutlass.Int64(kg_a2) * element_bytes
                        g_ptr_a2 = cute.make_ptr(
                            self.dtype,
                            mA_base_i64 + g_off_bytes_a2,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes,
                        )
                        src_a2 = cute.make_tensor(g_ptr_a2, (vec_size,))

                        s_linear_a2 = stage_idx * stage_stride_a + Int32(
                            sA_tile_layout((Int32(r_a2), k_a2))
                        )
                        s_off_bytes_a2 = cutlass.Int64(s_linear_a2) * element_bytes
                        s_ptr_a2 = cute.make_ptr(
                            self.dtype,
                            sA_base_i64 + s_off_bytes_a2,
                            cute.AddressSpace.smem,
                            assumed_align=align_bytes,
                        )
                        dst_a2 = cute.make_tensor(s_ptr_a2, (vec_size,))
                        if valid_a2:
                            cute.copy(atom_async_copy_a, src_a2, dst_a2)
                        else:
                            cute.copy(atom_async_copy_a, src_a2, dst_a2, pred=pred_false)

                    stage_off_elems_b2 = stage_idx * stage_stride_b
                    stage_off_bytes_b2 = cutlass.Int64(stage_off_elems_b2) * element_bytes
                    sB_stage_ptr2 = cute.make_ptr(
                        self.dtype,
                        sB_base_i64 + stage_off_bytes_b2,
                        cute.AddressSpace.smem,
                        assumed_align=align_bytes,
                    )
                    sB_stage2 = cute.make_tensor(sB_stage_ptr2, sB_tile_layout)
                    tBsB2 = gmem_thr_copy_B.partition_D(sB_stage2)

                    full_k_tile_next = (k_start_next + Int32(block_k)) <= K
                    gB_tile_next = cute.local_tile(
                        mB_expert,
                        (block_n, block_k),
                        (pid_n, next_tile),
                    )
                    tBgB_next = gmem_thr_copy_B.partition_S(gB_tile_next)
                    if full_k_tile_next:
                        if full_n_tile:
                            for n in cutlass.range_constexpr(cute.size(tBsB2.shape[1])):
                                cute.copy(
                                    gmem_tiled_copy_B,
                                    tBgB_next[None, n, None],
                                    tBsB2[None, n, None],
                                )
                        else:
                            for n in cutlass.range_constexpr(cute.size(tBsB2.shape[1])):
                                if n_start + t0BcB[0, n, 0][0] < N:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB_next[None, n, None],
                                        tBsB2[None, n, None],
                                    )
                    else:
                        tBpB_next = fa_utils.predicate_k(tBcB, limit=K - k_start_next)
                        if full_n_tile:
                            for n in cutlass.range_constexpr(cute.size(tBsB2.shape[1])):
                                cute.copy(
                                    gmem_tiled_copy_B,
                                    tBgB_next[None, n, None],
                                    tBsB2[None, n, None],
                                    pred=tBpB_next[None, n, None],
                                )
                        else:
                            for n in cutlass.range_constexpr(cute.size(tBsB2.shape[1])):
                                if n_start + t0BcB[0, n, 0][0] < N:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB_next[None, n, None],
                                        tBsB2[None, n, None],
                                        pred=tBpB_next[None, n, None],
                                    )

                    cute.arch.cp_async_commit_group()

                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Tail drain: `cp_async_wait_group(num_stages - 1)` becomes insufficient once
            # we stop prefetching (there are fewer groups in flight). Drain the remaining
            # tiles with wait values: num_stages-2, ..., 0.
            for drain_idx in cutlass.range_constexpr(num_stages - 1):
                cute.arch.cp_async_wait_group(num_stages - 2 - drain_idx)
                cute.arch.barrier()

                stage_off_bytes_a = (
                    cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes
                )
                sA_stage_ptr = cute.make_ptr(
                    self.dtype,
                    sA_base_i64 + stage_off_bytes_a,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                stage_off_elems_b = stage_idx * stage_stride_b
                stage_off_bytes_b = cutlass.Int64(stage_off_elems_b) * element_bytes
                sB_stage_ptr = cute.make_ptr(
                    self.dtype,
                    sB_base_i64 + stage_off_bytes_b,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                sA_stage = cute.make_tensor(sA_stage_ptr, sA_tile_layout)
                sB_stage = cute.make_tensor(sB_stage_ptr, sB_tile_layout)
                tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_stage))
                tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB_stage))
                tCsA = smem_thr_copy_A.partition_S(sA_stage)
                tCsB = smem_thr_copy_B.partition_S(sB_stage)
                tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
                tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

                cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

                k_frags = cute.size(tCsA.shape[2])
                for kf in cutlass.range_constexpr(k_frags):
                    if kf < k_frags - 1:
                        cute.copy(
                            smem_thr_copy_A,
                            tCsA[None, None, kf + 1],
                            tCrA_copy_view[None, None, kf + 1],
                        )
                        cute.copy(
                            smem_thr_copy_B,
                            tCsB[None, None, kf + 1],
                            tCrB_copy_view[None, None, kf + 1],
                        )
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf], tCrB[None, None, kf], acc)
                cute.arch.barrier()

                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Epilogue: apply routed weights (if needed), then store the fp32 accumulator to shared
            # in fp16/bf16, then scatter to gmem.
            #
            # NOTE: `retile` expects a Tensor (not TensorSSA), so we materialize a fragment for the
            # rmem -> smem store path.
            rC = cute.make_fragment_like(acc, self.dtype)
            if const_expr(self.mul_routed_weight):
                cC = cute.make_identity_tensor((block_m, block_n))
                tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
                tAcc = fa_utils.make_acc_tensor_mn_view(acc)
                tRC = fa_utils.make_acc_tensor_mn_view(rC)
                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    w32 = sW[m]
                    for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                        tRC[mi, ni] = self.dtype(Float32(tAcc[mi, ni]) * w32)
            else:
                rC.store(acc.load().to(self.dtype))
            smem_store_atom = fa_utils.get_smem_store_atom(90, self.dtype)
            if const_expr(block_n == 32):
                # When block_n=32, each warp effectively owns 32 / 4 = 8 columns of C.
                # fa_utils does `stmatrix.x4`, which is a wider store. This leads to
                # garbage data being written to SMEM. Use a narrower variant instead.
                smem_store_atom = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
                    self.dtype,
                )
            smem_thr_store = cute.make_tiled_copy_C(smem_store_atom, tiled_mma).get_slice(tx)
            tCsC = smem_thr_store.partition_D(sC)
            tCrC = smem_thr_store.retile(rC)
            cute.copy(smem_store_atom, tCrC, tCsC)
            cute.arch.barrier()

            gmem_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=copy_bits,
            )
            vec_n = block_n // vec_size
            total_vec_c = block_m * vec_n

            for vec_linear_c in range(tx, total_vec_c, num_threads):
                r_c = vec_linear_c // vec_n
                nvec_c = vec_linear_c - r_c * vec_n
                n0 = Int32(nvec_c * vec_size)
                aid_c = sAid[r_c]
                col0 = n_start + n0
                if (aid_c < num_valid_tokens) and (col0 < N):
                    s_linear_c = Int32(sC_layout((Int32(r_c), n0)))
                    s_off_bytes_c = cutlass.Int64(s_linear_c) * element_bytes
                    s_ptr_c = cute.make_ptr(
                        self.dtype,
                        sC_base_i64 + s_off_bytes_c,
                        cute.AddressSpace.smem,
                        assumed_align=align_bytes,
                    )
                    src_c = cute.make_tensor(s_ptr_c, (vec_size,))

                    g_off_bytes_c = sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes
                    g_ptr_c = cute.make_ptr(
                        self.dtype,
                        mC_base_i64 + g_off_bytes_c,
                        cute.AddressSpace.gmem,
                        assumed_align=align_bytes,
                    )
                    dst_c = cute.make_tensor(g_ptr_c, (vec_size,))

                    cute.copy(gmem_store_atom, src_c, dst_c)


class _FusedMoeMatmulCuTeFp8:
    """Routed MoE GEMM with FP8 activations+weights (W8A8) using SM90 WGMMA.

    Math:
      C = (A_fp8 @ (B_fp8)^T) * A_scale * B_scale   (and optionally * routed_weight)

    Storage format:
      - A_fp8: uint8 view of float8_e4m3fn values representing (A / A_scale)
      - A_scale: per-row fp32 scale
      - B_fp8: uint8 view of float8_e4m3fn values representing (W / B_scale_col)
      - B_scale: per-(expert, out_channel) fp32 scale
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        fp8_dtype: type[cutlass.Numeric],
        config: CuteMoeConfig,
        *,
        mul_routed_weight: bool,
        top_k: int,
    ) -> None:
        self.dtype = dtype
        self.fp8_dtype = fp8_dtype
        self.config = config
        self.mul_routed_weight = mul_routed_weight
        self.top_k = int(top_k)

    def _shared_storage_cls(self):
        block_m = self.config.block_m
        block_n = self.config.block_n
        block_k = self.config.block_k
        num_stages = self.config.num_stages

        # WGMMA requires warpgroup-compatible SMEM layouts.
        s_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.fp8_dtype, block_k),
            self.fp8_dtype,
        )
        sA_layout = cute.tile_to_shape(
            s_layout_atom, (block_m, block_k, num_stages), order=(0, 1, 2)
        )
        sB_layout = cute.tile_to_shape(
            s_layout_atom, (block_n, block_k, num_stages), order=(0, 1, 2)
        )

        sA_struct = cute.struct.Align[
            cute.struct.MemRange[self.fp8_dtype, cute.cosize(sA_layout)], 1024
        ]
        sB_struct = cute.struct.Align[
            cute.struct.MemRange[self.fp8_dtype, cute.cosize(sB_layout)], 1024
        ]

        sMeta_elems = block_m
        sAid_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        sAScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]

        sBScale_elems = block_n
        sBScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sBScale_elems], 16]

        @cute.struct
        class SharedStorage:
            sA: sA_struct
            sB: sB_struct
            sAid: sAid_struct
            sAScale: sAScale_struct
            sBScale: sBScale_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mAbits: cute.Tensor,  # (M_tokens_or_assignments, K) uint8 view of fp8 values
        mAScale: cute.Tensor,  # (M_tokens_or_assignments,) fp32
        mBbits: cute.Tensor,  # (E, N, K) uint8 view of fp8 values
        mBScale: cute.Tensor,  # (E, N) fp32
        mC: cute.Tensor,  # (M_assignments, N) BF16
        mTopkWeights: cute.Tensor,  # (M_assignments,) or empty
        mSortedTokenIds: cute.Tensor,  # (EM,)
        mExpertIds: cute.Tensor,  # (EM / block_m,)
        mNumTokensPostPadded: cute.Tensor,  # (1,)
        stream: cuda.CUstream,
    ):
        # SM90 warpgroup MMA.
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.fp8_dtype,
            self.fp8_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.config.block_m // 64, 1, 1),
            tiler_mn=(64, self.config.block_n),
        )

        block_m = self.config.block_m
        block_n = self.config.block_n
        grid_m = cute.ceil_div(Int32(mSortedTokenIds.shape[0]), block_m)
        grid_n = cute.ceil_div(Int32(mC.shape[1]), block_n)
        SharedStorage = self._shared_storage_cls()

        self.kernel(
            tiled_mma,
            mAbits,
            mAScale,
            mBbits,
            mBScale,
            mC,
            mTopkWeights,
            mSortedTokenIds,
            mExpertIds,
            mNumTokensPostPadded,
            SharedStorage,
        ).launch(
            grid=(grid_m, grid_n, 1),
            block=(self.config.num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mAbits: cute.Tensor,
        mAScale: cute.Tensor,
        mBbits: cute.Tensor,
        mBScale: cute.Tensor,
        mC: cute.Tensor,
        mTopkWeights: cute.Tensor,
        mSortedTokenIds: cute.Tensor,
        mExpertIds: cute.Tensor,
        mNumTokensPostPadded: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tx, _, _ = cute.arch.thread_idx()
        pid_m, pid_n, _ = cute.arch.block_idx()

        block_m = self.config.block_m
        block_n = self.config.block_n
        block_k = self.config.block_k
        num_threads = self.config.num_threads
        num_stages = self.config.num_stages

        n_start = pid_n * block_n
        row_start = pid_m * block_m

        num_tokens_post_padded = Int32(mNumTokensPostPadded[0])
        block_active = row_start < num_tokens_post_padded

        N = Int32(mC.shape[1])
        K = Int32(mAbits.shape[1])
        num_valid_tokens = Int32(mAbits.shape[0]) * Int32(self.top_k)
        expert_id = Int32(mExpertIds[pid_m])

        full_n_tile = (n_start + Int32(block_n)) <= N

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Shared routing/scale metadata for the M rows in this CTA.
        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        sAScale = storage.sAScale.get_tensor(s_meta_layout)
        sBScale = storage.sBScale.get_tensor(cute.make_layout((block_n,), stride=(1,)))

        # Shared-memory operand tensors (staged).
        s_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.fp8_dtype, block_k),
            self.fp8_dtype,
        )
        sA_layout = cute.tile_to_shape(
            s_layout_atom, (block_m, block_k, num_stages), order=(0, 1, 2)
        )
        sB_layout = cute.tile_to_shape(
            s_layout_atom, (block_n, block_k, num_stages), order=(0, 1, 2)
        )
        # WGMMA expects affine layouts; move swizzle to the pointer (recast) via `swizzle=...`.
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner, dtype=self.fp8_dtype)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner, dtype=self.fp8_dtype)

        element_bytes_a_g = cutlass.Int64(self.fp8_dtype.width // 8)
        element_bytes_c = cutlass.Int64(self.dtype.width // 8)
        stride_am_elems = cutlass.Int64(mAbits.stride[0])
        stride_cm_elems = cutlass.Int64(mC.stride[0])

        if block_active:
            if expert_id == -1:
                _ = Int32(0)

            else:
                # Load per-row routing metadata + activation scales once per CTA to avoid
                # redundant global loads across threads.
                if tx < Int32(block_m):
                    idx = row_start + tx
                    aid = Int32(num_valid_tokens)  # padded sentinel
                    if idx < num_tokens_post_padded:
                        aid = Int32(mSortedTokenIds[idx])
                    sAid[tx] = aid

                    row_scale = Float32(0.0)
                    if aid < num_valid_tokens:
                        tok = aid // Int32(self.top_k)
                        row_scale = Float32(mAScale[tok])
                        if const_expr(self.mul_routed_weight):
                            row_scale = row_scale * Float32(mTopkWeights[aid])
                    sAScale[tx] = row_scale

                # Prefetch per-output-channel scales for this expert + N tile into SMEM.
                #
                # Use cp.async so the gmem->smem transfer is counted as LDGSTS (like A/B)
                # instead of regular shared-memory stores. This helps match Triton's trace,
                # while still avoiding redundant gmem reads in the epilogue.
                if full_n_tile:
                    vec_size_scale = 4  # 16B
                    copy_bits_scale = vec_size_scale * Float32.width
                    atom_async_copy_scale = cute.make_copy_atom(
                        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                        Float32,
                        num_bits_per_copy=copy_bits_scale,
                    )
                    pred_scale_layout = cute.make_layout((vec_size_scale,), stride=(0,))
                    pred_scale = cute.make_rmem_tensor(pred_scale_layout, Boolean)
                    pred_scale[0] = True

                    align_bytes_scale = vec_size_scale * int(Float32.width // 8)
                    element_bytes_scale = cutlass.Int64(Float32.width // 8)
                    stride_bse = cutlass.Int64(mBScale.stride[0])
                    stride_bsn = cutlass.Int64(mBScale.stride[1])
                    mBScale_base_i64 = mBScale.iterator.toint()
                    sBScale_base_i64 = sBScale.iterator.toint()

                    num_vec_scale = block_n // vec_size_scale
                    for vec_scale in range(tx, num_vec_scale, num_threads):
                        n0 = Int32(vec_scale * vec_size_scale)
                        col0 = n_start + n0
                        g_off_elems_scale = (
                            cutlass.Int64(expert_id) * stride_bse
                            + cutlass.Int64(col0) * stride_bsn
                        )
                        g_ptr_scale = cute.make_ptr(
                            Float32,
                            mBScale_base_i64 + g_off_elems_scale * element_bytes_scale,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes_scale,
                        )
                        src_scale = cute.make_tensor(g_ptr_scale, (vec_size_scale,))

                        s_off_bytes_scale = cutlass.Int64(n0) * element_bytes_scale
                        s_ptr_scale = cute.make_ptr(
                            Float32,
                            sBScale_base_i64 + s_off_bytes_scale,
                            cute.AddressSpace.smem,
                            assumed_align=align_bytes_scale,
                        )
                        dst_scale = cute.make_tensor(s_ptr_scale, (vec_size_scale,))
                        cute.copy(atom_async_copy_scale, src_scale, dst_scale, pred=pred_scale)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_proxy(
                        ProxyKind.async_shared, space=SharedSpace.shared_cta
                    )

                else:
                    for n_local in range(tx, block_n, num_threads):
                        col = n_start + Int32(n_local)
                        scale = Float32(0.0)
                        if col < N:
                            scale = Float32(mBScale[expert_id, col])
                        sBScale[n_local] = scale

                cute.arch.barrier()

                # WGMMA accumulator.
                acc = cute.make_rmem_tensor(
                    tiled_mma.partition_shape_C((block_m, block_n)), Float32
                )
                acc.fill(0.0)

                # WGMMA operand fragments (indexed by stage).
                warp_group_idx = cute.arch.make_warp_uniform(tx // num_threads)
                warp_group_thread_layout = cute.make_layout((1,), stride=(num_threads,))
                wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
                tSrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA))
                tSrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB))

                # cp.async copy atoms for FP8 operands (16B vectors).
                vec_size_in = 16
                copy_bits_in = vec_size_in * self.fp8_dtype.width
                atom_async_copy_a = cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    self.fp8_dtype,
                    num_bits_per_copy=copy_bits_in,
                )
                atom_async_copy_b = cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    self.fp8_dtype,
                    num_bits_per_copy=copy_bits_in,
                )
                # Use a single predicated cp.async copy to avoid generating dual LDGSTS
                # instructions from control-flow `if/else` around `cute.copy`.
                # Make the predicate a broadcasted (stride-0) vector to match the expected
                # pred shape for vectorized copies without needing per-element fills.
                pred_in_layout = cute.make_layout((vec_size_in,), stride=(0,))
                pred_in = cute.make_rmem_tensor(pred_in_layout, Boolean)
                pred_in[0] = False

                align_bytes_in = vec_size_in * int(self.fp8_dtype.width // 8)
                element_bytes_fp8 = cutlass.Int64(self.fp8_dtype.width // 8)

                mC_base_i64 = mC.iterator.toint()
                mA_base_i64 = mAbits.iterator.toint()
                mB_base_i64 = mBbits.iterator.toint()

                stride_be_u8 = cutlass.Int64(mBbits.stride[0])
                stride_bn_u8 = cutlass.Int64(mBbits.stride[1])
                stride_bk_u8 = cutlass.Int64(mBbits.stride[2])

                block_vec_k = block_k // vec_size_in
                total_vec_a = block_m * block_vec_k
                total_vec_b = block_n * block_vec_k

                k_tiles = cute.ceil_div(K, block_k)

                # Cache per-thread row metadata for A to avoid reloading routing metadata
                # in the inner K-tile loops.
                iters_a = total_vec_a // num_threads
                r_base_a = Int32(tx // Int32(block_vec_k))
                r_stride_a = Int32(num_threads // block_vec_k)
                kvec_a = Int32(tx - r_base_a * Int32(block_vec_k))
                aid_row_r = cute.make_rmem_tensor((iters_a,), Int32)
                for it_a in cutlass.range_constexpr(iters_a):
                    r_a = r_base_a + Int32(it_a) * r_stride_a
                    aid_row_r[it_a] = sAid[r_a]

                # Prologue: prefetch the first `num_stages` K tiles.
                for stage_prefetch in cutlass.range_constexpr(num_stages):
                    tile_idx_prefetch = Int32(stage_prefetch)
                    k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                    tile_in_range = tile_idx_prefetch < k_tiles

                    # A tile: [block_m, block_k] via cp.async.
                    for it_a in cutlass.range_constexpr(iters_a):
                        r_a = r_base_a + Int32(it_a) * r_stride_a
                        k_a = Int32(kvec_a * Int32(vec_size_in))
                        kg_a = k_start_prefetch + k_a

                        aid_a = aid_row_r[it_a]
                        valid_row_a = aid_a < num_valid_tokens
                        tok_a = aid_a // Int32(self.top_k)

                        valid_a = tile_in_range and valid_row_a and (kg_a < K)
                        g_off_bytes_a = (
                            (cutlass.Int64(tok_a) * stride_am_elems + cutlass.Int64(kg_a))
                            * element_bytes_fp8
                        )
                        g_ptr_a = cute.make_ptr(
                            self.fp8_dtype,
                            mA_base_i64 + g_off_bytes_a,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes_in,
                        )
                        src_a = cute.make_tensor(g_ptr_a, (vec_size_in,))

                        s_linear_a = Int32(sA_layout((Int32(r_a), k_a, Int32(stage_prefetch))))
                        s_ptr_a = cute.make_ptr(
                            self.fp8_dtype,
                            sA.iterator.toint() + cutlass.Int64(s_linear_a) * element_bytes_fp8,
                            cute.AddressSpace.smem,
                            assumed_align=align_bytes_in,
                        )
                        dst_a = cute.make_tensor(s_ptr_a, (vec_size_in,))
                        pred_in[0] = valid_a
                        cute.copy(atom_async_copy_a, src_a, dst_a, pred=pred_in)

                    # B tile: [block_n, block_k] via cp.async.
                    iters_b = total_vec_b // num_threads
                    for it_b in cutlass.range_constexpr(iters_b):
                        vec_linear_b = tx + Int32(it_b * num_threads)
                        n_b = vec_linear_b // block_vec_k
                        kvec_b = vec_linear_b - n_b * block_vec_k
                        k_b = Int32(kvec_b * vec_size_in)
                        ng_b = n_start + Int32(n_b)
                        kg_b = k_start_prefetch + k_b

                        valid_b = tile_in_range and full_n_tile and (kg_b < K)
                        if not full_n_tile:
                            valid_b = tile_in_range and (ng_b < N) and (kg_b < K)
                        base_off_u8 = (
                            cutlass.Int64(expert_id) * stride_be_u8
                            + cutlass.Int64(ng_b) * stride_bn_u8
                            + cutlass.Int64(kg_b) * stride_bk_u8
                        )
                        g_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            mB_base_i64 + base_off_u8,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes_in,
                        )
                        src_b = cute.make_tensor(g_ptr_b, (vec_size_in,))

                        s_linear_b = Int32(sB_layout((Int32(n_b), k_b, Int32(stage_prefetch))))
                        s_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            sB.iterator.toint() + cutlass.Int64(s_linear_b) * element_bytes_fp8,
                            cute.AddressSpace.smem,
                            assumed_align=align_bytes_in,
                        )
                        dst_b = cute.make_tensor(s_ptr_b, (vec_size_in,))
                        pred_in[0] = valid_b
                        cute.copy(atom_async_copy_b, src_b, dst_b, pred=pred_in)

                    cute.arch.cp_async_commit_group()

                stage_idx = Int32(0)
                main_tiles = k_tiles - Int32(num_stages - 1)
                for tile_idx in cutlass.range(main_tiles, unroll=1):
                    cute.arch.cp_async_wait_group(num_stages - 1)
                    cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    cute.arch.barrier()

                    # MMA on the oldest stage in flight.
                    sm90_utils.gemm(
                        tiled_mma,
                        acc,
                        tSrA[None, None, None, stage_idx],
                        tSrB[None, None, None, stage_idx],
                        wg_wait=0,
                    )
                    cute.arch.barrier()

                    next_tile = tile_idx + Int32(num_stages)
                    if next_tile < k_tiles:
                        k_start_next = next_tile * Int32(block_k)

                        # Prefetch next A tile into the stage we just consumed.
                        for it_a2 in cutlass.range_constexpr(iters_a):
                            r_a2 = r_base_a + Int32(it_a2) * r_stride_a
                            k_a2 = Int32(kvec_a * Int32(vec_size_in))
                            kg_a2 = k_start_next + k_a2

                            aid_a2 = aid_row_r[it_a2]
                            valid_row_a2 = aid_a2 < num_valid_tokens
                            tok_a2 = aid_a2 // Int32(self.top_k)

                            valid_a2 = valid_row_a2 and (kg_a2 < K)
                            g_off_bytes_a2 = (
                                (cutlass.Int64(tok_a2) * stride_am_elems + cutlass.Int64(kg_a2))
                                * element_bytes_fp8
                            )
                            g_ptr_a2 = cute.make_ptr(
                                self.fp8_dtype,
                                mA_base_i64 + g_off_bytes_a2,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes_in,
                            )
                            src_a2 = cute.make_tensor(g_ptr_a2, (vec_size_in,))

                            s_linear_a2 = Int32(sA_layout((Int32(r_a2), k_a2, stage_idx)))
                            s_ptr_a2 = cute.make_ptr(
                                self.fp8_dtype,
                                sA.iterator.toint()
                                + cutlass.Int64(s_linear_a2) * element_bytes_fp8,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes_in,
                            )
                            dst_a2 = cute.make_tensor(s_ptr_a2, (vec_size_in,))
                            pred_in[0] = valid_a2
                            cute.copy(atom_async_copy_a, src_a2, dst_a2, pred=pred_in)

                        # Prefetch next B tile into the same stage.
                        iters_b2 = total_vec_b // num_threads
                        for it_b2 in cutlass.range_constexpr(iters_b2):
                            vec_linear_b2 = tx + Int32(it_b2 * num_threads)
                            n_b2 = vec_linear_b2 // block_vec_k
                            kvec_b2 = vec_linear_b2 - n_b2 * block_vec_k
                            k_b2 = Int32(kvec_b2 * vec_size_in)
                            ng_b2 = n_start + Int32(n_b2)
                            kg_b2 = k_start_next + k_b2

                            valid_b2 = full_n_tile and (kg_b2 < K)
                            if not full_n_tile:
                                valid_b2 = (ng_b2 < N) and (kg_b2 < K)
                            base_off_u8_2 = (
                                cutlass.Int64(expert_id) * stride_be_u8
                                + cutlass.Int64(ng_b2) * stride_bn_u8
                                + cutlass.Int64(kg_b2) * stride_bk_u8
                            )
                            g_ptr_b2 = cute.make_ptr(
                                self.fp8_dtype,
                                mB_base_i64 + base_off_u8_2,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes_in,
                            )
                            src_b2 = cute.make_tensor(g_ptr_b2, (vec_size_in,))

                            s_linear_b2 = Int32(sB_layout((Int32(n_b2), k_b2, stage_idx)))
                            s_ptr_b2 = cute.make_ptr(
                                self.fp8_dtype,
                                sB.iterator.toint()
                                + cutlass.Int64(s_linear_b2) * element_bytes_fp8,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes_in,
                            )
                            dst_b2 = cute.make_tensor(s_ptr_b2, (vec_size_in,))
                            pred_in[0] = valid_b2
                            cute.copy(atom_async_copy_b, src_b2, dst_b2, pred=pred_in)

                        cute.arch.cp_async_commit_group()

                    stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

                # Tail drain.
                for drain_idx in cutlass.range_constexpr(num_stages - 1):
                    cute.arch.cp_async_wait_group(num_stages - 2 - drain_idx)
                    cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
                    cute.arch.barrier()

                    sm90_utils.gemm(
                        tiled_mma,
                        acc,
                        tSrA[None, None, None, stage_idx],
                        tSrB[None, None, None, stage_idx],
                        wg_wait=0,
                    )
                    cute.arch.barrier()

                    stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

                # Epilogue: apply scales (and routed weight), then store/scatter.
                cC = cute.make_identity_tensor((block_m, block_n))
                thr_mma = tiled_mma.get_slice(tx)
                tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
                tAcc = fa_utils.make_acc_tensor_mn_view(acc)
                b_scale_r = cute.make_rmem_tensor((cute.size(tAcc.shape[1]),), Float32)
                for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                    n = Int32(tC_coords[0, ni][1])
                    b_scale_r[ni] = sBScale[n]
                gmem_store_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=2 * self.dtype.width,
                )
                gmem_store_atom_scalar = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=self.dtype.width,
                )
                vec_size_out = 2
                src_vec = cute.make_rmem_tensor((vec_size_out,), self.dtype)
                src_scalar = cute.make_rmem_tensor((1,), self.dtype)
                align_bytes_out = vec_size_out * int(self.dtype.width // 8)
                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    if aid < num_valid_tokens:
                        row_scale = Float32(sAScale[m])
                        row_off_bytes = cutlass.Int64(aid) * stride_cm_elems * element_bytes_c
                        if full_n_tile:
                            n_pairs = cute.size(tAcc.shape[1]) // vec_size_out
                            for pi in cutlass.range_constexpr(n_pairs):
                                ni0 = Int32(pi * vec_size_out)
                                n0 = Int32(tC_coords[mi, ni0][1])
                                col0 = n_start + n0
                                src_vec[0] = self.dtype(
                                    Float32(tAcc[mi, ni0]) * row_scale * b_scale_r[ni0]
                                )
                                src_vec[1] = self.dtype(
                                    Float32(tAcc[mi, ni0 + 1]) * row_scale * b_scale_r[ni0 + 1]
                                )
                                g_off_bytes_vec = (
                                    row_off_bytes + cutlass.Int64(col0) * element_bytes_c
                                )
                                g_ptr_vec = cute.make_ptr(
                                    self.dtype,
                                    mC_base_i64 + g_off_bytes_vec,
                                    cute.AddressSpace.gmem,
                                    assumed_align=align_bytes_out,
                                )
                                dst_vec = cute.make_tensor(g_ptr_vec, (vec_size_out,))
                                cute.copy(gmem_store_atom, src_vec, dst_vec)
                        else:
                            for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                                n = Int32(tC_coords[mi, ni][1])
                                col = n_start + n
                                if col < N:
                                    src_scalar[0] = self.dtype(
                                        Float32(tAcc[mi, ni]) * row_scale * b_scale_r[ni]
                                    )
                                    g_off_bytes_scalar = (
                                        row_off_bytes + cutlass.Int64(col) * element_bytes_c
                                    )
                                    g_ptr_scalar = cute.make_ptr(
                                        self.dtype,
                                        mC_base_i64 + g_off_bytes_scalar,
                                        cute.AddressSpace.gmem,
                                        assumed_align=int(self.dtype.width // 8),
                                    )
                                    dst_scalar = cute.make_tensor(g_ptr_scalar, (1,))
                                    cute.copy(gmem_store_atom_scalar, src_scalar, dst_scalar)


_CUTE_TOP_K_UP_DECODE = 8

# Cache compiled variants keyed by (kind, config). We only support two decode kernels:
# - up-proj: mul_routed_weight=False, top_k=8
# - down-proj: mul_routed_weight=True, top_k=1
_COMPILE_CACHE: Dict[Tuple[str, CuteMoeConfig], Any] = {}
_COMPILE_CACHE_FP8: Dict[Tuple[str, CuteMoeConfig], Any] = {}

# Enable JIT compilation for autotuning (set KESTREL_CUTE_MOE_JIT=1)
_ENABLE_JIT = os.environ.get("KESTREL_CUTE_MOE_JIT", "0") == "1"


def _invoke_cute_moe_impl(
    kind: str,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: CuteMoeConfig,
) -> None:
    if A.dtype != torch.bfloat16:
        raise ValueError(f"CuTe fused MoE supports bfloat16 only (got {A.dtype})")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A/B/C must be CUDA tensors")
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if B.ndim != 3:
        raise ValueError("B must be 3D [E, N, K]")
    if C.ndim != 3:
        raise ValueError("C must be 3D [M, top_k, N]")
    if B.stride(-1) != 1:
        raise ValueError("B must be contiguous in the last dimension (K)")
    if C.stride(-1) != 1:
        raise ValueError("C must be contiguous in the last dimension (N)")
    if B.shape[2] != A.shape[1]:
        raise ValueError("A and B must have the same K dimension")
    if sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids and expert_ids must be int32")
    if num_tokens_post_padded.dtype != torch.int32:
        raise ValueError("num_tokens_post_padded must be int32")
    if mul_routed_weight:
        if topk_weights is None:
            raise ValueError("topk_weights is required when mul_routed_weight=True")
    else:
        topk_weights = torch.empty((0,), device=A.device, dtype=A.dtype)
    if topk_weights is None:
        raise ValueError("topk_weights must be set (internal error)")

    # Match Triton's EM shrink for tiny decode batches: avoid launching blocks that will
    # immediately exit due to routing padding. This is safe because `num_tokens_post_padded`
    # is always <= num_valid_tokens * block_m.
    num_valid_tokens = int(A.shape[0]) * int(top_k)
    em_launch = min(int(sorted_token_ids.numel()), num_valid_tokens * int(config.block_m))
    if em_launch < int(sorted_token_ids.numel()):
        sorted_token_ids = sorted_token_ids[:em_launch]
        m_blocks = (em_launch + int(config.block_m) - 1) // int(config.block_m)
        expert_ids = expert_ids[:m_blocks]

    _ensure_cutlass_initialized()
    _maybe_set_device_cache_config()

    # Flatten output to [M_assignments, N] like the Triton kernel expects.
    C2d = C.view(-1, C.shape[-1])

    key = (kind, config)

    if key not in _COMPILE_CACHE:
        if _ENABLE_JIT:
            # JIT compile for autotuning
            from cutlass import BFloat16

            op = _FusedMoeMatmulCuTe(
                BFloat16, config, mul_routed_weight=mul_routed_weight, top_k=top_k
            )
            a_cute = _to_cute_tensor_2d_contig(A)
            b_cute = _to_cute_tensor_3d_last_contig(B)
            c_cute = _to_cute_tensor_2d_contig(C2d)
            sorted_cute = _to_cute_tensor_1d_i32(sorted_token_ids)
            expert_cute = _to_cute_tensor_1d_i32(expert_ids)
            post_cute = _to_cute_tensor_scalar_i32(num_tokens_post_padded)
            topk_w_cute = _to_cute_tensor_1d_contig(topk_weights)
            # Use env stream so TVM-FFI auto-picks current CUDA stream (matches precompiled)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            compiled = cute.compile(
                op,
                a_cute,
                b_cute,
                c_cute,
                topk_w_cute,
                sorted_cute,
                expert_cute,
                post_cute,
                stream_fake,
                options="--enable-tvm-ffi",
            )
            _set_compiled_kernel_shared_carveout(compiled)
            _COMPILE_CACHE[key] = compiled
        else:
            # Load precompiled kernel
            precompiled = _load_precompiled_kernel(kind, config)
            if precompiled is not None:
                _COMPILE_CACHE[key] = precompiled
            else:
                arch = _get_cuda_arch()
                raise RuntimeError(
                    f"No precompiled kernel for cute_moe(kind={kind}, config={config}, arch={arch}). "
                    f"Run precompile_cute_moe.py on this architecture to generate it."
                )

    # TVM-FFI handles PyTorch tensor conversion automatically
    _COMPILE_CACHE[key](
        A,
        B,
        C2d,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _invoke_cute_moe_fp8_impl(
    kind: str,
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: CuteMoeConfig,
) -> None:
    if A_fp8_bits.dtype != torch.uint8:
        raise ValueError(f"Expected FP8 activation bits as uint8 (got {A_fp8_bits.dtype})")
    if A_scale.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Expected A_scale float16/float32 (got {A_scale.dtype})")
    if B_fp8_bits.dtype != torch.uint8:
        raise ValueError(f"Expected FP8 weight bits as uint8 (got {B_fp8_bits.dtype})")
    if B_scale.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Expected B_scale float16/float32 (got {B_scale.dtype})")
    if C.dtype != torch.bfloat16:
        raise ValueError(f"CuTe fused MoE FP8 expects bfloat16 C (got {C.dtype})")
    if not (
        A_fp8_bits.is_cuda
        and A_scale.is_cuda
        and B_fp8_bits.is_cuda
        and B_scale.is_cuda
        and C.is_cuda
    ):
        raise ValueError("A_fp8_bits/A_scale/B_fp8_bits/B_scale/C must be CUDA tensors")
    if A_fp8_bits.ndim != 2:
        raise ValueError("A_fp8_bits must be 2D")
    if A_scale.ndim != 1:
        raise ValueError("A_scale must be 1D")
    if B_fp8_bits.ndim != 3:
        raise ValueError("B_fp8_bits must be 3D [E, N, K]")
    if B_scale.ndim != 2:
        raise ValueError("B_scale must be 2D [E, N]")
    if C.ndim != 3:
        raise ValueError("C must be 3D [M, top_k, N]")
    if A_fp8_bits.stride(-1) != 1:
        raise ValueError("A_fp8_bits must be contiguous in the last dimension (K)")
    if B_fp8_bits.stride(-1) != 1:
        raise ValueError("B_fp8_bits must be contiguous in the last dimension (K)")
    if C.stride(-1) != 1:
        raise ValueError("C must be contiguous in the last dimension (N)")
    if A_scale.shape[0] != A_fp8_bits.shape[0]:
        raise ValueError("A_scale must have length matching A_fp8_bits.shape[0]")
    if B_fp8_bits.shape[2] != A_fp8_bits.shape[1]:
        raise ValueError("A_fp8_bits and B_fp8_bits must have the same K dimension")
    if int(A_fp8_bits.shape[1]) % int(config.block_k) != 0:
        raise ValueError("CuTe FP8 kernel requires K divisible by block_k")
    if int(config.block_k) % 32 != 0:
        raise ValueError("CuTe FP8 kernel requires block_k divisible by 32 for WGMMA")
    if int(config.block_m) % 64 != 0:
        raise ValueError("CuTe FP8 kernel requires block_m divisible by 64 for WGMMA")
    if int(config.num_warps) != 4:
        raise ValueError("CuTe FP8 kernel requires num_warps=4 for a single warpgroup")
    if B_scale.shape[0] != B_fp8_bits.shape[0] or B_scale.shape[1] != B_fp8_bits.shape[1]:
        raise ValueError("B_scale must have shape [E, N] matching B_fp8_bits")
    if B_scale.stride(-1) != 1:
        raise ValueError("B_scale must be contiguous in the last dimension (N)")
    if sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids and expert_ids must be int32")
    if num_tokens_post_padded.dtype != torch.int32:
        raise ValueError("num_tokens_post_padded must be int32")
    if mul_routed_weight:
        if topk_weights is None:
            raise ValueError("topk_weights is required when mul_routed_weight=True")
    else:
        topk_weights = torch.empty((0,), device=C.device, dtype=C.dtype)
    if topk_weights is None:
        raise ValueError("topk_weights must be set (internal error)")

    num_valid_tokens = int(A_fp8_bits.shape[0]) * int(top_k)
    em_launch = min(int(sorted_token_ids.numel()), num_valid_tokens * int(config.block_m))
    if em_launch < int(sorted_token_ids.numel()):
        sorted_token_ids = sorted_token_ids[:em_launch]
        m_blocks = (em_launch + int(config.block_m) - 1) // int(config.block_m)
        expert_ids = expert_ids[:m_blocks]

    _ensure_cutlass_initialized()
    _maybe_set_device_cache_config()

    C2d = C.view(-1, C.shape[-1])

    dtype = cutlass.BFloat16
    fp8_dtype = cutlass.Float8E4M3FN
    key = (kind, config)

    a_bits_cute = _to_cute_tensor_2d_contig_u8(A_fp8_bits)
    a_scale_cute = _to_cute_tensor_1d_contig(A_scale.to(dtype=torch.float32))
    b_bits_cute = _to_cute_tensor_3d_last_contig_u8(B_fp8_bits)
    b_scale_cute = _to_cute_tensor_2d_contig(B_scale.to(dtype=torch.float32))
    c_cute = _to_cute_tensor_2d_contig(C2d)
    sorted_cute = _to_cute_tensor_1d_i32(sorted_token_ids)
    expert_cute = _to_cute_tensor_1d_i32(expert_ids)
    post_cute = _to_cute_tensor_scalar_i32(num_tokens_post_padded)
    topk_w_cute = _to_cute_tensor_1d_contig(topk_weights)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _COMPILE_CACHE_FP8:
        op = _FusedMoeMatmulCuTeFp8(
            dtype, fp8_dtype, config, mul_routed_weight=mul_routed_weight, top_k=top_k
        )
        compiled = cute.compile(
            op,
            a_bits_cute,
            a_scale_cute,
            b_bits_cute,
            b_scale_cute,
            c_cute,
            topk_w_cute,
            sorted_cute,
            expert_cute,
            post_cute,
            stream,
            options="--enable-tvm-ffi",
        )
        _set_compiled_kernel_shared_carveout(compiled)
        _COMPILE_CACHE_FP8[key] = compiled

    _COMPILE_CACHE_FP8[key](
        a_bits_cute,
        a_scale_cute,
        b_bits_cute,
        b_scale_cute,
        c_cute,
        topk_w_cute,
        sorted_cute,
        expert_cute,
        post_cute,
        stream,
    )


def invoke_cute_moe_up_fp8(
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE up-projection with FP8 activations+weights (W8A8).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A_fp8_bits: [M, hidden_size], B_fp8_bits: [E, intermediate_size*2, hidden_size]
        num_tokens = A_fp8_bits.shape[0]
        hidden_size = A_fp8_bits.shape[1]
        num_experts = B_fp8_bits.shape[0]
        intermediate_size = B_fp8_bits.shape[1] // 2
        config = get_cute_moe_config(
            "up", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype="fp8",
        )
    _invoke_cute_moe_fp8_impl(
        "up",
        A_fp8_bits,
        A_scale,
        B_fp8_bits,
        B_scale,
        C,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=_CUTE_TOP_K_UP_DECODE,
        config=config,
    )


def invoke_cute_moe_down_fp8(
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE down-projection with FP8 activations+weights (W8A8).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A_fp8_bits: [M*top_k, intermediate_size], B_fp8_bits: [E, hidden_size, intermediate_size]
        # C: [M, top_k, hidden_size]
        num_tokens = C.shape[0]
        hidden_size = B_fp8_bits.shape[1]
        num_experts = B_fp8_bits.shape[0]
        intermediate_size = B_fp8_bits.shape[2]
        config = get_cute_moe_config(
            "down", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype="fp8",
        )
    _invoke_cute_moe_fp8_impl(
        "down",
        A_fp8_bits,
        A_scale,
        B_fp8_bits,
        B_scale,
        C,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
    )


def invoke_cute_moe_up(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE up-projection (no routed-weight scaling).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A: [M, hidden_size], B: [E, intermediate_size*2, hidden_size]
        num_tokens = A.shape[0]
        hidden_size = A.shape[1]
        num_experts = B.shape[0]
        intermediate_size = B.shape[1] // 2  # gate+up are fused
        config = get_cute_moe_config(
            "up", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    _invoke_cute_moe_impl(
        "up",
        A,
        B,
        C,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=_CUTE_TOP_K_UP_DECODE,
        config=config,
    )


def invoke_cute_moe_down(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE down-projection (includes routed-weight scaling).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A: [M*top_k, intermediate_size], B: [E, hidden_size, intermediate_size]
        # C: [M, top_k, hidden_size] - use C to get num_tokens since A is expanded
        num_tokens = C.shape[0]
        hidden_size = B.shape[1]
        num_experts = B.shape[0]
        intermediate_size = B.shape[2]
        config = get_cute_moe_config(
            "down", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    _invoke_cute_moe_impl(
        "down",
        A,
        B,
        C,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
    )


__all__ = [
    "CuteMoeConfig",
    "get_cute_moe_config",
    "invoke_cute_moe_up",
    "invoke_cute_moe_down",
    "invoke_cute_moe_up_fp8",
    "invoke_cute_moe_down_fp8",
]
