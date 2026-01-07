"""CuTe MoE kernel utilities, PTX helpers, and initialization."""

import math
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack

from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.config import CuteMoeConfig, _get_cuda_arch

from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


# =============================================================================
# PTX Inline Assembly Helpers
# =============================================================================


@dsl_user_op
def store_streaming_b32(value: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store 32 bits (e.g. 2xBF16) with cache streaming hint (st.global.cs).

    Bypasses L1 and marks for early L2 eviction - useful for scattered write-only stores.
    """
    llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.cs.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_streaming_b128(
    v0: Int32, v1: Int32, v2: Int32, v3: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    """Store 128 bits (e.g. 8xBF16) with cache streaming hint (st.global.cs.v4.b32).

    Stores 4x32-bit values in a single 128-bit transaction.
    Bypasses L1 and marks for early L2 eviction - useful for scattered write-only stores.
    """
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shfl_sync_idx_b32(value: Int32, src_lane: Int32, *, loc=None, ip=None) -> Int32:
    """Warp shuffle - read value from src_lane within the warp.

    Uses shfl.sync.idx.b32 with full mask (0xffffffff) for all threads participating.
    The fourth operand 0x1f means width=32 (full warp).
    The fifth operand 0xffffffff is the membership mask (all threads participate).
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(value).ir_value(loc=loc, ip=ip), Int32(src_lane).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        "=r,r,r",
        has_side_effects=True,  # Synchronization point
        loc=loc,
        ip=ip,
    )
    return Int32(result)


def tiled_copy_2d_bypass(
    dtype: Type[cutlass.Numeric], major_mode_size: int, num_threads: int
) -> cute.TiledCopy:
    """Like copy_utils.tiled_copy_2d but with L1 cache bypass for async copies.

    Uses LoadCacheMode.GLOBAL which generates cp.async.cg (bypass L1, cache in L2),
    matching Triton's LDGSTS.E.BYPASS.128 pattern for gathered/scattered access.
    """
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


# =============================================================================
# WGMMA Helper
# =============================================================================


@cute.jit
def _wgmma_gemm_no_fence(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


# =============================================================================
# Global State and Initialization
# =============================================================================


_CUTLASS_INITIALIZED = False
_CUTE_KERNEL_ATTRS_SET: set[str] = set()
_DEVICE_CACHE_CONFIG_SET = False

# Precompiled kernel registry
_precompiled_cache: Dict[Tuple[str, CuteMoeConfig, int, int], Any] = {}
_precompiled_cache_fp8: Dict[Tuple[str, CuteMoeConfig], Any] = {}
_precompiled_dir = Path(__file__).parent.parent / "precompiled"


def _get_precompiled_kernel_path(
    kind: str, config: CuteMoeConfig, N: int, K: int
) -> Path | None:
    """Get path to precompiled kernel if it exists."""
    arch = _get_cuda_arch()
    filename = (
        f"cute_moe_{kind}_m{config.block_m}_n{config.block_n}_k{config.block_k}"
        f"_N{N}_K{K}_w{config.num_warps}_s{config.num_stages}_{arch}.so"
    )
    path = _precompiled_dir / filename
    return path if path.exists() else None


def _load_precompiled_kernel(kind: str, config: CuteMoeConfig, N: int, K: int):
    """Load a precompiled kernel if available, return None otherwise."""
    compile_key = (kind, config, N, K)

    # Check if already loaded
    if compile_key in _precompiled_cache:
        return _precompiled_cache[compile_key]

    # Check if precompiled file exists
    so_path = _get_precompiled_kernel_path(kind, config, N, K)
    if so_path is None:
        return None

    # Load the module
    mod = cute.runtime.load_module(str(so_path))

    # Get the function by its exported name
    arch = _get_cuda_arch()
    function_name = (
        f"cute_moe_{kind}_m{config.block_m}_n{config.block_n}_k{config.block_k}"
        f"_N{N}_K{K}_w{config.num_warps}_s{config.num_stages}_{arch}"
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


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================


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
