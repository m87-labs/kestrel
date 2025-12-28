"""CuTe DSL fused MoE GEMM kernel (H100 / SM90-focused).

This implements the core routed expert GEMM used by `kestrel.fused_moe`:
  out[sorted_token_ids] = x @ W_expert.T  (optionally scaled by routing weights)

The reference implementation lives in `kestrel/fused_moe/kernels.py` (Triton).
This file provides an equivalent kernel using NVIDIA's CuTe DSL.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from kestrel_kernels.flash_attn.cute import ampere_helpers
from kestrel_kernels.flash_attn.cute import utils as fa_utils

_CUTLASS_INITIALIZED = False
_CUTE_KERNEL_ATTRS_SET: set[str] = set()
_DEVICE_CACHE_CONFIG_SET = False


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
class FusedMoeCuTeConfig:
    # Tile sizes (tuned for Moondream MoE shapes; adjust via benchmarking).
    block_m: int = 16
    block_n: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2

    @property
    def num_threads(self) -> int:
        return 32 * self.num_warps


def _to_cute_tensor_1d_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 1D tensor")
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_1d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous 1D tensor")
    return from_dlpack(t.detach(), assumed_align=assumed_align).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_scalar_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or t.numel() != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 tensor with shape (1,)")
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_2d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 2 or not t.is_contiguous():
        raise ValueError("Expected row-major contiguous 2D tensor")
    # The MoE kernel relies on 128-bit vectorized accesses (vec_size=8 for BF16/FP16).
    # Add compactness/divisibility hints so the compiler can prove alignment for cp.async.
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=1, alignment=assumed_align, divisibility=8
    )


def _to_cute_tensor_3d_last_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 3 or t.stride(-1) != 1:
        raise ValueError("Expected 3D tensor contiguous in the last dim")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=2, alignment=assumed_align, divisibility=8
    )


class _FusedMoeMatmulCuTe:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        config: FusedMoeCuTeConfig,
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


_CUTE_TOP_K_UP_DECODE = 8

# Cache compiled variants keyed by (kind, config). We only support two decode kernels:
# - up-proj: mul_routed_weight=False, top_k=8
# - down-proj: mul_routed_weight=True, top_k=1
_COMPILE_CACHE: Dict[Tuple[str, FusedMoeCuTeConfig], Any] = {}


def _invoke_fused_moe_kernel_cute_impl(
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
    config: FusedMoeCuTeConfig,
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

    dtype = cutlass.BFloat16
    key = (kind, config)

    a_cute = _to_cute_tensor_2d_contig(A)
    b_cute = _to_cute_tensor_3d_last_contig(B)
    c_cute = _to_cute_tensor_2d_contig(C2d)
    sorted_cute = _to_cute_tensor_1d_i32(sorted_token_ids)
    expert_cute = _to_cute_tensor_1d_i32(expert_ids)
    post_cute = _to_cute_tensor_scalar_i32(num_tokens_post_padded)
    topk_w_cute = _to_cute_tensor_1d_contig(topk_weights)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if key not in _COMPILE_CACHE:
        op = _FusedMoeMatmulCuTe(dtype, config, mul_routed_weight=mul_routed_weight, top_k=top_k)
        compiled = cute.compile(
            op,
            a_cute,
            b_cute,
            c_cute,
            topk_w_cute,
            sorted_cute,
            expert_cute,
            post_cute,
            stream,
        )
        _set_compiled_kernel_shared_carveout(compiled)
        _COMPILE_CACHE[key] = compiled

    _COMPILE_CACHE[key](
        a_cute,
        b_cute,
        c_cute,
        topk_w_cute,
        sorted_cute,
        expert_cute,
        post_cute,
        stream,
    )


def invoke_fused_moe_kernel_cute_up_decode(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: FusedMoeCuTeConfig = FusedMoeCuTeConfig(),
) -> None:
    """Decode-specialized CuTe fused MoE up-projection (no routed-weight scaling)."""
    _invoke_fused_moe_kernel_cute_impl(
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


def invoke_fused_moe_kernel_cute_down_decode(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: FusedMoeCuTeConfig = FusedMoeCuTeConfig(),
) -> None:
    """Decode-specialized CuTe fused MoE down-projection (includes routed-weight scaling)."""
    _invoke_fused_moe_kernel_cute_impl(
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


def invoke_fused_moe_kernel_cute(
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
    config: FusedMoeCuTeConfig = FusedMoeCuTeConfig(),
) -> None:
    """Legacy wrapper for decode-only CuTe fused MoE.

    Supported combinations:
      - moe_up: mul_routed_weight=False, top_k=8
      - moe_down: mul_routed_weight=True, top_k=1
    """
    if mul_routed_weight:
        if int(top_k) != 1:
            raise ValueError("CuTe moe_down expects top_k=1")
        if topk_weights is None:
            raise ValueError("topk_weights is required when mul_routed_weight=True")
        return invoke_fused_moe_kernel_cute_down_decode(
            A,
            B,
            C,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=config,
        )
    if int(top_k) != _CUTE_TOP_K_UP_DECODE:
        raise ValueError(f"CuTe moe_up expects top_k={_CUTE_TOP_K_UP_DECODE}")
    return invoke_fused_moe_kernel_cute_up_decode(
        A,
        B,
        C,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=config,
    )


__all__ = [
    "FusedMoeCuTeConfig",
    "invoke_fused_moe_kernel_cute_up_decode",
    "invoke_fused_moe_kernel_cute_down_decode",
    "invoke_fused_moe_kernel_cute",
]
