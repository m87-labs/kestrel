"""CuTe MoE FP8 WGMMA kernel (SM90, block_m >= 64)."""

from typing import TYPE_CHECKING, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.arch import ProxyKind, SharedSpace
from cutlass import Boolean, Float32, Int32, const_expr
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.utils import LayoutEnum
from kestrel_kernels.flash_attn.cute import hopper_helpers as sm90_utils

from kestrel_kernels.flash_attn.cute import copy_utils
from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.utils import (
    store_streaming_b32,
    store_streaming_b128,
    shfl_sync_idx_b32,
    tiled_copy_2d_bypass,
)

if TYPE_CHECKING:
    from kestrel_kernels.cute_moe.config import CuteMoeConfig


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
        dtype: Type[cutlass.Numeric],
        fp8_dtype: Type[cutlass.Numeric],
        config: "CuteMoeConfig",
        *,
        mul_routed_weight: bool,
        top_k: int,
        N: int,
        K: int,
    ) -> None:
        self.dtype = dtype
        self.fp8_dtype = fp8_dtype
        self.config = config
        self.mul_routed_weight = mul_routed_weight
        self.top_k = int(top_k)
        self.N = int(N)
        self.K = int(K)

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
        sTok_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        sAScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]

        sBScale_elems = block_n
        sBScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sBScale_elems], 16]

        @cute.struct
        class SharedStorage:
            sA: sA_struct
            sB: sB_struct
            sAid: sAid_struct
            sTok: sTok_struct
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
            self.N,
            self.K,
            SharedStorage,
        ).launch(
            grid=(grid_n, grid_m, 1),  # N-first for better weight reuse
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
        N: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        tx, _, _ = cute.arch.thread_idx()
        pid_n, pid_m, _ = cute.arch.block_idx()  # Swapped for N-first scheduling

        block_m = self.config.block_m
        block_n = self.config.block_n
        block_k = self.config.block_k
        num_threads = self.config.num_threads
        num_stages = self.config.num_stages

        n_start = pid_n * block_n
        row_start = pid_m * block_m

        num_tokens_post_padded = Int32(mNumTokensPostPadded[0])
        block_active = row_start < num_tokens_post_padded

        N_const = Int32(N)
        K_const = Int32(K)
        num_valid_tokens = Int32(mAbits.shape[0]) * Int32(self.top_k)
        expert_id = Int32(mExpertIds[pid_m])

        if const_expr(N % block_n == 0):
            full_n_tile = True
        else:
            full_n_tile = (n_start + Int32(block_n)) <= N_const

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Shared routing/scale metadata for the M rows in this CTA.
        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        sTok = storage.sTok.get_tensor(s_meta_layout)
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

        element_bytes_c = cutlass.Int64(self.dtype.width // 8)
        stride_cm_elems = cutlass.Int64(mC.stride[0])

        if block_active:
            if expert_id == -1:
                _ = Int32(0)

            else:
                # Load per-row routing metadata + activation scales once per CTA.
                if tx < Int32(block_m):
                    idx = row_start + tx
                    aid = Int32(num_valid_tokens)  # padded sentinel
                    if idx < num_tokens_post_padded:
                        aid = Int32(mSortedTokenIds[idx])
                    sAid[tx] = aid

                    tok = Int32(0)
                    row_scale = Float32(0.0)
                    if aid < num_valid_tokens:
                        if const_expr(self.top_k == 1):
                            tok = aid
                        elif const_expr(self.top_k == 8):
                            tok = aid >> 3
                        else:
                            tok = aid // Int32(self.top_k)
                        row_scale = Float32(mAScale[tok])
                        if const_expr(self.mul_routed_weight):
                            row_scale = row_scale * Float32(mTopkWeights[aid])
                    sTok[tx] = tok
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
                # A warpgroup is 4 warps = 128 threads. Must compute warpgroup index correctly.
                num_threads_per_wg = 128
                warp_group_idx = cute.arch.make_warp_uniform(tx // Int32(num_threads_per_wg))
                warp_group_thread_layout = cute.make_layout(
                    num_threads // num_threads_per_wg,  # Number of warpgroups
                    stride=num_threads_per_wg,
                )
                wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
                tSrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA))
                tSrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB))

                # Use L1 bypass for gathered A tiles (avoids L1 cache thrashing).
                # Create TiledCopy with L1 bypass for FP8 operands.
                tiled_copy_A = tiled_copy_2d_bypass(self.fp8_dtype, block_k, num_threads)
                thr_copy_A = tiled_copy_A.get_slice(tx)
                # Create gather copy function that respects swizzle via partition_D
                num_tokens = Int32(mAbits.shape[0])
                copy_A = copy_utils.gather_m_get_copy_fn(
                    thr_copy_A,
                    mAbits,
                    sA,
                    sTok,  # Row indices (token IDs)
                    num_tokens,
                    K_const,
                )

                # cp.async copy atoms for FP8 B operands (16B vectors).
                vec_size_in = 16
                copy_bits_in = vec_size_in * self.fp8_dtype.width
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
                mB_base_i64 = mBbits.iterator.toint()

                stride_be_u8 = cutlass.Int64(mBbits.stride[0])
                stride_bn_u8 = cutlass.Int64(mBbits.stride[1])
                stride_bk_u8 = cutlass.Int64(mBbits.stride[2])

                block_vec_k = block_k // vec_size_in
                total_vec_b = block_n * block_vec_k

                k_tiles = cute.ceil_div(K, block_k)

                # With wg_wait=1, we need one free stage for writing (the WGMMA we just
                # issued might still be reading). So we only prefetch num_stages-1 tiles.
                use_wgmma_pipelining: cutlass.Constexpr[bool] = num_stages >= 3
                prologue_tiles: cutlass.Constexpr[int] = (
                    num_stages - 1 if use_wgmma_pipelining else num_stages
                )

                # Prologue: prefetch the first tiles.
                for stage_prefetch in cutlass.range_constexpr(prologue_tiles):
                    tile_idx_prefetch = Int32(stage_prefetch)
                    k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                    tile_in_range = tile_idx_prefetch < k_tiles

                    # A tile: use gather copy with L1 bypass to respect swizzled SMEM layout.
                    if tile_in_range:
                        if const_expr(K % block_k == 0):
                            copy_A(tile_idx_prefetch, stage_prefetch)
                        else:
                            copy_A(tile_idx_prefetch, stage_prefetch, pred=True)

                    # B tile: [block_n, block_k] via cp.async.
                    # FIX: use ceil to ensure all vectors are covered when total_vec_b % num_threads != 0
                    iters_b: cutlass.Constexpr[int] = (total_vec_b + num_threads - 1) // num_threads
                    for it_b in cutlass.range_constexpr(iters_b):
                        vec_linear_b = tx + Int32(it_b * num_threads)
                        if vec_linear_b < Int32(total_vec_b):
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

                # With wg_wait=1, we write to a different stage than we read from,
                # because our WGMMA might still be pending (reading the current stage).
                cp_async_wait_val: cutlass.Constexpr[int] = (
                    num_stages - 2 if use_wgmma_pipelining else num_stages - 1
                )
                prefetch_offset: cutlass.Constexpr[int] = (
                    num_stages - 1 if use_wgmma_pipelining else num_stages
                )

                for tile_idx in cutlass.range(main_tiles, unroll=1):
                    cute.arch.cp_async_wait_group(cp_async_wait_val)
                    # NOTE: fence_proxy omitted - cp_async data visible after wait_group.
                    cute.arch.barrier()

                    # MMA on the oldest stage in flight.
                    sm90_utils.gemm(
                        tiled_mma,
                        acc,
                        tSrA[None, None, None, stage_idx],
                        tSrB[None, None, None, stage_idx],
                        wg_wait=1 if use_wgmma_pipelining else 0,
                    )
                    cute.arch.barrier()

                    # Compute write stage: with wg_wait=1, write to a stage that's not
                    # being read by the current (potentially pending) WGMMA.
                    if const_expr(use_wgmma_pipelining):
                        write_stage = (
                            stage_idx - Int32(1)
                            if stage_idx > Int32(0)
                            else Int32(num_stages - 1)
                        )
                    else:
                        write_stage = stage_idx

                    next_tile = tile_idx + Int32(prefetch_offset)
                    if next_tile < k_tiles:
                        k_start_next = next_tile * Int32(block_k)

                        # Prefetch next A tile using gather copy with L1 bypass.
                        if const_expr(K % block_k == 0):
                            copy_A(next_tile, write_stage)
                        else:
                            copy_A(next_tile, write_stage, pred=True)

                        # Prefetch next B tile into the safe write stage.
                        # FIX: use ceil to ensure all vectors are covered
                        iters_b2: cutlass.Constexpr[int] = (total_vec_b + num_threads - 1) // num_threads
                        for it_b2 in cutlass.range_constexpr(iters_b2):
                            vec_linear_b2 = tx + Int32(it_b2 * num_threads)
                            if vec_linear_b2 < Int32(total_vec_b):
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

                                s_linear_b2 = Int32(sB_layout((Int32(n_b2), k_b2, write_stage)))
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
                    # NOTE: fence_proxy omitted - cp_async data visible after wait_group.
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

                # Epilogue: apply scales, then scatter stores using 128-bit coalesced writes.
                #
                # WGMMA accumulator layout for SM90:
                # - Every 4 consecutive threads (tid%4 == 0,1,2,3) share the same M row
                # - Within each 4-thread group, N values are distributed as:
                #   Thread 0: N=0,1, 8,9, 16,17, ...  (pairs at stride 8)
                #   Thread 1: N=2,3, 10,11, 18,19, ...
                #   Thread 2: N=4,5, 12,13, 20,21, ...
                #   Thread 3: N=6,7, 14,15, 22,23, ...
                # - Each thread has n_per_m/2 "chunks", each chunk = 2 consecutive N values
                # - For 128-bit stores (8 BF16), we gather from 4 threads via warp shuffle
                #
                cC = cute.make_identity_tensor((block_m, block_n))
                thr_mma = tiled_mma.get_slice(tx)
                tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
                tAcc = fa_utils.make_acc_tensor_mn_view(acc)

                # Load per-column B scales into registers
                b_scale_r = cute.make_rmem_tensor((cute.size(tAcc.shape[1]),), Float32)
                for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                    n = Int32(tC_coords[0, ni][1])
                    b_scale_r[ni] = sBScale[n]

                n_per_m: cutlass.Constexpr[int] = cute.size(tAcc.shape[1])
                num_chunks: cutlass.Constexpr[int] = n_per_m // 2  # Each chunk = 2 N values

                # Warp shuffle setup: threads 0-3, 4-7, etc. form groups
                lane = tx % 32
                lane_in_group = lane % 4
                group_base_lane = lane - lane_in_group

                # Precompute shuffle source lanes for 3-round ring exchange
                src_round1 = Int32(group_base_lane) + ((lane_in_group + 3) & 3)
                src_round2 = Int32(group_base_lane) + ((lane_in_group + 2) & 3)
                src_round3 = Int32(group_base_lane) + ((lane_in_group + 1) & 3)

                align_bytes_128 = 16
                element_bytes_const: cutlass.Constexpr[int] = 2  # BF16 = 2 bytes

                # 4-thread parallel stores: process 4 chunks at once
                num_chunk_groups: cutlass.Constexpr[int] = num_chunks // 4

                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    valid_row = aid < num_valid_tokens

                    # Use row_scale=0 for invalid rows so shuffles execute uniformly
                    row_scale = Float32(0.0)
                    if valid_row:
                        row_scale = Float32(sAScale[m])

                    row_off_bytes = cutlass.Int64(aid) * stride_cm_elems * element_bytes_c

                    if full_n_tile:
                        # Process 4 chunks at a time with all 4 threads storing in parallel
                        for chunk_group in cutlass.range_constexpr(num_chunk_groups):
                            base_chunk: cutlass.Constexpr[int] = chunk_group * 4

                            # Each thread computes packed values for all 4 chunks
                            ni0_c0: cutlass.Constexpr[int] = (base_chunk + 0) * 2
                            ni0_c1: cutlass.Constexpr[int] = (base_chunk + 1) * 2
                            ni0_c2: cutlass.Constexpr[int] = (base_chunk + 2) * 2
                            ni0_c3: cutlass.Constexpr[int] = (base_chunk + 3) * 2

                            # Apply row_scale and per-column b_scale, then pack
                            packed_c0 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c0]) * row_scale * b_scale_r[ni0_c0],
                                Float32(tAcc[mi, ni0_c0 + 1]) * row_scale * b_scale_r[ni0_c0 + 1],
                                self.dtype,
                            )
                            packed_c1 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c1]) * row_scale * b_scale_r[ni0_c1],
                                Float32(tAcc[mi, ni0_c1 + 1]) * row_scale * b_scale_r[ni0_c1 + 1],
                                self.dtype,
                            )
                            packed_c2 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c2]) * row_scale * b_scale_r[ni0_c2],
                                Float32(tAcc[mi, ni0_c2 + 1]) * row_scale * b_scale_r[ni0_c2 + 1],
                                self.dtype,
                            )
                            packed_c3 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c3]) * row_scale * b_scale_r[ni0_c3],
                                Float32(tAcc[mi, ni0_c3 + 1]) * row_scale * b_scale_r[ni0_c3 + 1],
                                self.dtype,
                            )

                            # 3-shuffle ring exchange to gather packed values
                            # Initialize with local contribution
                            s0 = packed_c0
                            s1 = packed_c1
                            s2 = packed_c2
                            s3 = packed_c3

                            # Round 1: offset = 1
                            send1 = packed_c1
                            if lane_in_group == 1:
                                send1 = packed_c2
                            elif lane_in_group == 2:
                                send1 = packed_c3
                            elif lane_in_group == 3:
                                send1 = packed_c0
                            recv1 = shfl_sync_idx_b32(send1, src_round1)
                            if lane_in_group == 0:
                                s3 = recv1
                            elif lane_in_group == 1:
                                s0 = recv1
                            elif lane_in_group == 2:
                                s1 = recv1
                            else:
                                s2 = recv1

                            # Round 2: offset = 2
                            send2 = packed_c2
                            if lane_in_group == 1:
                                send2 = packed_c3
                            elif lane_in_group == 2:
                                send2 = packed_c0
                            elif lane_in_group == 3:
                                send2 = packed_c1
                            recv2 = shfl_sync_idx_b32(send2, src_round2)
                            if lane_in_group == 0:
                                s2 = recv2
                            elif lane_in_group == 1:
                                s3 = recv2
                            elif lane_in_group == 2:
                                s0 = recv2
                            else:
                                s1 = recv2

                            # Round 3: offset = 3
                            send3 = packed_c3
                            if lane_in_group == 1:
                                send3 = packed_c0
                            elif lane_in_group == 2:
                                send3 = packed_c1
                            elif lane_in_group == 3:
                                send3 = packed_c2
                            recv3 = shfl_sync_idx_b32(send3, src_round3)
                            if lane_in_group == 0:
                                s1 = recv3
                            elif lane_in_group == 1:
                                s2 = recv3
                            elif lane_in_group == 2:
                                s3 = recv3
                            else:
                                s0 = recv3

                            # Compute store address: each thread stores to different chunk
                            my_n_offset = Int32(base_chunk * 8) + lane_in_group * 8
                            col0 = n_start + my_n_offset

                            if valid_row:
                                if const_expr(N % block_n == 0) or (col0 + 7 < N_const):
                                    g_off_bytes = row_off_bytes + cutlass.Int64(col0) * element_bytes_const
                                    g_ptr = cute.make_ptr(
                                        self.dtype,
                                        mC_base_i64 + g_off_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=align_bytes_128,
                                    )
                                    store_streaming_b128(s0, s1, s2, s3, g_ptr)
                                else:
                                    # Boundary: fall back to 32-bit stores
                                    for ni_local in cutlass.range_constexpr(4):
                                        col_check = col0 + ni_local * 2
                                        if col_check < N_const:
                                            g_off = row_off_bytes + cutlass.Int64(col_check) * element_bytes_const
                                            g_ptr_32 = cute.make_ptr(
                                                self.dtype,
                                                mC_base_i64 + g_off,
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            if const_expr(ni_local == 0):
                                                store_streaming_b32(s0, g_ptr_32)
                                            elif const_expr(ni_local == 1):
                                                store_streaming_b32(s1, g_ptr_32)
                                            elif const_expr(ni_local == 2):
                                                store_streaming_b32(s2, g_ptr_32)
                                            else:
                                                store_streaming_b32(s3, g_ptr_32)
                    else:
                        # Non-full tile: scalar stores with bounds checking
                        gmem_store_atom_scalar = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.dtype,
                            num_bits_per_copy=self.dtype.width,
                        )
                        src_scalar = cute.make_rmem_tensor((1,), self.dtype)
                        for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                            n = Int32(tC_coords[mi, ni][1])
                            col = n_start + n
                            if col < N_const:
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
