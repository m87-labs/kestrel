"""CuTe MoE BF16 WGMMA kernel (SM90, block_m >= 128)."""

from typing import TYPE_CHECKING, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import warpgroup

from kestrel_kernels.flash_attn.cute import copy_utils
from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.utils import (
    store_streaming_b32,
    store_streaming_b128,
    shfl_sync_idx_b32,
    tiled_copy_2d_bypass,
    _wgmma_gemm_no_fence,
)

if TYPE_CHECKING:
    from kestrel_kernels.cute_moe.config import CuteMoeConfig


def _should_use_wgmma_bf16(config: "CuteMoeConfig") -> bool:
    """Check if WGMMA BF16 kernel should be used for the given config."""
    return config.kernel_type == "wgmma"


class _FusedMoeMatmulCuTeWgmmaBf16:
    """Routed MoE GEMM with BF16 activations+weights using SM90 WGMMA.

    This is tuned for larger M tiles (block_m >= 64) where warpgroup MMA (WGMMA)
    tends to outperform warp-level MMA on H100.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        config: "CuteMoeConfig",
        *,
        mul_routed_weight: bool,
        top_k: int,
        N: int,
        K: int,
    ) -> None:
        self.dtype = dtype
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
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, block_k),
            self.dtype,
        )
        sA_layout = cute.tile_to_shape(
            s_layout_atom, (block_m, block_k, num_stages), order=(0, 1, 2)
        )
        sB_layout = cute.tile_to_shape(
            s_layout_atom, (block_n, block_k, num_stages), order=(0, 1, 2)
        )

        sA_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sA_layout)], 1024
        ]
        sB_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sB_layout)], 1024
        ]

        sMeta_elems = block_m
        sAid_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        sTok_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        # Pre-computed row base offsets in bytes for efficient gather.
        sArowBase_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int64, sMeta_elems], 16]
        sW_struct = None
        if self.mul_routed_weight:
            # Store routing weights as fp32 to avoid per-element bf16->fp32 conversion in the epilogue.
            sW_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]

        @cute.struct
        class SharedStorage:
            sA: sA_struct
            sB: sB_struct
            sAid: sAid_struct
            sTok: sTok_struct
            sArowBase: sArowBase_struct
            if self.mul_routed_weight:
                sW: sW_struct

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
        # SM90 warpgroup MMA (WGMMA).
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.config.block_m // 64, 1, 1),
            tiler_mn=(64, self.config.block_n),
        )

        block_m = self.config.block_m
        block_n = self.config.block_n
        grid_m = cute.ceil_div(Int32(mSortedTokenIds.shape[0]), block_m)
        grid_n = cute.ceil_div(Int32(self.N), block_n)
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
            self.N,
            self.K,
            SharedStorage,
        ).launch(
            grid=(grid_n, grid_m, 1),  # N-first for better weight reuse (like Triton GROUP_SIZE_M=1)
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
        num_threads_per_wg = 128

        N_const = Int32(N)
        K_const = Int32(K)
        m_count = Int32(mA.shape[0])
        if const_expr(self.top_k == 1):
            num_valid_tokens = m_count
        elif const_expr(self.top_k == 8):
            num_valid_tokens = m_count << 3
        else:
            num_valid_tokens = m_count * Int32(self.top_k)
        expert_id = Int32(mExpertIds[pid_m])
        if const_expr(N % block_n == 0):
            full_n_tile = True
        else:
            full_n_tile = (n_start + Int32(block_n)) <= N_const

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        sTok = storage.sTok.get_tensor(s_meta_layout)
        sArowBase = storage.sArowBase.get_tensor(s_meta_layout)
        if const_expr(self.mul_routed_weight):
            sW = storage.sW.get_tensor(s_meta_layout)
        else:
            sW = None

        # Shared-memory operand tensors (staged).
        s_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, block_k),
            self.dtype,
        )
        sA_layout = cute.tile_to_shape(
            s_layout_atom, (block_m, block_k, num_stages), order=(0, 1, 2)
        )
        sB_layout = cute.tile_to_shape(
            s_layout_atom, (block_n, block_k, num_stages), order=(0, 1, 2)
        )
        # WGMMA expects affine layouts; move swizzle to the pointer (recast) via `swizzle=...`.
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner, dtype=self.dtype)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner, dtype=self.dtype)
        element_bytes = cutlass.Int64(self.dtype.width // 8)
        stride_am_elems = cutlass.Int64(mA.stride[0])
        stride_cm_elems = cutlass.Int64(mC.stride[0])

        # Load per-row routing metadata once per CTA (unconditional, just guarded by tx < block_m).
        if tx < Int32(block_m):
            idx = row_start + tx
            aid = Int32(num_valid_tokens)  # padded sentinel
            if idx < num_tokens_post_padded:
                aid = Int32(mSortedTokenIds[idx])
            sAid[tx] = aid
            tok = Int32(0)
            if aid < num_valid_tokens:
                if const_expr(self.top_k == 1):
                    tok = aid
                elif const_expr(self.top_k == 8):
                    tok = aid >> 3
                else:
                    tok = aid // Int32(self.top_k)
            sTok[tx] = tok

            # Pre-compute row base offset in bytes for efficient gather.
            arow_base = cutlass.Int64(0)
            if aid < num_valid_tokens:
                arow_base = cutlass.Int64(tok) * stride_am_elems * element_bytes
            sArowBase[tx] = arow_base

            if const_expr(self.mul_routed_weight):
                w32 = Float32(0.0)
                if aid < num_valid_tokens:
                    w32 = Float32(mTopkWeights[aid])
                sW[tx] = w32

        cute.arch.barrier()

        # WGMMA accumulator (created unconditionally like warp-level kernel).
        acc = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((block_m, block_n)), Float32
        )
        acc.fill(0.0)

        mC_base_i64 = mC.iterator.toint()
        k_tiles = cute.ceil_div(K_const, block_k)

        # NOTE: Branch is uniform across the CTA, so barriers remain safe.
        # Pattern from warp-level kernel: each branch has its own complete setup.
        if (not block_active) or (expert_id == -1):
            # Inactive block or invalid expert: write zeros to output.
            # All setup done inside this block (matching warp-level pattern).
            vec_size_c = 8
            copy_bits_c = vec_size_c * self.dtype.width
            gmem_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=copy_bits_c,
            )
            element_bytes_c = cutlass.Int64(self.dtype.width // 8)
            align_bytes_c = vec_size_c * int(self.dtype.width // 8)
            zero_vec = cute.make_rmem_tensor((vec_size_c,), self.dtype)
            zero_vec.fill(self.dtype(0.0))
            vec_n_c = block_n // vec_size_c
            total_vec_c = block_m * vec_n_c
            for vec_linear_c in range(tx, total_vec_c, num_threads):
                r_c = vec_linear_c // vec_n_c
                nvec_c = vec_linear_c - r_c * vec_n_c
                n0 = Int32(nvec_c * vec_size_c)
                aid_c = sAid[r_c]
                if aid_c < num_valid_tokens:
                    col0 = n_start + n0
                    if const_expr(N % block_n == 0) or (col0 < N_const):
                        row_off_bytes = cutlass.Int64(aid_c) * stride_cm_elems * element_bytes_c
                        g_off_bytes_c = row_off_bytes + cutlass.Int64(col0) * element_bytes_c
                        g_ptr_c = cute.make_ptr(
                            self.dtype,
                            mC_base_i64 + g_off_bytes_c,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes_c,
                        )
                        dst_c = cute.make_tensor(g_ptr_c, (vec_size_c,))
                        cute.copy(gmem_store_atom, zero_vec, dst_c)
        else:
                # Active block: all copy setup done inside else-block (matching warp-level pattern).
                element_bytes = cutlass.Int64(self.dtype.width // 8)

                # Use tiled_copy_A with gather to respect swizzled SMEM layout.
                # Use L1 bypass for gathered A tiles (avoids L1 cache thrashing).
                tiled_copy_A = tiled_copy_2d_bypass(self.dtype, block_k, num_threads)
                thr_copy_A = tiled_copy_A.get_slice(tx)
                # Create gather copy function that respects swizzle via partition_D
                copy_A = copy_utils.gather_m_get_copy_fn(
                    thr_copy_A,
                    mA,
                    sA,
                    sTok,  # Row indices (token IDs)
                    num_valid_tokens,
                    K_const,
                )

                # Use L1 bypass for B tiles as well (streaming access pattern).
                tiled_copy_B = tiled_copy_2d_bypass(self.dtype, block_k, num_threads)
                thr_copy_B = tiled_copy_B.get_slice(tx)
                cB = cute.make_identity_tensor((block_n, block_k))
                tBcB = thr_copy_B.partition_S(cB)
                t0BcB = thr_copy_B.get_slice(0).partition_S(cB)

                mB_expert = mB[expert_id, None, None]

                # With wg_wait=1, we need one free stage for writing (the WGMMA we just
                # issued might still be reading). So we only prefetch num_stages-1 tiles.
                # With wg_wait=0, we can prefetch all num_stages tiles.
                use_wgmma_pipelining: cutlass.Constexpr[bool] = num_stages >= 3
                prologue_tiles: cutlass.Constexpr[int] = (
                    num_stages - 1 if use_wgmma_pipelining else num_stages
                )

                # Prologue: prefetch the first tiles.
                for stage_prefetch in cutlass.range_constexpr(prologue_tiles):
                    tile_idx_prefetch = Int32(stage_prefetch)
                    k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                    tile_in_range = tile_idx_prefetch < k_tiles

                    # A tile: use gather copy to respect swizzled SMEM layout.
                    if tile_in_range:
                        if const_expr(K % block_k == 0):
                            copy_A(tile_idx_prefetch, stage_prefetch)
                        else:
                            copy_A(tile_idx_prefetch, stage_prefetch, pred=True)

                    # B tile: only copy when tile is in range.
                    if tile_in_range:
                        if const_expr(K % block_k == 0):
                            full_k_tile = True
                        else:
                            full_k_tile = (k_start_prefetch + Int32(block_k)) <= K_const

                        sB_stage = sB[None, None, stage_prefetch]
                        gB_tile = cute.local_tile(
                            mB_expert,
                            (block_n, block_k),
                            (pid_n, tile_idx_prefetch),
                        )
                        tBgB = thr_copy_B.partition_S(gB_tile)
                        tBsB = thr_copy_B.partition_D(sB_stage)
                        if full_k_tile:
                            if full_n_tile:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    cute.copy(
                                        tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                    )
                            else:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    if n_start + t0BcB[0, n, 0][0] < N_const:
                                        cute.copy(
                                            tiled_copy_B,
                                            tBgB[None, n, None],
                                            tBsB[None, n, None],
                                        )
                        else:
                            tBpB = fa_utils.predicate_k(
                                tBcB, limit=K_const - k_start_prefetch
                            )
                            if full_n_tile:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    cute.copy(
                                        tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                        pred=tBpB[None, n, None],
                                    )
                            else:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    if n_start + t0BcB[0, n, 0][0] < N_const:
                                        cute.copy(
                                            tiled_copy_B,
                                            tBgB[None, n, None],
                                            tBsB[None, n, None],
                                            pred=tBpB[None, n, None],
                                        )

                    cute.arch.cp_async_commit_group()

                # WGMMA operand fragments (moved here from before prologue to avoid DSL tracing issues).
                warp_group_idx = cute.arch.make_warp_uniform(
                    tx // Int32(num_threads_per_wg)
                )
                warp_group_thread_layout = cute.make_layout(
                    num_threads // num_threads_per_wg,
                    stride=num_threads_per_wg,
                )
                wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
                tSrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA))
                tSrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB))

                stage_idx = Int32(0)
                main_tiles = k_tiles - Int32(num_stages - 1)

                # With wg_wait=1, we write to a different stage than we read from,
                # because our WGMMA might still be pending (reading the current stage).
                # The safe stage is (stage_idx + num_stages - 1) % num_stages.
                # We also prefetch one tile earlier and wait for one less group.
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
                    _wgmma_gemm_no_fence(
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
                    k_start_next = next_tile * Int32(block_k)
                    tile_in_range_next = next_tile < k_tiles

                    # A tile: use gather copy to respect swizzled SMEM layout.
                    if tile_in_range_next:
                        if const_expr(K % block_k == 0):
                            copy_A(next_tile, write_stage)
                        else:
                            copy_A(next_tile, write_stage, pred=True)

                    # B tile: only copy when tile is in range.
                    if tile_in_range_next:
                        if const_expr(K % block_k == 0):
                            full_k_tile_next = True
                        else:
                            full_k_tile_next = (k_start_next + Int32(block_k)) <= K_const

                        sB_stage = sB[None, None, write_stage]
                        gB_tile = cute.local_tile(
                            mB_expert,
                            (block_n, block_k),
                            (pid_n, next_tile),
                        )
                        tBgB = thr_copy_B.partition_S(gB_tile)
                        tBsB = thr_copy_B.partition_D(sB_stage)
                        if full_k_tile_next:
                            if full_n_tile:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    cute.copy(
                                        tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                    )
                            else:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    if n_start + t0BcB[0, n, 0][0] < N_const:
                                        cute.copy(
                                            tiled_copy_B,
                                            tBgB[None, n, None],
                                            tBsB[None, n, None],
                                        )
                        else:
                            tBpB = fa_utils.predicate_k(
                                tBcB, limit=K_const - k_start_next
                            )
                            if full_n_tile:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    cute.copy(
                                        tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                        pred=tBpB[None, n, None],
                                    )
                            else:
                                for n in cutlass.range_constexpr(
                                    cute.size(tBsB.shape[1])
                                ):
                                    if n_start + t0BcB[0, n, 0][0] < N_const:
                                        cute.copy(
                                            tiled_copy_B,
                                            tBgB[None, n, None],
                                            tBsB[None, n, None],
                                            pred=tBpB[None, n, None],
                                        )

                    cute.arch.cp_async_commit_group()

                    stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

                # Tail drain.
                for drain_idx in cutlass.range_constexpr(num_stages - 1):
                    cute.arch.cp_async_wait_group(num_stages - 2 - drain_idx)
                    # NOTE: fence_proxy omitted - cp_async data visible after wait_group.
                    cute.arch.barrier()

                    _wgmma_gemm_no_fence(
                        tiled_mma,
                        acc,
                        tSrA[None, None, None, stage_idx],
                        tSrB[None, None, None, stage_idx],
                        wg_wait=0,
                    )
                    cute.arch.barrier()

                    stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

                # Epilogue: Scatter stores to GMEM using 128-bit vectorized stores.
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
                thr_mma = tiled_mma.get_slice(tx)
                cC = cute.make_identity_tensor((block_m, block_n))
                tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
                tAcc = fa_utils.make_acc_tensor_mn_view(acc)

                n_per_m: cutlass.Constexpr[int] = cute.size(tAcc.shape[1])
                num_chunks: cutlass.Constexpr[int] = n_per_m // 2  # Each chunk = 2 N values per thread

                # Warp shuffle setup: threads 0-3, 4-7, etc. form groups
                lane = tx % 32  # Lane within warp
                lane_in_group = lane % 4  # Position within 4-thread group (0-3)
                group_base_lane = lane - lane_in_group  # First lane of this group

                # Precompute shuffle source lanes for 3-round ring exchange (hoisted from hot loop)
                # Round 1: src = (g + 3) % 4, Round 2: src = (g + 2) % 4, Round 3: src = (g + 1) % 4
                src_round1 = Int32(group_base_lane) + ((lane_in_group + 3) & 3)
                src_round2 = Int32(group_base_lane) + ((lane_in_group + 2) & 3)
                src_round3 = Int32(group_base_lane) + ((lane_in_group + 1) & 3)

                align_bytes_128 = 16  # 128-bit aligned
                element_bytes_const: cutlass.Constexpr[int] = 2  # BF16 = 2 bytes

                # 4-thread parallel stores: process 4 chunks at once, each thread stores one
                # This achieves 100% store coalescing (4 threads × 16 bytes = 64 bytes = 2 sectors)
                num_chunk_groups: cutlass.Constexpr[int] = num_chunks // 4

                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    valid_row = aid < num_valid_tokens

                    # CORRECTNESS: Use row_scale=0 for invalid lanes so packed values become 0.
                    # This allows shuffles to execute uniformly across all lanes (no divergent
                    # branches around shfl.sync with 0xffffffff mask). Only stores are predicated.
                    row_scale = Float32(0.0)
                    if valid_row:
                        if const_expr(self.mul_routed_weight):
                            row_scale = Float32(sW[m])
                        else:
                            row_scale = Float32(1.0)

                    # row_off_bytes only used when valid_row, but compute unconditionally
                    # to avoid divergent branch. Invalid lanes use aid=0 which is harmless.
                    row_off_bytes = cutlass.Int64(aid) * stride_cm_elems * element_bytes

                    # Process 4 chunks at a time with all 4 threads storing in parallel
                    for chunk_group in cutlass.range_constexpr(num_chunk_groups):
                            base_chunk: cutlass.Constexpr[int] = chunk_group * 4

                            # Each thread computes packed values for all 4 chunks
                            ni0_c0: cutlass.Constexpr[int] = (base_chunk + 0) * 2
                            ni0_c1: cutlass.Constexpr[int] = (base_chunk + 1) * 2
                            ni0_c2: cutlass.Constexpr[int] = (base_chunk + 2) * 2
                            ni0_c3: cutlass.Constexpr[int] = (base_chunk + 3) * 2

                            packed_c0 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c0]) * row_scale,
                                Float32(tAcc[mi, ni0_c0 + 1]) * row_scale,
                                self.dtype,
                            )
                            packed_c1 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c1]) * row_scale,
                                Float32(tAcc[mi, ni0_c1 + 1]) * row_scale,
                                self.dtype,
                            )
                            packed_c2 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c2]) * row_scale,
                                Float32(tAcc[mi, ni0_c2 + 1]) * row_scale,
                                self.dtype,
                            )
                            packed_c3 = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0_c3]) * row_scale,
                                Float32(tAcc[mi, ni0_c3 + 1]) * row_scale,
                                self.dtype,
                            )

                            # 3-shuffle ring exchange (+ 1 local) instead of 16 shuffles + 4-way branch
                            # Goal: lane g needs packed_c[g] from all 4 lanes in its group.
                            # Local contribution: s[g] = packed[g] (already have it)
                            # Ring rounds: for offset in 1,2,3:
                            #   dst = (g + offset) % 4  -> what value the destination lane needs from me
                            #   src = (g - offset) % 4  -> which lane to read from
                            #   send = packed[dst], recv = shfl(send, src) -> recv is packed[g] from lane src

                            # Initialize s0-s3 with local contribution based on lane_in_group
                            # s[lane_in_group] = packed[lane_in_group], others will be filled by shuffles
                            s0 = packed_c0  # Will be overwritten unless lane_in_group == 0
                            s1 = packed_c1  # Will be overwritten unless lane_in_group == 1
                            s2 = packed_c2  # Will be overwritten unless lane_in_group == 2
                            s3 = packed_c3  # Will be overwritten unless lane_in_group == 3

                            # Round 1: offset = 1
                            # dst = (g + 1) % 4, src = (g - 1) % 4 = (g + 3) % 4
                            # Lane 0: dst=1, src=3 -> send packed_c1, recv from lane 3
                            # Lane 1: dst=2, src=0 -> send packed_c2, recv from lane 0
                            # Lane 2: dst=3, src=1 -> send packed_c3, recv from lane 1
                            # Lane 3: dst=0, src=2 -> send packed_c0, recv from lane 2
                            send1 = packed_c1  # Default for lane 0
                            if lane_in_group == 1:
                                send1 = packed_c2
                            elif lane_in_group == 2:
                                send1 = packed_c3
                            elif lane_in_group == 3:
                                send1 = packed_c0
                            recv1 = shfl_sync_idx_b32(send1, src_round1)
                            # Store recv1 into s[src1 % 4] = s[(g + 3) % 4]
                            if lane_in_group == 0:
                                s3 = recv1
                            elif lane_in_group == 1:
                                s0 = recv1
                            elif lane_in_group == 2:
                                s1 = recv1
                            else:
                                s2 = recv1

                            # Round 2: offset = 2
                            # dst = (g + 2) % 4, src = (g + 2) % 4 (same! it's a swap)
                            # Lane 0: dst=2, src=2 -> send packed_c2, recv from lane 2
                            # Lane 1: dst=3, src=3 -> send packed_c3, recv from lane 3
                            # Lane 2: dst=0, src=0 -> send packed_c0, recv from lane 0
                            # Lane 3: dst=1, src=1 -> send packed_c1, recv from lane 1
                            send2 = packed_c2  # Default for lane 0
                            if lane_in_group == 1:
                                send2 = packed_c3
                            elif lane_in_group == 2:
                                send2 = packed_c0
                            elif lane_in_group == 3:
                                send2 = packed_c1
                            recv2 = shfl_sync_idx_b32(send2, src_round2)
                            # Store recv2 into s[(g + 2) % 4]
                            if lane_in_group == 0:
                                s2 = recv2
                            elif lane_in_group == 1:
                                s3 = recv2
                            elif lane_in_group == 2:
                                s0 = recv2
                            else:
                                s1 = recv2

                            # Round 3: offset = 3
                            # dst = (g + 3) % 4, src = (g + 1) % 4
                            # Lane 0: dst=3, src=1 -> send packed_c3, recv from lane 1
                            # Lane 1: dst=0, src=2 -> send packed_c0, recv from lane 2
                            # Lane 2: dst=1, src=3 -> send packed_c1, recv from lane 3
                            # Lane 3: dst=2, src=0 -> send packed_c2, recv from lane 0
                            send3 = packed_c3  # Default for lane 0
                            if lane_in_group == 1:
                                send3 = packed_c0
                            elif lane_in_group == 2:
                                send3 = packed_c1
                            elif lane_in_group == 3:
                                send3 = packed_c2
                            recv3 = shfl_sync_idx_b32(send3, src_round3)
                            # Store recv3 into s[(g + 1) % 4]
                            if lane_in_group == 0:
                                s1 = recv3
                            elif lane_in_group == 1:
                                s2 = recv3
                            elif lane_in_group == 2:
                                s3 = recv3
                            else:
                                s0 = recv3

                            # Compute store address: each thread stores to a different chunk
                            # Thread T stores chunk (base_chunk + T) at N offset (base_chunk + T) * 8
                            # WGMMA layout is linear: chunk C covers N = C*8 to C*8+7
                            # No shuffle needed - compute directly from base_chunk (constexpr)
                            my_n_offset = Int32(base_chunk * 8) + lane_in_group * 8
                            col0 = n_start + my_n_offset

                            # Only predicate the stores (shuffles above are uniform across all lanes)
                            if valid_row:
                                # Bounds check and store
                                # For aligned dimensions, skip bounds check
                                if const_expr(N % block_n == 0) or (col0 + 7 < N_const):
                                    g_off_bytes = row_off_bytes + cutlass.Int64(col0) * element_bytes
                                    g_ptr = cute.make_ptr(
                                        self.dtype,
                                        mC_base_i64 + g_off_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=align_bytes_128,
                                    )
                                    # ALL 4 threads store simultaneously - perfect coalescing!
                                    store_streaming_b128(s0, s1, s2, s3, g_ptr)
                                else:
                                    # Near boundary: fall back to individual 32-bit stores
                                    for ni_local in cutlass.range_constexpr(4):
                                        col_check = col0 + ni_local * 2
                                        if col_check < N_const:
                                            g_off = row_off_bytes + cutlass.Int64(col_check) * element_bytes
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
