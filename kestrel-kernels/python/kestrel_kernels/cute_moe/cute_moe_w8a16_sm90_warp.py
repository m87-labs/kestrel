"""CuTe MoE W8A16 warp-level MMA kernel (SM90, small block_m < 64).

FP8 weights with BF16 activations.

Key insight: Both A and B can use cp.async for gmem→smem.
- A: BF16 values in natural layout, then per-element BF16→F16 conversion during ldmatrix
- B: Packed FP8, unpacked to even/odd after ldmatrix

For A, since MMA expects F16 and we have BF16, we convert during the fragment load.
We don't need to match B's even/odd structure because A's K layout is contiguous
and B's unpacking creates the right structure.

IMPORTANT: This kernel treats A's BF16 values directly. The MMA uses F16, so we
convert BF16→F16 per element as we load A fragments from smem.
"""

from typing import TYPE_CHECKING, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from kestrel_kernels.flash_attn.cute import ampere_helpers
from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.utils import (
    store_streaming_b32,
    cvt_fp8x2_to_f16_both,
    cvt_bf16_to_f16,
)

if TYPE_CHECKING:
    from kestrel_kernels.cute_moe.config import CuteMoeConfig


class _FusedMoeMatmulCuTeWarpW8A16:
    """Routed MoE GEMM with FP8 weights and BF16 activations (W8A16) using warp-level MMA.

    Uses cp.async for both A and B for fast memory transfers.
    A is BF16, converted to F16 element-wise after ldmatrix.
    B is packed FP8, unpacked to F16 after ldmatrix.
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
        self.dtype = dtype  # Output/activation dtype (BF16)
        self.fp8_dtype = fp8_dtype  # FP8 dtype for weights (Float8E4M3)
        self.operand_dtype = cutlass.Float16  # MMA operand dtype
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

        half_k = block_k // 2
        # A: [block_m, block_k] as BF16 (same size as F16)
        sA_elems = num_stages * block_m * block_k
        # B: packed FP8 pairs [block_n, half_k]
        sB_fp8_elems = num_stages * block_n * half_k

        sMeta_elems = block_m

        sA_struct = cute.struct.Align[
            cute.struct.MemRange[self.operand_dtype, sA_elems], 1024
        ]
        sB_fp8_struct = cute.struct.Align[
            cute.struct.MemRange[self.operand_dtype, sB_fp8_elems], 1024
        ]
        sAid_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        sArowBase_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, sMeta_elems], 16
        ]
        sCrowBase_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, sMeta_elems], 16
        ]
        sW_struct = None
        if self.mul_routed_weight:
            sW_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]

        sBScale_elems = block_n
        sBScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sBScale_elems], 16]

        @cute.struct
        class SharedStorage:
            sA: sA_struct
            sB_fp8: sB_fp8_struct
            sAid: sAid_struct
            sArowBase: sArowBase_struct
            sCrowBase: sCrowBase_struct
            sBScale: sBScale_struct
            if self.mul_routed_weight:
                sW: sW_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M_tokens_or_assignments, K) BF16 activations
        mBbits: cute.Tensor,  # (E, N, K) uint8 view of fp8 values
        mBScale: cute.Tensor,  # (E, N) fp32
        mC: cute.Tensor,  # (M_assignments, N) BF16
        mTopkWeights: cute.Tensor,  # (M_assignments,) or empty
        mSortedTokenIds: cute.Tensor,  # (EM,)
        mExpertIds: cute.Tensor,  # (EM / block_m,)
        mNumTokensPostPadded: cute.Tensor,  # (1,)
        stream: cuda.CUstream,
    ):
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.operand_dtype, Float32, (16, 8, 16)),
            (1, self.config.num_warps, 1),
            permutation_mnk=(self.config.block_m, self.config.block_n, 16),
        )

        block_m = self.config.block_m
        block_n = self.config.block_n
        grid_m = cute.ceil_div(Int32(mSortedTokenIds.shape[0]), block_m)
        grid_n = cute.ceil_div(Int32(self.N), block_n)
        SharedStorage = self._shared_storage_cls()

        self.kernel(
            tiled_mma,
            mA,
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
            grid=(grid_n, grid_m, 1),
            block=(self.config.num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA: cute.Tensor,
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
        pid_n, pid_m, _ = cute.arch.block_idx()

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
        half_k = block_k // 2

        m_count = Int32(mA.shape[0])
        if const_expr(self.top_k == 1):
            num_valid_tokens = m_count
        elif const_expr(self.top_k == 8):
            num_valid_tokens = m_count << 3
        else:
            num_valid_tokens = m_count * Int32(self.top_k)

        expert_id = Int32(mExpertIds[pid_m])

        # Shared memory allocation
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        sArowBase = storage.sArowBase.get_tensor(s_meta_layout)
        sCrowBase = storage.sCrowBase.get_tensor(s_meta_layout)
        if const_expr(self.mul_routed_weight):
            sW = storage.sW.get_tensor(s_meta_layout)
        else:
            sW = None

        s_bscale_layout = cute.make_layout((block_n,), stride=(1,))
        sBScale = storage.sBScale.get_tensor(s_bscale_layout)

        # Precompute per-row metadata
        element_bytes_bf16 = cutlass.Int64(2)
        stride_am_elems = cutlass.Int64(mA.stride[0])
        stride_cm_elems = cutlass.Int64(mC.stride[0])

        if tx < Int32(block_m):
            idx = row_start + tx
            aid = Int32(num_valid_tokens)
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

            arow_base = cutlass.Int64(0)
            crow_base = cutlass.Int64(0)
            if aid < num_valid_tokens:
                arow_base = cutlass.Int64(tok) * stride_am_elems * element_bytes_bf16
                crow_base = cutlass.Int64(aid) * stride_cm_elems * element_bytes_bf16
            sArowBase[tx] = arow_base
            sCrowBase[tx] = crow_base

            if const_expr(self.mul_routed_weight):
                w32 = Float32(0.0)
                if aid < num_valid_tokens:
                    w32 = Float32(mTopkWeights[aid])
                sW[tx] = w32

        # Load B scales
        if expert_id != -1:
            for ni in range(tx, Int32(block_n), num_threads):
                col = n_start + ni
                if const_expr(N % block_n == 0) or (col < N_const):
                    sBScale[ni] = Float32(mBScale[expert_id, col])
                else:
                    sBScale[ni] = Float32(0.0)

        cute.arch.barrier()

        # Smem layouts
        # A: [block_m, block_k] BF16 values (stored as F16 for layout purposes)
        sA_layout_atom = ampere_helpers.get_smem_layout_atom(self.operand_dtype, block_k)
        sA_tile_layout = cute.tile_to_shape(sA_layout_atom, (block_m, block_k), (0, 1))

        # B: [block_n, half_k] packed FP8 pairs
        sB_layout_atom = ampere_helpers.get_smem_layout_atom(self.operand_dtype, half_k)
        sB_tile_layout = cute.tile_to_shape(sB_layout_atom, (block_n, half_k), (0, 1))

        # Get smem buffers
        sA_elems_per_stage = block_m * block_k
        sB_elems_per_stage = block_n * half_k

        sA = storage.sA.get_tensor(
            cute.make_layout((num_stages * sA_elems_per_stage,), stride=(1,))
        )
        sB_fp8 = storage.sB_fp8.get_tensor(
            cute.make_layout((num_stages * sB_elems_per_stage,), stride=(1,))
        )

        # Thread MMA setup
        thr_mma = tiled_mma.get_slice(tx)
        acc_shape = thr_mma.partition_shape_C((block_m, block_n))
        acc = cute.make_rmem_tensor(acc_shape, Float32)
        acc.fill(0.0)

        # Handle inactive blocks
        if (not block_active) or (expert_id == -1):
            vec_size = 8
            copy_bits = vec_size * self.dtype.width
            gmem_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=copy_bits,
            )
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
                if const_expr(N % block_n == 0):
                    if aid_c < num_valid_tokens:
                        g_off_bytes_c = sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes_bf16
                        g_ptr_c = cute.make_ptr(
                            self.dtype,
                            mC_base_i64 + g_off_bytes_c,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes,
                        )
                        dst_c = cute.make_tensor(g_ptr_c, (vec_size,))
                        cute.copy(gmem_store_atom, zero_vec, dst_c)
                else:
                    if (aid_c < num_valid_tokens) and (col0 < N_const):
                        g_off_bytes_c = sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes_bf16
                        g_ptr_c = cute.make_ptr(
                            self.dtype,
                            mC_base_i64 + g_off_bytes_c,
                            cute.AddressSpace.gmem,
                            assumed_align=align_bytes,
                        )
                        dst_c = cute.make_tensor(g_ptr_c, (vec_size,))
                        cute.copy(gmem_store_atom, zero_vec, dst_c)
        else:
            # Active block: run main GEMM loop
            # Both A and B use cp.async

            # Vectorized copy atoms
            vec_size_a = 8  # 8 BF16 = 128 bits
            copy_bits_a = vec_size_a * 16
            vec_size_b = 16  # 16 FP8 = 128 bits
            copy_bits_b = vec_size_b * 8

            atom_async_copy_a = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.dtype,  # BF16
                num_bits_per_copy=copy_bits_a,
            )
            atom_async_copy_b = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.fp8_dtype,
                num_bits_per_copy=copy_bits_b,
            )

            pred_false_a = cute.make_rmem_tensor((vec_size_a,), cutlass.Boolean)
            pred_false_a.fill(False)
            pred_false_b = cute.make_rmem_tensor((vec_size_b,), cutlass.Boolean)
            pred_false_b.fill(False)

            mA_base_i64 = mA.iterator.toint()
            mC_base_i64 = mC.iterator.toint()
            sA_base_i64 = sA.iterator.toint()
            sB_fp8_base_i64 = sB_fp8.iterator.toint()

            # Work distribution for A
            block_vec_k_a = block_k // vec_size_a
            total_vec_a = block_m * block_vec_k_a

            # Work distribution for B
            block_vec_k_b = block_k // vec_size_b
            total_vec_b = block_n * block_vec_k_b

            stage_stride_a = Int32(block_m * block_k)
            stage_stride_b = Int32(block_n * half_k)

            k_tiles = cute.ceil_div(K_const, block_k)
            if const_expr(N % block_n == 0):
                full_n_tile = True
            else:
                full_n_tile = (n_start + Int32(block_n)) <= N_const

            mB_expert = mBbits[expert_id, None, None]

            # Prologue: prefetch first stages
            for stage_prefetch in cutlass.range_constexpr(num_stages):
                tile_idx_prefetch = Int32(stage_prefetch)
                k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                tile_in_range = tile_idx_prefetch < k_tiles

                # A tile: cp.async (vectorized)
                for vec_linear_a in range(tx, total_vec_a, num_threads):
                    r_a = vec_linear_a // block_vec_k_a
                    kvec_a = vec_linear_a - r_a * block_vec_k_a
                    k_a = Int32(kvec_a * vec_size_a)
                    kg_a = k_start_prefetch + k_a

                    aid_a = sAid[r_a]
                    valid_row_a = aid_a < num_valid_tokens
                    arow_base = sArowBase[r_a]

                    if const_expr(K % block_k == 0):
                        valid_a = tile_in_range and valid_row_a
                    else:
                        valid_a = tile_in_range and valid_row_a and (kg_a < K_const)

                    g_off_bytes_a = arow_base + cutlass.Int64(kg_a) * element_bytes_bf16
                    g_ptr_a = cute.make_ptr(
                        self.dtype,
                        mA_base_i64 + g_off_bytes_a,
                        cute.AddressSpace.gmem,
                        assumed_align=vec_size_a * 2,
                    )
                    src_a = cute.make_tensor(g_ptr_a, (vec_size_a,))

                    s_linear_a = Int32(stage_prefetch) * stage_stride_a + Int32(
                        sA_tile_layout((Int32(r_a), k_a))
                    )
                    s_off_bytes_a = cutlass.Int64(s_linear_a) * element_bytes_bf16
                    s_ptr_a = cute.make_ptr(
                        self.dtype,
                        sA_base_i64 + s_off_bytes_a,
                        cute.AddressSpace.smem,
                        assumed_align=vec_size_a * 2,
                    )
                    dst_a = cute.make_tensor(s_ptr_a, (vec_size_a,))

                    if valid_a:
                        cute.copy(atom_async_copy_a, src_a, dst_a)
                    else:
                        cute.copy(atom_async_copy_a, src_a, dst_a, pred=pred_false_a)

                # B tile: cp.async (vectorized)
                if tile_in_range:
                    for vec_linear_b in range(tx, total_vec_b, num_threads):
                        r_b = vec_linear_b // block_vec_k_b
                        kvec_b = vec_linear_b - r_b * block_vec_k_b
                        k_b = Int32(kvec_b * vec_size_b)
                        kg_b = k_start_prefetch + k_b
                        col_b = n_start + Int32(r_b)

                        if const_expr(K % block_k == 0):
                            valid_k = True
                        else:
                            valid_k = kg_b < K_const

                        if const_expr(N % block_n == 0):
                            valid_b = valid_k
                        else:
                            valid_b = valid_k and (col_b < N_const)

                        g_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            mB_expert.iterator.toint() + cutlass.Int64(col_b) * cutlass.Int64(K) + cutlass.Int64(kg_b),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_b,
                        )
                        src_b = cute.make_tensor(g_ptr_b, (vec_size_b,))

                        s_linear_b = Int32(stage_prefetch) * stage_stride_b + Int32(
                            sB_tile_layout((Int32(r_b), k_b // 2))
                        )
                        s_off_bytes_b = cutlass.Int64(s_linear_b) * element_bytes_bf16
                        s_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            sB_fp8_base_i64 + s_off_bytes_b,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_b,
                        )
                        dst_b = cute.make_tensor(s_ptr_b, (vec_size_b,))
                        if valid_b:
                            cute.copy(atom_async_copy_b, src_b, dst_b)
                        else:
                            cute.copy(atom_async_copy_b, src_b, dst_b, pred=pred_false_b)

                cute.arch.cp_async_commit_group()

            stage_idx = Int32(0)

            # ldmatrix copy atoms
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.operand_dtype,
            )
            smem_thr_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma).get_slice(tx)
            smem_thr_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma).get_slice(tx)

            # Main loop
            main_tiles = k_tiles - Int32(num_stages - 1)
            for tile_idx in cutlass.range(main_tiles, unroll=1):
                cute.arch.cp_async_wait_group(num_stages - 1)
                cute.arch.barrier()

                # Get smem for current stage
                sA_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes_bf16
                sB_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_b) * element_bytes_bf16

                sA_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sA_base_i64 + sA_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sA_stage = cute.make_tensor(sA_ptr, sA_tile_layout)

                sB_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sB_fp8_base_i64 + sB_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sB_packed = cute.make_tensor(sB_packed_ptr, sB_tile_layout)

                # Create MMA fragments
                tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_stage))
                tCrB_lo = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))
                tCrB_hi = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))

                # Partition smem for copy
                tCsA = smem_thr_copy_A.partition_S(sA_stage)
                tCsB = smem_thr_copy_B.partition_S(sB_packed)

                tCrA_view = smem_thr_copy_A.retile(tCrA)
                tCrB_lo_view = smem_thr_copy_B.retile(tCrB_lo)
                tCrB_hi_view = smem_thr_copy_B.retile(tCrB_hi)

                k_frags = cute.size(tCsB.shape[2])
                num_a_regs = cute.size(tCrA_view.shape[0])
                num_b_regs = cute.size(tCrB_lo_view.shape[0])

                # Batched per k_frag: load A0+A1+B, convert A, unpack B, then MMA
                for kf_b in cutlass.range_constexpr(k_frags):
                    kf_a_0 = kf_b * 2
                    kf_a_1 = kf_b * 2 + 1

                    # Load all fragments for this k_frag
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf_a_0], tCrA_view[None, None, kf_a_0])
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf_a_1], tCrA_view[None, None, kf_a_1])
                    cute.copy(smem_thr_copy_B, tCsB[None, None, kf_b], tCrB_lo_view[None, None, kf_b])

                    # Convert all A values (both fragments)
                    for mi in cutlass.range_constexpr(num_a_regs):
                        bf16_val = tCrA_view[mi, 0, kf_a_0]
                        tCrA_view[mi, 0, kf_a_0] = cvt_bf16_to_f16(bf16_val)
                    for mi in cutlass.range_constexpr(num_a_regs):
                        bf16_val = tCrA_view[mi, 0, kf_a_1]
                        tCrA_view[mi, 0, kf_a_1] = cvt_bf16_to_f16(bf16_val)

                    # Unpack all B values
                    for ni in cutlass.range_constexpr(num_b_regs):
                        packed_fp8 = tCrB_lo_view[ni, 0, kf_b]
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrB_lo_view[ni, 0, kf_b] = f16_lo
                        tCrB_hi_view[ni, 0, kf_b] = f16_hi

                    # Both MMAs
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf_a_0], tCrB_lo[None, None, kf_b], acc)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf_a_1], tCrB_hi[None, None, kf_b], acc)

                # Prefetch next tile
                next_tile = tile_idx + Int32(num_stages)
                if next_tile < k_tiles:
                    k_start_next = next_tile * Int32(block_k)

                    for vec_linear_a in range(tx, total_vec_a, num_threads):
                        r_a = vec_linear_a // block_vec_k_a
                        kvec_a = vec_linear_a - r_a * block_vec_k_a
                        k_a = Int32(kvec_a * vec_size_a)
                        kg_a = k_start_next + k_a

                        aid_a = sAid[r_a]
                        valid_row_a = aid_a < num_valid_tokens
                        arow_base = sArowBase[r_a]

                        if const_expr(K % block_k == 0):
                            valid_a = valid_row_a
                        else:
                            valid_a = valid_row_a and (kg_a < K_const)

                        g_off_bytes_a = arow_base + cutlass.Int64(kg_a) * element_bytes_bf16
                        g_ptr_a = cute.make_ptr(
                            self.dtype,
                            mA_base_i64 + g_off_bytes_a,
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_a * 2,
                        )
                        src_a = cute.make_tensor(g_ptr_a, (vec_size_a,))

                        s_linear_a = stage_idx * stage_stride_a + Int32(
                            sA_tile_layout((Int32(r_a), k_a))
                        )
                        s_off_bytes_a = cutlass.Int64(s_linear_a) * element_bytes_bf16
                        s_ptr_a = cute.make_ptr(
                            self.dtype,
                            sA_base_i64 + s_off_bytes_a,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_a * 2,
                        )
                        dst_a = cute.make_tensor(s_ptr_a, (vec_size_a,))

                        if valid_a:
                            cute.copy(atom_async_copy_a, src_a, dst_a)
                        else:
                            cute.copy(atom_async_copy_a, src_a, dst_a, pred=pred_false_a)

                    for vec_linear_b2 in range(tx, total_vec_b, num_threads):
                        r_b2 = vec_linear_b2 // block_vec_k_b
                        kvec_b2 = vec_linear_b2 - r_b2 * block_vec_k_b
                        k_b2 = Int32(kvec_b2 * vec_size_b)
                        kg_b2 = k_start_next + k_b2
                        col_b2 = n_start + Int32(r_b2)

                        if const_expr(K % block_k == 0):
                            valid_k2 = True
                        else:
                            valid_k2 = kg_b2 < K_const

                        if const_expr(N % block_n == 0):
                            valid_b2 = valid_k2
                        else:
                            valid_b2 = valid_k2 and (col_b2 < N_const)

                        g_ptr_b2 = cute.make_ptr(
                            self.fp8_dtype,
                            mB_expert.iterator.toint() + cutlass.Int64(col_b2) * cutlass.Int64(K) + cutlass.Int64(kg_b2),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_b,
                        )
                        src_b2 = cute.make_tensor(g_ptr_b2, (vec_size_b,))

                        s_linear_b2 = stage_idx * stage_stride_b + Int32(
                            sB_tile_layout((Int32(r_b2), k_b2 // 2))
                        )
                        s_off_bytes_b2 = cutlass.Int64(s_linear_b2) * element_bytes_bf16
                        s_ptr_b2 = cute.make_ptr(
                            self.fp8_dtype,
                            sB_fp8_base_i64 + s_off_bytes_b2,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_b,
                        )
                        dst_b2 = cute.make_tensor(s_ptr_b2, (vec_size_b,))
                        if valid_b2:
                            cute.copy(atom_async_copy_b, src_b2, dst_b2)
                        else:
                            cute.copy(atom_async_copy_b, src_b2, dst_b2, pred=pred_false_b)

                    cute.arch.cp_async_commit_group()

                cute.arch.barrier()
                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Tail drain
            for drain_idx in cutlass.range_constexpr(num_stages - 1):
                cute.arch.cp_async_wait_group(num_stages - 2 - drain_idx)
                cute.arch.barrier()

                sA_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes_bf16
                sB_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_b) * element_bytes_bf16

                sA_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sA_base_i64 + sA_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sA_stage = cute.make_tensor(sA_ptr, sA_tile_layout)

                sB_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sB_fp8_base_i64 + sB_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sB_packed = cute.make_tensor(sB_packed_ptr, sB_tile_layout)

                tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_stage))
                tCrB_lo = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))
                tCrB_hi = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))

                tCsA = smem_thr_copy_A.partition_S(sA_stage)
                tCsB = smem_thr_copy_B.partition_S(sB_packed)
                tCrA_view = smem_thr_copy_A.retile(tCrA)
                tCrB_lo_view = smem_thr_copy_B.retile(tCrB_lo)
                tCrB_hi_view = smem_thr_copy_B.retile(tCrB_hi)

                k_frags = cute.size(tCsB.shape[2])
                num_a_regs = cute.size(tCrA_view.shape[0])
                num_b_regs = cute.size(tCrB_lo_view.shape[0])

                for kf_b in cutlass.range_constexpr(k_frags):
                    kf_a_0 = kf_b * 2
                    kf_a_1 = kf_b * 2 + 1

                    # Load all fragments for this k_frag
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf_a_0], tCrA_view[None, None, kf_a_0])
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf_a_1], tCrA_view[None, None, kf_a_1])
                    cute.copy(smem_thr_copy_B, tCsB[None, None, kf_b], tCrB_lo_view[None, None, kf_b])

                    # Convert all A values
                    for mi in cutlass.range_constexpr(num_a_regs):
                        bf16_val = tCrA_view[mi, 0, kf_a_0]
                        tCrA_view[mi, 0, kf_a_0] = cvt_bf16_to_f16(bf16_val)
                    for mi in cutlass.range_constexpr(num_a_regs):
                        bf16_val = tCrA_view[mi, 0, kf_a_1]
                        tCrA_view[mi, 0, kf_a_1] = cvt_bf16_to_f16(bf16_val)

                    # Unpack all B values
                    for ni in cutlass.range_constexpr(num_b_regs):
                        packed_fp8 = tCrB_lo_view[ni, 0, kf_b]
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrB_lo_view[ni, 0, kf_b] = f16_lo
                        tCrB_hi_view[ni, 0, kf_b] = f16_hi

                    # Both MMAs
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf_a_0], tCrB_lo[None, None, kf_b], acc)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kf_a_1], tCrB_hi[None, None, kf_b], acc)

                cute.arch.barrier()
                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Epilogue: apply B_scale and store
            cC = cute.make_identity_tensor((block_m, block_n))
            tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
            tAcc = fa_utils.make_acc_tensor_mn_view(acc)

            vec_size_out = 2
            align_bytes_out = vec_size_out * int(self.dtype.width // 8)

            if full_n_tile:
                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    if aid < num_valid_tokens:
                        row_scale = Float32(1.0)
                        if const_expr(self.mul_routed_weight):
                            row_scale = Float32(sW[m])

                        row_off_bytes = sCrowBase[m]
                        n_pairs = cute.size(tAcc.shape[1]) // vec_size_out
                        for pi in cutlass.range_constexpr(n_pairs):
                            ni0 = Int32(pi * vec_size_out)
                            n0 = Int32(tC_coords[mi, ni0][1])
                            col0 = n_start + n0

                            b_scale0 = sBScale[n0]
                            b_scale1 = sBScale[n0 + Int32(1)]
                            val0 = Float32(tAcc[mi, ni0]) * row_scale * b_scale0
                            val1 = Float32(tAcc[mi, ni0 + 1]) * row_scale * b_scale1

                            packed = fa_utils.cvt_f16x2_f32(val0, val1, self.dtype)
                            g_off_bytes_vec = row_off_bytes + cutlass.Int64(col0) * element_bytes_bf16
                            g_ptr_vec = cute.make_ptr(
                                self.dtype,
                                mC_base_i64 + g_off_bytes_vec,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes_out,
                            )
                            store_streaming_b32(packed, g_ptr_vec)
            else:
                gmem_store_atom_scalar = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=self.dtype.width,
                )
                src_scalar = cute.make_rmem_tensor((1,), self.dtype)
                align_bytes_scalar = int(self.dtype.width // 8)

                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    if aid < num_valid_tokens:
                        row_scale = Float32(1.0)
                        if const_expr(self.mul_routed_weight):
                            row_scale = Float32(sW[m])

                        row_off_bytes = sCrowBase[m]
                        for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                            n = Int32(tC_coords[mi, ni][1])
                            col = n_start + n
                            if col < N_const:
                                b_scale = sBScale[n]
                                src_scalar[0] = self.dtype(Float32(tAcc[mi, ni]) * row_scale * b_scale)
                                g_off_bytes_scalar = row_off_bytes + cutlass.Int64(col) * element_bytes_bf16
                                g_ptr_scalar = cute.make_ptr(
                                    self.dtype,
                                    mC_base_i64 + g_off_bytes_scalar,
                                    cute.AddressSpace.gmem,
                                    assumed_align=align_bytes_scalar,
                                )
                                dst_scalar = cute.make_tensor(g_ptr_scalar, (1,))
                                cute.copy(gmem_store_atom_scalar, src_scalar, dst_scalar)
