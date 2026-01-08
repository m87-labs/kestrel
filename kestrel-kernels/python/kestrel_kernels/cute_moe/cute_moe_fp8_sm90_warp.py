"""CuTe MoE FP8 warp-level MMA kernel (SM90, small block_m < 64).

Optimized version: loads FP8 via ldmatrix and converts in registers,
eliminating the extra smem roundtrip for FP8→BF16 conversion.
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
)

if TYPE_CHECKING:
    from kestrel_kernels.cute_moe.config import CuteMoeConfig


class _FusedMoeMatmulCuTeWarpFp8:
    """Routed MoE GEMM with FP8 activations+weights (W8A8) using warp-level MMA.

    Math:
      C = (A_fp8 @ (B_fp8)^T) * A_scale * B_scale   (and optionally * routed_weight)

    This kernel is tuned for small M tiles (block_m < 64) where warp-level MMA
    works well on SM90. For larger block_m, use the WGMMA-based FP8 kernel.

    Optimization: FP8 data is loaded via ldmatrix (treating pairs as 16-bit),
    then converted to BF16 in registers using PTX cvt instructions. This
    eliminates the extra smem write that a naive FP8→BF16 conversion would need.

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
        self.dtype = dtype  # Output dtype (BF16)
        self.fp8_dtype = fp8_dtype  # FP8 dtype (Float8E4M3)
        # Use F16 for MMA operands - enables single-instruction FP8→F16 conversion
        # (vs multi-step FP8→F16→F32→BF16 for BF16 operands)
        self.operand_dtype = cutlass.Float16
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

        # FP8 buffers stored as "packed pairs" for ldmatrix compatibility
        # Each pair of FP8 is treated as a 16-bit element for ldmatrix
        # So we store (block_m, block_k) FP8 as (block_m, block_k/2) "BF16"
        # Packed FP8 smem: stores 2 FP8 values per "BF16" element
        # This uses HALF the memory of equivalent BF16 buffers
        # Layout: (M or N, K/2) with swizzle for ldmatrix
        sA_fp8_elems = num_stages * block_m * (block_k // 2)  # Packed pairs
        sB_fp8_elems = num_stages * block_n * (block_k // 2)  # Packed pairs

        # Metadata buffers
        sMeta_elems = block_m

        # Use F16 dtype for smem to get correct alignment for ldmatrix
        # Each "F16" element actually holds 2 packed FP8 values
        # F16 enables single-instruction FP8→F16 conversion vs multi-step to BF16
        sA_fp8_struct = cute.struct.Align[
            cute.struct.MemRange[self.operand_dtype, sA_fp8_elems], 1024
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
        # Per-row A scale and routing weights
        sAScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]
        sW_struct = None
        if self.mul_routed_weight:
            sW_struct = cute.struct.Align[cute.struct.MemRange[Float32, sMeta_elems], 16]

        # Per-column B scale for current tile
        sBScale_elems = block_n
        sBScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, sBScale_elems], 16]

        @cute.struct
        class SharedStorage:
            sA_fp8: sA_fp8_struct
            sB_fp8: sB_fp8_struct
            sAid: sAid_struct
            sArowBase: sArowBase_struct
            sCrowBase: sCrowBase_struct
            sAScale: sAScale_struct
            sBScale: sBScale_struct
            if self.mul_routed_weight:
                sW: sW_struct

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
        # Warp-level MMA with F16 operands - enables single-instruction FP8→F16 conversion
        # (Triton uses this same approach: cvt.rn.f16x2.e4m3x2 + HMMA.16816.F32)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.operand_dtype, Float32, (16, 8, 16)),
            (1, self.config.num_warps, 1),  # (M, N, K) warp tiling
            permutation_mnk=(self.config.block_m, self.config.block_n, 16),
        )

        block_m = self.config.block_m
        block_n = self.config.block_n
        grid_m = cute.ceil_div(Int32(mSortedTokenIds.shape[0]), block_m)
        grid_n = cute.ceil_div(Int32(self.N), block_n)
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

        m_count = Int32(mAbits.shape[0])
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
        sAScale = storage.sAScale.get_tensor(s_meta_layout)
        if const_expr(self.mul_routed_weight):
            sW = storage.sW.get_tensor(s_meta_layout)
        else:
            sW = None

        s_bscale_layout = cute.make_layout((block_n,), stride=(1,))
        sBScale = storage.sBScale.get_tensor(s_bscale_layout)

        # Precompute per-row metadata
        element_bytes_fp8 = cutlass.Int64(1)  # FP8 is 1 byte
        element_bytes_bf16 = cutlass.Int64(2)  # BF16 is 2 bytes
        stride_am_elems = cutlass.Int64(mAbits.stride[0])
        stride_cm_elems = cutlass.Int64(mC.stride[0])

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

            arow_base = cutlass.Int64(0)
            crow_base = cutlass.Int64(0)
            a_scale = Float32(0.0)
            if aid < num_valid_tokens:
                arow_base = cutlass.Int64(tok) * stride_am_elems * element_bytes_fp8
                crow_base = cutlass.Int64(aid) * stride_cm_elems * element_bytes_bf16
                a_scale = Float32(mAScale[tok])
            sArowBase[tx] = arow_base
            sCrowBase[tx] = crow_base
            sAScale[tx] = a_scale

            if const_expr(self.mul_routed_weight):
                w32 = Float32(0.0)
                if aid < num_valid_tokens:
                    w32 = Float32(mTopkWeights[aid])
                sW[tx] = w32

        # Load B scales for this N tile
        if expert_id != -1:
            for ni in range(tx, Int32(block_n), num_threads):
                col = n_start + ni
                if const_expr(N % block_n == 0) or (col < N_const):
                    sBScale[ni] = Float32(mBScale[expert_id, col])
                else:
                    sBScale[ni] = Float32(0.0)

        cute.arch.barrier()

        # Swizzled SMEM layout for "packed FP8 pairs" (treated as F16 for ldmatrix)
        # The K dimension is halved because each "element" is 2 packed FP8
        sB_layout_atom = ampere_helpers.get_smem_layout_atom(self.operand_dtype, block_k // 2)
        sA_tile_layout = cute.tile_to_shape(sB_layout_atom, (block_m, block_k // 2), (0, 1))
        sB_tile_layout = cute.tile_to_shape(sB_layout_atom, (block_n, block_k // 2), (0, 1))

        # Get smem buffers - treated as "F16" for ldmatrix but really packed FP8
        # Packed FP8 smem as flat buffer for address calculations
        sA_fp8_elems_per_stage = block_m * block_k // 2  # "F16" elements (packed pairs)
        sA_fp8 = storage.sA_fp8.get_tensor(
            cute.make_layout((num_stages * sA_fp8_elems_per_stage,), stride=(1,))
        )
        sB_fp8_elems_per_stage = block_n * block_k // 2
        sB_fp8 = storage.sB_fp8.get_tensor(
            cute.make_layout((num_stages * sB_fp8_elems_per_stage,), stride=(1,))
        )

        # Thread MMA setup (using F16 operands after conversion)
        thr_mma = tiled_mma.get_slice(tx)
        acc_shape = thr_mma.partition_shape_C((block_m, block_n))
        acc = cute.make_rmem_tensor(acc_shape, Float32)
        acc.fill(0.0)

        # Handle inactive blocks
        if (not block_active) or (expert_id == -1):
            # Write zeros to output
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
            # FP8 uses 1 byte, load 16 bytes (128 bits) per cp.async
            vec_size_fp8 = 16
            copy_bits_fp8 = vec_size_fp8 * 8  # 128 bits
            atom_async_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.fp8_dtype,
                num_bits_per_copy=copy_bits_fp8,
            )

            pred_false = cute.make_rmem_tensor((vec_size_fp8,), cutlass.Boolean)
            pred_false.fill(False)

            mA_base_i64 = mAbits.iterator.toint()
            mC_base_i64 = mC.iterator.toint()
            sA_fp8_base_i64 = sA_fp8.iterator.toint()
            sB_fp8_base_i64 = sB_fp8.iterator.toint()

            block_vec_k_fp8 = block_k // vec_size_fp8
            total_vec_a_fp8 = block_m * block_vec_k_fp8
            total_vec_b_fp8 = block_n * block_vec_k_fp8

            # Stage strides in "BF16" elements (packed FP8 pairs)
            stage_stride_a = Int32(block_m * block_k // 2)
            stage_stride_b = Int32(block_n * block_k // 2)

            k_tiles = cute.ceil_div(K_const, block_k)
            if const_expr(N % block_n == 0):
                full_n_tile = True
            else:
                full_n_tile = (n_start + Int32(block_n)) <= N_const

            mB_expert = mBbits[expert_id, None, None]

            # Packed FP8 smem layout (K/2 dimension, treated as BF16 for ldmatrix)
            # Use swizzled layout for bank-conflict-free ldmatrix access
            sA_packed_layout = sA_tile_layout
            sB_packed_layout = sB_tile_layout

            # Prologue: prefetch the first `num_stages` K tiles (FP8 data)
            for stage_prefetch in cutlass.range_constexpr(num_stages):
                tile_idx_prefetch = Int32(stage_prefetch)
                k_start_prefetch = tile_idx_prefetch * Int32(block_k)
                tile_in_range = tile_idx_prefetch < k_tiles

                # A tile FP8: [block_m, block_k]
                for vec_linear_a in range(tx, total_vec_a_fp8, num_threads):
                    r_a = vec_linear_a // block_vec_k_fp8
                    kvec_a = vec_linear_a - r_a * block_vec_k_fp8
                    k_a = Int32(kvec_a * vec_size_fp8)
                    kg_a = k_start_prefetch + k_a

                    aid_a = sAid[r_a]
                    valid_row_a = aid_a < num_valid_tokens
                    arow_base = sArowBase[r_a]

                    if const_expr(K % block_k == 0):
                        valid_a = tile_in_range and valid_row_a
                    else:
                        valid_a = tile_in_range and valid_row_a and (kg_a < K_const)

                    g_off_bytes_a = arow_base + cutlass.Int64(kg_a) * element_bytes_fp8
                    g_ptr_a = cute.make_ptr(
                        self.fp8_dtype,
                        mA_base_i64 + g_off_bytes_a,
                        cute.AddressSpace.gmem,
                        assumed_align=vec_size_fp8,
                    )
                    src_a = cute.make_tensor(g_ptr_a, (vec_size_fp8,))

                    # Store to smem with SWIZZLED layout for bank-conflict-free ldmatrix
                    # k_a is in FP8 elements, divide by 2 to get "BF16" (packed pair) index
                    s_linear_a = Int32(stage_prefetch) * stage_stride_a + Int32(
                        sA_packed_layout((Int32(r_a), k_a // 2))
                    )
                    s_off_bytes_a = cutlass.Int64(s_linear_a) * element_bytes_bf16
                    s_ptr_a = cute.make_ptr(
                        self.fp8_dtype,
                        sA_fp8_base_i64 + s_off_bytes_a,
                        cute.AddressSpace.smem,
                        assumed_align=vec_size_fp8,
                    )
                    dst_a = cute.make_tensor(s_ptr_a, (vec_size_fp8,))
                    if valid_a:
                        cute.copy(atom_async_copy, src_a, dst_a)
                    else:
                        cute.copy(atom_async_copy, src_a, dst_a, pred=pred_false)

                # B tile FP8: [block_n, block_k]
                if tile_in_range:
                    for vec_linear_b in range(tx, total_vec_b_fp8, num_threads):
                        r_b = vec_linear_b // block_vec_k_fp8
                        kvec_b = vec_linear_b - r_b * block_vec_k_fp8
                        k_b = Int32(kvec_b * vec_size_fp8)
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

                        # B layout is (E, N, K), expert already selected
                        # Use col_b (global N index) not r_b (local index)!
                        g_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            mB_expert.iterator.toint() + cutlass.Int64(col_b) * cutlass.Int64(K) + cutlass.Int64(kg_b),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_fp8,
                        )
                        src_b = cute.make_tensor(g_ptr_b, (vec_size_fp8,))

                        # Store with SWIZZLED layout
                        s_linear_b = Int32(stage_prefetch) * stage_stride_b + Int32(
                            sB_packed_layout((Int32(r_b), k_b // 2))
                        )
                        s_off_bytes_b = cutlass.Int64(s_linear_b) * element_bytes_bf16
                        s_ptr_b = cute.make_ptr(
                            self.fp8_dtype,
                            sB_fp8_base_i64 + s_off_bytes_b,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_fp8,
                        )
                        dst_b = cute.make_tensor(s_ptr_b, (vec_size_fp8,))
                        if valid_b:
                            cute.copy(atom_async_copy, src_b, dst_b)
                        else:
                            cute.copy(atom_async_copy, src_b, dst_b, pred=pred_false)

                cute.arch.cp_async_commit_group()

            stage_idx = Int32(0)

            # Use ldmatrix for efficient vectorized smem→rmem loads
            # LdMatrix8x8x16bOp loads 4 matrices of 8x8 16-bit elements
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.operand_dtype,
            )

            # Create copy slices
            smem_thr_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma).get_slice(tx)
            smem_thr_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma).get_slice(tx)

            # Main loop
            main_tiles = k_tiles - Int32(num_stages - 1)
            for tile_idx in cutlass.range(main_tiles, unroll=1):
                cute.arch.cp_async_wait_group(num_stages - 1)
                cute.arch.barrier()

                # Get packed FP8 smem for current stage
                sA_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes_bf16
                sB_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_b) * element_bytes_bf16

                sA_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sA_fp8_base_i64 + sA_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sA_packed = cute.make_tensor(sA_packed_ptr, sA_packed_layout)

                sB_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sB_fp8_base_i64 + sB_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sB_packed = cute.make_tensor(sB_packed_ptr, sB_packed_layout)

                # Create TWO sets of MMA fragments - one for low FP8 values, one for high
                tCrA_lo = thr_mma.make_fragment_A(thr_mma.partition_A(sA_packed))
                tCrB_lo = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))
                tCrA_hi = thr_mma.make_fragment_A(thr_mma.partition_A(sA_packed))
                tCrB_hi = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))

                # Partition smem and create copy views
                tCsA = smem_thr_copy_A.partition_S(sA_packed)
                tCsB = smem_thr_copy_B.partition_S(sB_packed)
                tCrA_lo_view = smem_thr_copy_A.retile(tCrA_lo)
                tCrB_lo_view = smem_thr_copy_B.retile(tCrB_lo)
                tCrA_hi_view = smem_thr_copy_A.retile(tCrA_hi)
                tCrB_hi_view = smem_thr_copy_B.retile(tCrB_hi)

                # Number of K fragments (this is K/2 fragments since smem is packed)
                k_frags_packed = cute.size(tCsA.shape[2])
                num_a_regs = cute.size(tCrA_lo_view.shape[0])
                num_b_regs = cute.size(tCrB_lo_view.shape[0])

                # Fused load-convert-MMA: process one K fragment at a time
                for kf in cutlass.range_constexpr(k_frags_packed):
                    # Load this K fragment via ldmatrix
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf], tCrA_lo_view[None, None, kf])
                    cute.copy(smem_thr_copy_B, tCsB[None, None, kf], tCrB_lo_view[None, None, kf])

                    # Convert FP8 → F16 lo/hi using combined conversion (single cvt per pair)
                    for mi in cutlass.range_constexpr(num_a_regs):
                        packed_fp8 = tCrA_lo_view[mi, 0, kf]  # F16 slot holding 2 packed FP8
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrA_lo_view[mi, 0, kf] = f16_lo
                        tCrA_hi_view[mi, 0, kf] = f16_hi

                    for ni in cutlass.range_constexpr(num_b_regs):
                        packed_fp8 = tCrB_lo_view[ni, 0, kf]  # F16 slot holding 2 packed FP8
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrB_lo_view[ni, 0, kf] = f16_lo
                        tCrB_hi_view[ni, 0, kf] = f16_hi

                    # Execute MMA for this K fragment
                    cute.gemm(tiled_mma, acc, tCrA_lo[None, None, kf], tCrB_lo[None, None, kf], acc)
                    cute.gemm(tiled_mma, acc, tCrA_hi[None, None, kf], tCrB_hi[None, None, kf], acc)

                # Prefetch next tile (overlapped with register-only MMA above)
                next_tile = tile_idx + Int32(num_stages)
                if next_tile < k_tiles:
                    k_start_next = next_tile * Int32(block_k)

                    # A tile prefetch
                    for vec_linear_a2 in range(tx, total_vec_a_fp8, num_threads):
                        r_a2 = vec_linear_a2 // block_vec_k_fp8
                        kvec_a2 = vec_linear_a2 - r_a2 * block_vec_k_fp8
                        k_a2 = Int32(kvec_a2 * vec_size_fp8)
                        kg_a2 = k_start_next + k_a2

                        aid_a2 = sAid[r_a2]
                        valid_row_a2 = aid_a2 < num_valid_tokens
                        arow_base2 = sArowBase[r_a2]

                        if const_expr(K % block_k == 0):
                            valid_a2 = valid_row_a2
                        else:
                            valid_a2 = valid_row_a2 and (kg_a2 < K_const)

                        g_off_bytes_a2 = arow_base2 + cutlass.Int64(kg_a2) * element_bytes_fp8
                        g_ptr_a2 = cute.make_ptr(
                            self.fp8_dtype,
                            mA_base_i64 + g_off_bytes_a2,
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_fp8,
                        )
                        src_a2 = cute.make_tensor(g_ptr_a2, (vec_size_fp8,))

                        s_linear_a2 = stage_idx * stage_stride_a + Int32(
                            sA_packed_layout((Int32(r_a2), k_a2 // 2))
                        )
                        s_off_bytes_a2 = cutlass.Int64(s_linear_a2) * element_bytes_bf16
                        s_ptr_a2 = cute.make_ptr(
                            self.fp8_dtype,
                            sA_fp8_base_i64 + s_off_bytes_a2,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_fp8,
                        )
                        dst_a2 = cute.make_tensor(s_ptr_a2, (vec_size_fp8,))
                        if valid_a2:
                            cute.copy(atom_async_copy, src_a2, dst_a2)
                        else:
                            cute.copy(atom_async_copy, src_a2, dst_a2, pred=pred_false)

                    # B tile prefetch
                    for vec_linear_b2 in range(tx, total_vec_b_fp8, num_threads):
                        r_b2 = vec_linear_b2 // block_vec_k_fp8
                        kvec_b2 = vec_linear_b2 - r_b2 * block_vec_k_fp8
                        k_b2 = Int32(kvec_b2 * vec_size_fp8)
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

                        # Use col_b2 (global N index) not r_b2 (local index)!
                        g_ptr_b2 = cute.make_ptr(
                            self.fp8_dtype,
                            mB_expert.iterator.toint() + cutlass.Int64(col_b2) * cutlass.Int64(K) + cutlass.Int64(kg_b2),
                            cute.AddressSpace.gmem,
                            assumed_align=vec_size_fp8,
                        )
                        src_b2 = cute.make_tensor(g_ptr_b2, (vec_size_fp8,))

                        s_linear_b2 = stage_idx * stage_stride_b + Int32(
                            sB_packed_layout((Int32(r_b2), k_b2 // 2))
                        )
                        s_off_bytes_b2 = cutlass.Int64(s_linear_b2) * element_bytes_bf16
                        s_ptr_b2 = cute.make_ptr(
                            self.fp8_dtype,
                            sB_fp8_base_i64 + s_off_bytes_b2,
                            cute.AddressSpace.smem,
                            assumed_align=vec_size_fp8,
                        )
                        dst_b2 = cute.make_tensor(s_ptr_b2, (vec_size_fp8,))
                        if valid_b2:
                            cute.copy(atom_async_copy, src_b2, dst_b2)
                        else:
                            cute.copy(atom_async_copy, src_b2, dst_b2, pred=pred_false)

                    cute.arch.cp_async_commit_group()

                # Barrier ensures all threads done reading current stage before any overwrites
                cute.arch.barrier()
                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Tail drain - same register conversion approach as main loop
            for drain_idx in cutlass.range_constexpr(num_stages - 1):
                cute.arch.cp_async_wait_group(num_stages - 2 - drain_idx)
                cute.arch.barrier()

                sA_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_a) * element_bytes_bf16
                sB_stage_offset = cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_b) * element_bytes_bf16

                # Get packed FP8 smem tensors
                sA_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sA_fp8_base_i64 + sA_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sA_packed = cute.make_tensor(sA_packed_ptr, sA_packed_layout)

                sB_packed_ptr = cute.make_ptr(
                    self.operand_dtype,
                    sB_fp8_base_i64 + sB_stage_offset,
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                )
                sB_packed = cute.make_tensor(sB_packed_ptr, sB_packed_layout)

                # Create TWO sets of MMA fragments - one for low FP8 values, one for high
                tCrA_lo = thr_mma.make_fragment_A(thr_mma.partition_A(sA_packed))
                tCrB_lo = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))
                tCrA_hi = thr_mma.make_fragment_A(thr_mma.partition_A(sA_packed))
                tCrB_hi = thr_mma.make_fragment_B(thr_mma.partition_B(sB_packed))

                # Partition smem and create copy views
                tCsA = smem_thr_copy_A.partition_S(sA_packed)
                tCsB = smem_thr_copy_B.partition_S(sB_packed)
                tCrA_lo_view = smem_thr_copy_A.retile(tCrA_lo)
                tCrB_lo_view = smem_thr_copy_B.retile(tCrB_lo)
                tCrA_hi_view = smem_thr_copy_A.retile(tCrA_hi)
                tCrB_hi_view = smem_thr_copy_B.retile(tCrB_hi)

                k_frags_packed = cute.size(tCsA.shape[2])
                num_a_regs = cute.size(tCrA_lo_view.shape[0])
                num_b_regs = cute.size(tCrB_lo_view.shape[0])

                # Fused load-convert-MMA
                for kf in cutlass.range_constexpr(k_frags_packed):
                    cute.copy(smem_thr_copy_A, tCsA[None, None, kf], tCrA_lo_view[None, None, kf])
                    cute.copy(smem_thr_copy_B, tCsB[None, None, kf], tCrB_lo_view[None, None, kf])

                    # Convert FP8 → F16 lo/hi using combined conversion (single cvt per pair)
                    for mi in cutlass.range_constexpr(num_a_regs):
                        packed_fp8 = tCrA_lo_view[mi, 0, kf]  # F16 slot holding 2 packed FP8
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrA_lo_view[mi, 0, kf] = f16_lo
                        tCrA_hi_view[mi, 0, kf] = f16_hi

                    for ni in cutlass.range_constexpr(num_b_regs):
                        packed_fp8 = tCrB_lo_view[ni, 0, kf]  # F16 slot holding 2 packed FP8
                        f16_lo, f16_hi = cvt_fp8x2_to_f16_both(packed_fp8)
                        tCrB_lo_view[ni, 0, kf] = f16_lo
                        tCrB_hi_view[ni, 0, kf] = f16_hi

                    cute.gemm(tiled_mma, acc, tCrA_lo[None, None, kf], tCrB_lo[None, None, kf], acc)
                    cute.gemm(tiled_mma, acc, tCrA_hi[None, None, kf], tCrB_hi[None, None, kf], acc)

                cute.arch.barrier()
                stage_idx = stage_idx + 1 if stage_idx + 1 < num_stages else Int32(0)

            # Epilogue: apply scales and store
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
                        # Get per-row A scale and optional routing weight
                        a_scale = sAScale[m]
                        row_scale = a_scale
                        if const_expr(self.mul_routed_weight):
                            row_scale = a_scale * Float32(sW[m])

                        row_off_bytes = sCrowBase[m]
                        n_pairs = cute.size(tAcc.shape[1]) // vec_size_out
                        for pi in cutlass.range_constexpr(n_pairs):
                            ni0 = Int32(pi * vec_size_out)
                            n0 = Int32(tC_coords[mi, ni0][1])
                            col0 = n_start + n0

                            # Apply A_scale * B_scale for each output element
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
                # Non-full N tile: scalar stores with bounds checking
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
                        a_scale = sAScale[m]
                        row_scale = a_scale
                        if const_expr(self.mul_routed_weight):
                            row_scale = a_scale * Float32(sW[m])

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
