"""CuTe MoE BF16 warp-level MMA kernel (SM90, small block_m < 64)."""

from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from kestrel_kernels.flash_attn.cute import ampere_helpers
from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.config import CuteMoeConfig
from kestrel_kernels.cute_moe.utils import store_streaming_b32


class _FusedMoeMatmulCuTe:
    """Routed MoE GEMM with BF16 activations+weights using warp-level MMA.

    This is tuned for small M tiles (block_m < 64) where warp-level MMA works
    well on SM90.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        config: CuteMoeConfig,
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
        sA_elems = self.config.num_stages * self.config.block_m * self.config.block_k
        sB_elems = self.config.num_stages * self.config.block_n * self.config.block_k
        sMeta_elems = self.config.block_m
        # Match flash-attn's practice of aligning swizzled SMEM buffers to large boundaries.
        # This helps the compiler prove alignment for vectorized LDGSTS and reduces any
        # bank-conflict risk for LDSM (ldmatrix) loads.
        sA_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sA_elems], 1024]
        sB_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sB_elems], 1024]
        sAid_struct = cute.struct.Align[cute.struct.MemRange[Int32, sMeta_elems], 16]
        sW_struct = None
        if self.mul_routed_weight:
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
            sAid: sAid_struct
            sArowBase: sArowBase_struct
            sCrowBase: sCrowBase_struct
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
            grid=(grid_n, grid_m, 1),  # N-first for better weight reuse
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

        # NOTE: For this kernel, we assume K and N are static across the op invocation.
        N_const = Int32(N)
        K_const = Int32(K)

        # `num_valid_tokens` matches the Triton kernel semantics:
        # - When top_k > 1 (up projection): A is [M, K], valid assignments = M * top_k.
        # - When top_k == 1 (down projection): A is [M_assignments, K], valid assignments = M_assignments.
        m_count = Int32(mA.shape[0])
        if const_expr(self.top_k == 1):
            num_valid_tokens = m_count
        elif const_expr(self.top_k == 8):
            num_valid_tokens = m_count << 3
        else:
            num_valid_tokens = m_count * Int32(self.top_k)

        expert_id = Int32(mExpertIds[pid_m])

        # Shared memory.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Load assignment metadata once per CTA into shared memory:
        # - Avoid per-thread redundant loads from mSortedTokenIds.
        # - Avoid dynamic-indexed register arrays that frequently spill to local memory.
        s_meta_layout = cute.make_layout((block_m,), stride=(1,))
        sAid = storage.sAid.get_tensor(s_meta_layout)
        if const_expr(self.mul_routed_weight):
            sW = storage.sW.get_tensor(s_meta_layout)
        else:
            sW = None
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
                if const_expr(self.top_k == 1):
                    tok = aid
                elif const_expr(self.top_k == 8):
                    tok = aid >> 3
                else:
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
                if const_expr(N % block_n == 0):
                    if aid_c < num_valid_tokens:
                        g_off_bytes_c = (
                            sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes
                        )
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
                        g_off_bytes_c = (
                            sCrowBase[r_c] + cutlass.Int64(col0) * element_bytes
                        )
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

            block_vec_k = block_k // vec_size
            total_vec_a = block_m * block_vec_k

            stage_stride_a = Int32(block_m * block_k)
            stage_stride_b = Int32(block_n * block_k)

            k_tiles = cute.ceil_div(K_const, block_k)
            if const_expr(N % block_n == 0):
                full_n_tile = True
            else:
                full_n_tile = (n_start + Int32(block_n)) <= N_const

            vB_layout = cute.make_layout((1, vec_size))
            tB_shape_dim_1 = sB_layout_atom.outer.shape[1] // vec_size
            tB_layout = cute.make_ordered_layout(
                (num_threads // tB_shape_dim_1, tB_shape_dim_1),
                order=(1, 0),
            )
            gmem_tiled_copy_B = cute.make_tiled_copy_tv(
                atom_async_copy_b, tB_layout, vB_layout
            )
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

                    if const_expr(K % block_k == 0):
                        valid_a = tile_in_range and valid_row_a
                    else:
                        valid_a = tile_in_range and valid_row_a and (kg_a < K_const)
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
                    if const_expr(K % block_k == 0):
                        full_k_tile = True
                    else:
                        full_k_tile = (k_start_prefetch + Int32(block_k)) <= K_const
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
                                if n_start + t0BcB[0, n, 0][0] < N_const:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB[None, n, None],
                                        tBsB[None, n, None],
                                    )
                    else:
                        tBpB = fa_utils.predicate_k(tBcB, limit=K_const - k_start_prefetch)
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
                                if n_start + t0BcB[0, n, 0][0] < N_const:
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
                    if const_expr(K % block_k == 0):
                        full_k_tile_next = True
                    else:
                        full_k_tile_next = (k_start_next + Int32(block_k)) <= K_const

                    for vec_linear_a2 in range(tx, total_vec_a, num_threads):
                        r_a2 = vec_linear_a2 // block_vec_k
                        kvec_a2 = vec_linear_a2 - r_a2 * block_vec_k
                        k_a2 = Int32(kvec_a2 * vec_size)
                        kg_a2 = k_start_next + k_a2

                        aid_a2 = sAid[r_a2]
                        valid_row_a2 = aid_a2 < num_valid_tokens
                        arow_base2 = sArowBase[r_a2]

                        if const_expr(K % block_k == 0):
                            valid_a2 = valid_row_a2
                        else:
                            valid_a2 = valid_row_a2 and (kg_a2 < K_const)
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
                                if n_start + t0BcB[0, n, 0][0] < N_const:
                                    cute.copy(
                                        gmem_tiled_copy_B,
                                        tBgB_next[None, n, None],
                                        tBsB2[None, n, None],
                                    )
                    else:
                        tBpB_next = fa_utils.predicate_k(
                            tBcB, limit=K_const - k_start_next
                        )
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
                                if n_start + t0BcB[0, n, 0][0] < N_const:
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

            # Epilogue: apply routed weights (if needed), then store directly to gmem.
            # Use streaming stores (st.global.cs) which bypass L1 and mark for early L2
            # eviction - optimal for write-only scattered stores.
            cC = cute.make_identity_tensor((block_m, block_n))
            tC_coords = fa_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cC))
            tAcc = fa_utils.make_acc_tensor_mn_view(acc)

            gmem_store_atom_scalar = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=self.dtype.width,
            )
            vec_size_out = 2
            src_scalar = cute.make_rmem_tensor((1,), self.dtype)
            align_bytes_out = vec_size_out * int(self.dtype.width // 8)
            align_bytes_scalar = int(self.dtype.width // 8)

            if full_n_tile:
                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    if aid < num_valid_tokens:
                        row_scale = Float32(1.0)
                        if const_expr(self.mul_routed_weight):
                            row_scale = Float32(sW[m])
                        row_off_bytes = (
                            cutlass.Int64(aid) * stride_cm_elems * element_bytes
                        )
                        n_pairs = cute.size(tAcc.shape[1]) // vec_size_out
                        for pi in cutlass.range_constexpr(n_pairs):
                            ni0 = Int32(pi * vec_size_out)
                            n0 = Int32(tC_coords[mi, ni0][1])
                            col0 = n_start + n0
                            # Pack two BF16 values into Int32 and use streaming store
                            packed = fa_utils.cvt_f16x2_f32(
                                Float32(tAcc[mi, ni0]) * row_scale,
                                Float32(tAcc[mi, ni0 + 1]) * row_scale,
                                self.dtype,
                            )
                            g_off_bytes_vec = (
                                row_off_bytes + cutlass.Int64(col0) * element_bytes
                            )
                            g_ptr_vec = cute.make_ptr(
                                self.dtype,
                                mC_base_i64 + g_off_bytes_vec,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes_out,
                            )
                            store_streaming_b32(packed, g_ptr_vec)
            else:
                for mi in cutlass.range_constexpr(cute.size(tAcc.shape[0])):
                    m = Int32(tC_coords[mi, 0][0])
                    aid = sAid[m]
                    if aid < num_valid_tokens:
                        row_scale = Float32(1.0)
                        if const_expr(self.mul_routed_weight):
                            row_scale = Float32(sW[m])
                        row_off_bytes = (
                            cutlass.Int64(aid) * stride_cm_elems * element_bytes
                        )
                        for ni in cutlass.range_constexpr(cute.size(tAcc.shape[1])):
                            n = Int32(tC_coords[mi, ni][1])
                            col = n_start + n
                            if col < N_const:
                                src_scalar[0] = self.dtype(
                                    Float32(tAcc[mi, ni]) * row_scale
                                )
                                g_off_bytes_scalar = (
                                    row_off_bytes + cutlass.Int64(col) * element_bytes
                                )
                                g_ptr_scalar = cute.make_ptr(
                                    self.dtype,
                                    mC_base_i64 + g_off_bytes_scalar,
                                    cute.AddressSpace.gmem,
                                    assumed_align=align_bytes_scalar,
                                )
                                dst_scalar = cute.make_tensor(g_ptr_scalar, (1,))
                                cute.copy(gmem_store_atom_scalar, src_scalar, dst_scalar)
