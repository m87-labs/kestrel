# Persistent split-KV decode for SM90 paged KV cache (page_size == 1), with in-kernel combine.
#
# This kernel computes per-split partial outputs (Float32) + per-split LSE (Float32),
# then performs a per-(batch, head_group) reduction once all splits are done.
#
# Motivation: avoid the extra decode-combine kernel launch and overlap combine work
# with remaining split tasks, improving CUDA graph replay performance at small batch sizes.

import math
import operator
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync

from kestrel_kernels.flash_attn.cute import utils


class FlashAttentionDecodeSm90PersistentSplitFused:
    """SM90 decode-only split-KV kernel for paged KV, optimized for page_size == 1, with fused combine.

    Produces:
    - out (dtype): final attention output
    - lse (Float32, optional): final log-sum-exp

    Requires scratch:
    - out_partial (Float32): (num_splits, b, s_q, h, d), per-split normalized outputs
    - lse_partial (Float32): (num_splits, b, h, s_q), per-split lse in natural log
    - split_counters (Int32): (b, num_kv_heads * group_count), per-group split completion counters
    """

    arch = 90

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        qhead_per_kvhead: int,
        num_splits: int,
        *,
        dtype_kv: Optional[type[cutlass.Numeric]] = None,
        is_causal: bool,
        is_local: bool,
        split_tokens: int,
        persist_oversub: int = 1,
        max_qheads_per_block: int = 8,
        target_threads_per_block: int = 128,
        max_bdz: int = 32,
        num_stages_smem: int = 2,
        tile_size_per_bdx: int = 4,
    ):
        self.dtype = dtype
        self.dtype_kv = dtype if dtype_kv is None else dtype_kv
        self.head_dim = int(head_dim)
        self.qhead_per_kvhead = int(qhead_per_kvhead)
        self.is_causal = is_causal
        self.is_local = is_local
        self.num_splits = int(num_splits)
        if self.num_splits <= 1:
            raise ValueError(f"num_splits must be > 1 for fused split (got {num_splits})")
        if split_tokens <= 0:
            raise ValueError(f"split_tokens must be positive (got {split_tokens})")
        self.split_tokens = int(split_tokens)
        self.persist_oversub = max(1, int(persist_oversub))

        # Match the baseline SM90 decode fastpath geometry.
        if self.head_dim not in (64, 128):
            raise ValueError(
                "FlashAttentionDecodeSm90PersistentSplitFused only supports head_dim 64 or 128 "
                f"(got {head_dim})"
            )
        kv_bytes = int(self.dtype_kv.width // 8)
        if kv_bytes == 2:
            self.vec_size = 8
        elif kv_bytes == 1:
            self.vec_size = 16
        else:
            raise ValueError(
                f"Unsupported KV dtype width for SM90 decode fastpath: {self.dtype_kv.width} bits"
            )
        assert self.head_dim % self.vec_size == 0
        self.bdx = self.head_dim // self.vec_size
        assert self.bdx in (4, 8, 16)

        self.num_stages_smem = int(num_stages_smem)
        self.tile_size_per_bdx = int(tile_size_per_bdx)

        bdy_nominal = min(self.qhead_per_kvhead, max_qheads_per_block)
        bdy_min = (cute.arch.WARP_SIZE + self.bdx - 1) // self.bdx
        warp_multiple = cute.arch.WARP_SIZE // math.gcd(cute.arch.WARP_SIZE, self.bdx)
        if self.qhead_per_kvhead == 1:
            bdy = 1
            allow_warp_straddle = True
        else:
            bdy = max(bdy_nominal, bdy_min)
            bdy = ((bdy + warp_multiple - 1) // warp_multiple) * warp_multiple
            if bdy > max_qheads_per_block:
                bdy = (max_qheads_per_block // warp_multiple) * warp_multiple
                if bdy < bdy_min:
                    bdy = bdy_min
            allow_warp_straddle = False
        self.bdy = bdy
        self.group_count = (self.qhead_per_kvhead + self.bdy - 1) // self.bdy
        threads_xy = self.bdx * self.bdy
        if target_threads_per_block <= 0:
            raise ValueError("target_threads_per_block must be positive")
        if max_bdz <= 0:
            raise ValueError("max_bdz must be positive")
        self.bdz = max(1, min(int(max_bdz), int(target_threads_per_block) // threads_xy))
        if threads_xy % cute.arch.WARP_SIZE != 0 and not allow_warp_straddle:
            self.bdz = 1

        self.tile_tokens_per_tz = self.bdy * self.tile_size_per_bdx
        self.tile_tokens = self.bdz * self.tile_tokens_per_tz

    def _shared_storage_cls(self):
        cosize_kv_stage = self.num_stages_smem * self.tile_tokens * self.head_dim
        offset_cols = self.bdx + self.bdx // 4
        cosize_offsets = self.tile_tokens * offset_cols

        sK_struct = cute.struct.Align[cute.struct.MemRange[self.dtype_kv, cosize_kv_stage], 16]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype_kv, cosize_kv_stage], 16]
        sOffsets_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, cosize_offsets], 16
        ]

        # One flag per CTA: are we the last split for this (batch, head_group)?
        sIsLast_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 16]

        @cute.struct
        class SharedStorageDecode:
            sK: sK_struct
            sV: sV_struct
            sOffsets: sOffsets_struct
            sIsLast: sIsLast_struct

        return SharedStorageDecode

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (num_pages, page_size, h_k, d)
        mV: cute.Tensor,  # (num_pages, page_size, h_k, d)
        mO: cute.Tensor,  # (b, s_q, h, d) dtype
        mLSE: Optional[cute.Tensor],  # (b, h, s_q) float32
        mSeqUsedK: cute.Tensor,
        mPageTable: cute.Tensor,
        mO_partial: cute.Tensor,  # (num_splits, b, s_q, h, d) float32
        mLSE_partial: cute.Tensor,  # (num_splits, b, h, s_q) float32
        mSplitCounters: cute.Tensor,  # (b, num_kv_heads * group_count) int32
        softmax_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        window_size_left: Int32 | int | None,
        window_size_right: Int32 | int | None,
        stream: cuda.CUstream,
    ):
        assert mPageTable is not None, "PersistentSplitFused requires paged KV (mPageTable != None)"
        assert mO_partial.element_type == Float32, "out_partial must be fp32"
        assert mLSE_partial.element_type == Float32, "lse_partial must be fp32"
        assert mSplitCounters.element_type == cutlass.Int32, "split counters must be int32"

        # Assume all strides are divisible by 128 bits except the last stride.
        def _assume_divisible_stride(s, divby: int):
            return s if isinstance(s, int) else cute.assume(s, divby=divby)

        new_stride = lambda t: (
            *(_assume_divisible_stride(s, 128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )

        mQ, mK, mV, mO, mO_partial, mLSE_partial = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO, mO_partial, mLSE_partial)
        ]

        QO_layout_transpose = [1, 3, 2, 0]
        KV_layout_transpose = [1, 3, 2, 0]
        mQ, mO = [utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        mK, mV = [utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        mLSE = utils.select(mLSE, [2, 1, 0]) if const_expr(mLSE is not None) else None

        batch_size = mQ.shape[3]
        num_kv_heads = Int32(mK.shape[2])
        head_groups = num_kv_heads * Int32(self.group_count)
        total_tasks = batch_size * head_groups * Int32(self.num_splits)

        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        grid_cap = sm_count * Int32(self.persist_oversub)
        # Always cap the launched grid. A larger `total_tasks` increases the amount of work,
        # while `grid_cap` controls how many CTAs run concurrently (persistent scheduling).
        grid_x = Int32(cutlass.min(total_tasks, grid_cap))
        grid_dim = (grid_x, Int32(1), Int32(1))

        SharedStorage = self._shared_storage_cls()
        sK_layout = cute.make_layout(
            (self.num_stages_smem, self.tile_tokens, self.head_dim),
            stride=(self.tile_tokens * self.head_dim, self.head_dim, 1),
        )
        offset_cols = self.bdx + self.bdx // 4
        sOffsets_token_major_layout = cute.make_layout(
            (self.tile_tokens, offset_cols), stride=(offset_cols, 1)
        )
        sOMerge_layout = cute.make_layout(
            (self.bdz, self.bdy, self.head_dim),
            stride=(self.bdy * self.head_dim, self.head_dim, 1),
        )
        sMD_layout = cute.make_layout((self.bdz, self.bdy), stride=(self.bdy, 1))
        sIsLast_layout = cute.make_layout((1,), stride=(1,))

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mSeqUsedK,
            mPageTable,
            mO_partial,
            mLSE_partial,
            mSplitCounters,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            sK_layout,
            sOffsets_token_major_layout,
            sOMerge_layout,
            sMD_layout,
            sIsLast_layout,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.bdx, self.bdy, self.bdz),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mSeqUsedK: cute.Tensor,
        mPageTable: cute.Tensor,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mSplitCounters: cute.Tensor,
        softmax_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sK_layout: cute.Layout,
        sOffsets_token_major_layout: cute.Layout,
        sOMerge_layout: cute.Layout,
        sMD_layout: cute.Layout,
        sIsLast_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        tx, ty, tz = cute.arch.thread_idx()
        is_cta_leader = (tx == 0) and (ty == 0) and (tz == 0)
        task0 = Int32(cute.arch.block_idx()[0])
        grid_stride = Int32(cute.arch.grid_dim()[0])

        num_kv_heads = Int32(mK.shape[2])
        head_groups = num_kv_heads * Int32(self.group_count)
        num_splits = Int32(self.num_splits)
        tasks_per_batch = head_groups * num_splits
        total_tasks = Int32(mQ.shape[3]) * tasks_per_batch

        tasks_per_batch_div = FastDivmodDivisor(tasks_per_batch)
        num_splits_div = FastDivmodDivisor(num_splits)
        group_count_div = FastDivmodDivisor(Int32(self.group_count))

        bdx = const_expr(self.bdx)
        assert bdx in (4, 8, 16)
        offsets_row_shift = bdx.bit_length() - 1
        offsets_col_mask = bdx - 1
        tz_merge_div = cute.arch.WARP_SIZE // bdx
        tz_merge_offsets = tuple(bdx << i for i in range(tz_merge_div.bit_length() - 1))
        tz_merge_mask = tz_merge_div - 1
        tz_merge_shift = tz_merge_div.bit_length() - 1

        # Shared memory.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sK_layout)
        sOffsets_token_major = storage.sOffsets.get_tensor(sOffsets_token_major_layout)
        sIsLast = storage.sIsLast.get_tensor(sIsLast_layout)

        # Copy atoms.
        copy_bits_kv = self.vec_size * self.dtype_kv.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_kv,
            num_bits_per_copy=copy_bits_kv,
        )

        element_bytes = cutlass.Int64(self.dtype_kv.width // 8)
        feature_bytes = cutlass.Int64(tx) * cutlass.Int64(self.vec_size) * element_bytes
        offset_shift = Int32(4)  # offsets in units of 16 bytes (cp.async alignment)

        tile_tokens_i32 = Int32(self.tile_tokens)

        stride_page_elems = cutlass.Int64(mK.stride[3])
        stride_head_elems = cutlass.Int64(mK.stride[2])
        mK_base_i64 = mK.iterator.toint()
        mV_base_i64 = mV.iterator.toint()

        sK_base_i64 = sK.iterator.toint()
        sV_base_i64 = sV.iterator.toint()
        align_bytes = self.vec_size * (self.dtype_kv.width // 8)

        smem_feature_elems = tx * Int32(self.vec_size)
        stage_stride_elems = Int32(self.tile_tokens * self.head_dim)
        sKV_stage_layout = cute.make_layout(
            (self.tile_tokens, self.bdx, self.vec_size),
            stride=(self.head_dim, self.vec_size, 1),
        )

        # For full tiles, issue unpredicated cp.async loads.
        #
        # For partial tiles (tail), use a single predicated cp.async copy to avoid
        # generating dual LDGSTS instructions from control-flow `if/else` around
        # `cute.copy`, and to skip out-of-range loads.
        #
        # Make the predicate a broadcasted (stride-0) vector to match the expected
        # pred shape for vectorized copies without needing per-element fills.
        pred_in_layout = cute.make_layout((self.vec_size,), stride=(0,))
        pred_in = cute.make_rmem_tensor(pred_in_layout, cutlass.Boolean)
        pred_in[0] = False
        offset_bytes_vec = cute.make_rmem_tensor((self.tile_size_per_bdx,), cutlass.Int64)

        LOG2_E = math.log2(math.e)
        LOG2_E_F32 = Float32(LOG2_E)
        scale_log2 = softmax_scale * LOG2_E_F32 * k_scale
        split_tokens = Int32(self.split_tokens)
        num_stages_smem_i32 = Int32(self.num_stages_smem)
        cp_async_wait_n = Int32(self.num_stages_smem - 1)

        for task in cutlass.range(task0, total_tasks, grid_stride):
            batch_idx, rem1 = divmod(task, tasks_per_batch_div)
            block_y, split_idx = divmod(rem1, num_splits_div)

            kv_head_idx, group_id = divmod(block_y, group_count_div)
            q_in_kv = group_id * Int32(self.bdy) + ty
            qo_head_idx = kv_head_idx * Int32(self.qhead_per_kvhead) + q_in_kv
            qo_head_active = (q_in_kv < Int32(self.qhead_per_kvhead)) and (
                qo_head_idx < Int32(mQ.shape[2])
            )

            # KV length (per batch).
            kv_len = mSeqUsedK[batch_idx]
            max_kv_len = Int32(mPageTable.shape[1])
            if kv_len > max_kv_len:
                kv_len = max_kv_len

            # Restrict KV range to implement sliding window masking (baseline logic).
            q_pos = kv_len - 1
            chunk_start = Int32(0)
            chunk_end = kv_len
            if const_expr(self.is_local):
                if const_expr(window_size_left is not None):
                    start = q_pos - window_size_left
                    if start < 0:
                        start = Int32(0)
                    if start > chunk_start:
                        chunk_start = start
                if const_expr(window_size_right is not None):
                    end = q_pos + window_size_right + 1
                    if end < chunk_end:
                        chunk_end = end
            if const_expr(self.is_causal):
                end = q_pos + 1
                if end < chunk_end:
                    chunk_end = end
            if chunk_end < chunk_start:
                chunk_end = chunk_start

            # Split planning:
            # - Compile with a small fixed max num_splits (self.num_splits).
            # - Compute active_splits from the *actual* KV work (chunk_end - chunk_start),
            #   then partition the chunk across [0, active_splits).
            chunk_size_total = chunk_end - chunk_start
            active_splits = (chunk_size_total + split_tokens - 1) // split_tokens
            if active_splits < Int32(1):
                active_splits = Int32(1)
            if active_splits > num_splits:
                active_splits = num_splits

            active_split = split_idx < active_splits
            split_start = chunk_start
            split_end = chunk_start
            if active_split:
                base = chunk_size_total // active_splits
                rem = chunk_size_total - base * active_splits
                split_start = chunk_start + split_idx * base + cutlass.min(split_idx, rem)
                split_end = split_start + base + (Int32(1) if split_idx < rem else Int32(0))

            chunk_size = split_end - split_start
            num_iters = (chunk_size + Int32(self.tile_tokens) - 1) // Int32(self.tile_tokens)

            # Load Q vector.
            q_vec = cute.make_rmem_tensor((self.vec_size,), Float32)
            if qo_head_active and active_split:
                for i in cutlass.range_constexpr(self.vec_size):
                    q_vec[i] = Float32(mQ[0, tx * Int32(self.vec_size) + i, qo_head_idx, batch_idx])
            else:
                q_vec.fill(0.0)
            q_vec_ssa = q_vec.load()

            # Softmax accumulators.
            m = -Float32.inf
            d = Float32(0.0)
            o = cute.make_rmem_tensor((self.vec_size,), Float32)
            o.fill(0.0)

            # Fill the offset ring buffer for the first bdx slices (block_start = 0).
            if chunk_size > 0:
                for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                    token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                    linear = token_in_iter * Int32(self.bdx) + tx
                    token_off = linear
                    token_idx = split_start + token_off
                    if token_off < chunk_size:
                        page = mPageTable[batch_idx, token_idx]
                        offset_elems = cutlass.Int64(page) * stride_page_elems + cutlass.Int64(
                            kv_head_idx
                        ) * stride_head_elems
                        offset_bytes = offset_elems * element_bytes
                        sOffsets_token_major[token_in_iter, tx] = Int32(offset_bytes >> offset_shift)
                    else:
                        sOffsets_token_major[token_in_iter, tx] = 0
                cute.arch.barrier()

                # Preload K/V for the first num_stages iterations.
                stage_idx = Int32(0)
                for iter_prefetch in cutlass.range_constexpr(self.num_stages_smem):
                    iter_i = Int32(iter_prefetch)
                    slice_idx = iter_i % Int32(self.bdx)
                    stage_off_bytes = (
                        cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_elems) * element_bytes
                    )
                    sK_stage_i64 = sK_base_i64 + stage_off_bytes
                    sV_stage_i64 = sV_base_i64 + stage_off_bytes
                    iter_base = iter_i * tile_tokens_i32
                    full_tile = (iter_base + tile_tokens_i32) <= chunk_size
                    # K
                    if full_tile:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            linear = slice_idx * Int32(self.tile_tokens) + token_in_iter
                            row = linear >> offsets_row_shift
                            col = linear & offsets_col_mask
                            offset16_i32 = sOffsets_token_major[row, col]
                            offset16_u32 = cutlass.Uint32(offset16_i32)
                            offset_bytes = cutlass.Int64(offset16_u32) << offset_shift
                            offset_bytes_vec[j] = offset_bytes
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mK_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sK_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            cute.copy(atom_async_copy, src, dst)
                    else:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            token_off = iter_base + token_in_iter
                            valid = token_off < chunk_size
                            linear = slice_idx * Int32(self.tile_tokens) + token_in_iter
                            row = linear >> offsets_row_shift
                            col = linear & offsets_col_mask
                            offset16_i32 = sOffsets_token_major[row, col]
                            offset16_u32 = cutlass.Uint32(offset16_i32)
                            offset_bytes = cutlass.Int64(offset16_u32) << offset_shift
                            offset_bytes_vec[j] = offset_bytes
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mK_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sK_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            pred_in[0] = valid
                            cute.copy(atom_async_copy, src, dst, pred=pred_in)
                    # V
                    if full_tile:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            offset_bytes = offset_bytes_vec[j]
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mV_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sV_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            cute.copy(atom_async_copy, src, dst)
                    else:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            token_off = iter_base + token_in_iter
                            valid = token_off < chunk_size
                            offset_bytes = offset_bytes_vec[j]
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mV_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sV_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            pred_in[0] = valid
                            cute.copy(atom_async_copy, src, dst, pred=pred_in)
                    cute.arch.cp_async_commit_group()
                    stage_idx = (
                        stage_idx + 1
                        if stage_idx + 1 < num_stages_smem_i32
                        else Int32(0)
                    )

            stage_idx = Int32(0)
            s = cute.make_rmem_tensor((self.tile_tokens_per_tz,), Float32)

            for iter_idx in cutlass.range(num_iters, unroll=2):
                prefetch_iter = iter_idx + num_stages_smem_i32
                stage_off_bytes = (
                    cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_elems) * element_bytes
                )
                sK_stage_i64 = sK_base_i64 + stage_off_bytes
                sV_stage_i64 = sV_base_i64 + stage_off_bytes
                if prefetch_iter % Int32(self.bdx) == 0:
                    if prefetch_iter * Int32(self.tile_tokens) < chunk_size:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            linear = token_in_iter * Int32(self.bdx) + tx
                            token_off = prefetch_iter * Int32(self.tile_tokens) + linear
                            token_idx = split_start + token_off
                            if token_off < chunk_size:
                                page = mPageTable[batch_idx, token_idx]
                                offset_elems = cutlass.Int64(page) * stride_page_elems + cutlass.Int64(
                                    kv_head_idx
                                ) * stride_head_elems
                                offset_bytes = offset_elems * element_bytes
                                sOffsets_token_major[token_in_iter, tx] = Int32(offset_bytes >> offset_shift)
                            else:
                                sOffsets_token_major[token_in_iter, tx] = 0
                        cute.arch.barrier()

                cute.arch.cp_async_wait_group(cp_async_wait_n)
                cute.arch.barrier()
                sK_stage_ptr = cute.make_ptr(
                    self.dtype_kv,
                    sK_stage_i64,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                sK_stage = cute.make_tensor(sK_stage_ptr, sKV_stage_layout)
                m_prev = m
                for t in cutlass.range_constexpr(self.tile_tokens_per_tz):
                    token_in_stage = tz * Int32(self.tile_tokens_per_tz) + t
                    token_off = iter_idx * Int32(self.tile_tokens) + token_in_stage
                    dot = Float32(0.0)
                    if token_off < chunk_size and qo_head_active:
                        k_vec_f32 = sK_stage[token_in_stage, tx, None].load().to(Float32)
                        dot = Float32(
                            (k_vec_f32 * q_vec_ssa).reduce(
                                cute.ReductionOp.ADD, 0.0, reduction_profile=0
                            )
                        )
                    dot = utils.warp_reduce(dot, operator.add, width=self.bdx)
                    if qo_head_active:
                        if token_off < chunk_size:
                            s[t] = dot * scale_log2
                        else:
                            s[t] = -Float32.inf
                        m = utils.fmax(m, s[t])
                    else:
                        s[t] = Float32(0.0)

                if qo_head_active:
                    # If a tz plane has no valid tokens, keep (m=-inf, d=0) and avoid
                    # exp2f(-inf - -inf) which can yield NaNs and poison the merge.
                    if m != -Float32.inf:
                        o_scale = utils.exp2f(m_prev - m)
                        d *= o_scale
                        for i in cutlass.range_constexpr(self.vec_size):
                            o[i] *= o_scale
                        for t in cutlass.range_constexpr(self.tile_tokens_per_tz):
                            s[t] = utils.exp2f(s[t] - m)
                            d += s[t]
                cute.arch.sync_warp()

                if prefetch_iter < num_iters:
                    slice_idx = prefetch_iter % Int32(self.bdx)
                    prefetch_base = prefetch_iter * tile_tokens_i32
                    full_tile = (prefetch_base + tile_tokens_i32) <= chunk_size
                    if full_tile:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            linear = slice_idx * Int32(self.tile_tokens) + token_in_iter
                            row = linear >> offsets_row_shift
                            col = linear & offsets_col_mask
                            offset16_i32 = sOffsets_token_major[row, col]
                            offset16_u32 = cutlass.Uint32(offset16_i32)
                            offset_bytes = cutlass.Int64(offset16_u32) << offset_shift
                            offset_bytes_vec[j] = offset_bytes
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mK_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sK_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            cute.copy(atom_async_copy, src, dst)
                    else:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            token_off = prefetch_base + token_in_iter
                            valid = token_off < chunk_size
                            linear = slice_idx * Int32(self.tile_tokens) + token_in_iter
                            row = linear >> offsets_row_shift
                            col = linear & offsets_col_mask
                            offset16_i32 = sOffsets_token_major[row, col]
                            offset16_u32 = cutlass.Uint32(offset16_i32)
                            offset_bytes = cutlass.Int64(offset16_u32) << offset_shift
                            offset_bytes_vec[j] = offset_bytes
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mK_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sK_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            pred_in[0] = valid
                            cute.copy(atom_async_copy, src, dst, pred=pred_in)

                cute.arch.cp_async_wait_group(cp_async_wait_n)
                cute.arch.barrier()
                sV_stage_ptr = cute.make_ptr(
                    self.dtype_kv,
                    sV_stage_i64,
                    cute.AddressSpace.smem,
                    assumed_align=align_bytes,
                )
                sV_stage = cute.make_tensor(sV_stage_ptr, sKV_stage_layout)
                if qo_head_active:
                    for t in cutlass.range_constexpr(self.tile_tokens_per_tz):
                        token_in_stage = tz * Int32(self.tile_tokens_per_tz) + t
                        token_off = iter_idx * Int32(self.tile_tokens) + token_in_stage
                        if token_off < chunk_size:
                            wt = s[t]
                            v_vec_f32 = sV_stage[token_in_stage, tx, None].load().to(Float32)
                            for i in cutlass.range_constexpr(self.vec_size):
                                o[i] += wt * v_vec_f32[i]
                cute.arch.sync_warp()

                if prefetch_iter < num_iters:
                    prefetch_base = prefetch_iter * tile_tokens_i32
                    full_tile = (prefetch_base + tile_tokens_i32) <= chunk_size
                    if full_tile:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            offset_bytes = offset_bytes_vec[j]
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mV_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sV_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            cute.copy(atom_async_copy, src, dst)
                    else:
                        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                            token_in_iter = (tz * Int32(self.bdy) + ty) * Int32(self.tile_size_per_bdx) + j
                            token_off = prefetch_base + token_in_iter
                            valid = token_off < chunk_size
                            offset_bytes = offset_bytes_vec[j]
                            gmem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                mV_base_i64 + offset_bytes + feature_bytes,
                                cute.AddressSpace.gmem,
                                assumed_align=align_bytes,
                            )
                            src = cute.make_tensor(gmem_ptr, (self.vec_size,))

                            smem_row = token_in_iter * Int32(self.head_dim) + smem_feature_elems
                            smem_byte_off = cutlass.Int64(smem_row) * element_bytes
                            smem_ptr = cute.make_ptr(
                                self.dtype_kv,
                                sV_stage_i64 + smem_byte_off,
                                cute.AddressSpace.smem,
                                assumed_align=align_bytes,
                            )
                            dst = cute.make_tensor(smem_ptr, (self.vec_size,))
                            pred_in[0] = valid
                            cute.copy(atom_async_copy, src, dst, pred=pred_in)
                cute.arch.cp_async_commit_group()
                stage_idx = (
                    stage_idx + 1
                    if stage_idx + 1 < num_stages_smem_i32
                    else Int32(0)
                )

            if active_split:
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                # Merge (m, d, o) across threadIdx.z when bdz > 1 (FlashInfer-style).
                if const_expr(self.bdz > 1):
                    cosize_omerge = Int32(self.bdz) * Int32(self.bdy) * Int32(self.head_dim)
                    cosize_md = Int32(self.bdz) * Int32(self.bdy)
                    sOMerge_ptr = cute.make_ptr(
                        Float32,
                        sK_base_i64,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    sOMerge = cute.make_tensor(sOMerge_ptr, sOMerge_layout)
                    sM_ptr = cute.make_ptr(
                        Float32,
                        sK_base_i64 + cutlass.Int64(cosize_omerge) * cutlass.Int64(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    sM = cute.make_tensor(sM_ptr, sMD_layout)
                    sD_ptr = cute.make_ptr(
                        Float32,
                        sK_base_i64
                        + (cutlass.Int64(cosize_omerge) + cutlass.Int64(cosize_md)) * cutlass.Int64(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    sD = cute.make_tensor(sD_ptr, sMD_layout)

                    if const_expr(self.bdy == 1):
                        if qo_head_active:
                            o_other = cute.make_rmem_tensor((self.vec_size,), Float32)
                            for offset in tz_merge_offsets:
                                m2 = cute.arch.shuffle_sync_bfly(m, offset=offset)
                                d2 = cute.arch.shuffle_sync_bfly(d, offset=offset)
                                for i in cutlass.range_constexpr(self.vec_size):
                                    o_other[i] = cute.arch.shuffle_sync_bfly(o[i], offset=offset)
                                if d == Float32(0.0):
                                    m = m2
                                    d = d2
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        o[i] = o_other[i]
                                elif d2 != Float32(0.0):
                                    m_new = utils.fmax(m, m2)
                                    scale_self = utils.exp2f(m - m_new)
                                    scale_other = utils.exp2f(m2 - m_new)
                                    d = d * scale_self + d2 * scale_other
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        o[i] = o[i] * scale_self + o_other[i] * scale_other
                                    m = m_new

                            if (tz & Int32(tz_merge_mask)) == Int32(0):
                                warp_id = tz >> Int32(tz_merge_shift)
                                for i in cutlass.range_constexpr(self.vec_size):
                                    sOMerge[warp_id, ty, tx * Int32(self.vec_size) + i] = o[i]
                                if tx == 0:
                                    sM[warp_id, ty] = m
                                    sD[warp_id, ty] = d

                        cute.arch.barrier()

                        if tz == 0 and qo_head_active:
                            num_partials = self.bdz // tz_merge_div
                            m_merge = -Float32.inf
                            d_merge = Float32(0.0)
                            o_merge = cute.make_rmem_tensor((self.vec_size,), Float32)
                            o_merge.fill(0.0)
                            oz = cute.make_rmem_tensor((self.vec_size,), Float32)
                            for z in cutlass.range_constexpr(num_partials):
                                dz = sD[z, ty]
                                if dz > Float32(0.0):
                                    mz = sM[z, ty]
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        oz[i] = sOMerge[z, ty, tx * Int32(self.vec_size) + i]
                                    if d_merge == Float32(0.0):
                                        m_merge = mz
                                        d_merge = dz
                                        for i in cutlass.range_constexpr(self.vec_size):
                                            o_merge[i] = oz[i]
                                    else:
                                        m_new = utils.fmax(m_merge, mz)
                                        scale_old = utils.exp2f(m_merge - m_new)
                                        scale_z = utils.exp2f(mz - m_new)
                                        d_merge = d_merge * scale_old + dz * scale_z
                                        for i in cutlass.range_constexpr(self.vec_size):
                                            o_merge[i] = o_merge[i] * scale_old + oz[i] * scale_z
                                        m_merge = m_new
                            m = m_merge
                            d = d_merge
                            for i in cutlass.range_constexpr(self.vec_size):
                                o[i] = o_merge[i]
                    else:
                        if qo_head_active:
                            for i in cutlass.range_constexpr(self.vec_size):
                                sOMerge[tz, ty, tx * Int32(self.vec_size) + i] = o[i]
                            if tx == 0:
                                sM[tz, ty] = m
                                sD[tz, ty] = d
                        cute.arch.barrier()

                        if tz == 0 and qo_head_active:
                            m_merge = -Float32.inf
                            d_merge = Float32(0.0)
                            o_merge = cute.make_rmem_tensor((self.vec_size,), Float32)
                            o_merge.fill(0.0)
                            oz = cute.make_rmem_tensor((self.vec_size,), Float32)
                            for z in cutlass.range_constexpr(self.bdz):
                                dz = sD[z, ty]
                                if dz > Float32(0.0):
                                    mz = sM[z, ty]
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        oz[i] = sOMerge[z, ty, tx * Int32(self.vec_size) + i]
                                    if d_merge == Float32(0.0):
                                        m_merge = mz
                                        d_merge = dz
                                        for i in cutlass.range_constexpr(self.vec_size):
                                            o_merge[i] = oz[i]
                                    else:
                                        m_new = utils.fmax(m_merge, mz)
                                        scale_old = utils.exp2f(m_merge - m_new)
                                        scale_z = utils.exp2f(mz - m_new)
                                        d_merge = d_merge * scale_old + dz * scale_z
                                        for i in cutlass.range_constexpr(self.vec_size):
                                            o_merge[i] = o_merge[i] * scale_old + oz[i] * scale_z
                                        m_merge = m_new
                            m = m_merge
                            d = d_merge
                            for i in cutlass.range_constexpr(self.vec_size):
                                o[i] = o_merge[i]

            # Write out partials (only tz == 0 writes).
            # Apply v_scale here (fused with normalization) instead of per-token in the inner loop.
            if active_split and tz == 0 and qo_head_active:
                if chunk_size > 0:
                    inv_d = cute.arch.rcp_approx(d + Float32(1e-20)) * v_scale
                    for i in cutlass.range_constexpr(self.vec_size):
                        idx = Int32(self.bdx) * i + tx
                        mO_partial[split_idx, batch_idx, 0, qo_head_idx, idx] = (
                            o[i] * inv_d
                        )
                    if tx == 0:
                        LN2 = math.log(2.0)
                        mLSE_partial[split_idx, batch_idx, qo_head_idx, 0] = (
                            m + utils.log2f(d + Float32(1e-20))
                        ) * Float32(LN2)
                else:
                    for i in cutlass.range_constexpr(self.vec_size):
                        idx = Int32(self.bdx) * i + tx
                        mO_partial[split_idx, batch_idx, 0, qo_head_idx, idx] = Float32(
                            0.0
                        )
                    if tx == 0:
                        mLSE_partial[split_idx, batch_idx, qo_head_idx, 0] = -Float32.inf

            if self.bdx * self.bdy <= cute.arch.WARP_SIZE:
                # Warp-scoped epilogue:
                # When (bdx * bdy) <= 32, partial writes, atomic publish, and combine all live in a
                # single warp. Avoid CTA-wide sync_threads() barriers which stall unrelated warps.
                is_last = Int32(0)
                if active_split:
                    if is_cta_leader:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        old = utils.atomic_add_release_i32(Int32(1), ctr_ptr)
                        is_last = Int32(1) if old == (active_splits - 1) else Int32(0)
                    # Broadcast from lane 0 in each warp.
                    is_last = utils.shuffle_sync(is_last, 0)

                if active_split and is_last != Int32(0):
                    # Acquire to ensure we see all split partial writes for this group.
                    if tz == 0 and qo_head_active:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        _ = utils.ld_acquire_i32(ctr_ptr)

                        # Combine: only tz == 0 writes final outputs.
                        if const_expr(self.num_splits == 2):
                            has_s1 = active_splits > Int32(1)
                            lse0 = mLSE_partial[0, batch_idx, qo_head_idx, 0]
                            lse1 = -Float32.inf
                            if has_s1:
                                lse1 = mLSE_partial[1, batch_idx, qo_head_idx, 0]

                            lse_max = utils.fmax(lse0, lse1)
                            lse_max_cur = (
                                Float32(0.0) if lse_max == -Float32.inf else lse_max
                            )

                            w0 = utils.exp2f((lse0 - lse_max_cur) * LOG2_E_F32)
                            denom = w0

                            acc = cute.make_rmem_tensor((self.vec_size,), Float32)
                            for i in cutlass.range_constexpr(self.vec_size):
                                acc[i] = w0 * mO_partial[
                                    0,
                                    batch_idx,
                                    0,
                                    qo_head_idx,
                                    Int32(self.bdx) * i + tx,
                                ]

                            if has_s1:
                                w1 = utils.exp2f((lse1 - lse_max_cur) * LOG2_E_F32)
                                denom += w1
                                for i in cutlass.range_constexpr(self.vec_size):
                                    acc[i] += w1 * mO_partial[
                                        1,
                                        batch_idx,
                                        0,
                                        qo_head_idx,
                                        Int32(self.bdx) * i + tx,
                                    ]
                        else:
                            lse_max = -Float32.inf
                            for s_idx in cutlass.range_constexpr(self.num_splits):
                                lse_s = -Float32.inf
                                if Int32(s_idx) < active_splits:
                                    lse_s = mLSE_partial[s_idx, batch_idx, qo_head_idx, 0]
                                lse_max = utils.fmax(lse_max, lse_s)

                            lse_max_cur = (
                                Float32(0.0) if lse_max == -Float32.inf else lse_max
                            )
                            denom = Float32(0.0)
                            acc = cute.make_rmem_tensor((self.vec_size,), Float32)
                            acc.fill(0.0)
                            for s_idx in cutlass.range_constexpr(self.num_splits):
                                if Int32(s_idx) < active_splits:
                                    lse_s = mLSE_partial[s_idx, batch_idx, qo_head_idx, 0]
                                    wi = utils.exp2f((lse_s - lse_max_cur) * LOG2_E_F32)
                                    denom += wi
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        idx = Int32(self.bdx) * i + tx
                                        acc[i] += wi * mO_partial[
                                            s_idx,
                                            batch_idx,
                                            0,
                                            qo_head_idx,
                                            idx,
                                        ]
                        inv_denom = (
                            Float32(0.0)
                            if (denom == Float32(0.0) or denom != denom)
                            else Float32(1.0) / denom
                        )

                        for i in cutlass.range_constexpr(self.vec_size):
                            acc[i] *= inv_denom
                        for i in cutlass.range_constexpr(self.vec_size):
                            mO[0, tx * Int32(self.vec_size) + i, qo_head_idx, batch_idx] = acc[
                                i
                            ].to(self.dtype)
                        if const_expr(mLSE is not None):
                            if tx == 0:
                                mLSE[0, qo_head_idx, batch_idx] = utils.logf(denom) + lse_max

                    # Reset counter for this group so the buffer can be reused on next replay.
                    if is_cta_leader:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        utils.atomic_exch_i32(Int32(0), ctr_ptr)
            else:
                if active_split:
                    # Ensure partial writes are visible before we publish split completion.
                    cute.arch.sync_threads()

                    # One counter increment per (batch, head_group, split).
                    if is_cta_leader:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        # Publication ordering note:
                        # Partial outputs are written by multiple warps in this CTA.
                        # Using only a release atomic here was insufficient for correctness (we observed
                        # mismatches in head_dim=128, qhead_per_kvhead=4, return_lse=True).
                        # Use acq_rel as a stronger publish barrier before the last-split consumer reads.
                        old = utils.atomic_add_acq_rel_i32(Int32(1), ctr_ptr)
                        sIsLast[0] = Int32(1) if old == (active_splits - 1) else Int32(0)
                    cute.arch.sync_threads()

                if active_split and sIsLast[0] != Int32(0):
                    # Acquire to ensure we see all split partial writes for this group.
                    if tz == 0 and qo_head_active:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        _ = utils.ld_acquire_i32(ctr_ptr)
                    cute.arch.sync_threads()

                    # Combine: only tz == 0 writes final outputs.
                    if tz == 0 and qo_head_active:
                        # Compute lse_max and denom (unnormalized weights).
                        if const_expr(self.num_splits == 2):
                            has_s1 = active_splits > Int32(1)
                            lse0 = mLSE_partial[0, batch_idx, qo_head_idx, 0]
                            lse1 = -Float32.inf
                            if has_s1:
                                lse1 = mLSE_partial[1, batch_idx, qo_head_idx, 0]

                            lse_max = utils.fmax(lse0, lse1)
                            lse_max_cur = (
                                Float32(0.0) if lse_max == -Float32.inf else lse_max
                            )

                            w0 = utils.exp2f((lse0 - lse_max_cur) * LOG2_E_F32)
                            denom = w0

                            acc = cute.make_rmem_tensor((self.vec_size,), Float32)
                            for i in cutlass.range_constexpr(self.vec_size):
                                acc[i] = w0 * mO_partial[
                                    0,
                                    batch_idx,
                                    0,
                                    qo_head_idx,
                                    Int32(self.bdx) * i + tx,
                                ]

                            if has_s1:
                                w1 = utils.exp2f((lse1 - lse_max_cur) * LOG2_E_F32)
                                denom += w1
                                for i in cutlass.range_constexpr(self.vec_size):
                                    acc[i] += w1 * mO_partial[
                                        1,
                                        batch_idx,
                                        0,
                                        qo_head_idx,
                                        Int32(self.bdx) * i + tx,
                                    ]
                        else:
                            lse_max = -Float32.inf
                            for s_idx in cutlass.range_constexpr(self.num_splits):
                                lse_s = -Float32.inf
                                if Int32(s_idx) < active_splits:
                                    lse_s = mLSE_partial[s_idx, batch_idx, qo_head_idx, 0]
                                lse_max = utils.fmax(lse_max, lse_s)

                            lse_max_cur = (
                                Float32(0.0) if lse_max == -Float32.inf else lse_max
                            )
                            denom = Float32(0.0)
                            acc = cute.make_rmem_tensor((self.vec_size,), Float32)
                            acc.fill(0.0)
                            for s_idx in cutlass.range_constexpr(self.num_splits):
                                if Int32(s_idx) < active_splits:
                                    lse_s = mLSE_partial[s_idx, batch_idx, qo_head_idx, 0]
                                    wi = utils.exp2f((lse_s - lse_max_cur) * LOG2_E_F32)
                                    denom += wi
                                    for i in cutlass.range_constexpr(self.vec_size):
                                        idx = Int32(self.bdx) * i + tx
                                        acc[i] += wi * mO_partial[
                                            s_idx,
                                            batch_idx,
                                            0,
                                            qo_head_idx,
                                            idx,
                                        ]
                        inv_denom = (
                            Float32(0.0)
                            if (denom == Float32(0.0) or denom != denom)
                            else Float32(1.0) / denom
                        )

                        for i in cutlass.range_constexpr(self.vec_size):
                            acc[i] *= inv_denom
                        for i in cutlass.range_constexpr(self.vec_size):
                            mO[0, tx * Int32(self.vec_size) + i, qo_head_idx, batch_idx] = acc[
                                i
                            ].to(self.dtype)
                        if const_expr(mLSE is not None):
                            if tx == 0:
                                mLSE[0, qo_head_idx, batch_idx] = utils.logf(denom) + lse_max

                    cute.arch.sync_threads()

                    # Reset counter for this group so the buffer can be reused on next replay.
                    if is_cta_leader:
                        ctr_ptr = utils.elem_pointer(mSplitCounters, (batch_idx, block_y))
                        utils.atomic_exch_i32(Int32(0), ctr_ptr)
                        sIsLast[0] = Int32(0)
                    cute.arch.sync_threads()
