# Decode-only fast path for SM90 paged KV cache (page_size == 1).
# This is intentionally separate from the main FA3 forward kernel to keep the decode path small
# and closer to FlashInfer's tight cp.async pipeline structure.

import math
import operator
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from kestrel_kernels.flash_attn.cute import utils


class FlashAttentionDecodeSm90:
    """SM90 decode-only kernel for paged KV, optimized for page_size == 1.

    Supported (initial scope):
    - BF16 / FP16 Q/O
    - BF16 / FP16 / FP8 (E4M3FN) KV cache
    - seqlen_q == 1 (decode)
    - paged KV (page_table != None) with page_size == 1
    - MHA / GQA / MQA (qhead_per_kvhead arbitrary), by grouping q heads per KV head
    - causal + sliding window masking (via restricting the processed KV range)

    Not supported (falls back in interface):
    - score_mod / mask_mod
    - aux_tensors
    - block sparsity
    - head_dim_v != head_dim
    """

    arch = 90

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        qhead_per_kvhead: int,
        *,
        dtype_kv: Optional[type[cutlass.Numeric]] = None,
        is_causal: bool,
        is_local: bool,
        max_qheads_per_block: int = 8,
        target_threads_per_block: int = 128,
        max_bdz: int = 32,
        num_stages_smem: int = 2,
        tile_size_per_bdx: int = 4,
    ):
        self.dtype = dtype
        self.dtype_kv = dtype if dtype_kv is None else dtype_kv
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local

        # CuTe DSL's non-bulk cp.async atom currently supports 128-bit copies.
        # Choose vec_size so that vec_size * sizeof(dtype_kv) == 16 bytes:
        # - BF16/FP16 KV: vec_size=8
        # - FP8 KV: vec_size=16
        if head_dim not in (64, 128):
            raise ValueError(
                f"FlashAttentionDecodeSm90 only supports head_dim 64 or 128 for now (got {head_dim})"
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
        assert head_dim % self.vec_size == 0
        self.bdx = head_dim // self.vec_size
        assert self.bdx in (4, 8, 16)

        self.num_stages_smem = num_stages_smem
        self.tile_size_per_bdx = tile_size_per_bdx
        # Use `threadIdx.y` to represent query heads within a KV head group.
        # For MHA (qhead_per_kvhead == 1), keep bdy=1 and rely on bdz to fill the block.
        # This matches FlashInfer's geometry and avoids introducing inactive ty lanes.
        # For GQA/MQA, we pad `bdy` so that:
        # 1) `blockDim.x * blockDim.y >= 32` (at least one warp in the xy plane), and
        # 2) `blockDim.x * blockDim.y` is a multiple of 32 (warps don't straddle tz planes
        #    when we use `bdz > 1`).
        #
        # NOTE: The padded `ty` lanes are inactive for Q/O, but still participate in the
        # K/V gather + cp.async pipeline to keep the CTA "wide enough".
        bdy_nominal = min(qhead_per_kvhead, max_qheads_per_block)
        bdy_min = (cute.arch.WARP_SIZE + self.bdx - 1) // self.bdx
        warp_multiple = cute.arch.WARP_SIZE // math.gcd(cute.arch.WARP_SIZE, self.bdx)
        if qhead_per_kvhead == 1:
            bdy = 1
            allow_warp_straddle = True
        else:
            bdy = max(bdy_nominal, bdy_min)
            # Round up to a multiple so that threads_xy is warp-aligned.
            bdy = ((bdy + warp_multiple - 1) // warp_multiple) * warp_multiple
            if bdy > max_qheads_per_block:
                # Try to round down within the user's cap.
                bdy = (max_qheads_per_block // warp_multiple) * warp_multiple
                if bdy < bdy_min:
                    # Last-resort: allow padding beyond max_qheads_per_block.
                    bdy = bdy_min
            allow_warp_straddle = False
        self.bdy = bdy
        self.group_count = (qhead_per_kvhead + self.bdy - 1) // self.bdy
        threads_xy = self.bdx * self.bdy
        if target_threads_per_block <= 0:
            raise ValueError("target_threads_per_block must be positive")
        if max_bdz <= 0:
            raise ValueError("max_bdz must be positive")
        # Match FlashInfer-style launch geometry: use threadIdx.z (bdz) to reach a reasonable
        # threads-per-block even for MHA (qhead_per_kvhead == 1).
        self.bdz = max(1, min(max_bdz, target_threads_per_block // threads_xy))
        # Safety: if warps would straddle tz planes, force bdz=1 unless explicitly allowed.
        if threads_xy % cute.arch.WARP_SIZE != 0 and not allow_warp_straddle:
            self.bdz = 1

        # Tokens per iteration tile:
        # - bdy: query head lanes (also used to shard token loads)
        # - bdz: token lanes (required for MHA to avoid tiny CTAs)
        self.tile_tokens_per_tz = self.bdy * self.tile_size_per_bdx
        self.tile_tokens = self.bdz * self.tile_tokens_per_tz

    def _shared_storage_cls(self):
        cosize_kv_stage = self.num_stages_smem * self.tile_tokens * self.head_dim
        # Store offsets as 32-bit units (bytes / 16) to reduce shared-memory traffic and
        # eliminate 64-bit bank conflicts in the (token_in_iter, tx) store pattern.
        #
        # Padding is chosen so that rows separated by `tile_size_per_bdx` (4) map to disjoint
        # shared-memory bank sets for the supported bdx in {4, 8, 16}.
        offset_cols = self.bdx + self.bdx // 4
        cosize_offsets = self.tile_tokens * offset_cols

        sK_struct = cute.struct.Align[cute.struct.MemRange[self.dtype_kv, cosize_kv_stage], 16]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype_kv, cosize_kv_stage], 16]
        sOffsets_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int32, cosize_offsets], 16]

        @cute.struct
        class SharedStorageDecode:
            sK: sK_struct
            sV: sV_struct
            sOffsets: sOffsets_struct

        return SharedStorageDecode

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (num_pages, page_size, h_k, d) if page_table
        mV: cute.Tensor,  # (num_pages, page_size, h_k, d) if page_table
        mO: cute.Tensor,  # (b, s_q, h, d)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors=None,
        aux_tensors: Optional[list] = None,
    ):
        del mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, learnable_sink, blocksparse_tensors, aux_tensors

        assert mPageTable is not None, "DecodeSm90 requires paged KV (mPageTable != None)"

        # Assume all strides are divisible by 128 bits except the last stride.
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )

        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        QO_layout_transpose = [1, 3, 2, 0]
        KV_layout_transpose = [1, 3, 2, 0]
        mQ, mO = [utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        mK, mV = [utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        mLSE = utils.select(mLSE, [2, 1, 0]) if const_expr(mLSE is not None) else None

        # Internal layouts:
        # - Q/O: (seqlen_q, head_dim, num_q_heads, batch)
        # - K/V: (page_size, head_dim, num_kv_heads, num_pages)
        batch_size = mQ.shape[3]
        num_q_heads = mQ.shape[2]
        num_kv_heads = mK.shape[2]

        grid_dim = (batch_size, num_kv_heads * self.group_count, 1)
        SharedStorage = self._shared_storage_cls()
        sK_layout = cute.make_layout(
            (self.num_stages_smem, self.tile_tokens, self.head_dim),
            stride=(self.tile_tokens * self.head_dim, self.head_dim, 1),
        )

        # Offsets are filled in token-major order for coalesced writes.
        offset_cols = self.bdx + self.bdx // 4
        sOffsets_token_major_layout = cute.make_layout(
            (self.tile_tokens, offset_cols), stride=(offset_cols, 1)
        )
        # sOMerge layout for merging partial outputs across bdz partitions.
        # Bank conflict avoidance: we use XOR swizzle in the access pattern.
        # Swizzle formula: c' = c XOR ((c >> 4) * 3) distributes accesses across banks.
        # This is applied manually at each access site (see _swizzle_col).
        sOMerge_layout = cute.make_layout(
            (self.bdz, self.bdy, self.head_dim),
            stride=(self.bdy * self.head_dim, self.head_dim, 1),
        )
        sMD_layout = cute.make_layout((self.bdz, self.bdy), stride=(self.bdy, 1))

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mSeqUsedK,
            mPageTable,
            softmax_scale,
            k_scale,
            v_scale,
            window_size_left,
            window_size_right,
            sK_layout,
            sOffsets_token_major_layout,
            sOMerge_layout,
            sMD_layout,
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
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: cute.Tensor,
        softmax_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sK_layout: cute.Layout,
        sOffsets_token_major_layout: cute.Layout,
        sOMerge_layout: cute.Layout,
        sMD_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        tx, ty, tz = cute.arch.thread_idx()
        batch_idx, block_y, _ = cute.arch.block_idx()

        kv_head_idx = block_y // self.group_count
        group_id = block_y - kv_head_idx * self.group_count
        q_in_kv = group_id * self.bdy + ty
        qo_head_idx = kv_head_idx * self.qhead_per_kvhead + q_in_kv
        qo_head_active = (q_in_kv < self.qhead_per_kvhead) and (qo_head_idx < mQ.shape[2])

        # KV length (per batch) for decode.
        assert (
            mSeqUsedK is not None
        ), "FlashAttentionDecodeSm90 requires seqused_k (mSeqUsedK) for paged KV decode"
        kv_len = mSeqUsedK[batch_idx]
        max_kv_len = Int32(mPageTable.shape[1])
        if kv_len > max_kv_len:
            kv_len = max_kv_len

        # Restrict KV range to implement sliding window masking.
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
        # For decode, causal implies no future tokens; chunk_end is already <= kv_len.
        if const_expr(self.is_causal):
            end = q_pos + 1
            if end < chunk_end:
                chunk_end = end
        if chunk_end < chunk_start:
            chunk_end = chunk_start
        chunk_size = chunk_end - chunk_start

        # Shared memory.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sK_layout)
        sOffsets_token_major = storage.sOffsets.get_tensor(sOffsets_token_major_layout)
        bdx = const_expr(self.bdx)
        assert bdx in (4, 8, 16), "Offset swizzle assumes bdx is a power-of-two (4, 8, or 16)"

        # Compile-time constants for bdx-specific logic.
        offsets_row_shift = bdx.bit_length() - 1
        offsets_col_mask = bdx - 1
        tz_merge_div = cute.arch.WARP_SIZE // bdx
        tz_merge_offsets = tuple(bdx << i for i in range(tz_merge_div.bit_length() - 1))
        tz_merge_mask = tz_merge_div - 1
        tz_merge_shift = tz_merge_div.bit_length() - 1

        # Copy atoms.
        copy_bits = self.vec_size * self.dtype_kv.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_kv,
            num_bits_per_copy=copy_bits,
        )

        element_bytes = cutlass.Int64(self.dtype_kv.width // 8)
        feature_bytes = cutlass.Int64(tx) * cutlass.Int64(self.vec_size) * element_bytes
        # Offsets are stored in units of 16 bytes, matching the alignment requirement for cp.async.
        offset_shift = Int32(4)

        # Compute K/V byte strides for page + head.
        stride_page_elems = cutlass.Int64(mK.stride[3])
        stride_head_elems = cutlass.Int64(mK.stride[2])
        mK_base_i64 = mK.iterator.toint()
        mV_base_i64 = mV.iterator.toint()

        tile_tokens_i32 = Int32(self.tile_tokens)

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
        # Reuse offsets between K/V prefetch loops.
        offset_bytes_vec = cute.make_rmem_tensor((self.tile_size_per_bdx,), cutlass.Int64)

        # Load q into registers for active heads.
        q_vec = cute.make_rmem_tensor((self.vec_size,), Float32)
        q_vec.fill(0.0)
        if qo_head_active:
            for i in cutlass.range_constexpr(self.vec_size):
                q_vec[i] = Float32(mQ[0, tx * self.vec_size + i, qo_head_idx, batch_idx])
        q_vec_ssa = q_vec.load()

        # Softmax state (log2 domain).
        LOG2_E = math.log2(math.e)
        scale_log2 = softmax_scale * Float32(LOG2_E) * k_scale
        m = -Float32.inf
        d = Float32(0.0)
        o = cute.make_rmem_tensor((self.vec_size,), Float32)
        o.fill(0.0)

        # Iteration geometry.
        num_iters = cute.ceil_div(chunk_size, self.tile_tokens)

        # Fill the offset ring-buffer for the first bdx slices (block_start = 0).
        # Token-major fill for coalesced writes: sOffsets_token_major[token_in_iter, slice=tx].
        for j in cutlass.range_constexpr(self.tile_size_per_bdx):
            token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
            linear = token_in_iter * self.bdx + tx
            token_off = linear  # block_start == 0
            token_idx = chunk_start + token_off
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

        sK_base_i64 = sK.iterator.toint()
        sV_base_i64 = sV.iterator.toint()
        align_bytes = self.vec_size * (self.dtype_kv.width // 8)

        smem_feature_elems = tx * Int32(self.vec_size)
        stage_stride_elems = Int32(self.tile_tokens * self.head_dim)
        sKV_stage_layout = cute.make_layout(
            (self.tile_tokens, self.bdx, self.vec_size),
            stride=(self.head_dim, self.vec_size, 1),
        )

        # Preload K/V for the first num_stages iterations.
        stage_idx = Int32(0)
        for iter_prefetch in cutlass.range_constexpr(self.num_stages_smem):
            iter_i = Int32(iter_prefetch)
            slice_idx = iter_i % self.bdx
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
                    token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
                    linear = slice_idx * self.tile_tokens + token_in_iter
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
                    token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
                    token_off = iter_base + token_in_iter
                    valid = token_off < chunk_size
                    linear = slice_idx * self.tile_tokens + token_in_iter
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
                    token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
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
                    token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
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
            stage_idx = stage_idx + 1 if stage_idx + 1 < self.num_stages_smem else Int32(0)

        stage_idx = Int32(0)
        s = cute.make_rmem_tensor((self.tile_tokens_per_tz,), Float32)

        for iter_idx in cutlass.range(num_iters, unroll=2):
            prefetch_iter = iter_idx + self.num_stages_smem
            stage_off_bytes = (
                cutlass.Int64(stage_idx) * cutlass.Int64(stage_stride_elems) * element_bytes
            )
            sK_stage_i64 = sK_base_i64 + stage_off_bytes
            sV_stage_i64 = sV_base_i64 + stage_off_bytes
            # Update offset ring buffer at bdx boundaries for the prefetch iteration.
            if prefetch_iter % self.bdx == 0:
                if prefetch_iter * self.tile_tokens < chunk_size:
                    for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                        token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
                        linear = token_in_iter * self.bdx + tx
                        token_off = prefetch_iter * self.tile_tokens + linear
                        token_idx = chunk_start + token_off
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

            # Compute QK on current stage.
            cute.arch.cp_async_wait_group(self.num_stages_smem - 1)
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
                token_in_stage = tz * self.tile_tokens_per_tz + t
                token_off = iter_idx * self.tile_tokens + token_in_stage
                dot = Float32(0.0)
                if token_off < chunk_size and qo_head_active:
                    k_vec_f32 = sK_stage[token_in_stage, tx, None].load().to(Float32)
                    dot = Float32(
                        (k_vec_f32 * q_vec_ssa).reduce(
                            cute.ReductionOp.ADD, 0.0, reduction_profile=0
                        )
                    )
                # IMPORTANT: all lanes in the warp must execute the shuffle-based reduction.
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
                # Update running softmax stats and rescale output accumulator.
                #
                # When this tz lane has no valid tokens in the current tile, we can have
                # m_prev == m == -inf, which would otherwise produce NaNs in (m_prev - m).
                if m == -Float32.inf:
                    for t in cutlass.range_constexpr(self.tile_tokens_per_tz):
                        s[t] = Float32(0.0)
                else:
                    o_scale = utils.exp2f(m_prev - m)
                    d *= o_scale
                    for i in cutlass.range_constexpr(self.vec_size):
                        o[i] *= o_scale
                    for t in cutlass.range_constexpr(self.tile_tokens_per_tz):
                        s[t] = utils.exp2f(s[t] - m)
                        d += s[t]
            cute.arch.sync_warp()

            # Prefetch next K tile (overwrites current stage).
            if prefetch_iter < num_iters:
                slice_idx = prefetch_iter % self.bdx
                prefetch_base = prefetch_iter * tile_tokens_i32
                full_tile = (prefetch_base + tile_tokens_i32) <= chunk_size
                if full_tile:
                    for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                        token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
                        linear = slice_idx * self.tile_tokens + token_in_iter
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
                        token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
                        token_off = prefetch_base + token_in_iter
                        valid = token_off < chunk_size
                        linear = slice_idx * self.tile_tokens + token_in_iter
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

            # Update output using V for current stage.
            cute.arch.cp_async_wait_group(self.num_stages_smem - 1)
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
                    token_in_stage = tz * self.tile_tokens_per_tz + t
                    token_off = iter_idx * self.tile_tokens + token_in_stage
                    if token_off < chunk_size:
                        wt = s[t]
                        v_vec_f32 = sV_stage[token_in_stage, tx, None].load().to(Float32)
                        for i in cutlass.range_constexpr(self.vec_size):
                            o[i] += wt * v_vec_f32[i]
            cute.arch.sync_warp()

            # Prefetch next V tile.
            if prefetch_iter < num_iters:
                slice_idx = prefetch_iter % self.bdx
                prefetch_base = prefetch_iter * tile_tokens_i32
                full_tile = (prefetch_base + tile_tokens_i32) <= chunk_size
                if full_tile:
                    for j in cutlass.range_constexpr(self.tile_size_per_bdx):
                        token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
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
                        token_in_iter = (tz * self.bdy + ty) * self.tile_size_per_bdx + j
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

            # Advance stage.
            stage_idx = stage_idx + 1 if stage_idx + 1 < self.num_stages_smem else Int32(0)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Merge (m, d, o) across threadIdx.z when bdz > 1 (FlashInfer-style).
        if const_expr(self.bdz > 1):
            # Reuse the K/V shared-memory region for merge scratch instead of reserving a
            # dedicated float buffer for the entire kernel (FlashInfer-style).
            #
            # All cp.async traffic is complete and all threads are synced at this point,
            # so it is safe to overwrite sK/sV.
            cosize_omerge = self.bdz * self.bdy * self.head_dim
            cosize_md = self.bdz * self.bdy
            sOMerge_ptr = cute.make_ptr(
                Float32,
                sK_base_i64,
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            sOMerge = cute.make_tensor(sOMerge_ptr, sOMerge_layout)
            sM_ptr = cute.make_ptr(
                Float32,
                sK_base_i64 + cutlass.Int64(cosize_omerge * 4),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            sM = cute.make_tensor(sM_ptr, sMD_layout)
            sD_ptr = cute.make_ptr(
                Float32,
                sK_base_i64 + cutlass.Int64((cosize_omerge + cosize_md) * 4),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            sD = cute.make_tensor(sD_ptr, sMD_layout)

            if const_expr(self.bdy == 1):
                # In MHA geometry (bdy==1, bdx<32), warps straddle tz lanes. Instead of
                # writing all bdz slices to shared memory and having tz==0 merge 16 slices,
                # first merge tz within each warp via shuffles, then store one slice per warp.
                #
                # This reduces:
                # - Shared-memory store bank conflicts in the fp32 merge scratch.
                # - Shared-memory traffic + work in the final tz==0 merge loop.
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

                    # One writer per warp: tz % tz_merge_div == 0 stores to warp_id = tz // tz_merge_div.
                    if (tz & Int32(tz_merge_mask)) == Int32(0):
                        warp_id = tz >> Int32(tz_merge_shift)
                        for i in cutlass.range_constexpr(self.vec_size):
                            # XOR swizzle to avoid bank conflicts: c' = c ^ ((c >> 4) * 3)
                            c = tx * self.vec_size + i
                            chunk = c >> Int32(4)
                            c_swizzled = c ^ (chunk + (chunk << Int32(1)))
                            sOMerge[warp_id, ty, c_swizzled] = o[i]
                        if tx == 0:
                            sM[warp_id, ty] = m
                            sD[warp_id, ty] = d

                cute.arch.barrier()

                if tz == 0 and qo_head_active:
                    # Merge across warp partials (always 4 for the supported bdx/bdz pairs).
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
                                # XOR swizzle to match the store pattern
                                c = tx * self.vec_size + i
                                chunk = c >> Int32(4)
                                c_swizzled = c ^ (chunk + (chunk << Int32(1)))
                                oz[i] = sOMerge[z, ty, c_swizzled]
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
                # Generic merge path (no warp-level tz reduction).
                if qo_head_active:
                    for i in cutlass.range_constexpr(self.vec_size):
                        # XOR swizzle to avoid bank conflicts: c' = c ^ ((c >> 4) * 3)
                        c = tx * self.vec_size + i
                        chunk = c >> Int32(4)
                        c_swizzled = c ^ (chunk + (chunk << Int32(1)))
                        sOMerge[tz, ty, c_swizzled] = o[i]
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
                                # XOR swizzle to match the store pattern
                                c = tx * self.vec_size + i
                                chunk = c >> Int32(4)
                                c_swizzled = c ^ (chunk + (chunk << Int32(1)))
                                oz[i] = sOMerge[z, ty, c_swizzled]
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

        # Write out (only tz == 0 writes).
        # Apply v_scale here (fused with normalization) instead of per-token in the inner loop.
        if tz == 0 and qo_head_active:
            inv_d = cute.arch.rcp_approx(d + Float32(1e-20)) * v_scale
            for i in cutlass.range_constexpr(self.vec_size):
                mO[0, tx * self.vec_size + i, qo_head_idx, batch_idx] = (o[i] * inv_d).to(
                    self.dtype
                )
            if const_expr(mLSE is not None):
                if tx == 0:
                    LN2 = math.log(2.0)
                    mLSE[0, qo_head_idx, batch_idx] = (
                        m + utils.log2f(d + Float32(1e-20))
                    ) * Float32(LN2)
