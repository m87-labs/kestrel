"""MoE align block size kernel implementation.

This module contains the CuTe DSL kernel templates for MoE alignment operations.
It is NOT included in the distributed wheel - only used for JIT compilation
and precompilation during development.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_i32(a: int | Int32, ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    # `nvvm.atomicrmw` returns the old value (like CUDA's atomicAdd).
    return Int32(
        nvvm.atomicrmw(
            res=T.i32(),
            op=nvvm.AtomicOpKind.ADD,
            ptr=ptr.llvm_ptr,
            a=Int32(a).ir_value(loc=loc, ip=ip),
        )
    )


@dataclass(frozen=True)
class MoeAlignCuTeConfig:
    # Small path (single-CTA) tuned for decode-like TK.
    small_threads: int = 256
    # Large path (2 kernels): use a large block for the align kernel because the
    # grid is only 2 CTAs (one for counts/offsets, one for sentinel fill).
    large_align_threads: int = 1024
    large_scatter_threads: int = 256


class _MoeAlignBlockSizeCuTe:
    def __init__(
        self,
        *,
        num_experts: int,
        block_size: int,
        has_expert_map: bool,
        config: MoeAlignCuTeConfig,
    ) -> None:
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)
        self.has_expert_map = bool(has_expert_map)
        self.config = config

    @cute.jit
    def __call__(
        self,
        topk_ids: cute.Tensor,  # [T, K] integral
        sorted_token_ids: cute.Tensor,  # [max_num_tokens_padded] int32
        expert_ids: cute.Tensor,  # [max_num_m_blocks] int32
        num_tokens_post_pad: cute.Tensor,  # [1] int32
        expert_map: cute.Tensor,  # [num_experts] int32
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, expert_map).launch(
            grid=[1, 1, 1],
            block=[self.config.small_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        topk_ids: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_ids: cute.Tensor,
        num_tokens_post_pad: cute.Tensor,
        expert_map: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()

        # Flatten topk_ids to 1D [TK] so we can avoid div/mod in the hot loops.
        numel_i32 = Int32(cute.size(topk_ids.shape))
        topk_flat = cute.make_tensor(
            topk_ids.iterator,
            cute.make_layout((numel_i32,), stride=(1,)),
        )

        # Shared-memory bookkeeping (E <= 64 in Kestrel's typical MoE shapes).
        @cute.struct
        class SharedStorage:
            counts: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]
            offsets: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts + 1], 16]
            counters: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        counts = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
        offsets = storage.offsets.get_tensor(cute.make_layout((self.num_experts + 1,)))
        counters = storage.counters.get_tensor(cute.make_layout((self.num_experts,)))

        # Zero counts.
        if tidx < self.num_experts:
            counts[tidx] = Int32(0)
        cute.arch.sync_threads()

        # Count tokens per (mapped) expert.
        for i in range(tidx, numel_i32, self.config.small_threads):
            expert_id = Int32(topk_flat[i])
            if expert_id < self.num_experts:
                if const_expr(self.has_expert_map):
                    mapped = expert_map[expert_id]
                    if mapped != -1:
                        atomic_add_i32(1, elem_pointer(counts, mapped))
                else:
                    atomic_add_i32(1, elem_pointer(counts, expert_id))
        cute.arch.sync_threads()

        # Prefix sum over padded expert counts.
        if tidx == 0:
            offsets[0] = Int32(0)
            running = Int32(0)
            for e in cutlass.range(self.num_experts, unroll_full=True):
                c = counts[e]
                padded = ((c + (self.block_size - 1)) // self.block_size) * self.block_size
                running += padded
                offsets[e + 1] = running
            num_tokens_post_pad[0] = running
        cute.arch.sync_threads()

        total_padded_tokens = offsets[self.num_experts]
        total_blocks = total_padded_tokens // self.block_size

        # Initialize sorted_token_ids with sentinel values in the consumed region.
        for idx in range(tidx, total_padded_tokens, self.config.small_threads):
            sorted_token_ids[idx] = numel_i32

        # Write expert_ids per M-block.
        if tidx < self.num_experts:
            start = offsets[tidx]
            end = offsets[tidx + 1]
            for j in range(start, end, self.block_size):
                expert_ids[j // self.block_size] = tidx

        # Fill remaining expert_ids with inactive expert id (=0), matching the CUDA kernel.
        max_blocks = Int32(expert_ids.shape[0])
        for b in range(total_blocks + tidx, max_blocks, self.config.small_threads):
            expert_ids[b] = Int32(0)

        # Scatter token indices into the padded expert-major layout.
        if tidx < self.num_experts:
            counters[tidx] = Int32(0)
        cute.arch.sync_threads()

        for i in range(tidx, numel_i32, self.config.small_threads):
            expert_id = Int32(topk_flat[i])
            if expert_id < self.num_experts:
                if const_expr(self.has_expert_map):
                    mapped = expert_map[expert_id]
                    if mapped != -1:
                        pos = atomic_add_i32(1, elem_pointer(counters, mapped))
                        sorted_token_ids[offsets[mapped] + pos] = Int32(i)
                else:
                    pos = atomic_add_i32(1, elem_pointer(counters, expert_id))
                    sorted_token_ids[offsets[expert_id] + pos] = Int32(i)


class _MoeAlignBlockSizeCuTeLarge:
    def __init__(
        self,
        *,
        num_experts: int,
        block_size: int,
        has_expert_map: bool,
        config: MoeAlignCuTeConfig,
    ) -> None:
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)
        self.has_expert_map = bool(has_expert_map)
        self.config = config

    @cute.jit
    def __call__(
        self,
        topk_ids: cute.Tensor,  # [T, K] integral
        sorted_token_ids: cute.Tensor,  # [max_num_tokens_padded] int32
        expert_ids: cute.Tensor,  # [max_num_m_blocks] int32
        num_tokens_post_pad: cute.Tensor,  # [1] int32
        expert_map: cute.Tensor,  # [num_experts] int32
        cumsum_buffer: cute.Tensor,  # [num_experts] int32
        stream: cuda.CUstream,
    ) -> None:
        # Kernel 1: compute offsets/expert_ids, initialize cumsum_buffer, and fill sorted_token_ids with sentinels.
        self.align_kernel(
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map,
            cumsum_buffer,
        ).launch(
            grid=[2, 1, 1],
            block=[self.config.large_align_threads, 1, 1],
            stream=stream,
        )
        # Kernel 2: scatter token indices into the padded expert-major layout.
        num_blocks = Int32(cute.ceil_div(cute.size(topk_ids.shape), self.config.large_scatter_threads))
        if num_blocks > 0:
            self.scatter_kernel(topk_ids, sorted_token_ids, expert_map, cumsum_buffer).launch(
                grid=[num_blocks, Int32(1), Int32(1)],
                block=[self.config.large_scatter_threads, 1, 1],
                stream=stream,
            )

    @cute.kernel
    def align_kernel(
        self,
        topk_ids: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_ids: cute.Tensor,
        num_tokens_post_pad: cute.Tensor,
        expert_map: cute.Tensor,
        cumsum_buffer: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        numel_i32 = Int32(cute.size(topk_ids.shape))

        # Block 1: fill sorted_token_ids with sentinel values.
        if bidx == 1:
            max_sorted = Int32(sorted_token_ids.shape[0])
            for idx in range(tidx, max_sorted, self.config.large_align_threads):
                sorted_token_ids[idx] = numel_i32
        else:
            # Block 0: compute expert counts and padded offsets.
            topk_flat = cute.make_tensor(
                topk_ids.iterator,
                cute.make_layout((numel_i32,), stride=(1,)),
            )

            @cute.struct
            class SharedStorage:
                counts: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]
                offsets: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts + 1], 16]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage, 16)
            counts = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
            offsets = storage.offsets.get_tensor(cute.make_layout((self.num_experts + 1,)))

            if tidx < self.num_experts:
                counts[tidx] = Int32(0)
            cute.arch.sync_threads()

            for i in range(tidx, numel_i32, self.config.large_align_threads):
                expert_id = Int32(topk_flat[i])
                if expert_id < self.num_experts:
                    if const_expr(self.has_expert_map):
                        mapped = expert_map[expert_id]
                        if mapped != -1:
                            atomic_add_i32(1, elem_pointer(counts, mapped))
                    else:
                        atomic_add_i32(1, elem_pointer(counts, expert_id))
            cute.arch.sync_threads()

            if tidx == 0:
                offsets[0] = Int32(0)
                running = Int32(0)
                for e in cutlass.range(self.num_experts, unroll_full=True):
                    c = counts[e]
                    padded = ((c + (self.block_size - 1)) // self.block_size) * self.block_size
                    running += padded
                    offsets[e + 1] = running
                num_tokens_post_pad[0] = running
            cute.arch.sync_threads()

            # Initialize cumsum_buffer with base offsets (used as atomic counters in the scatter kernel).
            if tidx < self.num_experts:
                cumsum_buffer[tidx] = offsets[tidx]

            total_padded_tokens = offsets[self.num_experts]
            total_blocks = total_padded_tokens // self.block_size

            if tidx < self.num_experts:
                start = offsets[tidx]
                end = offsets[tidx + 1]
                for j in range(start, end, self.block_size):
                    expert_ids[j // self.block_size] = tidx

            max_blocks = Int32(expert_ids.shape[0])
            for b in range(total_blocks + tidx, max_blocks, self.config.large_align_threads):
                expert_ids[b] = Int32(0)

    @cute.kernel
    def scatter_kernel(
        self,
        topk_ids: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_map: cute.Tensor,
        cumsum_buffer: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_stride = Int32(cute.arch.grid_dim()[0])

        numel_i32 = Int32(cute.size(topk_ids.shape))
        topk_flat = cute.make_tensor(
            topk_ids.iterator,
            cute.make_layout((numel_i32,), stride=(1,)),
        )

        tid = Int32(bidx) * self.config.large_scatter_threads + tidx
        stride = Int32(self.config.large_scatter_threads) * grid_stride
        for i in range(tid, numel_i32, stride):
            expert_id = Int32(topk_flat[i])
            if expert_id < self.num_experts:
                if const_expr(self.has_expert_map):
                    mapped = expert_map[expert_id]
                    if mapped != -1:
                        rank = atomic_add_i32(1, elem_pointer(cumsum_buffer, mapped))
                        sorted_token_ids[rank] = Int32(i)
                else:
                    rank = atomic_add_i32(1, elem_pointer(cumsum_buffer, expert_id))
                    sorted_token_ids[rank] = Int32(i)


class _MoeAlignBlockSizeCuTeLora:
    def __init__(
        self,
        *,
        num_experts: int,
        block_size: int,
        top_k: int,
        has_expert_map: bool,
        config: MoeAlignCuTeConfig,
    ) -> None:
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.has_expert_map = bool(has_expert_map)
        self.config = config

    @cute.jit
    def __call__(
        self,
        topk_ids: cute.Tensor,  # [T, K] integral
        token_lora_mapping: cute.Tensor,  # [T] int32 (lora id per token)
        sorted_token_ids: cute.Tensor,  # [max_loras * max_num_tokens_padded] int32
        expert_ids: cute.Tensor,  # [max_loras * max_num_m_blocks] int32
        num_tokens_post_pad: cute.Tensor,  # [max_loras] int32
        sorted_stride: cute.Tensor,  # [1] int32
        expert_stride: cute.Tensor,  # [1] int32
        expert_map: cute.Tensor,  # [num_experts] int32
        stream: cuda.CUstream,
    ) -> None:
        # Grid over max_loras; each block handles one lora_id (dense: lora_id == bidx)
        max_loras = Int32(num_tokens_post_pad.shape[0])
        self.kernel(
            topk_ids,
            token_lora_mapping,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            sorted_stride,
            expert_stride,
            expert_map,
        ).launch(
            grid=[max_loras, 1, 1],
            block=[self.config.small_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        topk_ids: cute.Tensor,
        token_lora_mapping: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_ids: cute.Tensor,
        num_tokens_post_pad: cute.Tensor,
        sorted_stride: cute.Tensor,
        expert_stride: cute.Tensor,
        expert_map: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        numel_i32 = Int32(cute.size(topk_ids.shape))
        top_k = Int32(self.top_k)
        # Dense mapping: lora_id == block index (no indirection needed)
        lora_id_val = Int32(bidx)
        max_loras = Int32(num_tokens_post_pad.shape[0])

        if lora_id_val < max_loras:  # Always true with grid=[max_loras], but kept for safety
            max_tokens_padded = Int32(sorted_stride[0])
            max_blocks = Int32(expert_stride[0])
            row_offset = lora_id_val * max_tokens_padded
            block_offset = lora_id_val * max_blocks

            sorted_row = cute.make_tensor(
                sorted_token_ids.iterator + row_offset,
                cute.make_layout((max_tokens_padded,), stride=(1,)),
            )
            expert_row = cute.make_tensor(
                expert_ids.iterator + block_offset,
                cute.make_layout((max_blocks,), stride=(1,)),
            )

            topk_flat = cute.make_tensor(
                topk_ids.iterator,
                cute.make_layout((numel_i32,), stride=(1,)),
            )

            @cute.struct
            class SharedStorage:
                counts: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]
                offsets: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts + 1], 16]
                counters: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage, 16)
            counts = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
            offsets = storage.offsets.get_tensor(cute.make_layout((self.num_experts + 1,)))
            counters = storage.counters.get_tensor(cute.make_layout((self.num_experts,)))

            if tidx < self.num_experts:
                counts[tidx] = Int32(0)
            cute.arch.sync_threads()

            for i in range(tidx, numel_i32, self.config.small_threads):
                token_idx = i // top_k
                if token_lora_mapping[token_idx] == lora_id_val:
                    expert_id = Int32(topk_flat[i])
                    if expert_id < self.num_experts:
                        if const_expr(self.has_expert_map):
                            mapped = expert_map[expert_id]
                            if mapped != -1:
                                atomic_add_i32(1, elem_pointer(counts, mapped))
                        else:
                            atomic_add_i32(1, elem_pointer(counts, expert_id))
            cute.arch.sync_threads()

            if tidx == 0:
                offsets[0] = Int32(0)
                running = Int32(0)
                for e in cutlass.range(self.num_experts, unroll_full=True):
                    c = counts[e]
                    padded = ((c + (self.block_size - 1)) // self.block_size) * self.block_size
                    running += padded
                    offsets[e + 1] = running
                num_tokens_post_pad[lora_id_val] = running
            cute.arch.sync_threads()

            total_padded_tokens = offsets[self.num_experts]
            total_blocks = total_padded_tokens // self.block_size

            for idx in range(tidx, total_padded_tokens, self.config.small_threads):
                sorted_row[idx] = numel_i32

            if tidx < self.num_experts:
                start = offsets[tidx]
                end = offsets[tidx + 1]
                for j in range(start, end, self.block_size):
                    expert_row[j // self.block_size] = tidx

            for b in range(total_blocks + tidx, max_blocks, self.config.small_threads):
                expert_row[b] = Int32(0)

            if tidx < self.num_experts:
                counters[tidx] = Int32(0)
            cute.arch.sync_threads()

            for i in range(tidx, numel_i32, self.config.small_threads):
                token_idx = i // top_k
                if token_lora_mapping[token_idx] == lora_id_val:
                    expert_id = Int32(topk_flat[i])
                    if expert_id < self.num_experts:
                        if const_expr(self.has_expert_map):
                            mapped = expert_map[expert_id]
                            if mapped != -1:
                                pos = atomic_add_i32(1, elem_pointer(counters, mapped))
                                sorted_row[offsets[mapped] + pos] = Int32(i)
                        else:
                            pos = atomic_add_i32(1, elem_pointer(counters, expert_id))
                            sorted_row[offsets[expert_id] + pos] = Int32(i)


class _MoeAlignBlockSizeCuTeLargeLora:
    def __init__(
        self,
        *,
        num_experts: int,
        block_size: int,
        top_k: int,
        has_expert_map: bool,
        config: MoeAlignCuTeConfig,
    ) -> None:
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.has_expert_map = bool(has_expert_map)
        self.config = config

    @cute.jit
    def __call__(
        self,
        topk_ids: cute.Tensor,  # [T, K] integral
        token_lora_mapping: cute.Tensor,  # [T] int32
        sorted_token_ids: cute.Tensor,  # [max_loras * max_num_tokens_padded] int32
        expert_ids: cute.Tensor,  # [max_loras * max_num_m_blocks] int32
        num_tokens_post_pad: cute.Tensor,  # [max_loras] int32
        sorted_stride: cute.Tensor,  # [1] int32
        expert_stride: cute.Tensor,  # [1] int32
        expert_map: cute.Tensor,  # [num_experts] int32
        cumsum_buffer: cute.Tensor,  # [max_loras * num_experts] int32
        stream: cuda.CUstream,
    ) -> None:
        # Grid over max_loras; each block handles one lora_id (dense: lora_id == bidx)
        max_loras = Int32(num_tokens_post_pad.shape[0])
        self.align_kernel(
            topk_ids,
            token_lora_mapping,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            sorted_stride,
            expert_stride,
            expert_map,
            cumsum_buffer,
        ).launch(
            grid=[max_loras * 2, 1, 1],
            block=[self.config.large_align_threads, 1, 1],
            stream=stream,
        )
        num_blocks = Int32(cute.ceil_div(cute.size(topk_ids.shape), self.config.large_scatter_threads))
        if num_blocks > 0:
            self.scatter_kernel(
                topk_ids,
                token_lora_mapping,
                sorted_token_ids,
                sorted_stride,
                num_tokens_post_pad,
                expert_map,
                cumsum_buffer,
            ).launch(
                grid=[max_loras, num_blocks, Int32(1)],
                block=[self.config.large_scatter_threads, 1, 1],
                stream=stream,
            )

    @cute.kernel
    def align_kernel(
        self,
        topk_ids: cute.Tensor,
        token_lora_mapping: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_ids: cute.Tensor,
        num_tokens_post_pad: cute.Tensor,
        sorted_stride: cute.Tensor,
        expert_stride: cute.Tensor,
        expert_map: cute.Tensor,
        cumsum_buffer: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        numel_i32 = Int32(cute.size(topk_ids.shape))
        top_k = Int32(self.top_k)
        lora_idx = Int32(bidx) // Int32(2)
        phase = Int32(bidx) % Int32(2)
        # Dense mapping: lora_id == lora_idx (no indirection needed)
        lora_id_val = lora_idx
        max_loras = Int32(num_tokens_post_pad.shape[0])

        if lora_id_val < max_loras:  # Always true with grid=[max_loras*2], but kept for safety
            max_tokens_padded = Int32(sorted_stride[0])
            max_blocks = Int32(expert_stride[0])
            row_offset = lora_id_val * max_tokens_padded
            block_offset = lora_id_val * max_blocks

            sorted_row = cute.make_tensor(
                sorted_token_ids.iterator + row_offset,
                cute.make_layout((max_tokens_padded,), stride=(1,)),
            )
            expert_row = cute.make_tensor(
                expert_ids.iterator + block_offset,
                cute.make_layout((max_blocks,), stride=(1,)),
            )

            if phase == Int32(1):
                for idx in range(tidx, max_tokens_padded, self.config.large_align_threads):
                    sorted_row[idx] = numel_i32
            else:
                topk_flat = cute.make_tensor(
                    topk_ids.iterator,
                    cute.make_layout((numel_i32,), stride=(1,)),
                )

                @cute.struct
                class SharedStorage:
                    counts: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts], 16]
                    offsets: cute.struct.Align[cute.struct.MemRange[Int32, self.num_experts + 1], 16]

                smem = cutlass.utils.SmemAllocator()
                storage = smem.allocate(SharedStorage, 16)
                counts = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
                offsets = storage.offsets.get_tensor(cute.make_layout((self.num_experts + 1,)))

                if tidx < self.num_experts:
                    counts[tidx] = Int32(0)
                cute.arch.sync_threads()

                for i in range(tidx, numel_i32, self.config.large_align_threads):
                    token_idx = i // top_k
                    if token_lora_mapping[token_idx] == lora_id_val:
                        expert_id = Int32(topk_flat[i])
                        if expert_id < self.num_experts:
                            if const_expr(self.has_expert_map):
                                mapped = expert_map[expert_id]
                                if mapped != -1:
                                    atomic_add_i32(1, elem_pointer(counts, mapped))
                            else:
                                atomic_add_i32(1, elem_pointer(counts, expert_id))
                cute.arch.sync_threads()

                if tidx == 0:
                    offsets[0] = Int32(0)
                    running = Int32(0)
                    for e in cutlass.range(self.num_experts, unroll_full=True):
                        c = counts[e]
                        padded = ((c + (self.block_size - 1)) // self.block_size) * self.block_size
                        running += padded
                        offsets[e + 1] = running
                    num_tokens_post_pad[lora_id_val] = running
                cute.arch.sync_threads()

                # Write to per-LoRA row of cumsum_buffer to avoid race conditions
                if tidx < self.num_experts:
                    cumsum_buffer[lora_id_val * self.num_experts + tidx] = offsets[tidx]

                total_padded_tokens = offsets[self.num_experts]
                total_blocks = total_padded_tokens // self.block_size

                if tidx < self.num_experts:
                    start = offsets[tidx]
                    end = offsets[tidx + 1]
                    for j in range(start, end, self.block_size):
                        expert_row[j // self.block_size] = tidx

                for b in range(total_blocks + tidx, max_blocks, self.config.large_align_threads):
                    expert_row[b] = Int32(0)

    @cute.kernel
    def scatter_kernel(
        self,
        topk_ids: cute.Tensor,
        token_lora_mapping: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        sorted_stride: cute.Tensor,
        num_tokens_post_pad: cute.Tensor,
        expert_map: cute.Tensor,
        cumsum_buffer: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        grid_stride = Int32(cute.arch.grid_dim()[1])

        numel_i32 = Int32(cute.size(topk_ids.shape))
        top_k = Int32(self.top_k)
        # Dense mapping: lora_id == bidx_x (no indirection needed)
        lora_id_val = Int32(bidx_x)
        max_tokens_padded = Int32(sorted_stride[0])
        max_loras = Int32(num_tokens_post_pad.shape[0])
        topk_flat = cute.make_tensor(
            topk_ids.iterator,
            cute.make_layout((numel_i32,), stride=(1,)),
        )

        tid = Int32(bidx_y) * self.config.large_scatter_threads + tidx
        stride = Int32(self.config.large_scatter_threads) * grid_stride
        if lora_id_val < max_loras:  # Always true with grid=[max_loras, ...], but kept for safety
            row_offset = lora_id_val * max_tokens_padded
            sorted_row = cute.make_tensor(
                sorted_token_ids.iterator + row_offset,
                cute.make_layout((max_tokens_padded,), stride=(1,)),
            )
            # Use per-LoRA row of cumsum_buffer to avoid race conditions
            cumsum_row_offset = lora_id_val * self.num_experts
            for i in range(tid, numel_i32, stride):
                token_idx = i // top_k
                if token_lora_mapping[token_idx] == lora_id_val:
                    expert_id = Int32(topk_flat[i])
                    if expert_id < self.num_experts:
                        if const_expr(self.has_expert_map):
                            mapped = expert_map[expert_id]
                            if mapped != -1:
                                rank = atomic_add_i32(1, elem_pointer(cumsum_buffer, cumsum_row_offset + mapped))
                                sorted_row[rank] = Int32(i)
                        else:
                            rank = atomic_add_i32(1, elem_pointer(cumsum_buffer, cumsum_row_offset + expert_id))
                            sorted_row[rank] = Int32(i)


__all__ = [
    "MoeAlignCuTeConfig",
    "_MoeAlignBlockSizeCuTe",
    "_MoeAlignBlockSizeCuTeLarge",
    "_MoeAlignBlockSizeCuTeLora",
    "_MoeAlignBlockSizeCuTeLargeLora",
]
