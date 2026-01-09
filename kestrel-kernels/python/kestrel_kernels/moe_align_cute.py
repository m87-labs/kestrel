"""CuTe DSL implementation of `moe_align_block_size`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
import os


# Enable JIT compilation for autotuning (set KESTREL_CUTE_MOE_JIT=1)
_ENABLE_JIT = os.environ.get("KESTREL_CUTE_MOE_JIT", "0") == "1"

# Precompiled kernel registry
_precompiled_cache: Dict[Tuple, Any] = {}
_precompiled_dir = Path(__file__).parent / "precompiled"

# Cache the architecture string
_cuda_arch: Optional[str] = None


def _get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper, 'sm100' for Blackwell)."""
    global _cuda_arch
    if _cuda_arch is None:
        major, minor = torch.cuda.get_device_capability()
        _cuda_arch = f"sm{major}{minor}"
    return _cuda_arch


def _get_precompiled_kernel_path(
    kernel_type: str,  # "small", "large", "lora_small", "lora_large"
    topk_dtype: type,  # Int32 or Int64
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
) -> Optional[Path]:
    """Get path to precompiled kernel if it exists."""
    arch = _get_cuda_arch()
    dtype_name = "i32" if topk_dtype == Int32 else "i64"
    expert_map_str = "emap" if has_expert_map else "noemap"
    filename = f"moe_align_{kernel_type}_{dtype_name}_k{topk}_e{num_experts}_b{block_size}_{expert_map_str}_{arch}.so"
    path = _precompiled_dir / filename
    return path if path.exists() else None


def _get_precompiled_function_name(
    kernel_type: str,
    topk_dtype: type,
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
) -> str:
    """Get the exported function name for a precompiled kernel."""
    arch = _get_cuda_arch()
    dtype_name = "i32" if topk_dtype == Int32 else "i64"
    expert_map_str = "emap" if has_expert_map else "noemap"
    return f"moe_align_{kernel_type}_{dtype_name}_k{topk}_e{num_experts}_b{block_size}_{expert_map_str}_{arch}"


def _load_precompiled_kernel(
    kernel_type: str,
    topk_dtype: type,
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
) -> Optional[Any]:
    """Load a precompiled kernel if available, return None otherwise."""
    compile_key = (kernel_type, topk_dtype, topk, num_experts, block_size, has_expert_map)

    # Check if already loaded
    if compile_key in _precompiled_cache:
        return _precompiled_cache[compile_key]

    # Check if precompiled file exists
    so_path = _get_precompiled_kernel_path(
        kernel_type, topk_dtype, topk, num_experts, block_size, has_expert_map
    )
    if so_path is None:
        return None

    # Load the module
    mod = cute.runtime.load_module(str(so_path))

    # Get the function by its exported name
    function_name = _get_precompiled_function_name(
        kernel_type, topk_dtype, topk, num_experts, block_size, has_expert_map
    )
    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[compile_key] = kernel_fn
    return kernel_fn


def _jit_compile_small(
    topk_dtype: type,
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
    config: "MoeAlignCuTeConfig",
) -> Any:
    """JIT compile the small path kernel on demand."""
    t_sym = cute.sym_int()
    topk_ids_fake = cute.runtime.make_fake_tensor(
        topk_dtype,
        (t_sym, topk),
        stride=(topk, 1),
        assumed_align=topk_dtype.width // 8,
    )
    sorted_fake = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(),), stride=(1,), assumed_align=4,
    )
    expert_ids_fake = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(),), stride=(1,), assumed_align=4,
    )
    post_fake = cute.runtime.make_fake_tensor(
        Int32, (1,), stride=(1,), assumed_align=4,
    )
    expert_map_fake = cute.runtime.make_fake_tensor(
        Int32, (num_experts,), stride=(1,), assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Import here to avoid circular dependency at module load time
    op = _MoeAlignBlockSizeCuTe(
        num_experts=num_experts,
        block_size=block_size,
        has_expert_map=has_expert_map,
        config=config,
    )
    compiled = cute.compile(
        op,
        topk_ids_fake,
        sorted_fake,
        expert_ids_fake,
        post_fake,
        expert_map_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled


def _jit_compile_large(
    topk_dtype: type,
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
    config: "MoeAlignCuTeConfig",
) -> Any:
    """JIT compile the large path kernel on demand."""
    t_sym = cute.sym_int()
    topk_ids_fake = cute.runtime.make_fake_tensor(
        topk_dtype,
        (t_sym, topk),
        stride=(topk, 1),
        assumed_align=topk_dtype.width // 8,
    )
    sorted_fake = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(),), stride=(1,), assumed_align=4,
    )
    expert_ids_fake = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(),), stride=(1,), assumed_align=4,
    )
    post_fake = cute.runtime.make_fake_tensor(
        Int32, (1,), stride=(1,), assumed_align=4,
    )
    expert_map_fake = cute.runtime.make_fake_tensor(
        Int32, (num_experts,), stride=(1,), assumed_align=4,
    )
    cumsum_fake = cute.runtime.make_fake_tensor(
        Int32, (num_experts,), stride=(1,), assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    op = _MoeAlignBlockSizeCuTeLarge(
        num_experts=num_experts,
        block_size=block_size,
        has_expert_map=has_expert_map,
        config=config,
    )
    compiled = cute.compile(
        op,
        topk_ids_fake,
        sorted_fake,
        expert_ids_fake,
        post_fake,
        expert_map_fake,
        cumsum_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled


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


_COMPILE_CACHE_SMALL: Dict[Tuple[Any, int, int, int, bool, MoeAlignCuTeConfig], Any] = {}
_COMPILE_CACHE_LARGE: Dict[Tuple[Any, int, int, int, bool, MoeAlignCuTeConfig], Any] = {}
_COMPILE_CACHE_LORA_SMALL: Dict[Tuple[Any, int, int, int, bool, MoeAlignCuTeConfig], Any] = {}
_COMPILE_CACHE_LORA_LARGE: Dict[Tuple[Any, int, int, int, bool, MoeAlignCuTeConfig], Any] = {}
_CUMSUM_BUFFER_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}
_DUMMY_EXPERT_MAP_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_LORA_STRIDE_CACHE: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor,
) -> None:
    """CuTe DSL moe_align_block_size (CUDA-only)."""
    if topk_ids.device.type != "cuda":
        raise ValueError("topk_ids must be a CUDA tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")
    if sorted_token_ids.dtype != torch.int32 or sorted_token_ids.ndim != 1 or not sorted_token_ids.is_contiguous():
        raise ValueError("sorted_token_ids must be a contiguous int32 1D tensor")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1 or not expert_ids.is_contiguous():
        raise ValueError("expert_ids must be a contiguous int32 1D tensor")
    if (
        num_tokens_post_pad.dtype != torch.int32
        or num_tokens_post_pad.ndim != 1
        or num_tokens_post_pad.numel() != 1
        or not num_tokens_post_pad.is_contiguous()
    ):
        raise ValueError("num_tokens_post_pad must be a contiguous int32 tensor with shape (1,)")
    if expert_map.dtype != torch.int32 or expert_map.ndim != 1 or not expert_map.is_contiguous():
        raise ValueError("expert_map must be a contiguous int32 1D tensor")
    if expert_map.numel() not in (0, int(num_experts)):
        raise ValueError("expert_map must be empty or shape [num_experts]")

    topk = int(topk_ids.shape[1])
    has_expert_map = bool(expert_map.numel() > 0)
    topk_dtype = Int32 if topk_ids.dtype == torch.int32 else Int64
    numel = int(topk_ids.numel())
    cfg = MoeAlignCuTeConfig()
    # Match the CUDA extension fast path: a single-CTA shared-memory histogram +
    # scatter is best for decode-like TK and small E.
    small_batch_expert_mode = (int(num_experts) <= 64) and (numel < 1024)

    key = (topk_dtype, topk, int(num_experts), int(block_size), has_expert_map, cfg)
    dev_idx = int(topk_ids.device.index or 0)
    if has_expert_map:
        expert_map_arg = expert_map
    else:
        dummy_key = (dev_idx, int(num_experts))
        expert_map_arg = _DUMMY_EXPERT_MAP_CACHE.get(dummy_key)
        if expert_map_arg is None:
            expert_map_arg = torch.zeros(
                (int(num_experts),), device=topk_ids.device, dtype=torch.int32
            )
            _DUMMY_EXPERT_MAP_CACHE[dummy_key] = expert_map_arg

    if small_batch_expert_mode:
        if key not in _COMPILE_CACHE_SMALL:
            precompiled = _load_precompiled_kernel(
                "small", topk_dtype, topk, int(num_experts), int(block_size), has_expert_map
            )
            if precompiled is None:
                if _ENABLE_JIT:
                    # JIT compile on demand
                    precompiled = _jit_compile_small(
                        topk_dtype, topk, int(num_experts), int(block_size), has_expert_map, cfg
                    )
                else:
                    dtype_name = "int32" if topk_dtype == Int32 else "int64"
                    arch = _get_cuda_arch()
                    raise RuntimeError(
                        f"No precompiled kernel for moe_align_block_size(type=small, "
                        f"dtype={dtype_name}, topk={topk}, num_experts={num_experts}, "
                        f"block_size={block_size}, has_expert_map={has_expert_map}, arch={arch}). "
                        f"Run precompile_moe_align.py on this architecture to generate it."
                    )
            _COMPILE_CACHE_SMALL[key] = precompiled

        _COMPILE_CACHE_SMALL[key](
            topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, expert_map_arg
        )
        return

    # Large path: global cumsum buffer + multi-CTA scatter (matches CUDA kernel structure).
    if key not in _COMPILE_CACHE_LARGE:
        precompiled = _load_precompiled_kernel(
            "large", topk_dtype, topk, int(num_experts), int(block_size), has_expert_map
        )
        if precompiled is None:
            if _ENABLE_JIT:
                # JIT compile on demand
                precompiled = _jit_compile_large(
                    topk_dtype, topk, int(num_experts), int(block_size), has_expert_map, cfg
                )
            else:
                dtype_name = "int32" if topk_dtype == Int32 else "int64"
                arch = _get_cuda_arch()
                raise RuntimeError(
                    f"No precompiled kernel for moe_align_block_size(type=large, "
                    f"dtype={dtype_name}, topk={topk}, num_experts={num_experts}, "
                    f"block_size={block_size}, has_expert_map={has_expert_map}, arch={arch}). "
                    f"Run precompile_moe_align.py on this architecture to generate it."
                )
        _COMPILE_CACHE_LARGE[key] = precompiled

    # Reuse a per-stream scratch buffer to avoid per-call allocations on the hot path.
    # NOTE: The align kernel overwrites the buffer with base offsets each call, so it
    # is safe to reuse as long as calls on the same stream are sequential.
    stream_id = int(torch.cuda.current_stream(topk_ids.device).cuda_stream)
    buf_key = (dev_idx, stream_id, int(num_experts))
    cumsum_buffer = _CUMSUM_BUFFER_CACHE.get(buf_key)
    if cumsum_buffer is None:
        cumsum_buffer = torch.empty((int(num_experts),), device=topk_ids.device, dtype=torch.int32)
        _CUMSUM_BUFFER_CACHE[buf_key] = cumsum_buffer
    _COMPILE_CACHE_LARGE[key](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map_arg,
        cumsum_buffer,
    )


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """CuTe DSL moe_lora_align_block_size (CUDA-only).

    token_lora_mapping uses -1 for no-LoRA and [0, max_loras) for active LoRAs.
    Uses dense identity mapping for lora indices (lora_id == block_idx).
    """
    if topk_ids.device.type != "cuda":
        raise ValueError("topk_ids must be a CUDA tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")
    if token_lora_mapping.device != topk_ids.device:
        raise ValueError("token_lora_mapping must be on the same device as topk_ids")
    if token_lora_mapping.dtype != torch.int32:
        raise ValueError("token_lora_mapping must be int32")
    if token_lora_mapping.ndim != 1 or token_lora_mapping.shape[0] != topk_ids.shape[0]:
        raise ValueError("token_lora_mapping must have shape [num_tokens]")
    if not token_lora_mapping.is_contiguous():
        raise ValueError("token_lora_mapping must be contiguous")
    if sorted_token_ids.dtype != torch.int32 or sorted_token_ids.ndim != 2:
        raise ValueError("sorted_token_ids must be a contiguous int32 2D tensor")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 2:
        raise ValueError("expert_ids must be a contiguous int32 2D tensor")
    if num_tokens_post_pad.dtype != torch.int32 or num_tokens_post_pad.ndim != 1:
        raise ValueError("num_tokens_post_pad must be a contiguous int32 1D tensor")
    if not sorted_token_ids.is_contiguous():
        raise ValueError("sorted_token_ids must be contiguous")
    if not expert_ids.is_contiguous():
        raise ValueError("expert_ids must be contiguous")
    if not num_tokens_post_pad.is_contiguous():
        raise ValueError("num_tokens_post_pad must be contiguous")

    max_loras = int(sorted_token_ids.shape[0])
    if expert_ids.shape[0] != max_loras or num_tokens_post_pad.shape[0] != max_loras:
        raise ValueError("expert_ids and num_tokens_post_pad must have leading dim == max_loras")

    if expert_map is None or expert_map.numel() == 0:
        expert_map_arg = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
    else:
        if expert_map.device != topk_ids.device:
            raise ValueError("expert_map must be on the same device as topk_ids")
        if expert_map.dtype != torch.int32:
            raise ValueError("expert_map must be int32")
        if expert_map.ndim != 1 or expert_map.numel() != int(num_experts):
            raise ValueError("expert_map must be shape [num_experts]")
        if not expert_map.is_contiguous():
            raise ValueError("expert_map must be contiguous")
        expert_map_arg = expert_map

    topk = int(topk_ids.shape[1])
    topk_dtype = Int32 if topk_ids.dtype == torch.int32 else Int64
    numel = int(topk_ids.numel())
    cfg = MoeAlignCuTeConfig()
    has_expert_map = bool(expert_map_arg.numel() > 0)
    small_batch_expert_mode = (int(num_experts) <= 64) and (numel < 1024)

    num_tokens_post_pad.zero_()

    key = (topk_dtype, topk, int(num_experts), int(block_size), has_expert_map, cfg)
    dev_idx = int(topk_ids.device.index or 0)
    if not has_expert_map:
        dummy_key = (dev_idx, int(num_experts))
        dummy_map = _DUMMY_EXPERT_MAP_CACHE.get(dummy_key)
        if dummy_map is None:
            dummy_map = torch.zeros((int(num_experts),), device=topk_ids.device, dtype=torch.int32)
            _DUMMY_EXPERT_MAP_CACHE[dummy_key] = dummy_map
        expert_map_arg = dummy_map
    max_tokens_padded = int(sorted_token_ids.shape[1])
    max_blocks = int(expert_ids.shape[1])
    stride_key = (dev_idx, max_tokens_padded, max_blocks)
    stride_tensors = _LORA_STRIDE_CACHE.get(stride_key)
    if stride_tensors is None:
        sorted_stride = torch.tensor([max_tokens_padded], device=topk_ids.device, dtype=torch.int32)
        expert_stride = torch.tensor([max_blocks], device=topk_ids.device, dtype=torch.int32)
        stride_tensors = (sorted_stride, expert_stride)
        _LORA_STRIDE_CACHE[stride_key] = stride_tensors
    else:
        sorted_stride, expert_stride = stride_tensors

    sorted_flat = sorted_token_ids.view(-1)
    expert_flat = expert_ids.view(-1)

    if small_batch_expert_mode:
        if key not in _COMPILE_CACHE_LORA_SMALL:
            precompiled = _load_precompiled_kernel(
                "lora_small", topk_dtype, topk, int(num_experts), int(block_size), has_expert_map
            )
            if precompiled is None:
                dtype_name = "int32" if topk_dtype == Int32 else "int64"
                arch = _get_cuda_arch()
                raise RuntimeError(
                    f"No precompiled kernel for moe_lora_align_block_size(type=lora_small, "
                    f"dtype={dtype_name}, topk={topk}, num_experts={num_experts}, "
                    f"block_size={block_size}, has_expert_map={has_expert_map}, arch={arch}). "
                    f"Run precompile_moe_align.py on this architecture to generate it."
                )
            _COMPILE_CACHE_LORA_SMALL[key] = precompiled

        _COMPILE_CACHE_LORA_SMALL[key](
            topk_ids,
            token_lora_mapping,
            sorted_flat,
            expert_flat,
            num_tokens_post_pad,
            sorted_stride,
            expert_stride,
            expert_map_arg,
        )
        return

    if key not in _COMPILE_CACHE_LORA_LARGE:
        precompiled = _load_precompiled_kernel(
            "lora_large", topk_dtype, topk, int(num_experts), int(block_size), has_expert_map
        )
        if precompiled is None:
            dtype_name = "int32" if topk_dtype == Int32 else "int64"
            arch = _get_cuda_arch()
            raise RuntimeError(
                f"No precompiled kernel for moe_lora_align_block_size(type=lora_large, "
                f"dtype={dtype_name}, topk={topk}, num_experts={num_experts}, "
                f"block_size={block_size}, has_expert_map={has_expert_map}, arch={arch}). "
                f"Run precompile_moe_align.py on this architecture to generate it."
            )
        _COMPILE_CACHE_LORA_LARGE[key] = precompiled

    stream_id = int(torch.cuda.current_stream(topk_ids.device).cuda_stream)
    # For LoRA large path, use per-LoRA cumsum buffers to avoid race conditions
    buf_key = (dev_idx, stream_id, int(num_experts), max_loras)
    cumsum_buffer = _CUMSUM_BUFFER_CACHE.get(buf_key)
    required_size = max_loras * int(num_experts)
    if cumsum_buffer is None:
        cumsum_buffer = torch.empty((required_size,), device=topk_ids.device, dtype=torch.int32)
        _CUMSUM_BUFFER_CACHE[buf_key] = cumsum_buffer
    elif cumsum_buffer.numel() < required_size:
        raise RuntimeError(
            f"LoRA cumsum buffer overflow: requested {required_size} elements but "
            f"only {cumsum_buffer.numel()} allocated. This indicates the buffer "
            f"was not pre-allocated for sufficient max_loras before CUDA graph capture."
        )

    _COMPILE_CACHE_LORA_LARGE[key](
        topk_ids,
        token_lora_mapping,
        sorted_flat,
        expert_flat,
        num_tokens_post_pad,
        sorted_stride,
        expert_stride,
        expert_map_arg,
        cumsum_buffer,
    )


__all__ = ["MoeAlignCuTeConfig", "moe_align_block_size", "moe_lora_align_block_size"]
