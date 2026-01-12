"""MoE align block size kernel dispatch and loading.

This module provides the public API for MoE alignment operations and handles
loading precompiled kernels at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

import cutlass.cute as cute
from cutlass import Int32, Int64

from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module


# Enable JIT compilation for autotuning (set KESTREL_CUTE_MOE_JIT=1)
_ENABLE_JIT = os.environ.get("KESTREL_CUTE_MOE_JIT", "0") == "1"


@dataclass(frozen=True)
class MoeAlignCuTeConfig:
    """Configuration for MoE align kernels."""
    # Small path (single-CTA) tuned for decode-like TK.
    small_threads: int = 256
    # Large path (2 kernels): use a large block for the align kernel because the
    # grid is only 2 CTAs (one for counts/offsets, one for sentinel fill).
    large_align_threads: int = 1024
    large_scatter_threads: int = 256


# Precompiled kernel registry
_precompiled_cache: Dict[Tuple, Any] = {}


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

    # Build filename and function name
    arch = get_cuda_arch()
    dtype_name = "i32" if topk_dtype == Int32 else "i64"
    expert_map_str = "emap" if has_expert_map else "noemap"
    base_name = f"moe_align_{kernel_type}_{dtype_name}_k{topk}_e{num_experts}_b{block_size}_{expert_map_str}_{arch}"
    filename = f"{base_name}.so"
    function_name = base_name

    # Load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[compile_key] = kernel_fn
    return kernel_fn


def _jit_compile_small(
    topk_dtype: type,
    topk: int,
    num_experts: int,
    block_size: int,
    has_expert_map: bool,
    config: MoeAlignCuTeConfig,
) -> Any:
    """JIT compile the small path kernel on demand."""
    # Lazy import kernel class - only available in source installs
    try:
        from kestrel_kernels.moe_align.kernel import _MoeAlignBlockSizeCuTe
    except ImportError as e:
        raise RuntimeError(
            "JIT compilation requires source install. "
            "Install from source for JIT support."
        ) from e

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
    config: MoeAlignCuTeConfig,
) -> Any:
    """JIT compile the large path kernel on demand."""
    # Lazy import kernel class - only available in source installs
    try:
        from kestrel_kernels.moe_align.kernel import _MoeAlignBlockSizeCuTeLarge
    except ImportError as e:
        raise RuntimeError(
            "JIT compilation requires source install. "
            "Install from source for JIT support."
        ) from e

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
                    arch = get_cuda_arch()
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
                arch = get_cuda_arch()
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
                arch = get_cuda_arch()
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
            arch = get_cuda_arch()
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
