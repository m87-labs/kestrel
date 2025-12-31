"""Routing helpers for fused MoE kernels."""

from __future__ import annotations

import torch

from kestrel_kernels.moe_align_cute import (
    moe_align_block_size as moe_align_block_size_cute,
    moe_lora_align_block_size as moe_lora_align_block_size_cute,
)


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align token distribution across experts to block boundaries.

    This matches vLLM's moe_align_block_size semantics.
    """
    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a torch.Tensor")
    if topk_ids.device.type != "cuda":
        raise ValueError("topk_ids must be a CUDA tensor")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive int")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive int")
    if expert_map is not None:
        if not isinstance(expert_map, torch.Tensor):
            raise TypeError("expert_map must be a torch.Tensor")
        if expert_map.device != topk_ids.device:
            raise ValueError("expert_map must be on the same device as topk_ids")
        if expert_map.dtype != torch.int32:
            raise ValueError("expert_map must be int32")
        if expert_map.ndim != 1 or expert_map.numel() != num_experts:
            raise ValueError("expert_map must be shape [num_experts]")
        if not expert_map.is_contiguous():
            raise ValueError("expert_map must be contiguous")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = _round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    if expert_map is None or not ignore_invalid_experts:
        expert_map_arg = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
    else:
        expert_map_arg = expert_map

    moe_align_block_size_cute(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map_arg,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    block_size: int,
    num_experts: int,
    max_loras: int,
    *,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align token distribution across experts for each LoRA id.

    token_lora_mapping uses -1 for no-LoRA and [0, max_loras) for active LoRAs.
    Uses dense identity mapping (lora_id == grid block index).
    """
    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a torch.Tensor")
    if topk_ids.device.type != "cuda":
        raise ValueError("topk_ids must be a CUDA tensor")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive int")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive int")
    if not isinstance(max_loras, int) or max_loras <= 0:
        raise ValueError("max_loras must be a positive int")
    if token_lora_mapping.device != topk_ids.device:
        raise ValueError("token_lora_mapping must be on the same device as topk_ids")
    if token_lora_mapping.dtype != torch.int32:
        raise ValueError("token_lora_mapping must be int32")
    if token_lora_mapping.ndim != 1 or token_lora_mapping.shape[0] != topk_ids.shape[0]:
        raise ValueError("token_lora_mapping must have shape [num_tokens]")
    if not token_lora_mapping.is_contiguous():
        raise ValueError("token_lora_mapping must be contiguous")
    if expert_map is not None:
        if not isinstance(expert_map, torch.Tensor):
            raise TypeError("expert_map must be a torch.Tensor")
        if expert_map.device != topk_ids.device:
            raise ValueError("expert_map must be on the same device as topk_ids")
        if expert_map.dtype != torch.int32:
            raise ValueError("expert_map must be int32")
        if expert_map.ndim != 1 or expert_map.numel() != num_experts:
            raise ValueError("expert_map must be shape [num_experts]")
        if not expert_map.is_contiguous():
            raise ValueError("expert_map must be contiguous")

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = _round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )

    sorted_ids = torch.empty(
        (max_loras, max_num_tokens_padded),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        (max_loras, max_num_m_blocks),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty(
        (max_loras,), dtype=torch.int32, device=topk_ids.device
    )

    moe_lora_align_block_size_cute(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map=expert_map,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


__all__ = ["moe_align_block_size", "moe_lora_align_block_size"]
