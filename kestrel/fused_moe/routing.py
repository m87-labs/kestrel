from __future__ import annotations

import os
import torch


def _compileable_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    counts = torch.zeros(minlength, dtype=torch.int64, device=x.device)
    if x.numel() == 0:
        return counts
    ones = torch.ones_like(x, dtype=torch.int64)
    counts.scatter_add_(0, x, ones)
    return counts


def _round_up(x: torch.Tensor, align: int) -> torch.Tensor:
    return ((x + align - 1) // align) * align


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reimplementation of vLLM's moe_align_block_size.

    The function sorts token assignments by expert, pads each expert's token
    count to ``block_size``, and returns metadata tensors consumed by the fused
    Triton GEMMs. Padding entries are filled with a sentinel index equal to the
    flattened token count so downstream kernels can mask them out without having
    to change tensor shapes between iterations.
    """

    if topk_ids.dim() != 2:
        raise ValueError("topk_ids must be 2D [num_tokens, top_k]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    device = topk_ids.device
    num_tokens, top_k = topk_ids.shape
    assignments = topk_ids.reshape(-1).to(dtype=torch.int64)
    total_assignments = assignments.numel()
    sentinel = total_assignments

    if total_assignments == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        return empty, empty, zero

    flat_indices = torch.arange(total_assignments, device=device, dtype=torch.int32)
    sorted_experts, order = torch.sort(assignments)
    sorted_indices = flat_indices.index_select(0, order)

    expert_counts = _compileable_bincount(sorted_experts, minlength=num_experts)
    padded_counts = _round_up(expert_counts, block_size)
    total_padded = padded_counts.sum()
    max_tokens_padded = total_assignments + num_experts * (block_size - 1)

    sorted_token_ids = torch.full(
        (max_tokens_padded,), sentinel, dtype=torch.int32, device=device
    )

    if sorted_indices.numel() > 0:
        expert_offsets = torch.cumsum(padded_counts, 0) - padded_counts
        expert_actual_offsets = torch.cumsum(expert_counts, 0) - expert_counts

        positional = torch.arange(
            sorted_indices.numel(), device=device, dtype=torch.int64
        )
        padded_base = expert_offsets.index_select(0, sorted_experts)
        actual_base = expert_actual_offsets.index_select(0, sorted_experts)
        destination = padded_base + (positional - actual_base)
        sorted_token_ids.index_copy_(0, destination.to(torch.int64), sorted_indices)

    block_counts = (padded_counts // block_size).to(dtype=torch.int32)
    max_num_m_blocks = (max_tokens_padded + (block_size - 1)) // block_size
    expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    block_boundaries = torch.cumsum(block_counts, 0)
    block_indices = torch.arange(max_num_m_blocks, device=device, dtype=torch.int64)
    assigned = torch.bucketize(block_indices, block_boundaries, right=True)
    total_blocks = (
        block_boundaries[-1]
        if block_boundaries.numel() > 0
        else torch.zeros((), device=device, dtype=torch.int64)
    )
    mask = block_indices < total_blocks
    expert_ids = torch.where(mask, assigned.to(torch.int32), expert_ids)
    if os.getenv("KESTREL_COMPARE_MOE_BACKENDS") == "1" and block_counts.sum() > 0:
        expected = torch.repeat_interleave(
            torch.arange(num_experts, device=device, dtype=torch.int32), block_counts
        )
        compare_len = min(expected.size(0), int(total_blocks.item()))
        if compare_len > 0:
            meta_diff = (expert_ids[:compare_len] - expected[:compare_len]).abs().max()
            print(
                "[FusedMoE][compare] block_ids max_diff="
                f"{meta_diff.item():.6f}",
                flush=True,
            )

    num_tokens_post_padded = total_padded.to(torch.int32).reshape(1)
    return sorted_token_ids, expert_ids, num_tokens_post_padded
