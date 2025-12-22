"""Correctness tests for moe_align_block_size."""

import pytest
import torch

from kestrel.fused_moe.routing import moe_align_block_size


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = ((max_num_tokens_padded + block_size - 1) // block_size) * block_size
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos : current_pos + num_expert_tokens] = expert_tokens
            expert_blocks_needed = expert_padded_counts[expert_id] // block_size
            expert_id_new = expert_id if expert_map is None else expert_map[expert_id].item()
            expert_ids[current_block : current_block + expert_blocks_needed] = expert_id_new
            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor([total_padded_tokens], dtype=torch.int32, device=topk_ids.device)

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    actual_expert_ids: torch.Tensor,
    block_size: int,
    num_tokens_post_pad: int,
    total_tokens: int,
) -> None:
    actual_expert_tokens = {}
    golden_expert_tokens = {}

    num_blocks = num_tokens_post_pad // block_size
    for block_idx in range(num_blocks):
        expert_id = actual_expert_ids[block_idx].item()
        if expert_id < 0:
            continue

        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        actual_block_tokens = actual_sorted_ids[start_idx:end_idx]
        actual_block_tokens = actual_block_tokens[actual_block_tokens < total_tokens]
        actual_expert_tokens.setdefault(expert_id, []).extend(actual_block_tokens.tolist())

        golden_block_tokens = golden_sorted_ids[start_idx:end_idx]
        golden_block_tokens = golden_block_tokens[golden_block_tokens < total_tokens]
        golden_expert_tokens.setdefault(expert_id, []).extend(golden_block_tokens.tolist())

    for expert_id in actual_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        torch.testing.assert_close(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        )


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_topk_ids(num_tokens: int, topk: int, num_experts: int, device: torch.device) -> torch.Tensor:
    topk_ids = torch.zeros((num_tokens, topk), device=device, dtype=torch.int32)
    for i in range(num_tokens):
        topk_ids[i] = torch.randperm(num_experts, device=device)[:topk]
    return topk_ids


def test_moe_align_block_size_matches_torch(device):
    torch.manual_seed(0)
    num_tokens = 128
    topk = 8
    num_experts = 64
    block_size = 16

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)

    actual_sorted, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    golden_sorted, golden_expert_ids, golden_num_tokens = torch_moe_align_block_size(
        topk_ids, block_size, num_experts
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted,
        golden_sorted,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        num_tokens * topk,
    )


def test_moe_align_block_size_with_expert_map(device):
    torch.manual_seed(0)
    num_tokens = 128
    topk = 8
    num_experts = 64
    block_size = 16

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)

    expert_map = torch.full((num_experts,), -1, device=device, dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    actual_sorted, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    golden_sorted, golden_expert_ids, golden_num_tokens = torch_moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        expert_map=expert_map,
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted,
        golden_sorted,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        num_tokens * topk,
    )


def test_moe_align_block_size_rejects_bad_dtype(device):
    topk_ids = torch.zeros((4, 2), device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="topk_ids must be int32"):
        moe_align_block_size(topk_ids, 16, 64)


def test_moe_align_block_size_small_batch_matches_torch(device):
    torch.manual_seed(0)
    num_tokens = 4
    topk = 8
    num_experts = 64
    block_size = 16

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)

    actual_sorted, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    golden_sorted, golden_expert_ids, golden_num_tokens = torch_moe_align_block_size(
        topk_ids, block_size, num_experts
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted,
        golden_sorted,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        num_tokens * topk,
    )
