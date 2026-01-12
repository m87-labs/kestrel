"""Correctness tests for CuTe moe_align_block_size vs a torch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kestrel-kernels" / "python"))


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _alloc_outputs(
    *,
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = int(topk_ids.numel()) + int(num_experts) * (int(block_size) - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = _round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(int(topk_ids.numel()) * int(block_size), max_num_tokens_padded)

    sorted_token_ids = torch.empty((max_num_tokens_padded,), device=topk_ids.device, dtype=torch.int32)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty((max_num_m_blocks,), device=topk_ids.device, dtype=torch.int32)
    num_tokens_post_pad = torch.empty((1,), device=topk_ids.device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _alloc_outputs_lora(
    *,
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = int(topk_ids.numel()) + int(num_experts) * (int(block_size) - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = _round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(int(topk_ids.numel()) * int(block_size), max_num_tokens_padded)

    sorted_token_ids = torch.empty(
        (max_loras, max_num_tokens_padded),
        device=topk_ids.device,
        dtype=torch.int32,
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        (max_loras, max_num_m_blocks),
        device=topk_ids.device,
        dtype=torch.int32,
    )
    num_tokens_post_pad = torch.empty((max_loras,), device=topk_ids.device, dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _make_topk_ids(num_tokens: int, topk: int, num_experts: int, device: torch.device) -> torch.Tensor:
    topk_ids = torch.zeros((num_tokens, topk), device=device, dtype=torch.int32)
    for i in range(num_tokens):
        topk_ids[i] = torch.randperm(num_experts, device=device)[:topk]
    return topk_ids


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
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


def torch_moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    block_size: int,
    num_experts: int,
    max_loras: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_token_ids, expert_ids, num_tokens_post_pad = _alloc_outputs_lora(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        max_loras=max_loras,
    )

    sentinel = num_experts
    for lora_id in range(max_loras):
        mask = token_lora_mapping == lora_id
        if not mask.any():
            num_tokens_post_pad[lora_id] = 0
            continue
        masked_topk_ids = torch.where(mask[:, None], topk_ids, sentinel)
        sorted_ref, expert_ref, post_ref = torch_moe_align_block_size(
            masked_topk_ids, block_size, num_experts, expert_map=expert_map
        )
        sorted_token_ids[lora_id].copy_(sorted_ref)
        expert_ids[lora_id].copy_(expert_ref)
        num_tokens_post_pad[lora_id] = post_ref[0]

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _verify_same_expert_buckets(
    *,
    sorted_a: torch.Tensor,
    sorted_b: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    num_tokens_post_pad: int,
    total_tokens: int,
) -> None:
    num_blocks = num_tokens_post_pad // block_size
    actual_expert_tokens = {}
    expected_expert_tokens = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        if expert_id < 0:
            continue

        start_idx = block_idx * block_size
        end_idx = start_idx + block_size

        a_block = sorted_a[start_idx:end_idx]
        a_block = a_block[a_block < total_tokens]
        actual_expert_tokens.setdefault(expert_id, []).extend(a_block.tolist())

        b_block = sorted_b[start_idx:end_idx]
        b_block = b_block[b_block < total_tokens]
        expected_expert_tokens.setdefault(expert_id, []).extend(b_block.tolist())

    for expert_id, actual in actual_expert_tokens.items():
        expected = expected_expert_tokens.get(expert_id, [])
        torch.testing.assert_close(
            torch.sort(torch.tensor(actual, device=sorted_a.device))[0],
            torch.sort(torch.tensor(expected, device=sorted_a.device))[0],
        )


@pytest.mark.parametrize("num_tokens,block_size", [
    (1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16),
    (9, 16), (15, 16), (16, 16), (17, 16), (31, 16), (32, 16), (33, 16),
    (128, 64), (127, 64), (129, 64),
])
def test_moe_align_block_size_cute_matches_cuda(device, num_tokens: int, block_size: int):
    torch.manual_seed(0)
    topk = 8
    num_experts = 64

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)

    from kestrel_kernels.moe_align import moe_align_block_size as moe_align_block_size_cute

    sorted_cute, expert_cute, post_cute = _alloc_outputs(
        topk_ids=topk_ids, num_experts=num_experts, block_size=block_size
    )
    expert_map_empty = torch.empty((0,), device=device, dtype=torch.int32)

    moe_align_block_size_cute(
        topk_ids, num_experts, block_size, sorted_cute, expert_cute, post_cute, expert_map_empty
    )

    sorted_ref, expert_ref, post_ref = torch_moe_align_block_size(
        topk_ids, block_size, num_experts
    )

    torch.testing.assert_close(post_cute, post_ref, atol=0, rtol=0)
    torch.testing.assert_close(expert_cute, expert_ref, atol=0, rtol=0)
    _verify_same_expert_buckets(
        sorted_a=sorted_cute,
        sorted_b=sorted_ref,
        expert_ids=expert_ref,
        block_size=block_size,
        num_tokens_post_pad=int(post_ref.item()),
        total_tokens=num_tokens * topk,
    )


def test_moe_align_block_size_cute_matches_cuda_with_expert_map(device):
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

    from kestrel_kernels.moe_align import moe_align_block_size as moe_align_block_size_cute

    sorted_cute, expert_cute, post_cute = _alloc_outputs(
        topk_ids=topk_ids, num_experts=num_experts, block_size=block_size
    )
    moe_align_block_size_cute(
        topk_ids, num_experts, block_size, sorted_cute, expert_cute, post_cute, expert_map
    )

    sorted_ref, expert_ref, post_ref = torch_moe_align_block_size(
        topk_ids, block_size, num_experts, expert_map=expert_map
    )

    torch.testing.assert_close(post_cute, post_ref, atol=0, rtol=0)
    torch.testing.assert_close(expert_cute, expert_ref, atol=0, rtol=0)
    _verify_same_expert_buckets(
        sorted_a=sorted_cute,
        sorted_b=sorted_ref,
        expert_ids=expert_ref,
        block_size=block_size,
        num_tokens_post_pad=int(post_ref.item()),
        total_tokens=num_tokens * topk,
    )


def test_moe_lora_align_block_size_single_lora(device):
    torch.manual_seed(0)
    num_tokens = 128
    topk = 8
    num_experts = 64
    block_size = 16
    max_loras = 4

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)
    token_lora_mapping = torch.zeros((num_tokens,), device=device, dtype=torch.int32)

    from kestrel_kernels.moe_align import (
        moe_lora_align_block_size as moe_lora_align_block_size_cute,
    )

    sorted_cute, expert_cute, post_cute = _alloc_outputs_lora(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        max_loras=max_loras,
    )
    expert_map_empty = torch.empty((0,), device=device, dtype=torch.int32)

    moe_lora_align_block_size_cute(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        sorted_cute,
        expert_cute,
        post_cute,
        expert_map=expert_map_empty,
    )

    sorted_ref, expert_ref, post_ref = torch_moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        block_size,
        num_experts,
        max_loras,
    )

    torch.testing.assert_close(post_cute[0], post_ref[0], atol=0, rtol=0)
    torch.testing.assert_close(expert_cute[0], expert_ref[0], atol=0, rtol=0)
    _verify_same_expert_buckets(
        sorted_a=sorted_cute[0],
        sorted_b=sorted_ref[0],
        expert_ids=expert_ref[0],
        block_size=block_size,
        num_tokens_post_pad=int(post_ref[0].item()),
        total_tokens=num_tokens * topk,
    )


def test_moe_lora_align_block_size_multi_lora(device):
    torch.manual_seed(0)
    num_tokens = 32
    topk = 8
    num_experts = 16
    block_size = 16
    max_loras = 3

    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, device)
    token_lora_mapping = torch.tensor(
        [0, 1, 2, -1] * (num_tokens // 4),
        device=device,
        dtype=torch.int32,
    )

    from kestrel_kernels.moe_align import (
        moe_lora_align_block_size as moe_lora_align_block_size_cute,
    )

    sorted_cute, expert_cute, post_cute = _alloc_outputs_lora(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        max_loras=max_loras,
    )
    expert_map_empty = torch.empty((0,), device=device, dtype=torch.int32)

    moe_lora_align_block_size_cute(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        sorted_cute,
        expert_cute,
        post_cute,
        expert_map=expert_map_empty,
    )

    sorted_ref, expert_ref, post_ref = torch_moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        block_size,
        num_experts,
        max_loras,
    )

    for lora_id in range(max_loras):
        torch.testing.assert_close(post_cute[lora_id], post_ref[lora_id], atol=0, rtol=0)
        torch.testing.assert_close(expert_cute[lora_id], expert_ref[lora_id], atol=0, rtol=0)
        if post_ref[lora_id] > 0:
            _verify_same_expert_buckets(
                sorted_a=sorted_cute[lora_id],
                sorted_b=sorted_ref[lora_id],
                expert_ids=expert_ref[lora_id],
                block_size=block_size,
                num_tokens_post_pad=int(post_ref[lora_id].item()),
                total_tokens=num_tokens * topk,
            )
