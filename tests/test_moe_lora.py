"""MoE LoRA application smoke tests against a PyTorch reference."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _max_tokens_padded(topk_ids: torch.Tensor, block_size: int, num_experts: int) -> int:
    padded = int(topk_ids.numel()) + int(num_experts) * (int(block_size) - 1)
    if topk_ids.numel() < num_experts:
        padded = min(int(topk_ids.numel()) * int(block_size), padded)
    return padded


def _alloc_lora_route(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_tokens = _max_tokens_padded(topk_ids, block_size, num_experts)
    max_blocks = _round_up(max_tokens, block_size) // block_size
    return (
        torch.empty((max_loras, max_tokens), dtype=torch.int32, device=topk_ids.device),
        torch.empty((max_loras, max_blocks), dtype=torch.int32, device=topk_ids.device),
        torch.empty((max_loras,), dtype=torch.int32, device=topk_ids.device),
    )


def _lora_route(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from kestrel_kernels.moe_align import moe_lora_align_block_size

    sorted_ids, expert_ids, post = _alloc_lora_route(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        max_loras=max_loras,
    )
    moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        post,
    )
    return sorted_ids, expert_ids, post


def _reference_moe_lora(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
    num_experts: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    out = torch.zeros(
        (num_tokens, top_k, lora_b.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    for token in range(num_tokens):
        slot = int(lora_slot_ids[token].item())
        if slot <= 0:
            continue
        lora_id = slot - 1
        for route in range(top_k):
            expert = int(topk_ids[token, route].item())
            super_expert = lora_id * int(num_experts) + expert
            hidden = F.linear(x[token : token + 1], lora_a[super_expert])
            delta = F.linear(hidden, lora_b[super_expert]).squeeze(0)
            if mul_routed_weight:
                delta = delta * topk_weights[token, route]
            out[token, route] = delta
    return out


def _apply_moe_lora(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    max_loras: int,
    mul_routed_weight: bool = False,
    active_lora_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    from kestrel_kernels.moe.lora import apply_moe_lora_batched

    token_lora_mapping = torch.where(
        lora_slot_ids > 0,
        lora_slot_ids - 1,
        torch.tensor(-1, device=x.device, dtype=torch.int32),
    ).to(torch.int32)
    sorted_ids, expert_ids, post = _lora_route(
        topk_ids,
        token_lora_mapping,
        block_size=block_size,
        num_experts=num_experts,
        max_loras=max_loras,
    )
    output = torch.zeros(
        (topk_ids.shape[0], topk_ids.shape[1], lora_b.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    scratch = torch.empty(
        (topk_ids.numel(), lora_a.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    apply_moe_lora_batched(
        x=x,
        topk_weights=topk_weights,
        output=output,
        lora_a=lora_a,
        lora_b=lora_b,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=post,
        top_k=topk_ids.shape[1],
        num_experts=num_experts,
        block_size_m=block_size,
        scratch=scratch,
        mul_routed_weight=mul_routed_weight,
        active_lora_ids=active_lora_ids,
    )
    return output


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def test_moe_lora_application_matches_reference(device: torch.device) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_tokens = 8
    top_k = 8
    num_experts = 64
    max_loras = 3
    rank = 8
    hidden_dim = 32
    out_dim = 48
    block_size = 16

    x = (torch.randn((num_tokens, hidden_dim), device=device) * 0.25).to(dtype)
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device=device, dtype=torch.int32)[:top_k]
            for _ in range(num_tokens)
        ]
    ).contiguous()
    topk_weights = torch.rand((num_tokens, top_k), dtype=dtype, device=device)
    lora_slot_ids = torch.tensor([0, 1, 2, 3, 1, 0, 2, 3], dtype=torch.int32, device=device)
    lora_a = (torch.randn((max_loras * num_experts, rank, hidden_dim), device=device) * 0.2).to(dtype)
    lora_b = (torch.randn((max_loras * num_experts, out_dim, rank), device=device) * 0.2).to(dtype)

    actual = _apply_moe_lora(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_slot_ids=lora_slot_ids,
        block_size=block_size,
        num_experts=num_experts,
        max_loras=max_loras,
        mul_routed_weight=True,
    )
    expected = _reference_moe_lora(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_slot_ids=lora_slot_ids,
        num_experts=num_experts,
        mul_routed_weight=True,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_moe_lora_no_active_adapters_is_noop(device: torch.device) -> None:
    dtype = torch.bfloat16
    topk_ids = torch.randint(0, 64, (4, 8), dtype=torch.int32, device=device)
    topk_weights = torch.ones((4, 8), dtype=dtype, device=device)
    lora_slot_ids = torch.ones((4,), dtype=torch.int32, device=device)
    lora_a = torch.randn((64, 8, 16), dtype=dtype, device=device)
    lora_b = torch.randn((64, 24, 8), dtype=dtype, device=device)

    output = _apply_moe_lora(
        x=torch.randn((4, 16), dtype=dtype, device=device),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_slot_ids=lora_slot_ids,
        block_size=16,
        num_experts=64,
        max_loras=1,
        active_lora_ids=torch.empty((0,), dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_moe_lora_cudagraph_replay_is_stable(device: torch.device) -> None:
    from kestrel_kernels.moe.lora import apply_moe_lora_batched
    from kestrel_kernels.moe_align import moe_lora_align_block_size

    torch.manual_seed(1)
    dtype = torch.bfloat16
    num_tokens = 4
    top_k = 8
    num_experts = 64
    rank = 8
    block_size = 16

    x = torch.randn((num_tokens, 16), dtype=dtype, device=device)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.ones((num_tokens, top_k), dtype=dtype, device=device)
    lora_slot_ids = torch.tensor([1, 2, 0, 1], dtype=torch.int32, device=device)
    lora_a = torch.randn((2 * num_experts, rank, 16), dtype=dtype, device=device)
    lora_b = torch.randn((2 * num_experts, 24, rank), dtype=dtype, device=device)
    token_lora_mapping = torch.where(
        lora_slot_ids > 0,
        lora_slot_ids - 1,
        torch.tensor(-1, device=device, dtype=torch.int32),
    ).to(torch.int32)
    sorted_ids, expert_ids, post = _alloc_lora_route(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        max_loras=2,
    )
    output = torch.empty((num_tokens, top_k, 24), dtype=dtype, device=device)
    scratch = torch.empty((num_tokens * top_k, rank), dtype=dtype, device=device)

    expected = _reference_moe_lora(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_slot_ids=lora_slot_ids,
        num_experts=num_experts,
        mul_routed_weight=False,
    )

    def run() -> None:
        output.zero_()
        moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            post,
        )
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=post,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size,
            scratch=scratch,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
