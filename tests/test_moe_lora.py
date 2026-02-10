"""Tests for MoE LoRA kernel using moe_lora_align_block_size."""

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn


def naive_moe_lora_batched(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
    num_experts: int,
    mul_routed_weight: bool = False,
) -> torch.Tensor:
    """Naive per-token MoE LoRA using F.linear.

    Uses the batched mapping: slot 0 -> no LoRA, slot N (N>0) -> lora_id N-1.
    Workspace is sized (max_slots-1)*num_experts with no slot 0 weights.

    Args:
        x: Input activations [num_tokens, hidden_dim]
        topk_ids: Expert assignments [num_tokens, top_k]
        topk_weights: Router weights [num_tokens, top_k]
        lora_a: LoRA A weights [(max_slots-1) * num_experts, rank, hidden_dim]
        lora_b: LoRA B weights [(max_slots-1) * num_experts, out_dim, rank]
        lora_slot_ids: Per-token slot indices [num_tokens]
        num_experts: Number of experts
        mul_routed_weight: Whether to multiply by router weights

    Returns:
        LoRA delta [num_tokens, top_k, out_dim]
    """
    num_tokens, top_k = topk_ids.shape
    out_dim = lora_b.shape[1]
    output = torch.zeros(num_tokens, top_k, out_dim, dtype=x.dtype, device=x.device)

    for i in range(num_tokens):
        slot = lora_slot_ids[i].item()
        if slot == 0:
            # Slot 0 = no LoRA, output stays zero
            continue
        for k in range(top_k):
            expert = topk_ids[i, k].item()
            # Batched mapping: slot N -> (N-1) * num_experts + expert
            super_expert = (slot - 1) * num_experts + expert
            a = lora_a[super_expert]  # [rank, hidden_dim]
            b = lora_b[super_expert]  # [out_dim, rank]
            h = F.linear(x[i : i + 1], a)  # [1, rank]
            delta = F.linear(h, b).squeeze(0)  # [out_dim]
            if mul_routed_weight:
                delta = delta * topk_weights[i, k]
            output[i, k] = delta

    return output


class DummyExperts(nn.Module):
    """Minimal expert container for FusedMoEModule tests."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_features, in_features, dtype=dtype, device=device)
            * 0.02,
            requires_grad=False,
        )


@pytest.fixture
def device():
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def dtype():
    return torch.bfloat16


@pytest.fixture(scope="module", autouse=True)
def _preallocate_lora_intermediate_buffers() -> None:
    """Pre-size global LoRA intermediates to cover this test module.

    LoRA kernels keep module-global FixedBuffer instances. Without up-front
    sizing, an early small-shape test can lock in tiny storage and cause later
    larger-shape tests in this module to fail with overflow.

    Use a suite-wide upper bound so this module composes with test_dense_lora,
    which shares the same global buffers.
    """
    from kestrel.fused_moe.lora_kernels import preallocate_lora_buffers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda", torch.cuda.current_device())
    preallocate_lora_buffers(
        max_num_tokens=4096,
        top_k=8,
        max_lora_rank=256,
        device=device,
        dtype=torch.bfloat16,
    )


class TestMoELoRA:
    """Test MoE LoRA kernel against naive implementation."""

    def test_single_slot_single_token(self, device, dtype):
        """Test with a single token using a single slot."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2

        # Workspace sized for (max_slots - 1) * num_experts
        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (1, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(1, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1], dtype=torch.int32, device=device)

        # Expected: naive implementation
        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        # Actual: use batched kernel with moe_lora_align_block_size
        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        # Convert slot IDs to lora mapping: slot 0 -> -1, slot N -> N-1
        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
        block_size_m = 16

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            block_size_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(1, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_slot_zero_no_lora(self, device, dtype):
        """Test that slot 0 produces zero delta."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 4

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
        block_size_m = 16

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            block_size_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        # Should be all zeros since slot 0 is filtered
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_mixed_slots(self, device, dtype):
        """Test tokens with different slot assignments (mixed adapters)."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 8

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        # Mix of slot 0 (no LoRA), slot 1, slot 2, slot 3
        lora_slot_ids = torch.tensor([0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
        block_size_m = 16

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            block_size_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_with_router_weights(self, device, dtype):
        """Test that router weights are applied correctly."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 4

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts, mul_routed_weight=True
        )

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
        block_size_m = 16

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            block_size_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=True,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_lora_actually_changes_output(self, device, dtype):
        """Negative test: verify that LoRA actually changes the output."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 4

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        block_size_m = 16
        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)

        # Output without LoRA (slot 0 for all)
        slot_ids_no_lora = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        mapping_no = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)

        sorted_no, expert_ids_no, num_tokens_no = moe_lora_align_block_size(
            topk_ids, mapping_no, block_size_m, num_experts, max_loras        )

        output_no_lora = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output_no_lora,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_no,
            expert_ids=expert_ids_no,
            num_tokens_post_padded=num_tokens_no,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        # Output with LoRA (slot 1 for all)
        slot_ids_with_lora = torch.ones(num_tokens, dtype=torch.int32, device=device)
        mapping_with = torch.zeros(num_tokens, dtype=torch.int32, device=device)  # lora_id=0

        sorted_with, expert_ids_with, num_tokens_with = moe_lora_align_block_size(
            topk_ids, mapping_with, block_size_m, num_experts, max_loras        )

        output_with_lora = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output_with_lora,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_with,
            expert_ids=expert_ids_with,
            num_tokens_post_padded=num_tokens_with,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        # They should NOT be equal
        assert not torch.allclose(output_no_lora, output_with_lora), \
            "LoRA should produce different outputs than no-LoRA"

        # The no-LoRA output should be zeros
        torch.testing.assert_close(output_no_lora, torch.zeros_like(output_no_lora))

        # The with-LoRA output should NOT be zeros
        assert output_with_lora.abs().sum() > 0, "LoRA output should be non-zero"

    def test_different_slots_produce_different_outputs(self, device, dtype):
        """Negative test: verify that different slots produce different outputs."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (1, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(1, top_k, dtype=dtype, device=device)

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        block_size_m = 16
        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)

        outputs = []
        for slot in range(1, max_slots):  # Skip slot 0
            # lora_id = slot - 1
            lora_id = slot - 1
            mapping = torch.tensor([lora_id], dtype=torch.int32, device=device)

            sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
                topk_ids, mapping, block_size_m, num_experts, max_loras            )

            output = torch.zeros(1, top_k, out_dim, dtype=dtype, device=device)
            apply_moe_lora_batched(
                x=x,
                topk_weights=topk_weights,
                output=output,
                lora_a=lora_a,
                lora_b=lora_b,
                sorted_token_ids=sorted_lora,
                expert_ids=expert_ids_lora,
                num_tokens_post_padded=num_tokens_lora,
                    top_k=top_k,
                num_experts=num_experts,
                block_size_m=block_size_m,
                mul_routed_weight=False,
            )
            outputs.append(output.clone())

        # Each slot should produce a different output
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Slots {i+1} and {j+1} should produce different outputs"

    def test_prefill_path_uses_split_kernels(self, device, dtype):
        """Test that larger token counts use split shrink/expand with PDL."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        # Use more than 256 tokens to trigger split path
        num_tokens = 512

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
        block_size_m = 16

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            block_size_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestSingleLoRA:
    """Test single-LoRA path (optimized for prefill)."""

    def test_single_lora_matches_naive(self, device, dtype):
        """Test apply_moe_lora_single matches naive implementation."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 32

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        # All tokens use slot 2 (lora_id = 1)
        lora_slot = 2
        lora_id = lora_slot - 1
        lora_slot_ids = torch.full((num_tokens,), lora_slot, dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_single

        block_size_m = 16
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, num_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_single(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            lora_id=lora_id,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_single_lora_matches_batched(self, device, dtype):
        """Test that single-LoRA path produces same results as batched path."""
        max_slots = 8
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 128

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)

        # All tokens use slot 3 (lora_id = 2)
        lora_slot = 3
        lora_id = lora_slot - 1
        lora_slot_ids = torch.full((num_tokens,), lora_slot, dtype=torch.int32, device=device)

        from kestrel.fused_moe.routing import moe_align_block_size, moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_single, apply_moe_lora_batched

        block_size_m = 16

        # Single-LoRA path
        sorted_single, expert_ids_single, num_post_single = moe_align_block_size(
            topk_ids, block_size_m, num_experts
        )
        output_single = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_single(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output_single,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_single,
            expert_ids=expert_ids_single,
            num_tokens_post_padded=num_post_single,
            lora_id=lora_id,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=True,
        )

        # Batched path
        token_lora_mapping = (lora_slot_ids - 1).to(torch.int32)
        sorted_batch, expert_ids_batch, num_post_batch = moe_lora_align_block_size(
            topk_ids, token_lora_mapping, block_size_m, num_experts, max_loras
        )
        output_batched = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output_batched,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_batch,
            expert_ids=expert_ids_batch,
            num_tokens_post_padded=num_post_batch,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=True,
        )

        torch.testing.assert_close(output_single, output_batched, rtol=1e-2, atol=1e-2)

    def test_single_lora_prefill_split_path(self, device, dtype):
        """Test single-LoRA with large token count (triggers split shrink/expand)."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        # Use more than 256 tokens to trigger split path
        num_tokens = 512

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        lora_slot = 1
        lora_id = lora_slot - 1
        lora_slot_ids = torch.full((num_tokens,), lora_slot, dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_single

        block_size_m = 16
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, num_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_single(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            lora_id=lora_id,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_different_lora_ids_produce_different_outputs(self, device, dtype):
        """Test that different lora_ids produce different outputs."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 16

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_single

        block_size_m = 16
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, num_experts
        )

        outputs = []
        for lora_id in range(max_loras):
            output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
            apply_moe_lora_single(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=output,
                lora_a=lora_a,
                lora_b=lora_b,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                lora_id=lora_id,
                top_k=top_k,
                num_experts=num_experts,
                block_size_m=block_size_m,
                mul_routed_weight=False,
            )
            outputs.append(output.clone())

        # Each lora_id should produce a different output
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"lora_id {i} and {j} should produce different outputs"


class TestMoELoRACudaGraph:
    """Test MoE LoRA kernel with CUDA graph capture."""

    def test_moe_lora_cudagraph(self, device, dtype):
        """Test that MoE LoRA batched kernel can be captured in a CUDA graph."""
        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 8

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        # Graph-owned input buffers
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([0, 1, 2, 3, 1, 2, 0, 1], dtype=torch.int32, device=device)
        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)

        block_size_m = 16

        if torch.cuda.get_device_capability(device)[0] != 9:
            pytest.skip("moe_lora_align_block_size CUDA graph test requires SM90 kernels")

        def run_kernel():
            # Use graph-safe pattern: (lora_slot_ids - 1) instead of torch.where
            token_lora_mapping = (lora_slot_ids - 1).to(torch.int32)

            sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
                topk_ids,
                token_lora_mapping,
                block_size_m,
                num_experts,
                max_loras,
            )

            output.zero_()
            apply_moe_lora_batched(
                x=x,
                topk_weights=topk_weights,
                output=output,
                lora_a=lora_a,
                lora_b=lora_b,
                sorted_token_ids=sorted_lora,
                expert_ids=expert_ids_lora,
                num_tokens_post_padded=num_tokens_lora,
                top_k=top_k,
                num_experts=num_experts,
                block_size_m=block_size_m,
                mul_routed_weight=False,
            )
            return output.clone()

        # Warmup
        for _ in range(3):
            out_eager = run_kernel()
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                run_kernel()
        torch.cuda.synchronize()

        # Replay and compare
        g.replay()
        torch.cuda.synchronize()
        out_graph = output.clone()

        torch.testing.assert_close(out_graph, out_eager, rtol=1e-5, atol=1e-5)

        # Test with different inputs
        x.copy_(torch.randn_like(x))
        lora_slot_ids.copy_(torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device=device))

        out_eager2 = run_kernel()
        g.replay()
        torch.cuda.synchronize()
        out_graph2 = output.clone()

        torch.testing.assert_close(out_graph2, out_eager2, rtol=1e-5, atol=1e-5)


def _to_power_of_2(x: int) -> int:
    """Round x down to the nearest power of 2.

    Triton's tl.arange requires the range to be a power of 2.
    We round DOWN to ensure we use a block size that has precompiled kernels.
    (The CuTe MoE configs define block_m values like 192 which aren't power of 2,
    but rounding down to 128 ensures we have precompiled routing kernels.)
    """
    if x <= 0:
        return 1
    if x & (x - 1) == 0:
        return x  # Already a power of 2
    # Round down: find the highest set bit
    return 1 << (x.bit_length() - 1)


class TestToPowerOf2Helper:
    """Unit tests for the _to_power_of_2 helper function."""

    def test_already_power_of_2(self):
        """Power-of-2 values should be unchanged."""
        assert _to_power_of_2(1) == 1
        assert _to_power_of_2(2) == 2
        assert _to_power_of_2(4) == 4
        assert _to_power_of_2(8) == 8
        assert _to_power_of_2(16) == 16
        assert _to_power_of_2(32) == 32
        assert _to_power_of_2(64) == 64
        assert _to_power_of_2(128) == 128
        assert _to_power_of_2(256) == 256

    def test_round_down_non_power_of_2(self):
        """Non-power-of-2 values should round down."""
        # Values from actual CuTe MoE configs
        assert _to_power_of_2(192) == 128  # FP8 config value
        # Other potential values
        assert _to_power_of_2(96) == 64
        assert _to_power_of_2(48) == 32
        assert _to_power_of_2(24) == 16
        # Edge cases near powers of 2
        assert _to_power_of_2(3) == 2
        assert _to_power_of_2(5) == 4
        assert _to_power_of_2(7) == 4
        assert _to_power_of_2(9) == 8
        assert _to_power_of_2(15) == 8
        assert _to_power_of_2(17) == 16
        assert _to_power_of_2(31) == 16
        assert _to_power_of_2(33) == 32
        assert _to_power_of_2(63) == 32
        assert _to_power_of_2(65) == 64
        assert _to_power_of_2(127) == 64
        assert _to_power_of_2(129) == 128
        assert _to_power_of_2(255) == 128
        assert _to_power_of_2(257) == 256

    def test_edge_cases(self):
        """Edge cases for small and zero values."""
        assert _to_power_of_2(0) == 1
        assert _to_power_of_2(-1) == 1
        assert _to_power_of_2(-100) == 1


class TestNonPowerOf2BlockSize:
    """Test that LoRA kernels work with non-power-of-2 block sizes.

    Triton's tl.arange requires power-of-2 range. The CuTe MoE FP8 configs
    can return non-power-of-2 block_m values (e.g., 192 for large token counts).
    These tests verify that the LoRA kernels handle this correctly by
    rounding down to the nearest power of 2.

    Config analysis:
    - FP8 block_m values: 16, 64, 128, 192 (only 192 is non-power-of-2)
    - BF16 block_m values: 16, 32, 64, 128 (all power-of-2)
    - Precompiled routing kernels: 16, 32, 64, 128, 192
    - 192 → 128 (precompiled ✓)
    """

    # Test the actual config value (192) plus other non-power-of-2 values
    @pytest.mark.parametrize("block_size_m", [192, 96, 48, 24])
    def test_batched_lora_non_power_of_2_block_size(self, device, dtype, block_size_m):
        """Test batched MoE LoRA kernel with non-power-of-2 block_size_m.

        This is a regression test for the bug where block_m=192 (from FP8 config
        for large token counts) caused Triton compilation to fail with:
        'arange's range must be a power of 2'

        The fix rounds block_size_m DOWN to power-of-2 for BOTH routing and kernel.
        """
        max_slots = 4
        num_experts = 64  # Match Moondream MoE config
        rank = 8
        hidden_dim = 128
        out_dim = 256
        top_k = 8  # Match Moondream MoE top_k
        num_tokens = 128  # Enough tokens to exercise the block size

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        # Mix of different slots
        lora_slot_ids = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        # Round DOWN to power-of-2 for BOTH routing and kernel
        lora_block_m = _to_power_of_2(block_size_m)
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            lora_block_m,  # Power-of-2 for routing
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=lora_block_m,  # Power-of-2 for kernel
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_size_m", [192, 96, 48, 24])
    def test_single_lora_non_power_of_2_block_size(self, device, dtype, block_size_m):
        """Test single MoE LoRA kernel with non-power-of-2 block_size_m."""
        max_slots = 4
        num_experts = 64
        rank = 8
        hidden_dim = 128
        out_dim = 256
        top_k = 8  # Match Moondream MoE top_k
        num_tokens = 128

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        # All tokens use slot 2 (lora_id = 1)
        lora_slot = 2
        lora_id = lora_slot - 1
        lora_slot_ids = torch.full((num_tokens,), lora_slot, dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_single

        # Round DOWN to power-of-2 for BOTH routing and kernel
        lora_block_m = _to_power_of_2(block_size_m)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, lora_block_m, num_experts  # Power-of-2 for routing
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_single(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            lora_id=lora_id,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=lora_block_m,  # Power-of-2 for kernel
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    # Also test that power-of-2 values still work (regression protection)
    @pytest.mark.parametrize("block_size_m", [16, 32, 64, 128])
    def test_batched_lora_power_of_2_block_size(self, device, dtype, block_size_m):
        """Test that power-of-2 block sizes still work correctly."""
        max_slots = 4
        num_experts = 64
        rank = 8
        hidden_dim = 128
        out_dim = 256
        top_k = 8
        num_tokens = 128

        max_loras = max_slots - 1
        num_super_experts = max_loras * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32, device=device)

        expected = naive_moe_lora_batched(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from kestrel.fused_moe.routing import moe_lora_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora_batched

        token_lora_mapping = torch.where(
            lora_slot_ids > 0,
            lora_slot_ids - 1,
            torch.tensor(-1, device=device, dtype=lora_slot_ids.dtype),
        ).to(torch.int32)

        # Power-of-2 values should be unchanged
        lora_block_m = _to_power_of_2(block_size_m)
        assert lora_block_m == block_size_m, f"Power-of-2 {block_size_m} should be unchanged"

        sorted_lora, expert_ids_lora, num_tokens_lora = moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            lora_block_m,
            num_experts,
            max_loras,
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            num_experts=num_experts,
            block_size_m=lora_block_m,
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestLoRAStream:
    """Validate batched LoRA path with a dedicated stream."""

    def test_batched_stream_matches_default(self, device, dtype):
        torch.manual_seed(0)

        from kestrel.fused_moe.module import FusedMoEModule
        from kestrel.moondream.lora_workspace import MoELoRALayerWorkspace

        num_experts = 64
        top_k = 8
        d_model = 2048
        d_expert = 1024
        rank = 8
        max_slots = 3
        max_loras = max_slots - 1
        num_tokens = 8

        up_experts = DummyExperts(
            num_experts=num_experts,
            in_features=d_model,
            out_features=d_expert * 2,
            dtype=dtype,
            device=device,
        )
        down_experts = DummyExperts(
            num_experts=num_experts,
            in_features=d_expert,
            out_features=d_model,
            dtype=dtype,
            device=device,
        )
        moe = FusedMoEModule(
            up_experts=up_experts,
            down_experts=down_experts,
            top_k=top_k,
            hidden_size=d_expert,
            input_size=d_model,
            num_experts=num_experts,
        )

        hidden_states = torch.randn(num_tokens, d_model, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, dtype=dtype, device=device), dim=-1
        )

        num_super_experts = max_loras * num_experts
        lora_workspace = MoELoRALayerWorkspace(
            up_a=torch.randn(num_super_experts, rank, d_model, dtype=dtype, device=device) * 0.1,
            up_b=torch.randn(num_super_experts, d_expert * 2, rank, dtype=dtype, device=device) * 0.1,
            down_a=torch.randn(num_super_experts, rank, d_expert, dtype=dtype, device=device) * 0.1,
            down_b=torch.randn(num_super_experts, d_model, rank, dtype=dtype, device=device) * 0.1,
            num_experts=num_experts,
        )

        lora_slot_ids = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32, device=device)

        lora_workspace.stream = None
        out_default = moe(
            hidden_states,
            topk_weights,
            topk_ids,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
        )
        torch.cuda.synchronize()

        lora_workspace.stream = torch.cuda.Stream(device=device)
        out_stream = moe(
            hidden_states,
            topk_weights,
            topk_ids,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(out_stream, out_default, rtol=2e-2, atol=2e-2)
