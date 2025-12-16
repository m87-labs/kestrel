"""Tests for MoE mixed-slot LoRA comparing against naive implementation."""

import pytest
import torch
import torch.nn.functional as F


def naive_moe_lora(
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

    Args:
        x: Input activations [num_tokens, hidden_dim]
        topk_ids: Expert assignments [num_tokens, top_k]
        topk_weights: Router weights [num_tokens, top_k]
        lora_a: LoRA A weights [max_slots * num_experts, rank, hidden_dim]
        lora_b: LoRA B weights [max_slots * num_experts, out_dim, rank]
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
        for k in range(top_k):
            expert = topk_ids[i, k].item()
            # Super-expert index: slot * num_experts + expert
            super_expert = slot * num_experts + expert
            a = lora_a[super_expert]  # [rank, hidden_dim]
            b = lora_b[super_expert]  # [out_dim, rank]
            # x[i] @ A.T @ B.T
            h = F.linear(x[i:i+1], a)  # [1, rank]
            delta = F.linear(h, b).squeeze(0)  # [out_dim]
            if mul_routed_weight:
                delta = delta * topk_weights[i, k]
            output[i, k] = delta

    return output


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.bfloat16


class TestMoELoRA:
    """Test MoE mixed-slot LoRA against naive implementation."""

    def test_single_slot_single_token(self, device, dtype):
        """Test with a single token using a single slot."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2

        # Create super-expert shaped tensors
        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        # Slot 0 super-experts should be zeros (no LoRA)
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (1, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(1, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1], dtype=torch.int32, device=device)

        # Expected: naive implementation
        expected = naive_moe_lora(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        # Actual: use apply_moe_lora via super-expert routing
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        combined_topk_ids = topk_ids + expanded_slots * num_experts

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(1, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
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

        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        combined_topk_ids = topk_ids + expanded_slots * num_experts

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
            mul_routed_weight=False,
        )

        # Should be all zeros since slot 0 weights are zeros
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

        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        # Mix of slot 0 (no LoRA), slot 1, slot 2, slot 3
        lora_slot_ids = torch.tensor([0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=device)

        expected = naive_moe_lora(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        combined_topk_ids = topk_ids + expanded_slots * num_experts

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
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

        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device=device)

        expected = naive_moe_lora(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts, mul_routed_weight=True
        )

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        combined_topk_ids = topk_ids + expanded_slots * num_experts

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
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

        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        # Output without LoRA (slot 0 for all)
        slot_ids_no_lora = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        expanded_slots_no = slot_ids_no_lora.unsqueeze(1).expand(-1, top_k)
        combined_no = topk_ids + expanded_slots_no * num_experts

        block_size_m = 16
        sorted_no, expert_ids_no, num_tokens_no = moe_align_block_size(
            combined_no, block_size_m, num_super_experts
        )

        output_no_lora = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_no,
            topk_weights=topk_weights,
            output=output_no_lora,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_no,
            expert_ids=expert_ids_no,
            num_tokens_post_padded=num_tokens_no,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
            mul_routed_weight=False,
        )

        # Output with LoRA (slot 1 for all)
        slot_ids_with_lora = torch.ones(num_tokens, dtype=torch.int32, device=device)
        expanded_slots_with = slot_ids_with_lora.unsqueeze(1).expand(-1, top_k)
        combined_with = topk_ids + expanded_slots_with * num_experts

        sorted_with, expert_ids_with, num_tokens_with = moe_align_block_size(
            combined_with, block_size_m, num_super_experts
        )

        output_with_lora = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_with,
            topk_weights=topk_weights,
            output=output_with_lora,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_with,
            expert_ids=expert_ids_with,
            num_tokens_post_padded=num_tokens_with,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
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

        num_super_experts = max_slots * num_experts
        lora_a = torch.randn(num_super_experts, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(num_super_experts, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[:num_experts].zero_()
        lora_b[:num_experts].zero_()

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(0, num_experts, (1, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.ones(1, top_k, dtype=dtype, device=device)

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from kestrel.fused_moe.lora_kernels import apply_moe_lora

        outputs = []
        for slot in range(1, max_slots):  # Skip slot 0
            slot_ids = torch.tensor([slot], dtype=torch.int32, device=device)
            expanded_slots = slot_ids.unsqueeze(1).expand(-1, top_k)
            combined = topk_ids + expanded_slots * num_experts

            block_size_m = 16
            sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
                combined, block_size_m, num_super_experts
            )

            output = torch.zeros(1, top_k, out_dim, dtype=dtype, device=device)
            apply_moe_lora(
                x=x,
                topk_ids=combined,
                topk_weights=topk_weights,
                output=output,
                lora_a=lora_a,
                lora_b=lora_b,
                sorted_token_ids=sorted_lora,
                expert_ids=expert_ids_lora,
                num_tokens_post_padded=num_tokens_lora,
                top_k=top_k,
                config={"BLOCK_SIZE_M": block_size_m},
                mul_routed_weight=False,
            )
            outputs.append(output.clone())

        # Each slot should produce a different output
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Slots {i+1} and {j+1} should produce different outputs"


def naive_moe_lora_sentinel(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
    num_experts: int,
    mul_routed_weight: bool = False,
) -> torch.Tensor:
    """Naive MoE LoRA with sentinel-based slot 0 filtering.

    Uses the mapping: slot 0 -> no LoRA, slot N (N>0) -> super-expert (N-1)*num_experts + expert.
    Workspace is sized (max_slots-1)*num_experts with no slot 0 weights.
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
            # New mapping: slot N -> (N-1) * num_experts + expert
            super_expert = (slot - 1) * num_experts + expert
            a = lora_a[super_expert]  # [rank, hidden_dim]
            b = lora_b[super_expert]  # [out_dim, rank]
            h = F.linear(x[i : i + 1], a)  # [1, rank]
            delta = F.linear(h, b).squeeze(0)  # [out_dim]
            if mul_routed_weight:
                delta = delta * topk_weights[i, k]
            output[i, k] = delta

    return output


class TestMoELoRASentinel:
    """Test MoE LoRA with sentinel-based slot 0 filtering.

    This tests the pattern used to avoid the 1024 super-expert limit in
    moe_align_block_size. By using a sentinel value for slot 0 tokens,
    they are filtered out by the kernel (expert_id >= num_experts is skipped).
    This allows the workspace to exclude slot 0, reducing max_super_experts
    from max_slots * num_experts to (max_slots - 1) * num_experts.
    """

    def test_sentinel_filters_slot_zero(self, device, dtype):
        """Test that slot 0 tokens are filtered via sentinel (no workspace slot 0)."""
        max_slots = 4  # Workspace only has slots 1-3 (indices 0-2 in workspace)
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 8

        # Workspace sized for (max_slots - 1) super-experts - NO slot 0
        num_super_experts = (max_slots - 1) * num_experts  # 24, not 32
        lora_a = torch.randn(
            num_super_experts, rank, hidden_dim, dtype=dtype, device=device
        ) * 0.1
        lora_b = torch.randn(
            num_super_experts, out_dim, rank, dtype=dtype, device=device
        ) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        # Mix of slot 0 (no LoRA) and slots 1, 2, 3
        lora_slot_ids = torch.tensor(
            [0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=device
        )

        # Expected output from naive implementation
        expected = naive_moe_lora_sentinel(
            x, topk_ids, topk_weights, lora_a, lora_b, lora_slot_ids, num_experts
        )

        # Actual: use sentinel-based mapping
        from kestrel.fused_moe.lora_kernels import apply_moe_lora
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        sentinel = num_super_experts  # Value >= num_super_experts will be filtered
        combined_topk_ids = torch.where(
            expanded_slots > 0,
            topk_ids + (expanded_slots - 1) * num_experts,
            sentinel,
        ).to(torch.int32)

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
            mul_routed_weight=False,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_sentinel_slot_zero_output_is_zero(self, device, dtype):
        """Test that slot 0 tokens produce zero output (not processed)."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 4

        num_super_experts = (max_slots - 1) * num_experts
        lora_a = torch.randn(
            num_super_experts, rank, hidden_dim, dtype=dtype, device=device
        ) * 0.1
        lora_b = torch.randn(
            num_super_experts, out_dim, rank, dtype=dtype, device=device
        ) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        topk_weights = torch.ones(num_tokens, top_k, dtype=dtype, device=device)
        # All slot 0
        lora_slot_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

        from kestrel.fused_moe.lora_kernels import apply_moe_lora
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        sentinel = num_super_experts
        combined_topk_ids = torch.where(
            expanded_slots > 0,
            topk_ids + (expanded_slots - 1) * num_experts,
            sentinel,
        ).to(torch.int32)

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
            mul_routed_weight=False,
        )

        # All slot 0 -> all outputs should be zero
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_sentinel_with_router_weights(self, device, dtype):
        """Test sentinel filtering with router weight multiplication."""
        max_slots = 4
        num_experts = 8
        rank = 8
        hidden_dim = 64
        out_dim = 128
        top_k = 2
        num_tokens = 4

        num_super_experts = (max_slots - 1) * num_experts
        lora_a = torch.randn(
            num_super_experts, rank, hidden_dim, dtype=dtype, device=device
        ) * 0.1
        lora_b = torch.randn(
            num_super_experts, out_dim, rank, dtype=dtype, device=device
        ) * 0.1

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        expected = naive_moe_lora_sentinel(
            x,
            topk_ids,
            topk_weights,
            lora_a,
            lora_b,
            lora_slot_ids,
            num_experts,
            mul_routed_weight=True,
        )

        from kestrel.fused_moe.lora_kernels import apply_moe_lora
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, top_k)
        sentinel = num_super_experts
        combined_topk_ids = torch.where(
            expanded_slots > 0,
            topk_ids + (expanded_slots - 1) * num_experts,
            sentinel,
        ).to(torch.int32)

        block_size_m = 16
        sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
            combined_topk_ids, block_size_m, num_super_experts
        )

        output = torch.zeros(num_tokens, top_k, out_dim, dtype=dtype, device=device)
        apply_moe_lora(
            x=x,
            topk_ids=combined_topk_ids,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_a,
            lora_b=lora_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=top_k,
            config={"BLOCK_SIZE_M": block_size_m},
            mul_routed_weight=True,
        )

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
