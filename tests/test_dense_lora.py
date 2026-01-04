"""Tests for apply_dense_lora comparing against naive F.linear implementation."""

import pytest
import torch
import torch.nn.functional as F

from kestrel.moondream.layers import apply_dense_lora
from kestrel.moondream.lora_workspace import DenseLoRALayerWorkspace


def naive_dense_lora(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
) -> torch.Tensor:
    """Naive per-token LoRA using F.linear.

    Args:
        x: Input activations [num_tokens, hidden_dim]
        lora_a: LoRA A weights [max_slots, rank, hidden_dim]
        lora_b: LoRA B weights [max_slots, out_dim, rank]
        lora_slot_ids: Per-token slot indices [num_tokens]

    Returns:
        LoRA delta [num_tokens, out_dim]
    """
    num_tokens = x.shape[0]
    out_dim = lora_b.shape[1]
    output = torch.zeros(num_tokens, out_dim, dtype=x.dtype, device=x.device)

    for i in range(num_tokens):
        slot = lora_slot_ids[i].item()
        # A is [rank, hidden_dim], B is [out_dim, rank]
        a = lora_a[slot]  # [rank, hidden_dim]
        b = lora_b[slot]  # [out_dim, rank]
        # x[i] @ A.T @ B.T = x[i] @ A.T @ B.T
        h = F.linear(x[i:i+1], a)  # [1, rank]
        output[i] = F.linear(h, b).squeeze(0)  # [out_dim]

    return output


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.bfloat16


class TestApplyDenseLoRA:
    """Test apply_dense_lora against naive implementation."""

    def test_single_slot_single_token(self, device, dtype):
        """Test with a single token using a single slot."""
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        # Create workspace-like tensors
        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        # Slot 0 should be zeros (no LoRA)
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1], dtype=torch.int32, device=device)

        # Expected: naive implementation
        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)

        # Actual: apply_dense_lora accumulates into output
        output = torch.zeros(1, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_slot_zero_no_lora(self, device, dtype):
        """Test that slot 0 produces zero delta."""
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(4, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.zeros(4, dtype=torch.int32, device=device)  # All slot 0

        output = torch.zeros(4, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        # Should be all zeros since slot 0 is zeros
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_multiple_tokens_same_slot(self, device, dtype):
        """Test multiple tokens all using the same slot."""
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 16

        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.full((num_tokens,), 2, dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)

        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_mixed_slots(self, device, dtype):
        """Test tokens with different slot assignments (mixed adapters)."""
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 8

        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        # Mix of slot 0 (no LoRA), slot 1, slot 2, slot 3
        lora_slot_ids = torch.tensor([0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)

        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_writes_to_output(self, device, dtype):
        """Test that apply_dense_lora writes to output buffer (caller adds to base)."""
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 4

        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device=device)

        # Start with zero output - kernel writes delta directly
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)

        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_large_batch(self, device, dtype):
        """Test with a larger batch size typical of production."""
        max_slots = 8
        rank = 16
        hidden_dim = 1024
        out_dim = 4096
        num_tokens = 64

        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.01
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.01
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        # Random slot assignments
        lora_slot_ids = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)

        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_lora_actually_changes_output(self, device, dtype):
        """Negative test: verify that LoRA actually changes the output.

        This ensures we're not just passing a trivial test where both implementations
        happen to produce zeros or the same wrong answer.
        """
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 4

        # Create non-trivial LoRA weights
        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

        # Output without LoRA (slot 0 for all)
        output_without_lora = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        slot_ids_no_lora = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        apply_dense_lora(x, output_without_lora, lora_a, lora_b, slot_ids_no_lora)

        # Output with LoRA (slot 1 for all)
        output_with_lora = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        slot_ids_with_lora = torch.ones(num_tokens, dtype=torch.int32, device=device)
        apply_dense_lora(x, output_with_lora, lora_a, lora_b, slot_ids_with_lora)

        # They should NOT be equal - LoRA should have a real effect
        assert not torch.allclose(output_without_lora, output_with_lora), \
            "LoRA should produce different outputs than no-LoRA"

        # The no-LoRA output should be zeros
        torch.testing.assert_close(output_without_lora, torch.zeros_like(output_without_lora))

        # The with-LoRA output should NOT be zeros
        assert output_with_lora.abs().sum() > 0, "LoRA output should be non-zero"

    def test_different_slots_produce_different_outputs(self, device, dtype):
        """Negative test: verify that different slots produce different outputs.

        This ensures the slot routing is actually working and not just using
        the same weights for all slots.
        """
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        # Create distinct LoRA weights for each slot
        lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * 0.1
        lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * 0.1
        lora_a[0].zero_()
        lora_b[0].zero_()

        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)

        outputs = []
        for slot in range(1, max_slots):  # Skip slot 0
            output = torch.zeros(1, out_dim, dtype=dtype, device=device)
            slot_ids = torch.tensor([slot], dtype=torch.int32, device=device)
            apply_dense_lora(x, output, lora_a, lora_b, slot_ids)
            outputs.append(output.clone())

        # Each slot should produce a different output
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Slots {i+1} and {j+1} should produce different outputs"


class TestMlpCudaGraph:
    """Test mlp function with CUDA graph capture."""

    def test_mlp_with_lora_cudagraph(self, device, dtype):
        """Test that mlp with LoRA can be captured in a CUDA graph."""
        from kestrel.moondream.layers import mlp, MLPWeights, LinearWeights

        batch = 2
        seq_len = 4
        d_model = 64
        d_ffn = 128
        max_slots = 4
        rank = 8

        # Create MLP weights
        fc1_weight = torch.randn(d_ffn, d_model, dtype=dtype, device=device)
        fc1_bias = torch.randn(d_ffn, dtype=dtype, device=device)
        fc2_weight = torch.randn(d_model, d_ffn, dtype=dtype, device=device)
        fc2_bias = torch.randn(d_model, dtype=dtype, device=device)
        mlp_weights = MLPWeights(
            fc1=LinearWeights(weight=fc1_weight, bias=fc1_bias),
            fc2=LinearWeights(weight=fc2_weight, bias=fc2_bias),
        )

        # Create LoRA workspace
        lora_workspace = DenseLoRALayerWorkspace(
            up_a=torch.randn(max_slots, rank, d_model, dtype=dtype, device=device) * 0.1,
            up_b=torch.randn(max_slots, d_ffn, rank, dtype=dtype, device=device) * 0.1,
            down_a=torch.randn(max_slots, rank, d_ffn, dtype=dtype, device=device) * 0.1,
            down_b=torch.randn(max_slots, d_model, rank, dtype=dtype, device=device) * 0.1,
        )
        # Zero out slot 0 (no LoRA)
        lora_workspace.up_a[0].zero_()
        lora_workspace.up_b[0].zero_()
        lora_workspace.down_a[0].zero_()
        lora_workspace.down_b[0].zero_()

        # Graph-owned input/output buffers
        x = torch.randn(batch, seq_len, d_model, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)

        # Warmup
        for _ in range(3):
            out = mlp(x, mlp_weights, lora_workspace=lora_workspace, lora_slot_ids=lora_slot_ids)
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                out_graph = mlp(x, mlp_weights, lora_workspace=lora_workspace, lora_slot_ids=lora_slot_ids)

        # Replay and verify output matches eager
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()

        out_eager = mlp(x, mlp_weights, lora_workspace=lora_workspace, lora_slot_ids=lora_slot_ids)
        torch.testing.assert_close(out_graph, out_eager, rtol=1e-3, atol=1e-3)
