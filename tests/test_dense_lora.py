"""Tests for dense LoRA against a naive F.linear reference."""

import pytest
import torch
import torch.nn.functional as F

from kestrel.dense_lora import (
    apply_dense_lora,
    create_mlp_scratch,
    prepare_dense_lora_batch,
)
from kestrel.moondream.lora_workspace import DenseLoRALayerWorkspace


def naive_dense_lora(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
) -> torch.Tensor:
    """Naive per-token LoRA using F.linear."""
    num_tokens = x.shape[0]
    out_dim = lora_b.shape[1]
    output = torch.zeros(num_tokens, out_dim, dtype=x.dtype, device=x.device)

    for i in range(num_tokens):
        slot = int(lora_slot_ids[i].item())
        h = F.linear(x[i : i + 1], lora_a[slot])
        output[i] = F.linear(h, lora_b[slot]).squeeze(0)

    return output


def make_lora_weights(
    max_slots: int,
    rank: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    lora_a = torch.randn(max_slots, rank, hidden_dim, dtype=dtype, device=device) * scale
    lora_b = torch.randn(max_slots, out_dim, rank, dtype=dtype, device=device) * scale
    lora_a[0].zero_()
    lora_b[0].zero_()
    return lora_a, lora_b


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.bfloat16


class TestApplyDenseLoRA:
    """Dense LoRA correctness for the supported backend."""

    def test_single_slot_single_token(self, device: torch.device, dtype: torch.dtype) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1], dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)
        output = torch.zeros(1, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_slot_zero_no_lora(self, device: torch.device, dtype: torch.dtype) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(4, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.zeros(4, dtype=torch.int32, device=device)

        output = torch.zeros(4, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_multiple_tokens_same_slot(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 16

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.full((num_tokens,), 2, dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_mixed_slots(self, device: torch.device, dtype: torch.dtype) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 8

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor(
            [0, 1, 2, 3, 1, 2, 0, 3],
            dtype=torch.int32,
            device=device,
        )

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_writes_to_output(self, device: torch.device, dtype: torch.dtype) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 4

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device=device)

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_large_batch(self, device: torch.device, dtype: torch.dtype) -> None:
        max_slots = 8
        rank = 16
        hidden_dim = 1024
        out_dim = 4096
        num_tokens = 64

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
            scale=0.01,
        )
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        lora_slot_ids = torch.randint(
            0,
            max_slots,
            (num_tokens,),
            dtype=torch.int32,
            device=device,
        )

        expected = naive_dense_lora(x, lora_a, lora_b, lora_slot_ids)
        output = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, lora_slot_ids)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_sequence_segmented_prepare_batch(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        batch = 3
        seq_len = 4
        max_slots = 5
        hidden_dim = 64
        out_dim = 128

        lora_a, lora_b = make_lora_weights(
            max_slots,
            8,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(batch * seq_len, hidden_dim, dtype=dtype, device=device)
        sequence_slot_ids = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
        token_slot_ids = sequence_slot_ids.repeat_interleave(seq_len)

        expected = naive_dense_lora(x, lora_a, lora_b, token_slot_ids)
        prepared_batch = prepare_dense_lora_batch(
            sequence_slot_ids,
            segment_len=seq_len,
        )
        output = torch.zeros(batch * seq_len, out_dim, dtype=dtype, device=device)
        apply_dense_lora(x, output, lora_a, lora_b, prepared_batch=prepared_batch)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_lora_actually_changes_output(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128
        num_tokens = 4

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

        output_without_lora = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        slot_ids_no_lora = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        apply_dense_lora(x, output_without_lora, lora_a, lora_b, slot_ids_no_lora)

        output_with_lora = torch.zeros(num_tokens, out_dim, dtype=dtype, device=device)
        slot_ids_with_lora = torch.ones(num_tokens, dtype=torch.int32, device=device)
        apply_dense_lora(x, output_with_lora, lora_a, lora_b, slot_ids_with_lora)

        assert not torch.allclose(output_without_lora, output_with_lora)
        torch.testing.assert_close(output_without_lora, torch.zeros_like(output_without_lora))
        assert output_with_lora.abs().sum() > 0

    def test_different_slots_produce_different_outputs(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        max_slots = 4
        rank = 8
        hidden_dim = 64
        out_dim = 128

        lora_a, lora_b = make_lora_weights(
            max_slots,
            rank,
            hidden_dim,
            out_dim,
            dtype=dtype,
            device=device,
        )
        x = torch.randn(1, hidden_dim, dtype=dtype, device=device)

        outputs = []
        for slot in range(1, max_slots):
            output = torch.zeros(1, out_dim, dtype=dtype, device=device)
            apply_dense_lora(
                x,
                output,
                lora_a,
                lora_b,
                torch.tensor([slot], dtype=torch.int32, device=device),
            )
            outputs.append(output.clone())

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j])


class TestMlpCudaGraph:
    """CUDA graph capture coverage for the dense LoRA path."""

    def test_mlp_with_lora_scratch_matches_eager(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        from kestrel.moondream.layers import LinearWeights, MLPWeights, mlp

        batch = 2
        seq_len = 4
        d_model = 64
        d_ffn = 128
        max_slots = 4
        rank = 8

        fc1_weight = torch.randn(d_ffn, d_model, dtype=dtype, device=device)
        fc1_bias = torch.randn(d_ffn, dtype=dtype, device=device)
        fc2_weight = torch.randn(d_model, d_ffn, dtype=dtype, device=device)
        fc2_bias = torch.randn(d_model, dtype=dtype, device=device)
        mlp_weights = MLPWeights(
            fc1=LinearWeights(weight=fc1_weight, bias=fc1_bias),
            fc2=LinearWeights(weight=fc2_weight, bias=fc2_bias),
        )

        lora_workspace = DenseLoRALayerWorkspace(
            up_a=torch.randn(max_slots, rank, d_model, dtype=dtype, device=device) * 0.1,
            up_b=torch.randn(max_slots, d_ffn, rank, dtype=dtype, device=device) * 0.1,
            down_a=torch.randn(max_slots, rank, d_ffn, dtype=dtype, device=device) * 0.1,
            down_b=torch.randn(max_slots, d_model, rank, dtype=dtype, device=device) * 0.1,
        )
        lora_workspace.up_a[0].zero_()
        lora_workspace.up_b[0].zero_()
        lora_workspace.down_a[0].zero_()
        lora_workspace.down_b[0].zero_()

        x = torch.randn(batch, seq_len, d_model, dtype=dtype, device=device)
        lora_slot_ids = torch.tensor([1, 2], dtype=torch.int32, device=device)
        lora_scratch = create_mlp_scratch(
            max_segments=batch,
            max_segment_len=seq_len,
            max_rank=rank,
            d_model=d_model,
            d_ffn=d_ffn,
            device=device,
            dtype=dtype,
        )

        out_eager = mlp(
            x,
            mlp_weights,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
        )
        out_scratch = mlp(
            x,
            mlp_weights,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
            lora_scratch=lora_scratch,
        )

        torch.testing.assert_close(out_scratch, out_eager, rtol=1e-3, atol=1e-3)

    def test_mlp_with_lora_cudagraph_dynamic_slot_values(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        from kestrel.moondream.layers import LinearWeights, MLPWeights, mlp

        batch = 2
        seq_len = 4
        d_model = 64
        d_ffn = 128
        max_slots = 4
        rank = 8

        fc1_weight = torch.randn(d_ffn, d_model, dtype=dtype, device=device)
        fc1_bias = torch.randn(d_ffn, dtype=dtype, device=device)
        fc2_weight = torch.randn(d_model, d_ffn, dtype=dtype, device=device)
        fc2_bias = torch.randn(d_model, dtype=dtype, device=device)
        mlp_weights = MLPWeights(
            fc1=LinearWeights(weight=fc1_weight, bias=fc1_bias),
            fc2=LinearWeights(weight=fc2_weight, bias=fc2_bias),
        )

        lora_workspace = DenseLoRALayerWorkspace(
            up_a=torch.randn(max_slots, rank, d_model, dtype=dtype, device=device) * 0.1,
            up_b=torch.randn(max_slots, d_ffn, rank, dtype=dtype, device=device) * 0.1,
            down_a=torch.randn(max_slots, rank, d_ffn, dtype=dtype, device=device) * 0.1,
            down_b=torch.randn(max_slots, d_model, rank, dtype=dtype, device=device) * 0.1,
        )
        lora_workspace.up_a[0].zero_()
        lora_workspace.up_b[0].zero_()
        lora_workspace.down_a[0].zero_()
        lora_workspace.down_b[0].zero_()
        lora_scratch = create_mlp_scratch(
            max_segments=batch,
            max_segment_len=seq_len,
            max_rank=rank,
            d_model=d_model,
            d_ffn=d_ffn,
            device=device,
            dtype=dtype,
        )

        x = torch.randn(batch, seq_len, d_model, dtype=dtype, device=device)
        lora_slot_ids = torch.zeros(batch, dtype=torch.int32, device=device)

        for _ in range(3):
            mlp(
                x,
                mlp_weights,
                lora_workspace=lora_workspace,
                lora_slot_ids=lora_slot_ids,
                lora_scratch=lora_scratch,
            )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                out_graph = mlp(
                    x,
                    mlp_weights,
                    lora_workspace=lora_workspace,
                    lora_slot_ids=lora_slot_ids,
                    lora_scratch=lora_scratch,
                )

        x.copy_(torch.randn_like(x))
        lora_slot_ids.copy_(torch.tensor([1, 2], dtype=torch.int32, device=device))
        g.replay()
        torch.cuda.synchronize()
        out_eager = mlp(
            x,
            mlp_weights,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
            lora_scratch=lora_scratch,
        )
        torch.testing.assert_close(out_graph, out_eager, rtol=1e-3, atol=1e-3)

        x.copy_(torch.randn_like(x))
        lora_slot_ids.copy_(torch.tensor([2, 1], dtype=torch.int32, device=device))
        g.replay()
        torch.cuda.synchronize()
        out_eager = mlp(
            x,
            mlp_weights,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
            lora_scratch=lora_scratch,
        )
        torch.testing.assert_close(out_graph, out_eager, rtol=1e-3, atol=1e-3)
