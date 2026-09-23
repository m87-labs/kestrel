"""Focused checks for invariant codec checkpoint transforms."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from kestrel.models.qwen3_tts.codec import (
    _EuclideanCodebook,
    _MLP,
    _SnakeBeta,
)
from kestrel.models.qwen3_tts.config import Qwen3TTSCodecConfig


@torch.inference_mode()
def test_codec_mlp_packing_preserves_checkpoint_projection():
    torch.manual_seed(0)
    model = _MLP(replace(Qwen3TTSCodecConfig(), hidden_size=32, intermediate_size=64)).bfloat16()
    inputs = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    expected = model.down_proj(F.silu(model.gate_proj(inputs)) * model.up_proj(inputs))
    model.prepare_for_inference()
    torch.testing.assert_close(model(inputs), expected)


def test_codec_static_values_are_materialized_without_drift() -> None:
    snake = _SnakeBeta(3)
    snake.alpha.data.copy_(torch.tensor((-0.5, 0.0, 0.5)))
    snake.beta.data.copy_(torch.tensor((0.25, -0.25, 0.0)))
    hidden = torch.randn(2, 3, 5)
    expected_snake = hidden + torch.sin(
        hidden * snake.alpha.exp().view(1, -1, 1)
    ).square() / (snake.beta.exp().view(1, -1, 1) + 1e-9)

    codebook = _EuclideanCodebook(3, 4).to(torch.bfloat16)
    codebook.cluster_usage.data.copy_(torch.tensor((0.3, 0.7, 1.3, 3.1)))
    codebook.embedding_sum.data.normal_()
    codes = torch.tensor(((0, 2, 3), (1, 0, 2)))
    expected_codes = F.embedding(
        codes,
        (
            codebook.embedding_sum.float()
            / codebook.cluster_usage.float().clamp(min=1e-5).unsqueeze(1)
        ).bfloat16(),
    )

    snake.prepare_for_inference()
    codebook.prepare_for_inference()

    torch.testing.assert_close(snake(hidden), expected_snake)
    torch.testing.assert_close(codebook.decode(codes), expected_codes, rtol=0, atol=0)
