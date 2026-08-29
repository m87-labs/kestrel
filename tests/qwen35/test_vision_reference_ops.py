from __future__ import annotations

import torch
import torch.nn.functional as F

from kestrel.models.qwen35.qwen_config import Qwen3_5VisionConfig
from kestrel.models.qwen35.qwen_model import (
    Qwen3_5VisionAttention,
    Qwen3_5VisionPatchEmbed,
    _apply_vision_rotary_embedding,
)


def _config() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(
        depth=1,
        hidden_size=8,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
    )


def test_patch_embed_matches_reference_conv3d_accumulation() -> None:
    torch.manual_seed(7)
    module = Qwen3_5VisionPatchEmbed(_config())
    hidden_states = torch.randn(5, 3 * 2 * 2 * 2)

    expected = F.conv3d(
        hidden_states.view(-1, 3, 2, 2, 2),
        module.proj.weight.view(8, 3, 2, 2, 2),
        module.proj.bias,
        stride=(2, 2, 2),
    ).view(-1, 8)

    torch.testing.assert_close(module(hidden_states), expected)


def test_vision_rotary_matches_fp32_reference_for_bfloat16_inputs() -> None:
    torch.manual_seed(11)
    query = torch.randn(6, 2, 4).bfloat16()
    key = torch.randn(6, 2, 4).bfloat16()
    angles = torch.randn(6, 4)
    cos = angles.cos()
    sin = angles.sin()

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    expected_query = (
        query.float() * cos.unsqueeze(-2)
        + rotate_half(query.float()) * sin.unsqueeze(-2)
    ).bfloat16()
    expected_key = (
        key.float() * cos.unsqueeze(-2) + rotate_half(key.float()) * sin.unsqueeze(-2)
    ).bfloat16()

    actual_query, actual_key = _apply_vision_rotary_embedding(
        query,
        key,
        cos,
        sin,
    )

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


def test_vision_attention_keeps_packed_sequences_isolated() -> None:
    torch.manual_seed(13)
    module = Qwen3_5VisionAttention(_config())
    hidden_states = torch.randn(5, 8)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    position_embeddings = (torch.ones(5, 4), torch.zeros(5, 4))

    baseline = module(hidden_states, cu_seqlens, position_embeddings)
    changed = hidden_states.clone()
    changed[2:] = torch.randn_like(changed[2:])
    perturbed = module(changed, cu_seqlens, position_embeddings)

    torch.testing.assert_close(perturbed[:2], baseline[:2])
