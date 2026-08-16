from __future__ import annotations

from types import SimpleNamespace

import torch

from kestrel.models.gemma4.model import (
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
)


def test_patch_embedder_applies_reference_pixel_scaling() -> None:
    config = SimpleNamespace(
        patch_size=1,
        hidden_size=3,
        position_embedding_size=2,
    )
    embedder = Gemma4VisionPatchEmbedder(config)
    with torch.no_grad():
        embedder.input_proj.weight.copy_(torch.eye(3))
        embedder.position_embedding_table.zero_()

    output = embedder(
        torch.zeros((1, 1, 3)),
        torch.zeros((1, 1, 2), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.bool),
    )

    torch.testing.assert_close(output, -torch.ones_like(output))


def test_vision_pooler_keeps_fp32_until_standardization() -> None:
    pooler = Gemma4VisionPooler(SimpleNamespace(hidden_size=4))
    hidden_states = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0]]],
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])

    output, mask = pooler(
        hidden_states,
        positions,
        torch.zeros((1, 4), dtype=torch.bool),
        output_length=1,
    )

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.tensor([[[5.0]]]))
    assert mask.tolist() == [[True]]
