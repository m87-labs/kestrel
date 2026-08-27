from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kestrel.models.gemma4.loader import load_weights
from kestrel.runtime.bounded_projection import PackedLinear


class _ToyShardedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = PackedLinear(
            2,
            (2, 2),
            source_names=("gate", "up"),
        )
        self.exact = nn.Linear(2, 2, bias=False)


def test_load_weights_streams_packed_projection_parts_across_shards(tmp_path) -> None:
    gate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    exact = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    save_file(
        {"gate.weight": gate, "exact.weight": exact},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"up.weight": up},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "gate.weight": "model-00001-of-00002.safetensors",
                    "exact.weight": "model-00001-of-00002.safetensors",
                    "up.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    model = _ToyShardedModel()

    load_weights(tmp_path, model)

    torch.testing.assert_close(model.packed.weight, torch.cat((gate, up), dim=0))
    torch.testing.assert_close(model.exact.weight, exact)


def test_load_weights_rejects_incomplete_packed_projection(tmp_path) -> None:
    save_file(
        {
            "gate.weight": torch.ones((2, 2)),
            "exact.weight": torch.ones((2, 2)),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(KeyError, match="missing source weights"):
        load_weights(tmp_path, _ToyShardedModel())


def test_load_weights_rejects_unexpected_checkpoint_tensor(tmp_path) -> None:
    save_file(
        {
            "gate.weight": torch.ones((2, 2)),
            "up.weight": torch.ones((2, 2)),
            "exact.weight": torch.ones((2, 2)),
            "unexpected.weight": torch.ones((1,)),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        load_weights(tmp_path, _ToyShardedModel())
