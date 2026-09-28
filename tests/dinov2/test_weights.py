from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from kestrel.models.dinov2.config import Dinov2Config
from kestrel.models.dinov2.model import Dinov2Model
from kestrel.models.dinov2.weights import (
    CONFIG_FILENAME,
    PROCESSOR_CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    _copy_checkpoint_into_model,
    resolve_checkpoint_files,
)

from ._fixtures import MODEL_CONFIG


def _small_model() -> Dinov2Model:
    config = replace(
        Dinov2Config.from_dict(MODEL_CONFIG),
        hidden_size=16,
        image_size=8,
        mlp_ratio=2,
        num_attention_heads=4,
        num_hidden_layers=1,
        patch_size=2,
    )
    return Dinov2Model(config)


def _checkpoint_for(model: Dinov2Model) -> dict[str, torch.Tensor]:
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    state["embeddings.mask_token"] = torch.zeros(1, model.config.hidden_size)
    return state


def test_checkpoint_copy_is_strict_and_ignores_only_mask_token() -> None:
    torch.manual_seed(23)
    source = _small_model()
    checkpoint = _checkpoint_for(source)
    destination = _small_model()
    for parameter in destination.parameters():
        parameter.data.zero_()

    _copy_checkpoint_into_model(destination, checkpoint)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], tensor)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "mask"])
def test_checkpoint_copy_refuses_invalid_tensors(fault: str) -> None:
    model = _small_model()
    checkpoint = _checkpoint_for(model)
    if fault == "missing":
        del checkpoint["layernorm.weight"]
        pattern = "missing"
    elif fault == "unexpected":
        checkpoint["classifier.weight"] = torch.zeros(1)
        pattern = "unexpected"
    elif fault == "shape":
        checkpoint["layernorm.weight"] = torch.zeros(17)
        pattern = "checkpoint.*model"
    else:
        del checkpoint["embeddings.mask_token"]
        pattern = "training-only"
    with pytest.raises(RuntimeError, match=pattern):
        _copy_checkpoint_into_model(model, checkpoint)


def test_checkpoint_file_path_preserves_hub_symlink_name(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    (snapshot / CONFIG_FILENAME).write_text("{}")
    (snapshot / PROCESSOR_CONFIG_FILENAME).write_text("{}")
    blob = blobs / "0123456789abcdef"
    blob.write_bytes(b"checkpoint")
    weights = snapshot / WEIGHTS_FILENAME
    weights.symlink_to(blob)

    files = resolve_checkpoint_files(weights)
    assert files.root == snapshot.resolve()
    assert files.weights.name == WEIGHTS_FILENAME
    assert files.weights.resolve() == blob.resolve()
