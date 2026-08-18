from __future__ import annotations

import math

import torch
from safetensors.torch import save_file

import pytest

from kestrel.models.whisper.weights import (
    WhisperCheckpointError,
    expected_whisper_checkpoint_shapes,
    load_whisper_safetensors,
    named_whisper_tensors,
    validate_whisper_weight_tree,
)


def test_shipping_checkpoint_manifest_matches_pinned_header() -> None:
    from kestrel.models.whisper.config import WhisperTurboConfig

    shapes = expected_whisper_checkpoint_shapes(WhisperTurboConfig())
    assert len(shapes) == 587
    assert sum(math.prod(shape) for shape in shapes.values()) == 808_878_080
    assert "proj_out.weight" not in shapes


def _save(path, tensors) -> None:
    save_file(tensors, path, metadata={"format": "pt"})


def test_tiny_transformers_state_owns_every_expected_tensor(
    tiny_whisper_config,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    expected = expected_whisper_checkpoint_shapes(tiny_whisper_config)
    assert set(tiny_checkpoint_tensors) == set(expected)
    assert len(expected) == 89

    path = tmp_path / "model.safetensors"
    _save(path, tiny_checkpoint_tensors)
    weights = load_whisper_safetensors(
        path,
        tiny_whisper_config,
        checkpoint_dtype=torch.float32,
    )
    named = named_whisper_tensors(weights)
    assert set(named) == set(expected)
    assert weights.decoder.output_projection is weights.decoder.token_embedding
    assert validate_whisper_weight_tree(weights, tiny_whisper_config) == (
        torch.device("cpu"),
        torch.float32,
    )
    converted = weights.to(dtype=torch.bfloat16)
    assert validate_whisper_weight_tree(converted, tiny_whisper_config) == (
        torch.device("cpu"),
        torch.bfloat16,
    )
    assert converted.decoder.output_projection is converted.decoder.token_embedding


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "dtype"])
def test_loader_rejects_checkpoint_drift(
    failure,
    tiny_whisper_config,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    tensors = dict(tiny_checkpoint_tensors)
    if failure == "missing":
        tensors.pop("model.encoder.conv1.bias")
    elif failure == "unexpected":
        tensors["training_only.weight"] = torch.zeros(1)
    elif failure == "shape":
        tensors["model.encoder.conv1.bias"] = tensors["model.encoder.conv1.bias"][
            :-1
        ].contiguous()
    else:
        tensors["model.encoder.conv1.bias"] = tensors["model.encoder.conv1.bias"].to(
            torch.float64
        )
    path = tmp_path / f"{failure}.safetensors"
    _save(path, tensors)

    with pytest.raises(
        WhisperCheckpointError,
        match=failure if failure in {"missing", "unexpected"} else "conv1.bias",
    ):
        load_whisper_safetensors(
            path,
            tiny_whisper_config,
            checkpoint_dtype=torch.float32,
        )


def test_optional_projection_must_be_exactly_tied(
    tiny_whisper_config,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    tied = dict(tiny_checkpoint_tensors)
    tied["proj_out.weight"] = tied["model.decoder.embed_tokens.weight"].clone()
    tied_path = tmp_path / "tied.safetensors"
    _save(tied_path, tied)
    weights = load_whisper_safetensors(
        tied_path,
        tiny_whisper_config,
        checkpoint_dtype=torch.float32,
    )
    assert weights.decoder.output_projection is weights.decoder.token_embedding

    untied = dict(tied)
    untied["proj_out.weight"] = tied["proj_out.weight"].clone()
    untied["proj_out.weight"][0, 0] += 1.0
    untied_path = tmp_path / "untied.safetensors"
    _save(untied_path, untied)
    with pytest.raises(WhisperCheckpointError, match="exactly tied"):
        load_whisper_safetensors(
            untied_path,
            tiny_whisper_config,
            checkpoint_dtype=torch.float32,
        )


def test_loader_requires_pytorch_safetensors_metadata(
    tiny_whisper_config,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    path = tmp_path / "no-format.safetensors"
    save_file(tiny_checkpoint_tensors, path)
    with pytest.raises(WhisperCheckpointError, match="format='pt'"):
        load_whisper_safetensors(
            path,
            tiny_whisper_config,
            checkpoint_dtype=torch.float32,
        )
