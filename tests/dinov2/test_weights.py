"""Checkpoint tensor loading requires no model construction."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kestrel.models.dinov2.weights import (
    CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, WEIGHTS_FILENAME,
    load_dinov2, resolve_checkpoint_files,
)
from ._fixtures import MODEL_CONFIG, PROCESSOR_CONFIG


def _checkpoint(root, tensors):
    (root / CONFIG_FILENAME).write_text(json.dumps(MODEL_CONFIG))
    (root / PROCESSOR_CONFIG_FILENAME).write_text(json.dumps(PROCESSOR_CONFIG))
    save_file(tensors, str(root / WEIGHTS_FILENAME))


def test_direct_loader_keeps_fp32_storage_and_does_not_construct_a_model(tmp_path, monkeypatch):
    state = {"embeddings.cls_token": torch.randn(1, 1, 384),
             "layernorm.weight": torch.ones(384),
             "embeddings.mask_token": torch.zeros(1, 384)}
    _checkpoint(tmp_path, state)
    monkeypatch.setattr("safetensors.torch.load_file", lambda *args, **kwargs: state)

    def no_model(*args, **kwargs):
        raise AssertionError("checkpoint loading constructed a model")

    monkeypatch.setattr(torch.nn.Module, "__init__", no_model)
    loaded = load_dinov2(tmp_path)
    assert loaded.state_dict["embeddings.cls_token"] is state["embeddings.cls_token"]
    assert loaded.state_dict["layernorm.weight"] is state["layernorm.weight"]
    assert "embeddings.mask_token" not in loaded.state_dict
    assert loaded.model_config.hidden_size == 384


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_checkpoint_tensors_load_on_cpu_as_fp32(tmp_path, dtype):
    expected = torch.randn(384).to(dtype)
    _checkpoint(tmp_path, {"layernorm.weight": expected})
    loaded = load_dinov2(tmp_path)
    assert loaded.state_dict["layernorm.weight"].dtype is torch.float32
    assert loaded.state_dict["layernorm.weight"].device.type == "cpu"
    torch.testing.assert_close(loaded.state_dict["layernorm.weight"], expected.float())


def test_nonfloating_checkpoint_tensor_is_rejected(tmp_path):
    _checkpoint(tmp_path, {"layernorm.weight": torch.ones(384, dtype=torch.int64)})
    with pytest.raises(ValueError, match="must be floating point"):
        load_dinov2(tmp_path)


def test_pinned_hub_download_requests_only_the_checkpoint_files(tmp_path, monkeypatch):
    from kestrel.models.dinov2.metadata import DEFAULT_DINOV2_REPO_ID, DEFAULT_DINOV2_REVISION

    _checkpoint(tmp_path, {"layernorm.weight": torch.ones(384)})
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    load_dinov2()
    assert calls == [(DEFAULT_DINOV2_REPO_ID, dict(
        revision=DEFAULT_DINOV2_REVISION,
        allow_patterns=[CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, WEIGHTS_FILENAME],
        local_files_only=False,
    ))]


def test_checkpoint_file_path_preserves_hub_symlink_name(tmp_path: Path):
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


def test_empty_inference_checkpoint_is_rejected(tmp_path):
    _checkpoint(tmp_path, {"embeddings.mask_token": torch.zeros(1, 384)})
    with pytest.raises(ValueError, match="no inference tensors"):
        load_dinov2(tmp_path)
