"""The ternary student runs on CPU/MPS only: CUDA requests are redirected."""

import torch

from kestrel.models.parakeet_tdt.weights import (
    TERNARY_MODEL_ID,
    is_ternary_checkpoint,
    ternary_runtime_device,
)


def test_cuda_request_is_redirected(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert ternary_runtime_device(torch.device("cuda", 0)) == torch.device("cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert ternary_runtime_device(torch.device("cuda", 0)) == torch.device("mps")
    assert ternary_runtime_device(torch.device("cpu")) == torch.device("cpu")
    assert ternary_runtime_device(torch.device("mps")) == torch.device("mps")


def test_ternary_checkpoint_detection(tmp_path) -> None:
    assert is_ternary_checkpoint(None, TERNARY_MODEL_ID)
    assert not is_ternary_checkpoint(None, "nvidia/parakeet-tdt-0.6b-v3")
    assert not is_ternary_checkpoint(tmp_path)
    (tmp_path / "ternary.json").write_text("{}")
    assert is_ternary_checkpoint(tmp_path)
