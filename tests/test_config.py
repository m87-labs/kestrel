from pathlib import Path

import pytest

from kestrel.config import RuntimeConfig
import kestrel.config as config_mod


def test_runtime_config_rejects_invalid_service_name_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ensure_model_weights(model: str) -> Path:
        calls.append(model)
        raise AssertionError("ensure_model_weights should not be called")

    import kestrel.model_download as model_download

    monkeypatch.setattr(model_download, "ensure_model_weights", fake_ensure_model_weights)

    with pytest.raises(ValueError, match=r"service_name must match \[A-Za-z0-9_-\]\+"):
        RuntimeConfig(
            model_path=None,
            device="cpu",
            service_name="bad.name",
        )

    assert calls == []


def test_runtime_config_explains_torch_cuda_newer_than_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(config_mod.torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(config_mod.torch.version, "cuda", "13.0")
    monkeypatch.setattr(config_mod, "_torch_cuda_driver_version", lambda: "12.8")

    with pytest.raises(RuntimeError) as exc_info:
        RuntimeConfig(
            model_path="/unused",
            device="cuda",
        )

    msg = str(exc_info.value)
    assert "PyTorch is built for CUDA 13.0" in msg
    assert "driver reports CUDA 12.8 support" in msg
    assert "newer than the installed NVIDIA driver supports" in msg


def test_runtime_config_explains_cuda_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(config_mod.torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(config_mod.torch.version, "cuda", "12.8")
    monkeypatch.setattr(config_mod, "_torch_cuda_driver_version", lambda: None)

    with pytest.raises(RuntimeError) as exc_info:
        RuntimeConfig(
            model_path="/unused",
            device="cuda",
        )

    msg = str(exc_info.value)
    assert "PyTorch could not initialize CUDA" in msg
    assert "Installed PyTorch is built for CUDA 12.8" in msg
    assert "Verify with:" in msg


def test_runtime_config_allows_cuda_minor_compatibility_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(config_mod.torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(config_mod.torch.version, "cuda", "12.8")
    monkeypatch.setattr(config_mod, "_torch_cuda_driver_version", lambda: "12.4")

    with pytest.raises(RuntimeError) as exc_info:
        RuntimeConfig(
            model_path="/unused",
            device="cuda",
        )

    msg = str(exc_info.value)
    assert "Installed PyTorch is built for CUDA 12.8" in msg
    assert "driver reports CUDA 12.4 support" in msg
    assert "newer than the installed NVIDIA driver supports" not in msg
