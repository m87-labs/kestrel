from pathlib import Path

import pytest

from kestrel.config import RuntimeConfig


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
