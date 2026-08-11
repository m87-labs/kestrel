from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub

import kestrel.models
from kestrel.model_download import ensure_model_weights, probe_supported_model_configs


def test_config_probe_only_resolves_configured_model_ids(monkeypatch) -> None:
    resolved: list[str] = []
    downloads: list[tuple[str, str, dict[str, object]]] = []
    specs = {
        "configured-hf": SimpleNamespace(
            repo_id="owner/configured-hf", revision="pinned-revision"
        ),
        "configured-local": SimpleNamespace(repo_id=None, revision=None),
    }

    def get_spec(model_id: str):
        resolved.append(model_id)
        return specs[model_id]

    def hf_hub_download(repo_id: str, *, filename: str, **kwargs: object) -> None:
        downloads.append((repo_id, filename, kwargs))

    monkeypatch.setattr(kestrel.models, "get_spec", get_spec)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)

    probe_supported_model_configs(
        ("configured-hf", "configured-local", "configured-hf")
    )

    assert resolved == ["configured-hf", "configured-local"]
    assert downloads == [
        (
            "owner/configured-hf",
            "config.json",
            {"etag_timeout": 2, "revision": "pinned-revision"},
        )
    ]


def test_config_probe_without_ids_keeps_legacy_all_model_behavior(monkeypatch) -> None:
    resolved: list[str] = []

    monkeypatch.setattr(kestrel.models, "known_models", lambda: ["registered"])
    monkeypatch.setattr(
        kestrel.models,
        "get_spec",
        lambda model_id: (
            resolved.append(model_id)
            or SimpleNamespace(repo_id=None, revision=None)
        ),
    )

    probe_supported_model_configs()

    assert resolved == ["registered"]


def test_weight_download_uses_modelspec_revision(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []
    spec = SimpleNamespace(
        repo_id="owner/model",
        filename="model.safetensors",
        revision="pinned-revision",
    )
    downloaded = tmp_path / "model.safetensors"

    monkeypatch.setattr(kestrel.models, "get_spec", lambda _model_id: spec)

    def hf_hub_download(
        repo_id: str, *, filename: str, **kwargs: object
    ) -> str:
        calls.append((repo_id, filename, kwargs))
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)

    assert ensure_model_weights("configured-model") == downloaded
    assert calls == [
        (
            "owner/model",
            "model.safetensors",
            {"revision": "pinned-revision"},
        )
    ]
