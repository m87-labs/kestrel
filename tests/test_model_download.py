from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub

import kestrel.models
from kestrel.model_download import ensure_model_weights, probe_supported_model_configs
from kestrel.models.gemma4 import loader as gemma_loader
from kestrel.models.qwen35 import qwen_loader


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


def test_gemma_snapshot_uses_declared_revision(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(gemma_loader, "snapshot_download", snapshot_download)

    assert gemma_loader._snapshot(
        "owner/gemma", revision="immutable-commit"
    ) == tmp_path
    assert calls == [
        (
            "owner/gemma",
            {
                "allow_patterns": [
                    "config.json",
                    "*.safetensors",
                    "model.safetensors.index.json",
                ],
                "revision": "immutable-commit",
            },
        )
    ]


def test_gemma_snapshot_keeps_local_source_local(monkeypatch, tmp_path) -> None:
    def unexpected_download(*args: object, **kwargs: object) -> str:
        raise AssertionError("local Gemma source must not use the Hub")

    monkeypatch.setattr(gemma_loader, "snapshot_download", unexpected_download)

    assert gemma_loader._snapshot(
        tmp_path, revision="immutable-commit"
    ) == tmp_path


def test_qwen_shard_download_uses_declared_revision(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def hf_hub_download(
        repo_id: str, filename: str, **kwargs: object
    ) -> str:
        calls.append((repo_id, filename, kwargs))
        return "/cache/model.safetensors"

    monkeypatch.setattr(qwen_loader, "hf_hub_download", hf_hub_download)

    assert qwen_loader._resolve_checkpoint_file(
        "owner/qwen",
        "model-00001-of-00002.safetensors",
        revision="immutable-commit",
    ) == "/cache/model.safetensors"
    assert calls == [
        (
            "owner/qwen",
            "model-00001-of-00002.safetensors",
            {"revision": "immutable-commit"},
        )
    ]


def test_qwen_checkpoint_keeps_local_source_local(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text("{}", encoding="utf-8")

    def unexpected_download(*args: object, **kwargs: object) -> str:
        raise AssertionError("local Qwen source must not use the Hub")

    monkeypatch.setattr(qwen_loader, "hf_hub_download", unexpected_download)

    assert qwen_loader._resolve_checkpoint_file(
        tmp_path,
        "config.json",
        revision="immutable-commit",
    ) == str(checkpoint)
