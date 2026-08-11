"""ModelSpec capability metadata + the SkillRegistry it returns.

Skills are declared on the ModelSpec so the engine can resolve a model's
capabilities without building its (GPU) runtime. A model with no
autoregressive skills must be expressible — the default ``skills`` factory
returns an empty registry rather than raising.
"""

from __future__ import annotations

import pytest

from kestrel.models import get_spec, known_models
from kestrel.models.moondream import (
    DEFAULT_MOONDREAM3_CONFIG,
    MoondreamRuntime,
    build_skill_registry,
)
from kestrel.models.registry import ModelSpec, _REGISTRY
import kestrel.models.registry as registry_module
from kestrel.skills import SkillRegistry


def _spec(**overrides: object) -> ModelSpec:
    base = dict(
        name="m",
        repo_id="r",
        filename="f",
        checkpoint_format="c",
        default_config={},
        tokenizer_id="t",
        runtime=lambda *a, **k: None,
    )
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


def test_empty_skill_registry_is_allowed() -> None:
    """A model can have zero skills (single-pass); the registry must not
    reject an empty iterable."""
    registry = SkillRegistry([])
    assert registry.names() == ()
    with pytest.raises(ValueError, match="Unknown skill"):
        registry.resolve("anything")


def test_modelspec_default_skills_is_empty_not_raising() -> None:
    """The default ``ModelSpec.skills`` factory (for skill-less models) must
    return an empty registry, not raise — pre-start capability paths call
    ``get_spec(...).skills()`` before the runtime can advertise its tasks."""
    spec = _spec()  # skills omitted -> default factory
    registry = spec.skills()
    assert isinstance(registry, SkillRegistry)
    assert registry.names() == ()


def test_modelspec_skills_factory_is_honored() -> None:
    sentinel = SkillRegistry([])
    spec = _spec(skills=lambda: sentinel)
    assert spec.skills() is sentinel


def test_moondream31_a2b_uses_md3_runtime_metadata() -> None:
    spec = get_spec("moondream3.1-9B-A2B")

    assert "moondream3.1-9B-A2B" in known_models()
    assert spec.repo_id == "moondream/moondream3.1-9B-A2B"
    assert spec.filename == "model.safetensors"
    assert spec.checkpoint_format == "md3"
    assert spec.default_config == DEFAULT_MOONDREAM3_CONFIG
    assert spec.tokenizer_id == "moondream/starmie-v1"
    assert spec.runtime is MoondreamRuntime
    assert spec.skills is build_skill_registry


def test_external_entry_point_discovery_selects_only_requested_name(
    monkeypatch,
) -> None:
    selections: list[dict[str, str]] = []

    def entry_points(**selection: str):
        selections.append(selection)
        return ()

    monkeypatch.setattr(registry_module, "entry_points", entry_points)
    assert registry_module._external_model_entry_points("configured-model") == ()
    assert selections == [
        {"group": "kestrel.models", "name": "configured-model"}
    ]


def test_external_model_is_discovered_by_matching_entry_point(monkeypatch) -> None:
    calls: list[str] = []
    external_spec = _spec(name="external-model")

    class EntryPoint:
        name = "external-model"
        value = "external_package:register_models"

        def load(self):
            calls.append("load")

            def registrar() -> None:
                calls.append("register")
                registry_module.register(external_spec)

            return registrar

    monkeypatch.setattr(
        registry_module,
        "_external_model_entry_points",
        lambda name=None: (EntryPoint(),),
    )
    try:
        assert "external-model" in known_models()
        assert get_spec("external-model") is external_spec
        assert calls == ["load", "register"]
        # Registry lookup is idempotent and does not reload the package.
        assert get_spec("external-model") is external_spec
        assert calls == ["load", "register"]
    finally:
        _REGISTRY.pop("external-model", None)


def test_external_entry_point_must_register_its_own_name(monkeypatch) -> None:
    wrong_spec = _spec(name="wrong-model")

    class EntryPoint:
        name = "broken-model"
        value = "broken_package:register_models"

        @staticmethod
        def load():
            return lambda: registry_module.register(wrong_spec)

    monkeypatch.setattr(
        registry_module,
        "_external_model_entry_points",
        lambda name=None: (EntryPoint(),),
    )
    try:
        with pytest.raises(RuntimeError, match="did not register"):
            get_spec("broken-model")
    finally:
        _REGISTRY.pop("wrong-model", None)


def test_duplicate_external_model_providers_fail_explicitly(monkeypatch) -> None:
    class EntryPoint:
        name = "duplicate-model"

        def __init__(self, value: str) -> None:
            self.value = value

        @staticmethod
        def load():  # pragma: no cover - lookup must fail before import
            raise AssertionError("duplicate providers must not be imported")

    monkeypatch.setattr(
        registry_module,
        "_external_model_entry_points",
        lambda name=None: (
            EntryPoint("provider_a:register_models"),
            EntryPoint("provider_b:register_models"),
        ),
    )

    with pytest.raises(RuntimeError, match="Multiple model entry points") as exc_info:
        get_spec("duplicate-model")

    message = str(exc_info.value)
    assert "provider_a:register_models" in message
    assert "provider_b:register_models" in message
