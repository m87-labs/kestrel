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
from kestrel.models.registry import ModelSpec
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
