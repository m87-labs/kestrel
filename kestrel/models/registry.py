"""Registry of model families supported by Kestrel.

A ``ModelSpec`` carries the small handful of facts the engine needs to
bootstrap a model: the HuggingFace download coordinates, the default
config dict, the checkpoint format tag (consumed by the weight loader),
the HF tokenizer hub id, and the runtime constructor.

New model families register themselves at import time from their
package's ``__init__.py`` (see ``kestrel/models/moondream/__init__.py``).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List

if TYPE_CHECKING:
    from kestrel.config import RuntimeConfig
    from kestrel.runtime import Runtime
    from kestrel.skills import SkillRegistry


@dataclass(frozen=True)
class ModelSpec:
    """Bootstrap metadata for a supported model."""

    name: str
    repo_id: str
    filename: str
    checkpoint_format: str
    default_config: Dict[str, Any]
    tokenizer_id: str
    # Constructor invoked as ``runtime(cfg, **kwargs)`` by the engine to
    # produce a concrete :class:`~kestrel.runtime.Runtime` for this
    # model. Kwargs (e.g. ``max_lora_rank``) are forwarded from the
    # engine's runtime-construction path.
    runtime: Callable[..., "Runtime"]
    # Factory for the model's capabilities. Returns the
    # :class:`~kestrel.skills.SkillRegistry` this model serves. Static
    # metadata — callable without building the (GPU) runtime — so the
    # engine can validate inputs and report ``tasks`` before startup.
    # Models with no autoregressive skills (e.g. single-pass) leave this
    # at the default empty registry and advertise tasks via the runtime.
    skills: Callable[[], "SkillRegistry"] = lambda: _empty_skill_registry()


def _empty_skill_registry() -> "SkillRegistry":
    """Default ``ModelSpec.skills`` factory: a model with no skills.

    Imported lazily so the registry module stays free of a hard
    dependency on the skill package.
    """
    from kestrel.skills import SkillRegistry

    return SkillRegistry([])


_REGISTRY: Dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    """Add a model to the registry."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model by name."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return _REGISTRY[name]


def known_models() -> List[str]:
    """Return the names of all registered models, sorted."""
    return sorted(_REGISTRY)


__all__ = [
    "ModelSpec",
    "get_spec",
    "known_models",
    "register",
]
