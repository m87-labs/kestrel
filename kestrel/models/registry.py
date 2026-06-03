"""Registry of model families supported by Kestrel.

A ``ModelSpec`` carries the facts the engine needs to bootstrap a model.
Only ``name``, the ``runtime`` constructor, and the ``skills`` factory are
universal; the rest are autoregressive/HuggingFace bootstrap hints (download
coordinates, checkpoint-format tag, tokenizer id, default config) consumed by
a specific runtime family. A single-pass model whose runtime factory owns its
own loading leaves them unset.

New model families register themselves at import time from their
package's ``__init__.py`` (see ``kestrel/models/moondream/__init__.py``).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from kestrel.config import RuntimeConfig
    from kestrel.runtime import Runtime
    from kestrel.skills import SkillRegistry


@dataclass(frozen=True)
class ModelSpec:
    """Bootstrap metadata for a supported model."""

    name: str
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

    # --- Autoregressive / HuggingFace bootstrap hints (optional) ---
    # Consumed by a specific runtime family's weight loader + tokenizer
    # (Moondream's). A single-pass spec whose factory owns loading omits
    # them; the kernel never reads these — it only calls ``runtime``.
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    checkpoint_format: Optional[str] = None
    tokenizer_id: Optional[str] = None
    default_config: Dict[str, Any] = field(default_factory=dict)


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
