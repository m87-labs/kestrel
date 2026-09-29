"""Registry of model families supported by Kestrel.

A ``ModelSpec`` carries the facts the engine needs to bootstrap a model.
Only ``name``, the ``runtime`` constructor, and the ``skills`` factory are
universal; the rest are autoregressive/HuggingFace bootstrap hints (download
coordinates, checkpoint-format tag, tokenizer id, default config) consumed by
a specific runtime family. A single-pass model whose runtime factory owns its
own loading leaves them unset.

Model packages advertise names at import time; their registration modules
resolve concrete specs only when selected. Application specs can still be
registered directly.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

if TYPE_CHECKING:
    from kestrel.runtime import Runtime
    from kestrel.skills import CapabilityOrchestrator, SkillRegistry


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
    # Consumed by an autoregressive runtime family's weight loader and
    # tokenizer. A single-pass spec whose factory owns loading omits them;
    # the kernel never reads these — it only calls ``runtime``.
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    checkpoint_format: Optional[str] = None
    tokenizer_id: Optional[str] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    # Immutable revision for artifacts hosted by ``repo_id``. Runtimes also
    # apply it to the tokenizer when ``tokenizer_id`` names that same repo.
    revision: Optional[str] = None
    # Optional model-owned composition around ordinary capability requests.
    # This is execution-shape independent: each leaf still runs through the
    # model's normal autoregressive or single-pass lane.
    orchestrators: Callable[
        [], "Mapping[str, CapabilityOrchestrator]"
    ] = lambda: _empty_orchestrators()


def _empty_skill_registry() -> "SkillRegistry":
    """Default ``ModelSpec.skills`` factory: a model with no skills.

    Imported lazily so the registry module stays free of a hard
    dependency on the skill package.
    """
    from kestrel.skills import SkillRegistry

    return SkillRegistry([])


def _empty_orchestrators() -> "Mapping[str, CapabilityOrchestrator]":
    return {}


_REGISTRY: Dict[str, ModelSpec] = {}
_LAZY_REGISTRY: Dict[str, str] = {}


def register_lazy(names: list[str], module: str) -> None:
    """Advertise a family's names without importing its implementation."""
    for name in names:
        _LAZY_REGISTRY[name] = module


def register(spec: ModelSpec) -> None:
    """Add a model to the registry."""
    _REGISTRY[spec.name] = spec


def register_builtin(spec: ModelSpec) -> None:
    """Deferred defaults must not replace an explicit application registration."""
    _REGISTRY.setdefault(spec.name, spec)


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model by name."""
    if name not in _REGISTRY and name in _LAZY_REGISTRY:
        import_module(_LAZY_REGISTRY[name])
    if name not in _REGISTRY:
        known = ", ".join(known_models())
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return _REGISTRY[name]


def known_models() -> List[str]:
    """Return the names of all registered models, sorted."""
    return sorted(_REGISTRY.keys() | _LAZY_REGISTRY.keys())


__all__ = [
    "ModelSpec",
    "get_spec",
    "known_models",
    "register",
]
