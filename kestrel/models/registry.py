"""Registry of model families supported by Kestrel.

A ``ModelSpec`` carries the facts the engine needs to bootstrap a model.
Only ``name``, the ``runtime`` constructor, and the ``skills`` factory are
universal; the rest are autoregressive/HuggingFace bootstrap hints (download
coordinates, checkpoint-format tag, tokenizer id, default config) consumed by
a specific runtime family. A single-pass model whose runtime factory owns its
own loading leaves them unset.

Built-in model families register themselves at import time from their
package's ``__init__.py`` (see ``kestrel/models/moondream/__init__.py``).
Separately installed families expose a zero-argument registrar through the
``kestrel.models`` entry-point group. The entry-point name is the model name,
so looking up an external model imports only the package that owns it.
"""

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
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
    revision: Optional[str] = None
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
_EXTERNAL_MODEL_GROUP = "kestrel.models"


def _external_model_entry_points(name: str | None = None) -> tuple[EntryPoint, ...]:
    """Return installed model registrars in deterministic order.

    A named lookup asks importlib.metadata for that name directly. This keeps
    startup discovery scoped to configured models; the unnarrowed form exists
    only for the explicit ``known_models()`` introspection API.
    """

    selection = {"group": _EXTERNAL_MODEL_GROUP}
    if name is not None:
        selection["name"] = name
    return tuple(
        sorted(
            entry_points(**selection),
            key=lambda entry_point: (entry_point.name, entry_point.value),
        )
    )


def _load_external_model(name: str) -> None:
    """Load the installed registrar whose entry-point name is ``name``."""

    if name in _REGISTRY:
        return
    matches = [ep for ep in _external_model_entry_points(name) if ep.name == name]
    if len(matches) > 1:
        providers = ", ".join(entry_point.value for entry_point in matches)
        raise RuntimeError(
            f"Multiple model entry points are installed for {name!r}: {providers}"
        )
    if not matches:
        return

    entry_point = matches[0]
    registrar = entry_point.load()
    if not callable(registrar):
        raise TypeError(
            f"Model entry point {entry_point.name!r} must resolve to a "
            "zero-argument registrar"
        )
    registrar()
    if name not in _REGISTRY:
        raise RuntimeError(
            f"Model entry point {name!r} did not register a model with the same name"
        )


def register(spec: ModelSpec) -> None:
    """Add a model to the registry."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model by name."""
    if name not in _REGISTRY:
        _load_external_model(name)
    if name not in _REGISTRY:
        known = ", ".join(known_models())
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return _REGISTRY[name]


def known_models() -> List[str]:
    """Return built-in and installed external model names, sorted."""

    external = (entry_point.name for entry_point in _external_model_entry_points())
    return sorted(set(_REGISTRY).union(external))


__all__ = [
    "ModelSpec",
    "get_spec",
    "known_models",
    "register",
]
