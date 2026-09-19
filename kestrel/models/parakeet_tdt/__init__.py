"""Lightweight parakeet_tdt registration and lazy public exports."""

from importlib import import_module
from kestrel.models.registry import register_lazy
from .metadata import MODEL_ID, REVISION

register_lazy([MODEL_ID], __name__ + ".registration")

__all__ = ["MODEL_ID", "REVISION", "ParakeetTdtRuntime", "load_parakeet_tdt"]

_LAZY_EXPORTS = {
    "ParakeetTdtRuntime": ".runtime",
    "load_parakeet_tdt": ".weights"
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
