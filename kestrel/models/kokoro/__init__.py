"""Inference-only direct Kokoro-82M support."""

from importlib import import_module

from kestrel.models.registry import register_lazy

from .metadata import (
    DEFAULT_KOKORO_MODEL,
    DEFAULT_KOKORO_REPO_ID,
    DEFAULT_KOKORO_REVISION,
)


register_lazy([DEFAULT_KOKORO_MODEL], __name__ + ".registration")


__all__ = [
    "DEFAULT_KOKORO_MODEL",
    "DEFAULT_KOKORO_REPO_ID",
    "DEFAULT_KOKORO_REVISION",
    "KokoroRuntime",
    "create_kokoro_runtime",
]


def __getattr__(name):
    if name not in ("KokoroRuntime", "create_kokoro_runtime"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".runtime", __name__), name)
    globals()[name] = value
    return value
