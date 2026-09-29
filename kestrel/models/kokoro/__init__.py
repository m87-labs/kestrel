"""Inference-only direct Kokoro-82M support."""

from kestrel.models.registry import ModelSpec, register

from .runtime import KokoroRuntime, create_kokoro_runtime
from .weights import (
    DEFAULT_KOKORO_MODEL,
    DEFAULT_KOKORO_REPO_ID,
    DEFAULT_KOKORO_REVISION,
)


def _build_orchestrators():
    from .orchestrator import build_orchestrators

    return build_orchestrators()


register(
    ModelSpec(
        name=DEFAULT_KOKORO_MODEL,
        repo_id=DEFAULT_KOKORO_REPO_ID,
        revision=DEFAULT_KOKORO_REVISION,
        runtime=create_kokoro_runtime,
        orchestrators=_build_orchestrators,
    )
)


__all__ = [
    "DEFAULT_KOKORO_MODEL",
    "DEFAULT_KOKORO_REPO_ID",
    "DEFAULT_KOKORO_REVISION",
    "KokoroRuntime",
    "create_kokoro_runtime",
]
