"""NVIDIA Parakeet TDT 0.6B v3 support for Kestrel."""

from kestrel.models.registry import ModelSpec, register

from .runtime import ParakeetTdtRuntime
from .weights import MODEL_ID, REVISION, load_parakeet_tdt


def _build_orchestrators():
    from .longform import ParakeetLongFormOrchestrator

    return {"transcribe": ParakeetLongFormOrchestrator()}


register(
    ModelSpec(
        name=MODEL_ID,
        repo_id=MODEL_ID,
        revision=REVISION,
        runtime=ParakeetTdtRuntime,
        orchestrators=_build_orchestrators,
    )
)

__all__ = ["MODEL_ID", "REVISION", "ParakeetTdtRuntime", "load_parakeet_tdt"]
