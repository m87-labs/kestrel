"""NVIDIA Parakeet TDT 0.6B v3 support for Kestrel."""

from kestrel.models.registry import ModelSpec, register

from .runtime import ParakeetTdtRuntime
from .weights import MODEL_ID, REVISION, TERNARY_MODEL_ID, TERNARY_REVISION, load_parakeet_tdt


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

# The 2-bit (ternary) student: same runtime, same contract, weights at ``repo_id`` (private for now);
# ``RuntimeConfig(model=TERNARY_MODEL_ID)`` downloads them, ``model_path=<export dir>`` loads a local export.
register(
    ModelSpec(
        name=TERNARY_MODEL_ID,
        repo_id=TERNARY_MODEL_ID,
        revision=TERNARY_REVISION,
        runtime=ParakeetTdtRuntime,
        orchestrators=_build_orchestrators,
    )
)

__all__ = ["MODEL_ID", "REVISION", "TERNARY_MODEL_ID", "ParakeetTdtRuntime", "load_parakeet_tdt"]
