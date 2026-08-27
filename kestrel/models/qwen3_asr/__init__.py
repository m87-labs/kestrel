"""Qwen3-ASR 0.6B and 1.7B support for Kestrel."""

from kestrel.models.registry import ModelSpec, register

from .alignment import (
    MODEL_ID as _ALIGNER_MODEL_ID,
    REVISION as _ALIGNER_REVISION,
    Qwen3ForcedAlignerRuntime,
)
from .runtime import Qwen3AsrRuntime
from .weights import QWEN3_ASR_MODELS, load_qwen3_asr


def _build_skill_registry():
    from .skill import build_skill_registry

    return build_skill_registry()


for _model_name, _revision in QWEN3_ASR_MODELS.items():
    register(
        ModelSpec(
            name=_model_name,
            repo_id=_model_name,
            revision=_revision,
            runtime=Qwen3AsrRuntime,
            skills=_build_skill_registry,
        )
    )

register(
    ModelSpec(
        name=_ALIGNER_MODEL_ID,
        repo_id=_ALIGNER_MODEL_ID,
        revision=_ALIGNER_REVISION,
        runtime=Qwen3ForcedAlignerRuntime,
    )
)

__all__ = [
    "QWEN3_ASR_MODELS",
    "Qwen3AsrRuntime",
    "Qwen3ForcedAlignerRuntime",
    "load_qwen3_asr",
]
