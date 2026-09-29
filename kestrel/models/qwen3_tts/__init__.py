"""Qwen3-TTS CustomVoice support for Kestrel."""

from kestrel.models.registry import ModelSpec, register

from .config import SUPPORTED_CHECKPOINTS
from .runtime import Qwen3TTSRuntime


def _build_skill_registry():
    from .skill import build_skill_registry

    return build_skill_registry()


for repo_id, revision in SUPPORTED_CHECKPOINTS.items():
    register(
        ModelSpec(
            name=repo_id,
            repo_id=repo_id,
            revision=revision,
            runtime=Qwen3TTSRuntime,
            skills=_build_skill_registry,
        )
    )


__all__ = ["SUPPORTED_CHECKPOINTS", "Qwen3TTSRuntime"]
