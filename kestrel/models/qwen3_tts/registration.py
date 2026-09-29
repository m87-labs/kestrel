"""Resolve Qwen3-TTS implementation only when selected."""

from kestrel.models.registry import ModelSpec, register_builtin

from .config import SUPPORTED_CHECKPOINTS
from .runtime import Qwen3TTSRuntime
from .skill import build_skill_registry

for repo_id, revision in SUPPORTED_CHECKPOINTS.items():
    register_builtin(ModelSpec(
        name=repo_id,
        repo_id=repo_id,
        revision=revision,
        runtime=Qwen3TTSRuntime,
        skills=build_skill_registry,
    ))
