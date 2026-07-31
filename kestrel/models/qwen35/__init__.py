"""Qwen 3.5 model support for the Kestrel inference engine."""

from kestrel.models.registry import ModelSpec, register

from .prompt_template import Qwen35PromptTemplate
from .runtime import Qwen35Runtime
from .skills import build_skill_registry


_VARIANTS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-35B-A3B-FP8",
]

for _repo_id in _VARIANTS:
    register(
        ModelSpec(
            name=_repo_id,
            repo_id=_repo_id,
            checkpoint_format="qwen3_5",
            default_config={},
            tokenizer_id=_repo_id,
            runtime=Qwen35Runtime,
            skills=build_skill_registry,
        )
    )


__all__ = ["Qwen35PromptTemplate", "Qwen35Runtime"]
