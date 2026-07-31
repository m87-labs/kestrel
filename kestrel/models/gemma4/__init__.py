"""Gemma 4 model support for the Kestrel inference engine."""

from kestrel.models.registry import ModelSpec, register

from .prompt_template import Gemma4PromptTemplate
from .runtime import create_gemma4_runtime
from .skills import build_skill_registry


_VARIANTS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E2B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-31B-it",
    "google/gemma-4-31B",
]

for _repo_id in _VARIANTS:
    register(
        ModelSpec(
            name=_repo_id,
            repo_id=_repo_id,
            filename="config.json",
            checkpoint_format="gemma4",
            default_config={},
            tokenizer_id=_repo_id,
            runtime=create_gemma4_runtime,
            skills=build_skill_registry,
        )
    )


__all__ = ["Gemma4PromptTemplate", "create_gemma4_runtime"]
