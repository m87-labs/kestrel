"""Qwen 3.5/3.6 hybrid model support for the Kestrel inference engine."""

from kestrel.models.registry import ModelSpec, register_builtin

from .prompt_template import Qwen35PromptTemplate
from .runtime import Qwen35Runtime
from .skills import build_skill_registry


from . import _VARIANTS


for _repo_id in _VARIANTS:
    register_builtin(
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
