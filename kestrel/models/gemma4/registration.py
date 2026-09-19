"""Gemma 4 model support for the Kestrel inference engine."""

from kestrel.models.registry import ModelSpec, register_builtin

from .prompt_template import Gemma4PromptTemplate
from .runtime import Gemma4Runtime
from .skills import build_skill_registry


from . import _VARIANTS


for _repo_id in _VARIANTS:
    register_builtin(
        ModelSpec(
            name=_repo_id,
            repo_id=_repo_id,
            checkpoint_format="gemma4",
            default_config={},
            tokenizer_id=_repo_id,
            runtime=Gemma4Runtime,
            skills=build_skill_registry,
        )
    )


__all__ = ["Gemma4PromptTemplate", "Gemma4Runtime"]
