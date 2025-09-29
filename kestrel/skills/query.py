"""Query skill leveraging the existing text generation flow."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


class QuerySkill(SkillSpec):
    """Default skill emitting plain text answers."""

    def __init__(self) -> None:
        super().__init__(name="query")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> Tensor:
        template = runtime.config.tokenizer.templates["query"]
        prefix: Sequence[int] = template["prefix"]
        suffix: Sequence[int] = template["suffix"]
        encoded = runtime.tokenizer.encode(prompt).ids if prompt else []
        ids = [*prefix, *encoded, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
    ) -> "QuerySkillState":
        return QuerySkillState(self, request)


class QuerySkillState(SkillState):
    """Skill state that buffers tokens and exposes plain text outputs."""

    def __init__(self, spec: SkillSpec, request: "GenerationRequest") -> None:
        super().__init__(spec, request)

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        self.append_token(step.token)
        return None

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        text = runtime.tokenizer.decode(list(self.tokens)) if self.tokens else ""
        return SkillFinalizeResult(text=text, tokens=list(self.tokens))
