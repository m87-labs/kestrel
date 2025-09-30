"""Query skill leveraging the existing text generation flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import TextToken

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class QuerySettings:
    """Sampling parameters supplied with a query invocation."""

    temperature: float
    top_p: float


@dataclass(slots=True)
class QueryRequest:
    """Validated query payload aligned with the fal_inference API."""

    question: str
    image: Optional[pyvips.Image]
    reasoning: bool
    stream: bool
    settings: QuerySettings


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
        *,
        context: Optional[object] = None,
    ) -> "QuerySkillState":
        if context is None:
            context = QueryRequest(
                question=request.prompt,
                image=request.image,
                reasoning=False,
                stream=False,
                settings=QuerySettings(
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
            )
        if not isinstance(context, QueryRequest):
            raise ValueError("QuerySkill.create_state requires a QueryRequest context")
        return QuerySkillState(self, request, context)


class QuerySkillState(SkillState):
    """Skill state that buffers tokens and exposes plain text outputs."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        query_request: QueryRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = query_request

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
        text_tokens = [
            token.token_id for token in self.tokens if isinstance(token, TextToken)
        ]
        text = runtime.tokenizer.decode(text_tokens) if text_tokens else ""
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens),
            extras={"answer": text},
        )
