"""Point skill that extracts spatial coordinates from model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken, Token

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class PointSettings:
    """Sampling parameters supplied with a point invocation."""

    temperature: float
    top_p: float


@dataclass(slots=True)
class PointRequest:
    """Validated point payload aligned with the planned API."""

    object: str
    image: Optional[pyvips.Image]
    stream: bool
    settings: PointSettings


class PointSkill(SkillSpec):
    """Skill that returns model-indicated points as normalized coordinates."""

    def __init__(self) -> None:
        super().__init__(name="point")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Tensor:
        if not isinstance(request_context, PointRequest):
            raise ValueError("PointSkill.build_prompt_tokens requires a PointRequest")
        template = runtime.config.tokenizer.templates["point"]
        if template is None:
            raise ValueError("Model configuration does not include point templates")
        prefix: Sequence[int] = template["prefix"]
        suffix: Sequence[int] = template["suffix"]
        prompt = request_context.object
        object_tokens = runtime.tokenizer.encode(prompt).ids if prompt else []
        ids = [*prefix, *object_tokens, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: "PointRequest",
    ) -> "PointSkillState":
        if not isinstance(request_context, PointRequest):
            raise ValueError("PointSkill.create_state requires a PointRequest context")
        return PointSkillState(self, request, request_context)


class PointSkillState(SkillState):
    """Skill state that aggregates emitted points from coord tokens."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        point_request: PointRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = point_request
        self._awaiting_y = False

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        self.append_token(step.token)
        if isinstance(step.token, CoordToken):
            self._awaiting_y = not self._awaiting_y
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

        points = _extract_points(self.tokens)
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens),
            output={"points": points},
        )

    def allowed_token_ids(self, runtime: "MoondreamRuntime") -> Sequence[int]:
        tokenizer = runtime.config.tokenizer
        if self._awaiting_y:
            return [tokenizer.coord_id]
        return [tokenizer.coord_id, tokenizer.eos_id]


def _extract_points(tokens: Sequence[Token]) -> list[Dict[str, float]]:
    points: list[Dict[str, float]] = []
    pending_x: Optional[float] = None
    for token in tokens:
        if isinstance(token, CoordToken):
            if pending_x is None:
                pending_x = float(token.pos)
            else:
                points.append({"x": pending_x, "y": float(token.pos)})
                pending_x = None
        elif isinstance(token, SizeToken):
            if points:
                current = points[-1]
                current.setdefault("width", float(token.width))
                current.setdefault("height", float(token.height))
    return points



__all__ = [
    "PointRequest",
    "PointSettings",
    "PointSkill",
]
