"""Detect skill that extracts bounding boxes from model outputs."""


from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pyvips
import torch
import numpy as np
from torch import Tensor

from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken, Token

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class DetectSettings:
    """Sampling parameters supplied with a detect invocation."""

    temperature: float
    top_p: float


@dataclass(slots=True)
class DetectRequest:
    """Detect payload used internally by the scheduler."""

    object: str
    image: Optional[pyvips.Image | np.ndarray]
    stream: bool
    settings: DetectSettings
    max_objects: int


class DetectSkill(SkillSpec):
    """Skill that returns bounding boxes for a queried object."""

    def __init__(self) -> None:
        super().__init__(name="detect")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Tensor:
        if not isinstance(request_context, DetectRequest):
            raise ValueError("DetectSkill.build_prompt_tokens requires a DetectRequest")
        template = runtime.config.tokenizer.templates["detect"]
        if template is None:
            raise ValueError("Model configuration does not include detect templates")
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
        request_context: "DetectRequest",
    ) -> "DetectSkillState":
        if not isinstance(request_context, DetectRequest):
            raise ValueError("DetectSkill.create_state requires a DetectRequest context")
        return DetectSkillState(self, request, request_context)


class DetectSkillState(SkillState):
    """Skill state that decodes x/y/size triples into bounding boxes."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        detect_request: DetectRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = detect_request
        self._stage: str = "x"  # cycle: x -> y -> size -> x

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        self.append_token(step.token)
        if isinstance(step.token, CoordToken):
            if self._stage == "x":
                self._stage = "y"
            elif self._stage == "y":
                self._stage = "size"
        elif isinstance(step.token, SizeToken):
            self._stage = "x"
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
        objects = _extract_objects(self.tokens)
        return SkillFinalizeResult(
            text=text,
            tokens=list(self.tokens),
            output={"objects": objects},
        )

    def allowed_token_ids(self, runtime: "MoondreamRuntime") -> Sequence[int]:
        tokenizer = runtime.config.tokenizer
        if self._stage == "x":
            return [tokenizer.coord_id, tokenizer.eos_id]
        if self._stage == "y":
            return [tokenizer.coord_id]
        return [tokenizer.size_id]


def _extract_objects(tokens: Sequence[Token]) -> list[Dict[str, float]]:
    objects: list[Dict[str, float]] = []
    pending_x: Optional[float] = None
    pending_y: Optional[float] = None
    for token in tokens:
        if isinstance(token, CoordToken):
            if pending_x is None:
                pending_x = float(token.pos)
            elif pending_y is None:
                pending_y = float(token.pos)
            else:
                # Unexpected extra coordinate; reset state.
                pending_x = float(token.pos)
                pending_y = None
        elif isinstance(token, SizeToken):
            if pending_x is None or pending_y is None:
                continue
            width = float(token.width)
            height = float(token.height)
            half_w = width / 2.0
            half_h = height / 2.0
            x_min = max(pending_x - half_w, 0.0)
            y_min = max(pending_y - half_h, 0.0)
            x_max = min(pending_x + half_w, 1.0)
            y_max = min(pending_y + half_h, 1.0)
            objects.append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
            pending_x = None
            pending_y = None
    return objects


__all__ = [
    "DetectRequest",
    "DetectSettings",
    "DetectSkill",
]
