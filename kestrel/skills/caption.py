"""Caption skill that generates text summaries for images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import TextToken

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class CaptionSettings:
    """Sampling parameters for caption generation."""

    temperature: float
    top_p: float


@dataclass(slots=True)
class CaptionRequest:
    """Caption payload carried through the scheduler."""

    length: str
    image: Optional[pyvips.Image]
    stream: bool
    settings: CaptionSettings


class CaptionSkill(SkillSpec):
    """Skill that returns plain text captions."""

    VALID_LENGTHS = {"short", "normal", "long"}

    def __init__(self) -> None:
        super().__init__(name="caption")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
    ) -> Tensor:
        templates = runtime.config.tokenizer.templates["caption"]
        if templates is None:
            raise ValueError("Model configuration does not include caption templates")
        if prompt not in templates:
            valid = ", ".join(sorted(templates.keys()))
            raise ValueError(f"Unsupported caption length '{prompt}'. Expected one of: {valid}")
        tokens = templates[prompt]
        if not tokens:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor([tokens], dtype=torch.long)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        *,
        context: Optional[object] = None,
    ) -> "CaptionSkillState":
        if context is None:
            context = CaptionRequest(
                length=request.prompt,
                image=request.image,
                stream=False,
                settings=CaptionSettings(
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
            )
        if not isinstance(context, CaptionRequest):
            raise ValueError("CaptionSkill.create_state requires a CaptionRequest context")
        return CaptionSkillState(self, request, context)


class CaptionSkillState(SkillState):
    """Skill state that accumulates caption text."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        caption_request: CaptionRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = caption_request

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
        caption = runtime.tokenizer.decode(text_tokens) if text_tokens else ""
        return SkillFinalizeResult(
            text=caption,
            tokens=list(self.tokens),
            extras={"caption": caption},
        )


__all__ = [
    "CaptionRequest",
    "CaptionSettings",
    "CaptionSkill",
]
