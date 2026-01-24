"""Caption skill that generates text summaries for images."""


from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from kestrel.moondream.runtime import TextToken, Token

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
    image: Optional[np.ndarray | bytes]
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
        request_context: object,
    ) -> Sequence["Token"]:
        if not isinstance(request_context, CaptionRequest):
            raise ValueError("CaptionSkill.build_prompt_tokens requires a CaptionRequest")
        templates = runtime.config.tokenizer.templates["caption"]
        if templates is None:
            raise ValueError("Model configuration does not include caption templates")
        length_key = request_context.length
        if length_key not in templates:
            valid = ", ".join(sorted(templates.keys()))
            raise ValueError(
                f"Unsupported caption length '{length_key}'. Expected one of: {valid}"
            )
        tokens = templates[length_key]
        return [TextToken(token_id=int(tid)) for tid in tokens]

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: "CaptionRequest",
    ) -> "CaptionSkillState":
        if not isinstance(request_context, CaptionRequest):
            raise ValueError("CaptionSkill.create_state requires a CaptionRequest context")
        return CaptionSkillState(self, request, request_context)


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
        self._streaming = bool(caption_request.stream)
        self._stream_offset = 0

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        self.append_token(step.token)
        return None

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        if not self._streaming:
            return None
        text_tokens = [
            token.token_id for token in self.tokens if isinstance(token, TextToken)
        ]
        if not text_tokens:
            return None
        caption = runtime.tokenizer.decode(text_tokens)
        if len(caption) <= self._stream_offset:
            return None
        chunk = caption[self._stream_offset :]
        self._stream_offset = len(caption)
        if not chunk:
            return None
        return chunk

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
            output={"caption": caption},
        )


__all__ = [
    "CaptionRequest",
    "CaptionSettings",
    "CaptionSkill",
]
