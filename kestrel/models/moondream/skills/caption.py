"""Caption skill that generates text summaries for images."""


from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..runtime import TextToken, Token

from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    BuiltRequest,
    DecodeStep,
    SkillFinalizeResult,
    SkillSpec,
    SkillState,
    parse_settings,
)
from typing import Mapping

if False:  # pragma: no cover - type-checking imports
    from ..runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class CaptionRequest:
    """Caption payload — the carrier read by this skill's decode."""

    length: str
    image: Optional[np.ndarray | bytes]
    stream: bool


class CaptionSkill(SkillSpec):
    """Skill that returns plain text captions."""

    VALID_LENGTHS = {"short", "normal", "long"}

    def __init__(self) -> None:
        super().__init__(name="caption")

    def build_request(
        self,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        if image is None:
            raise ValueError("image must be provided for captioning")
        length = str(prompt.get("length", "normal")).strip().lower() or "normal"
        if length not in self.VALID_LENGTHS:
            valid = ", ".join(sorted(self.VALID_LENGTHS))
            raise ValueError(f"length must be one of: {valid}")
        s = parse_settings(
            settings,
            temperature=AR_DEFAULT_TEMPERATURE,
            top_p=AR_DEFAULT_TOP_P,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        request = CaptionRequest(
            length=length,
            image=image,
            stream=bool(prompt.get("stream", False)),
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
        )

    def prompt_text(self, request_context: object) -> str:
        return getattr(request_context, "length", "")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Sequence["Token"]:
        if not isinstance(request_context, CaptionRequest):
            raise ValueError("CaptionSkill.build_prompt_tokens requires a CaptionRequest")
        pt = runtime.prompt_template
        token_ids = pt.caption(request_context.length)
        if token_ids is None:
            raise ValueError("Model does not include caption templates")
        tokens = [TextToken(token_id=int(pt.bos_id))]
        tokens.extend(TextToken(token_id=int(tid)) for tid in token_ids)
        return tokens

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

    # The caption mask is position-independent: ``suppressed_token_ids`` returns
    # the same constant set (``[answer_id]`` for moondream2, else ``None``) at
    # every decode position -- it never transitions within a committed run. The
    # spec scheduler treats any active constraint as stateful unless told
    # otherwise, so declare the mask constant here; without this, the constant
    # ``answer_id`` suppression would force caption spec decode to one committed
    # token per macro-step (``commit_caps = 1``), throwing away the multi-token
    # speculative accept for no correctness benefit.
    mask_is_stateful = False

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

    def suppressed_token_ids(
        self, runtime: "MoondreamRuntime"
    ) -> Optional[Sequence[int]]:
        if runtime.model_name != "moondream2":
            return None
        return [runtime.prompt_template.answer_id]

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
    "CaptionSkill",
]
