"""Detect skill that extracts bounding boxes from model outputs."""


from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from ..runtime import CoordToken, SizeToken, TextToken, Token

from kestrel.skills.base import (
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


# Default object cap when the caller doesn't specify one.
_DETECT_DEFAULT_MAX_OBJECTS = 50


@dataclass(slots=True)
class DetectRequest:
    """Detect payload — the carrier read by this skill's decode."""

    object: str
    image: Optional[np.ndarray | bytes]
    stream: bool
    max_objects: int


class DetectSkill(SkillSpec):
    """Skill that returns bounding boxes for a queried object."""

    def __init__(self) -> None:
        super().__init__(name="detect")

    def build_request(
        self,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        obj = str(prompt.get("object", "")).strip()
        if not obj:
            raise ValueError("object must be a non-empty string")
        max_objects = _DETECT_DEFAULT_MAX_OBJECTS
        if settings is not None and "max_objects" in settings:
            max_objects = max(1, int(settings["max_objects"]))  # type: ignore[arg-type]
        # detect derives its token budget from max_objects and ignores any
        # settings["max_tokens"] (matching the original engine behavior), so
        # parse only the sampling knobs. Each object consumes up to 3 tokens
        # (x, y, size); one extra for EOS.
        max_tokens = max(3 * max_objects + 1, 3)
        s = parse_settings(
            {k: v for k, v in (settings or {}).items() if k != "max_tokens"},
            temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        )
        request = DetectRequest(
            object=obj, image=image, stream=False, max_objects=max_objects,
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
        )

    def prompt_text(self, request_context: object) -> str:
        return getattr(request_context, "object", "")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Sequence["Token"]:
        if not isinstance(request_context, DetectRequest):
            raise ValueError("DetectSkill.build_prompt_tokens requires a DetectRequest")
        pt = runtime.prompt_template
        template = pt.detect()
        if template is None:
            raise ValueError("Model does not include a detect template")
        prompt = request_context.object
        object_tokens = runtime.tokenizer.encode(" " + prompt).ids if prompt else []
        ids = [*template.prefix, *object_tokens, *template.suffix]
        tokens = [TextToken(token_id=int(pt.bos_id))]
        tokens.extend(TextToken(token_id=int(tid)) for tid in ids)
        return tokens

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
        pt = runtime.prompt_template
        if self._stage == "x":
            return [pt.coord_id, pt.eos_id]
        if self._stage == "y":
            return [pt.coord_id]
        return [pt.size_id]


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
    "DetectSkill",
]
