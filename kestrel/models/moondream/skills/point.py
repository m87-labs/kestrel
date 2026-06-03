"""Point skill that extracts spatial coordinates from model outputs."""


from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..runtime import CoordToken, SizeToken, TextToken, Token
from kestrel.utils.spatial_refs import build_spatial_tokens, normalize_spatial_refs

from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
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
class PointRequest:
    """Validated point payload — the carrier read by this skill's decode."""

    object: str
    image: Optional[np.ndarray | bytes]
    stream: bool
    spatial_refs: Optional[Sequence[Sequence[float]]] = None


class PointSkill(SkillSpec):
    """Skill that returns model-indicated points as normalized coordinates."""

    def __init__(self) -> None:
        super().__init__(name="point")

    def build_request(
        self,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        obj = str(prompt.get("object", "")).strip()
        if not obj:
            raise ValueError("object must be a non-empty string")
        if image is None:
            raise ValueError("image must be provided for pointing")
        # point decodes greedily; max_objects (if given) caps the budget.
        s = parse_settings(
            settings, temperature=0.0, top_p=1.0,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        max_tokens = s.max_tokens
        if settings is not None and "max_objects" in settings:
            max_objects = max(1, int(settings["max_objects"]))  # type: ignore[arg-type]
            max_tokens = max(2 * max_objects + 1, 2)
        request = PointRequest(
            object=obj,
            image=image,
            stream=False,
            spatial_refs=normalize_spatial_refs(prompt.get("spatial_refs")),
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
        if not isinstance(request_context, PointRequest):
            raise ValueError("PointSkill.build_prompt_tokens requires a PointRequest")
        pt = runtime.prompt_template
        template = pt.point()
        if template is None:
            raise ValueError("Model does not include a point template")
        prompt = request_context.object
        object_tokens = runtime.tokenizer.encode(" " + prompt).ids if prompt else []
        tokens: List[Token] = [TextToken(token_id=int(pt.bos_id))]
        tokens.extend(TextToken(token_id=int(tid)) for tid in template.prefix)
        tokens.extend(build_spatial_tokens(request_context.spatial_refs))
        tokens.extend(TextToken(token_id=int(tid)) for tid in object_tokens)
        tokens.extend(TextToken(token_id=int(tid)) for tid in template.suffix)
        return tokens

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
        pt = runtime.prompt_template
        if self._awaiting_y:
            return [pt.coord_id]
        return [pt.coord_id, pt.eos_id]


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
    "PointSkill",
]
