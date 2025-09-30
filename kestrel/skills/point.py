"""Point skill that extracts spatial coordinates from model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

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


@dataclass(slots=True)
class PointInvocation:
    """Point request plus resolved defaults for scheduling."""

    request: PointRequest
    max_new_tokens: int


@dataclass(slots=True)
class PointDefaults:
    """Default values applied while parsing external point payloads."""

    max_new_tokens: int
    temperature: float
    top_p: float


class PointSkill(SkillSpec):
    """Skill that returns model-indicated points as normalized coordinates."""

    def __init__(self) -> None:
        super().__init__(name="point")

    def parse_http_payload(
        self,
        payload: Mapping[str, Any],
        *,
        image_decoder: Callable[[str], pyvips.Image],
        defaults: PointDefaults,
    ) -> PointInvocation:
        if not isinstance(payload, Mapping):
            raise ValueError("Request body must be a JSON object")

        raw_object = payload.get("object")
        if not isinstance(raw_object, str) or not raw_object.strip():
            raise ValueError("Field 'object' must be a non-empty string")
        object_name = raw_object.strip()

        raw_stream = payload.get("stream", False)
        if not isinstance(raw_stream, bool):
            raise ValueError("Field 'stream' must be a boolean if provided")
        if raw_stream:
            raise ValueError("Point requests do not support streaming responses")

        settings_payload = payload.get("settings")
        if settings_payload is None:
            temperature = defaults.temperature
            top_p = defaults.top_p
        elif isinstance(settings_payload, Mapping):
            temperature = _parse_float(
                settings_payload.get("temperature", defaults.temperature),
                field="settings.temperature",
                minimum=0.0,
            )
            top_p = _parse_float(
                settings_payload.get("top_p", defaults.top_p),
                field="settings.top_p",
                minimum_exclusive=0.0,
                maximum=1.0,
            )
        else:
            raise ValueError("Field 'settings' must be an object if provided")

        max_new_tokens = _parse_int(
            payload.get("max_new_tokens", defaults.max_new_tokens),
            field="max_new_tokens",
            minimum=1,
        )

        image_data = payload.get("image_url")
        if image_data is None:
            image: Optional[pyvips.Image] = None
        elif isinstance(image_data, str):
            image = image_decoder(image_data)
        else:
            raise ValueError("Field 'image_url' must be a string if provided")

        request = PointRequest(
            object=object_name,
            image=image,
            stream=raw_stream,
            settings=PointSettings(temperature=temperature, top_p=top_p),
        )
        return PointInvocation(request=request, max_new_tokens=max_new_tokens)

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
    ) -> Tensor:
        template = runtime.config.tokenizer.templates["point"]
        if template is None:
            raise ValueError("Model configuration does not include point templates")
        prefix: Sequence[int] = template["prefix"]
        suffix: Sequence[int] = template["suffix"]
        object_tokens = runtime.tokenizer.encode(prompt).ids if prompt else []
        ids = [*prefix, *object_tokens, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        *,
        context: Optional[object] = None,
    ) -> "PointSkillState":
        if context is None:
            context = PointRequest(
                object=request.prompt,
                image=request.image,
                stream=False,
                settings=PointSettings(
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
            )
        if not isinstance(context, PointRequest):
            raise ValueError("PointSkill.create_state requires a PointRequest context")
        return PointSkillState(self, request, context)


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
            extras={"points": points},
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


def _parse_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Field '{field}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"Field '{field}' must be >= {minimum}, received {value}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"Field '{field}' must be <= {maximum}, received {value}"
        )
    return value


def _parse_float(
    value: Any,
    *,
    field: str,
    minimum: Optional[float] = None,
    minimum_exclusive: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        raise ValueError(f"Field '{field}' must be a number")
    if minimum is not None and out < minimum:
        raise ValueError(
            f"Field '{field}' must be >= {minimum}, received {out}"
        )
    if minimum_exclusive is not None and out <= minimum_exclusive:
        raise ValueError(
            f"Field '{field}' must be > {minimum_exclusive}, received {out}"
        )
    if maximum is not None and out > maximum:
        raise ValueError(
            f"Field '{field}' must be <= {maximum}, received {out}"
        )
    return out


__all__ = [
    "PointDefaults",
    "PointInvocation",
    "PointRequest",
    "PointSettings",
    "PointSkill",
]
