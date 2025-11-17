"""Segmentation skill that returns SVG paths and bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken, Token
from kestrel.utils.svg import (
    decode_svg_token_strings,
    svg_path_from_token_ids,
)

from .base import DecodeStep, SkillFinalizeResult, SkillSpec, SkillState

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class SegmentSettings:
    """Sampling parameters supplied with a segment invocation."""

    temperature: float
    top_p: float
    max_tokens: int


@dataclass(slots=True)
class SegmentRequest:
    """Segment payload used internally by the scheduler."""

    label: str
    image: Optional[pyvips.Image]
    stream: bool
    settings: SegmentSettings
    spatial_refs: Optional[Sequence[Sequence[float]]] = None


class SegmentSkill(SkillSpec):
    """Skill that emits SVG paths for the requested label."""

    def __init__(self) -> None:
        super().__init__(name="segment")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Tensor:
        if not isinstance(request_context, SegmentRequest):
            raise ValueError("SegmentSkill.build_prompt_tokens requires a SegmentRequest")
        template = runtime.config.tokenizer.templates.get("segment")
        if template is None:
            raise ValueError("Model configuration does not include segment templates")
        prefix: Sequence[int] = template.get("prefix", [])
        suffix: Sequence[int] = template.get("suffix", [])
        label = request_context.label
        label_tokens = runtime.tokenizer.encode(label).ids if label else []
        ids = [*prefix, *label_tokens, *suffix]
        if not ids:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: "SegmentRequest",
    ) -> "SegmentSkillState":
        if not isinstance(request_context, SegmentRequest):
            raise ValueError("SegmentSkill.create_state requires a SegmentRequest context")
        return SegmentSkillState(self, request, request_context)


class SegmentSkillState(SkillState):
    """Skill state that interprets coord/size tokens and SVG path text."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        segment_request: SegmentRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = segment_request
        self._text_token_ids: List[int] = []
        self._coord_values: List[float] = []
        self._size_values: List[Tuple[float, float]] = []
        self._streaming: bool = bool(segment_request.stream)
        self._stream_offset: int = 0
        self._pending_stream: Optional[str] = None

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        token = step.token
        self.append_token(token)
        if isinstance(token, TextToken):
            self._text_token_ids.append(token.token_id)
            self._update_stream(runtime)
        elif isinstance(token, CoordToken):
            self._coord_values.append(float(token.pos))
        elif isinstance(token, SizeToken):
            self._size_values.append((float(token.width), float(token.height)))
        return None

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        tokenizer = runtime.tokenizer
        raw_text = (
            tokenizer.decode(self._text_token_ids) if self._text_token_ids else ""
        )

        try:
            svg_path, decoded_tokens = svg_path_from_token_ids(
                tokenizer, self._text_token_ids
            )
            parse_error: Optional[str] = None
        except Exception as exc:  # pragma: no cover - defensive
            decoded_tokens = decode_svg_token_strings(tokenizer, self._text_token_ids)
            svg_path = ""
            parse_error = str(exc)

        points = _coords_to_points(self._coord_values)
        bbox = _build_bbox(self._coord_values, self._size_values)

        segment: Dict[str, object] = {
            "label": self._request.label,
            "text": raw_text.strip(),
            "svg_path": svg_path,
            "path_tokens": decoded_tokens,
            "token_ids": list(self._text_token_ids),
            "points": points,
        }
        if parse_error:
            segment["parse_error"] = parse_error
        if bbox is not None:
            segment["bbox"] = bbox
        if self._size_values:
            segment["sizes"] = [
                {"width": max(min(w, 1.0), 0.0), "height": max(min(h, 1.0), 0.0)}
                for w, h in self._size_values
            ]
        if self._request.spatial_refs is not None:
            segment["spatial_refs"] = [list(ref) for ref in self._request.spatial_refs]

        return SkillFinalizeResult(
            text=svg_path or raw_text,
            tokens=list(self.tokens),
            output={"segments": [segment]},
        )

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        if not self._streaming:
            return None
        if not self._pending_stream:
            return None
        delta = self._pending_stream
        self._pending_stream = None
        return delta

    def _update_stream(self, runtime: "MoondreamRuntime") -> None:
        if not self._streaming:
            return
        try:
            path, _ = svg_path_from_token_ids(runtime.tokenizer, self._text_token_ids)
        except Exception:
            return  # Don't stream until the path is parseable.
        if not path:
            return
        if len(path) > self._stream_offset:
            self._pending_stream = path[self._stream_offset :]
            self._stream_offset = len(path)


def _coords_to_points(coords: Sequence[float]) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for i in range(0, len(coords) - 1, 2):
        x = float(coords[i])
        y = float(coords[i + 1])
        points.append({"x": _clamp_unit(x), "y": _clamp_unit(y)})
    return points


def _build_bbox(
    coords: Sequence[float], sizes: Sequence[Tuple[float, float]]
) -> Optional[Dict[str, float]]:
    if len(coords) < 2 or not sizes:
        return None
    cx = _clamp_unit(float(coords[0]))
    cy = _clamp_unit(float(coords[1]))
    width = _clamp_unit(float(sizes[0][0]))
    height = _clamp_unit(float(sizes[0][1]))
    half_w = width / 2.0
    half_h = height / 2.0
    return {
        "x_center": cx,
        "y_center": cy,
        "width": width,
        "height": height,
        "x_min": max(cx - half_w, 0.0),
        "x_max": min(cx + half_w, 1.0),
        "y_min": max(cy - half_h, 0.0),
        "y_max": min(cy + half_h, 1.0),
    }


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "SegmentRequest",
    "SegmentSettings",
    "SegmentSkill",
]
