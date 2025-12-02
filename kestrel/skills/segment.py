"""Segmentation skill that returns SVG paths and bounding boxes."""


from dataclasses import dataclass
from typing import Iterable
import json
from typing import Dict, List, Optional, Sequence, Tuple

import pyvips
import torch
import numpy as np
from torch import Tensor

from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken, Token
from kestrel.utils.svg import (
    decode_svg_token_strings,
    parse_svg_tokens,
    scale_svg_path_tokens,
    svg_path_from_token_ids,
    PATH_COMMANDS,
)
from ..seg_refiner import refine_segmentation
from ..head_seg_refiner import refine_segmentation_with_head

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

    object: str
    image: Optional[pyvips.Image | np.ndarray]
    stream: bool
    settings: SegmentSettings
    spatial_refs: Optional[Sequence[Sequence[float]]] = None


class SegmentSkill(SkillSpec):
    """Skill that emits SVG paths for the requested object."""

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
        object_name = request_context.object
        object_tokens = runtime.tokenizer.encode(object_name).ids if object_name else []

        # Assemble ids in vixtral order:
        # <prefix> <spatial placeholders> object <suffix>
        ids: List[int] = [*prefix]

        refs = request_context.spatial_refs
        if refs:
            coord_id = runtime.config.tokenizer.coord_id
            size_id = runtime.config.tokenizer.size_id
            for ref in refs:
                n = len(ref)
                if n == 2:
                    ids.extend([coord_id, coord_id])
                elif n == 4:
                    ids.extend([coord_id, coord_id, size_id])
                else:
                    raise ValueError(
                        "Each spatial_ref must contain 2 (point) or 4 (bbox) values"
                    )

        ids.extend(object_tokens)
        ids.extend(suffix)

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
        self._stream_cursor: int = 0  # index into decoded tokens processed for streaming
        self._pending_stream: Optional[str] = None
        self._bbox_sent: bool = False

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

        coarse_path = svg_path
        coarse_bbox = bbox

        if svg_path and bbox and not parse_error and self._request.image is not None:
            if runtime.head_refiner is not None:
                refined_path, refined_bbox = refine_segmentation_with_head(
                    self._request.image,
                    svg_path,
                    bbox,
                    runtime.head_refiner,
                    runtime.model.vision,
                    iters=6,
                )
            else:
                refined_path, refined_bbox = refine_segmentation(
                    self._request.image,
                    svg_path,
                    bbox,
                    runtime.sam_model,
                    iters=6,
                )
            if refined_path is not None and refined_bbox is not None:
                svg_path = refined_path
                bbox = refined_bbox

        segment: Dict[str, object] = {
            "object": self._request.object,
            "text": raw_text.strip(),
            "svg_path": svg_path,
            "path_tokens": decoded_tokens,
            "token_ids": list(self._text_token_ids),
            "points": points,
            "coarse_path": coarse_path,
        }
        if parse_error:
            segment["parse_error"] = parse_error
        if bbox is not None:
            segment["bbox"] = bbox
        if coarse_bbox is not None:
            segment["coarse_bbox"] = coarse_bbox
        if self._size_values:
            segment["sizes"] = [
                {"width": max(min(w, 1.0), 0.0), "height": max(min(h, 1.0), 0.0)}
                for w, h in self._size_values
            ]
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

        if not self._bbox_sent:
            bbox = _build_bbox(self._coord_values, self._size_values)
            if bbox:
                self._pending_stream = "__BBOX__" + json.dumps(bbox)
                self._bbox_sent = True
                return

        # Emit one path chunk per call: previous command + its args.
        decoded_tokens = decode_svg_token_strings(runtime.tokenizer, self._text_token_ids)
        if not decoded_tokens:
            return
        try:
            parsed_tokens = parse_svg_tokens(decoded_tokens)
        except Exception:
            return
        scaled_tokens = scale_svg_path_tokens(parsed_tokens)
        end, chunk = _next_command_chunk(scaled_tokens, self._stream_cursor)
        if chunk and end > self._stream_cursor:
            first = self._stream_cursor == 0
            self._pending_stream = _format_chunk(chunk, first=first)
            self._stream_cursor = end


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
        "x_min": max(cx - half_w, 0.0),
        "x_max": min(cx + half_w, 1.0),
        "y_min": max(cy - half_h, 0.0),
        "y_max": min(cy + half_h, 1.0),
    }


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


# Number of coordinate values each SVG path command expects.
_COMMAND_ARITY = {
    "Z": 0, "z": 0,
    "H": 1, "h": 1, "V": 1, "v": 1,
    "M": 2, "m": 2, "L": 2, "l": 2, "T": 2, "t": 2,
    "S": 4, "s": 4, "Q": 4, "q": 4,
    "C": 6, "c": 6,
    "A": 7, "a": 7,
}


def _next_command_chunk(tokens: Sequence[str], start: int) -> tuple[int, str]:
    """Return a command-plus-args chunk ending just before the next command."""

    n = len(tokens)
    if start >= n:
        return start, ""

    # Find the next command
    cmd_idx = start
    while cmd_idx < n and tokens[cmd_idx] not in PATH_COMMANDS:
        cmd_idx += 1
    if cmd_idx >= n:
        return start, ""

    # Collect until the following command (or end)
    end = cmd_idx + 1
    while end < n and tokens[end] not in PATH_COMMANDS:
        end += 1

    # Require enough args for the command (1 for command + arity)
    cmd = tokens[cmd_idx]
    required = 1 + _COMMAND_ARITY.get(cmd, 2)
    if end - cmd_idx < required:
        return start, ""

    chunk = " ".join(tokens[cmd_idx:end])
    return end, chunk


def _format_chunk(chunk: str, *, first: bool) -> str:
    if not chunk:
        return ""
    return chunk if first else " " + chunk


__all__ = [
    "SegmentRequest",
    "SegmentSettings",
    "SegmentSkill",
]
