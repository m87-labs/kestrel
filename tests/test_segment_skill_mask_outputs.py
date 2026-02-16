"""Tests for SegmentSkill bitmap mask outputs when return_base64 is enabled."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import kestrel.skills.segment as segment_mod
from kestrel.skills.segment import (
    SegmentRequest,
    SegmentSettings,
    SegmentSkill,
    SegmentSkillState,
)


@dataclass(slots=True)
class _DummyRefineResult:
    refined_svg_path: str | None
    refined_bbox: dict | None
    coarse_mask_base64: str | None
    refined_mask_base64: str | None


class _DummyTokenizer:
    def decode(self, _token_ids) -> str:
        return "dummy"


class _DummySegRefiner:
    def refine_with_bitmaps(self, _image, _svg_path, _bbox, *, return_base64: bool):
        assert return_base64 is True
        return _DummyRefineResult(
            refined_svg_path="REFINED_PATH",
            refined_bbox={"x_min": 0.1, "y_min": 0.2, "x_max": 0.9, "y_max": 0.8},
            coarse_mask_base64="COARSE_MASK",
            refined_mask_base64="REFINED_MASK",
        )


def test_segment_skill_attaches_masks_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        segment_mod,
        "svg_path_from_token_ids",
        lambda _tokenizer, _token_ids: ("M.0,.0z", []),
    )

    segment_request = SegmentRequest(
        object="dog",
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        stream=False,
        settings=SegmentSettings(temperature=0.0, top_p=1.0, max_tokens=16),
        return_base64=True,
    )
    state = SegmentSkillState(SegmentSkill(), request=object(), segment_request=segment_request)
    state._text_token_ids = [1]
    state._coord_values = [0.5, 0.5]
    state._size_values = [(0.2, 0.2)]

    runtime = type("Runtime", (), {})()
    runtime.tokenizer = _DummyTokenizer()
    runtime.seg_refiner = _DummySegRefiner()

    finalized = state.finalize(runtime, reason="stop")
    segment = finalized.output["segments"][0]

    assert segment["svg_path"] == "REFINED_PATH"
    assert segment["bbox"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.9, "y_max": 0.8}
    assert segment["coarse_mask_base64"] == "COARSE_MASK"
    assert segment["refined_mask_base64"] == "REFINED_MASK"

