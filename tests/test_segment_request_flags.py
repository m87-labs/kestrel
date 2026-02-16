"""Tests for segment request flags and defaults."""

from __future__ import annotations

from kestrel.skills.segment import SegmentRequest, SegmentSettings


def test_segment_request_return_base64_defaults_false() -> None:
    request = SegmentRequest(
        object="dog",
        image=None,
        stream=False,
        settings=SegmentSettings(temperature=0.0, top_p=1.0, max_tokens=16),
    )
    assert request.return_base64 is False

