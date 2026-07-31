"""_media_to_legacy_image: BuiltRequest.media -> the legacy image value.

The adapter is the temporary compatibility boundary between the generic
skill media contract and the image-specific AR path behind
``submit(image=...)``: images pass through with object identity and order
preserved, anything else is refused. It validates modality only — payload
type checks stay downstream.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from kestrel.engine.core import InferenceEngine
from kestrel.models.moondream.skills import QuerySkill
from kestrel.skills import MediaInput, SkillRegistry

_adapt = InferenceEngine._media_to_legacy_image


def test_empty_media_maps_to_none() -> None:
    assert _adapt(()) is None


def test_single_image_preserves_object_identity() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert _adapt((MediaInput(kind="image", data=image),)) is image


def test_multiple_images_preserve_order_in_a_tuple() -> None:
    first, second = b"first", b"second"
    out = _adapt(
        (MediaInput(kind="image", data=first), MediaInput(kind="image", data=second))
    )
    assert isinstance(out, tuple)
    assert out[0] is first
    assert out[1] is second


def test_rejects_audio() -> None:
    with pytest.raises(NotImplementedError):
        _adapt((MediaInput(kind="audio", data=b"pcm"),))


def test_rejects_video() -> None:
    with pytest.raises(NotImplementedError):
        _adapt((MediaInput(kind="video", data=b"frames"),))


def test_rejects_mixed_image_and_non_image_media() -> None:
    with pytest.raises(NotImplementedError):
        _adapt(
            (
                MediaInput(kind="image", data=b"img"),
                MediaInput(kind="audio", data=b"pcm"),
            )
        )


def test_public_image_request_submits_ordered_media() -> None:
    """End to end above the queue: a public ``query`` with an image flows
    prompt -> skill media -> ``submit(_media=...)``, delivering the caller's
    exact image object. The legacy image value is derived from this media
    at request submission (transitionally) and, ultimately, at admission."""
    eng = object.__new__(InferenceEngine)
    eng._skills_override = SkillRegistry([QuerySkill()])
    captured: dict[str, Any] = {}

    async def fake_submit(request_context: object, **kwargs: Any) -> str:
        captured["request_context"] = request_context
        captured.update(kwargs)
        return "RESULT"

    eng.submit = fake_submit  # type: ignore[method-assign]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    out = asyncio.run(eng.query(image=image, question="what is this?"))
    assert out == "RESULT"
    (item,) = captured["_media"]
    assert item.kind == "image"
    assert item.data is image
