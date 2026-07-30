"""Skills emit their request media as ordered ``BuiltRequest.media``.

The media tuple is the engine-facing projection of the skill-owned request:
one ``MediaInput`` per input, in prompt order. These tests pin that every
public image-capable skill populates it from the same objects retained by its
request context, and that chat's order matches the parsed ``image_index`` order.
"""

from __future__ import annotations

import numpy as np
import pytest

from kestrel.models.moondream.skills import (
    CaptionSkill,
    DetectSkill,
    PointSkill,
    QuerySkill,
    SegmentSkill,
)
from kestrel.skills import ChatSkill, MediaInput

_IMG_A = "data:image/png;base64,aGk="  # decodes to b"hi"
_IMG_B = "data:image/png;base64,eW8="  # decodes to b"yo"


def _image_prompt_cases():
    return [
        (QuerySkill(), {"question": "what is this?"}),
        (CaptionSkill(), {"length": "normal"}),
        (DetectSkill(), {"object": "cat"}),
        (PointSkill(), {"object": "cat"}),
        (SegmentSkill(), {"object": "cat"}),
    ]


@pytest.mark.parametrize(
    "skill,prompt", _image_prompt_cases(), ids=lambda c: getattr(c, "name", None)
)
def test_image_skill_input_produces_media(skill, prompt) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    built = skill.build_request(image, prompt, None)
    assert len(built.media) == 1
    (item,) = built.media
    assert isinstance(item, MediaInput)
    assert item.kind == "image"
    context_image = getattr(built.request_context, "image")
    assert context_image is image
    assert item.data is context_image


@pytest.mark.parametrize(
    "skill,prompt",
    [
        (QuerySkill(), {"question": "just text?"}),
        (DetectSkill(), {"object": "cat"}),
    ],
    ids=lambda c: getattr(c, "name", None),
)
def test_text_only_input_produces_empty_media(skill, prompt) -> None:
    built = skill.build_request(None, prompt, None)
    assert built.media == ()


def test_chat_without_images_produces_empty_media() -> None:
    built = ChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, None
    )
    assert built.media == ()


def test_chat_media_order_matches_image_index_order() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMG_A}},
                {"type": "text", "text": "first?"},
            ],
        },
        {"role": "assistant", "content": "a"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "second?"},
                {"type": "image_url", "image_url": {"url": _IMG_B}},
            ],
        },
    ]
    built = ChatSkill().build_request(None, {"messages": msgs}, None)
    ctx = built.request_context
    assert all(m.kind == "image" for m in built.media)
    assert len(built.media) == len(ctx.images)
    for item, image in zip(built.media, ctx.images):
        assert item.data is image

    # media[i] is the same image that ``image_index == i`` parts (and the
    # runtime's ImageMarker indices) refer to.
    assert [m.data for m in built.media] == list(ctx.images) == [b"hi", b"yo"]
    assert ctx.messages[0].parts[0].image_index == 0
    assert ctx.messages[2].parts[1].image_index == 1
