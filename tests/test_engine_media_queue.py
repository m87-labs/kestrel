"""Queued AR requests carry ordered media.

``_submit_request`` projects the stable ``submit(image=...)`` argument into
ordered ``MediaInput`` values, and passes skill-originated
``BuiltRequest.media`` through untouched. These tests pin that the queued
``_AutoregressiveRequest.media`` preserves object identity and order from
both entry paths; the legacy image representation begins only at admission.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

from kestrel.engine.core import InferenceEngine
from kestrel.models.moondream.skills import QuerySkill
from kestrel.skills import MediaInput, SkillRegistry


def _engine() -> InferenceEngine:
    """A minimal engine whose ``_submit_request`` runs for real: requests
    land on ``_queue`` instead of a live scheduler."""
    eng = object.__new__(InferenceEngine)
    eng._shutdown = False
    eng._initialized = True
    eng._scheduler_error = None
    eng._init_task = None
    eng._request_ids = iter(range(1, 1_000_000))
    eng._adapter_provider = None
    eng._skills_override = SkillRegistry([QuerySkill()])
    eng._default_model = "m"
    eng._runtimes = {
        "m": SimpleNamespace(
            image_prefix_length=729,
            max_seq_length=4096,
            prompt_template=None,
            device=SimpleNamespace(type="cuda"),
        )
    }
    eng._queue = asyncio.Queue()
    return eng


class _StubSkill(QuerySkill):
    """Query skill that skips prompt-token construction (no runtime needed)."""

    def build_prompt_tokens(self, runtime: Any, request_context: Any) -> list:
        return []


def _submit(eng: InferenceEngine, **kwargs: Any):
    async def go():
        future, _ = await eng._submit_request(
            max_new_tokens=8,
            request_context=object(),
            adapter=None,
            temperature=None,
            top_p=None,
            return_logprobs=None,
            generated_prefix=eng._normalize_generated_prefix(None, "x"),
            suppress_next_token_ids=None,
            stream_queue=None,
            skill="query",
            **kwargs,
        )
        payload = eng._queue.get_nowait()
        future.cancel()
        return payload

    return asyncio.run(go())


def _stub_engine() -> InferenceEngine:
    eng = _engine()
    eng._skills_override = SkillRegistry([_StubSkill()])
    return eng


def test_no_image_queues_empty_media() -> None:
    payload = _submit(_stub_engine(), image=None)
    assert payload.media == ()


def test_one_image_queues_one_media_input() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = _submit(_stub_engine(), image=image)
    (item,) = payload.media
    assert item.kind == "image"
    assert item.data is image


def test_multiple_images_queue_ordered_media() -> None:
    first, second = b"first", b"second"
    payload = _submit(_stub_engine(), image=[first, second])
    assert [m.kind for m in payload.media] == ["image", "image"]
    assert payload.media[0].data is first
    assert payload.media[1].data is second


def test_skill_media_reaches_queued_request_unchanged() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    media = (MediaInput(kind="image", data=image),)
    payload = _submit(_stub_engine(), image=None, media=media)
    assert payload.media is media  # the exact tuple, not a copy


def test_image_and_media_are_mutually_exclusive() -> None:
    """The two engine-entry representations must not silently compete:
    supplying both is rejected before either is normalized or queued."""
    eng = _stub_engine()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="image and media are mutually exclusive"):
        _submit(
            eng,
            image=image,
            media=(MediaInput(kind="image", data=image),),
        )
    assert eng._queue.empty()  # nothing was queued


def test_public_query_media_reaches_queue() -> None:
    """End to end above the queue: a public image query's media lands on the
    queued request with the caller's exact image object."""
    eng = _stub_engine()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    captured: dict[str, Any] = {}

    async def go():
        task = asyncio.ensure_future(
            eng.query(image=image, question="what is this?")
        )
        payload = await eng._queue.get()
        captured["payload"] = payload
        payload.future.cancel()
        task.cancel()

    asyncio.run(go())
    payload = captured["payload"]
    (item,) = payload.media
    assert item.data is image
