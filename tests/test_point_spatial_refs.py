from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from kestrel.engine import InferenceEngine
from kestrel.moondream.runtime import CoordToken, SizeToken, TextToken
from kestrel.skills.point import PointRequest, PointSettings, PointSkill


class _FakeTokenizer:
    def encode(self, text: str) -> SimpleNamespace:
        assert text == " gaze"
        return SimpleNamespace(ids=[42])


def _fake_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(
                bos_id=1,
                templates={"point": {"prefix": [10, 11], "suffix": [12]}},
            )
        ),
        tokenizer=_FakeTokenizer(),
    )


def test_point_prompt_tokens_include_spatial_refs_before_object_text() -> None:
    request = PointRequest(
        object="gaze",
        image=None,
        stream=False,
        settings=PointSettings(temperature=0.0, top_p=1.0),
        spatial_refs=((0.2, 0.3), (0.1, 0.2, 0.5, 0.6)),
    )

    tokens = PointSkill().build_prompt_tokens(_fake_runtime(), request)

    assert tokens[:3] == [TextToken(1), TextToken(10), TextToken(11)]
    assert tokens[3] == CoordToken(pos=pytest.approx(0.2))
    assert tokens[4] == CoordToken(pos=pytest.approx(0.3))
    assert tokens[5] == CoordToken(pos=pytest.approx(0.3))
    assert tokens[6] == CoordToken(pos=pytest.approx(0.4))
    assert tokens[7] == SizeToken(width=pytest.approx(0.4), height=pytest.approx(0.4))
    assert tokens[8:] == [TextToken(42), TextToken(12)]


def test_engine_point_normalizes_spatial_refs_for_request() -> None:
    engine = object.__new__(InferenceEngine)
    engine._default_max_new_tokens = 128
    captured: dict[str, object] = {}

    async def fake_submit(request_context: object, **kwargs: object) -> SimpleNamespace:
        captured["request_context"] = request_context
        captured["kwargs"] = kwargs
        return SimpleNamespace(output={})

    engine.submit = fake_submit  # type: ignore[method-assign]
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    asyncio.run(
        engine.point(
            image,
            " gaze ",
            settings={"max_tokens": 32},
            spatial_refs=[[0.2, 0.3], [0.1, 0.2, 0.5, 0.6]],
        )
    )

    request = captured["request_context"]
    assert isinstance(request, PointRequest)
    assert request.object == "gaze"
    assert request.spatial_refs == ((0.2, 0.3), (0.1, 0.2, 0.5, 0.6))
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["skill"] == "point"
    assert kwargs["max_new_tokens"] == 32


def test_engine_point_keeps_positional_settings_compatibility() -> None:
    engine = object.__new__(InferenceEngine)
    engine._default_max_new_tokens = 128
    captured: dict[str, object] = {}

    async def fake_submit(request_context: object, **kwargs: object) -> SimpleNamespace:
        captured["request_context"] = request_context
        captured["kwargs"] = kwargs
        return SimpleNamespace(output={})

    engine.submit = fake_submit  # type: ignore[method-assign]
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    asyncio.run(engine.point(image, "person", {"max_tokens": 12}))

    request = captured["request_context"]
    assert isinstance(request, PointRequest)
    assert request.spatial_refs is None
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_new_tokens"] == 12


def test_engine_point_requires_image_before_spatial_ref_validation() -> None:
    engine = object.__new__(InferenceEngine)

    with pytest.raises(ValueError, match="image must be provided for pointing"):
        asyncio.run(engine.point(None, "gaze", spatial_refs=[[0.2, 0.3]]))

    refs = np.array([[0.2, 0.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="image must be provided for pointing"):
        asyncio.run(engine.point(None, "gaze", spatial_refs=refs))


def test_engine_query_rejects_array_spatial_refs_without_image() -> None:
    engine = object.__new__(InferenceEngine)

    refs = np.array([[0.2, 0.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="spatial_refs can only be used with an image"):
        asyncio.run(engine.query(question="Where?", spatial_refs=refs))


def test_engine_segment_requires_image_before_spatial_ref_validation() -> None:
    engine = object.__new__(InferenceEngine)

    refs = np.array([[0.2, 0.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="image must be provided for segmentation"):
        asyncio.run(engine.segment(None, "person", spatial_refs=refs))
