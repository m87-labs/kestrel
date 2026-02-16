"""HTTP /v1/segment response shape tests for return_base64."""

from __future__ import annotations

import base64

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from kestrel.config import RuntimeConfig
from kestrel.engine import EngineMetrics, EngineResult
from kestrel.server.http import _ServerConfig, _ServerState


class _StubEngine:
    def __init__(self, result: EngineResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    async def segment(
        self,
        image,
        object: str,
        *,
        spatial_refs=None,
        return_base64: bool = False,
        settings=None,
    ) -> EngineResult:
        self.calls.append(
            {
                "object": object,
                "return_base64": return_base64,
                "has_image": image is not None,
            }
        )
        return self._result


def _make_app(engine: _StubEngine) -> Starlette:
    cfg = _ServerConfig(
        runtime_cfg=RuntimeConfig(model_path="/tmp/weights.pt", device="cpu"),
        default_max_new_tokens=128,
        default_temperature=0.2,
        default_top_p=0.9,
    )
    state = _ServerState(cfg)
    state.engine = engine
    app = Starlette(routes=[Route("/v1/segment", state.handle_segment, methods=["POST"])])
    return app


def _dummy_result() -> EngineResult:
    metrics = EngineMetrics(
        input_tokens=0,
        output_tokens=0,
        prefill_time_ms=0.0,
        decode_time_ms=0.0,
        ttft_ms=0.0,
    )
    return EngineResult(
        request_id=1,
        tokens=[],
        finish_reason="stop",
        metrics=metrics,
        output={
            "segments": [
                {
                    "svg_path": "M.0,.0z",
                    "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                    "coarse_mask_base64": "COARSE",
                    "refined_mask_base64": "REFINED",
                }
            ]
        },
    )


def test_segment_return_base64_includes_masks() -> None:
    engine = _StubEngine(_dummy_result())
    app = _make_app(engine)
    client = TestClient(app)

    image_b64 = base64.b64encode(b"not-an-image").decode("ascii")
    resp = client.post(
        "/v1/segment",
        json={
            "object": "dog",
            "image_url": image_b64,
            "return_base64": True,
            "settings": {"max_tokens": 8},
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["coarse_mask_base64"] == "COARSE"
    assert payload["refined_mask_base64"] == "REFINED"
    assert engine.calls and engine.calls[0]["return_base64"] is True


def test_segment_return_base64_false_omits_masks() -> None:
    engine = _StubEngine(_dummy_result())
    app = _make_app(engine)
    client = TestClient(app)

    image_b64 = base64.b64encode(b"not-an-image").decode("ascii")
    resp = client.post(
        "/v1/segment",
        json={
            "object": "dog",
            "image_url": image_b64,
            "return_base64": False,
            "settings": {"max_tokens": 8},
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "coarse_mask_base64" not in payload
    assert "refined_mask_base64" not in payload


def test_segment_return_base64_requires_image() -> None:
    engine = _StubEngine(_dummy_result())
    app = _make_app(engine)
    client = TestClient(app)

    resp = client.post(
        "/v1/segment",
        json={
            "object": "dog",
            "return_base64": True,
            "settings": {"max_tokens": 8},
        },
    )
    assert resp.status_code == 400, resp.text
    assert engine.calls == []

