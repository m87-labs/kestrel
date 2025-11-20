"""ASGI application exposing the Kestrel inference engine over HTTP."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pyvips
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from kestrel.config import RuntimeConfig
from kestrel.engine import EngineMetrics, InferenceEngine
from kestrel.skills.segment import SegmentRequest, SegmentSettings
from kestrel.utils.image import load_vips_from_base64

# ---------------------------------------------------------------------------
# Helpers

logger = logging.getLogger(__name__)


STREAM_CHUNK_SIZE = 3
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _format_metrics(result_metrics: EngineMetrics) -> Dict[str, float]:
    return {
        "input_tokens": result_metrics.input_tokens,
        "output_tokens": result_metrics.output_tokens,
        "prefill_time_ms": result_metrics.prefill_time_ms,
        "decode_time_ms": result_metrics.decode_time_ms,
        "ttft_ms": result_metrics.ttft_ms,
    }


def _sse_payload(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@dataclass(slots=True)
class _ServerConfig:
    runtime_cfg: RuntimeConfig
    default_max_new_tokens: int
    default_temperature: float
    default_top_p: float


class _ServerState:
    """Container that owns the inference engine shared by all requests."""

    __slots__ = ("config", "engine")

    def __init__(self, config: _ServerConfig) -> None:
        self.config = config
        self.engine: Optional[InferenceEngine] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks

    async def startup(self) -> None:
        if self.engine is not None:
            return
        logger.info("Starting inference engine")
        self.engine = await InferenceEngine.create(self.config.runtime_cfg)
        logger.info("Inference engine ready")

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        logger.info("Shutting down inference engine")
        await self.engine.shutdown()
        self.engine = None

    # ------------------------------------------------------------------
    # Request handlers

    async def handle_health(self, _request: Request) -> Response:
        if self.engine is None or not self.engine.is_running:
            return JSONResponse({"status": "starting"}, status_code=503)
        return JSONResponse({"status": "ok"})

    async def handle_segment(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            object_name = _parse_required_str(payload, "object")
            settings_payload = payload.get("settings") or {}
            if not isinstance(settings_payload, dict):
                raise ValueError("Field 'settings' must be an object if provided")
            temperature = _parse_float(
                settings_payload.get("temperature", self.config.default_temperature),
                "settings.temperature",
                minimum=0.0,
            )
            top_p = _parse_float(
                settings_payload.get("top_p", self.config.default_top_p),
                "settings.top_p",
                minimum_exclusive=0.0,
                maximum=1.0,
            )
            max_tokens = _parse_int(
                settings_payload.get("max_tokens", self.config.default_max_new_tokens),
                "settings.max_tokens",
                minimum=1,
            )

            image_data = payload.get("image_url")
            if image_data is None:
                raise ValueError("Field 'image_url' must be provided")
            if not isinstance(image_data, str):
                raise ValueError("Field 'image_url' must be a string")
            image = load_vips_from_base64(image_data)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        engine = self.engine
        assert engine is not None
        try:
            result = await engine.segment(
                image=image,
                object=object_name,
                settings={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )

        segment = (result.output.get("segments") or [{}])[0]
        bbox = segment.get("bbox")
        path = segment.get("svg_path") or ""
        if bbox and set(bbox.keys()) == {"x_center", "y_center", "width", "height", "x_min", "x_max", "y_min", "y_max"}:
            bbox = {
                "x_min": bbox["x_min"],
                "y_min": bbox["y_min"],
                "x_max": bbox["x_max"],
                "y_max": bbox["y_max"],
            }

        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "bbox": bbox,
            "path": path,
            "metrics": _format_metrics(result.metrics),
        }
        parse_error = segment.get("parse_error")
        if parse_error:
            response_payload["parse_error"] = parse_error

        return JSONResponse(response_payload)

    async def handle_query(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            question = _parse_required_str(payload, "question")
            reasoning = _parse_bool(payload.get("reasoning", False), "reasoning")

            stream = _parse_bool(payload.get("stream", False), "stream")

            settings_payload = payload.get("settings")
            if settings_payload is None:
                settings_payload = {}
            if isinstance(settings_payload, dict):
                temperature = _parse_float(
                    settings_payload.get("temperature", self.config.default_temperature),
                    "settings.temperature",
                    minimum=0.0,
                )
                top_p = _parse_float(
                    settings_payload.get("top_p", self.config.default_top_p),
                    "settings.top_p",
                    minimum_exclusive=0.0,
                    maximum=1.0,
                )
                max_tokens = _parse_int(
                    settings_payload.get("max_tokens", self.config.default_max_new_tokens),
                    "settings.max_tokens",
                    minimum=1,
                )
            else:
                raise ValueError("Field 'settings' must be an object if provided")

            image_data = payload.get("image_url")
            if image_data is None:
                image: Optional[pyvips.Image] = None
            elif isinstance(image_data, str):
                image = load_vips_from_base64(image_data)
            else:
                raise ValueError("Field 'image_url' must be a string if provided")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if stream and reasoning:
            return JSONResponse(
                {"error": "Streaming is not supported when reasoning is enabled"},
                status_code=400,
            )

        engine = self.engine
        assert engine is not None
        settings_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if stream:
            try:
                query_stream = await engine.query(
                    image=image,
                    question=question,
                    reasoning=reasoning,
                    stream=True,
                    settings=settings_dict,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                logger.exception("Streaming query failed to start")
                return JSONResponse(
                    {"error": "Inference failed", "detail": str(exc)},
                    status_code=500,
                )

            async def event_generator():
                chunk_buffer: list[str] = []
                tokens_emitted = 0
                try:
                    async for update in query_stream:
                        text = update.text
                        if not text:
                            continue
                        chunk_buffer.append(text)
                        tokens_emitted += 1
                        if tokens_emitted % STREAM_CHUNK_SIZE == 0:
                            payload = {
                                "chunk": "".join(chunk_buffer),
                                "completed": False,
                                "token_index": update.token_index,
                            }
                            chunk_buffer.clear()
                            yield _sse_payload(payload)
                    result = await query_stream.result()
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.exception("Streaming query errored")
                    raise exc

                final_chunk = "".join(chunk_buffer)
                chunk_buffer.clear()
                metrics = _format_metrics(result.metrics)
                payload = {
                    "chunk": final_chunk,
                    "completed": True,
                    "request_id": str(result.request_id),
                    "finish_reason": result.finish_reason,
                    "answer": result.output.get("answer", ""),
                    "metrics": metrics,
                }
                if "reasoning" in result.output:
                    payload["reasoning"] = result.output["reasoning"]
                yield _sse_payload(payload)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=STREAM_HEADERS,
            )

        try:
            result = await engine.query(
                image=image,
                question=question,
                reasoning=reasoning,
                stream=False,
                settings=settings_dict,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )
        metrics = _format_metrics(result.metrics)
        output = result.output
        answer = output.get("answer", "")
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "answer": answer,
            "metrics": metrics,
        }
        if "reasoning" in output:
            response_payload["reasoning"] = output["reasoning"]
        return JSONResponse(response_payload)

    async def handle_point(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            object_name = _parse_required_str(payload, "object")
            settings_payload = payload.get("settings")
            if settings_payload is None:
                settings_payload = {}
            if isinstance(settings_payload, dict):
                max_objects = _parse_int(
                    settings_payload.get("max_objects", 150),
                    "settings.max_objects",
                    minimum=1,
                )
            else:
                raise ValueError("Field 'settings' must be an object if provided")

            image_data = payload.get("image_url")
            if image_data is None:
                image: Optional[pyvips.Image] = None
            elif isinstance(image_data, str):
                image = load_vips_from_base64(image_data)
            else:
                raise ValueError("Field 'image_url' must be a string if provided")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        try:
            engine = self.engine
            assert engine is not None
            result = await engine.point(
                image=image,
                object=object_name,
                settings={"max_objects": max_objects},
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )

        metrics_payload = _format_metrics(result.metrics)
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "points": result.output.get("points"),
            "metrics": metrics_payload,
        }
        return JSONResponse(response_payload)

    async def handle_caption(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            length = payload.get("length", "normal")
            if not isinstance(length, str):
                raise ValueError("Field 'length' must be a string")
            stream = _parse_bool(payload.get("stream", False), "stream")

            settings_payload = payload.get("settings")
            if settings_payload is None:
                settings_payload = {}
            if not isinstance(settings_payload, dict):
                raise ValueError("Field 'settings' must be an object if provided")
            temperature = _parse_float(
                settings_payload.get("temperature", self.config.default_temperature),
                "settings.temperature",
                minimum=0.0,
            )
            top_p = _parse_float(
                settings_payload.get("top_p", self.config.default_top_p),
                "settings.top_p",
                minimum_exclusive=0.0,
                maximum=1.0,
            )
            max_tokens = _parse_int(
                settings_payload.get("max_tokens", self.config.default_max_new_tokens),
                "settings.max_tokens",
                minimum=1,
            )

            image_data = payload.get("image_url")
            if image_data is None:
                raise ValueError("Field 'image_url' must be provided")
            if not isinstance(image_data, str):
                raise ValueError("Field 'image_url' must be a string")
            image = load_vips_from_base64(image_data)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        engine = self.engine
        assert engine is not None
        settings_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if stream:
            try:
                caption_stream = await engine.caption(
                    image=image,
                    length=length,
                    stream=True,
                    settings=settings_dict,
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("Streaming caption failed to start")
                return JSONResponse(
                    {"error": "Inference failed", "detail": str(exc)},
                    status_code=500,
                )

            async def event_generator():
                chunk_buffer: list[str] = []
                tokens_emitted = 0
                try:
                    async for update in caption_stream:
                        text = update.text
                        if not text:
                            continue
                        chunk_buffer.append(text)
                        tokens_emitted += 1
                        if tokens_emitted % STREAM_CHUNK_SIZE == 0:
                            payload = {
                                "chunk": "".join(chunk_buffer),
                                "completed": False,
                                "token_index": update.token_index,
                            }
                            chunk_buffer.clear()
                            yield _sse_payload(payload)
                    result = await caption_stream.result()
                except Exception as exc:  # pragma: no cover
                    logger.exception("Streaming caption errored")
                    raise exc

                final_chunk = "".join(chunk_buffer)
                chunk_buffer.clear()
                metrics = _format_metrics(result.metrics)
                payload = {
                    "chunk": final_chunk,
                    "completed": True,
                    "request_id": str(result.request_id),
                    "finish_reason": result.finish_reason,
                    "caption": result.output.get("caption", ""),
                    "metrics": metrics,
                }
                yield _sse_payload(payload)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=STREAM_HEADERS,
            )

        try:
            result = await engine.caption(
                image=image,
                length=length,
                stream=False,
                settings=settings_dict,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )

        metrics = _format_metrics(result.metrics)
        caption = result.output.get("caption", "")
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "caption": caption,
            "metrics": metrics,
        }
        return JSONResponse(response_payload)

    async def handle_detect(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            object_name = _parse_required_str(payload, "object")
            settings_payload = payload.get("settings")
            if settings_payload is None:
                settings_payload = {}
            if isinstance(settings_payload, dict):
                max_objects = _parse_int(
                    settings_payload.get("max_objects", 150),
                    "settings.max_objects",
                    minimum=1,
                )
            else:
                raise ValueError("Field 'settings' must be an object if provided")

            image_data = payload.get("image_url")
            if image_data is None:
                image: Optional[pyvips.Image] = None
            elif isinstance(image_data, str):
                image = load_vips_from_base64(image_data)
            else:
                raise ValueError("Field 'image_url' must be a string if provided")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        try:
            engine = self.engine
            assert engine is not None
            result = await engine.detect(
                image=image,
                object=object_name,
                settings={"max_objects": max_objects},
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )

        metrics_payload = _format_metrics(result.metrics)
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "objects": result.output.get("objects"),
            "metrics": metrics_payload,
        }
        return JSONResponse(response_payload)

    async def handle_segment(self, request: Request) -> Response:
        if self.engine is None:
            return JSONResponse({"error": "Engine is not ready"}, status_code=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be an object"}, status_code=400)

        try:
            object_name = _parse_required_str(payload, "object")
            stream = _parse_bool(payload.get("stream", False), "stream")
            settings_payload = payload.get("settings")
            if settings_payload is None:
                settings_payload = {}
            if isinstance(settings_payload, dict):
                temperature = _parse_float(
                    settings_payload.get("temperature", 0.0),
                    "settings.temperature",
                    minimum=0.0,
                )
                top_p = _parse_float(
                    settings_payload.get("top_p", 1.0),
                    "settings.top_p",
                    minimum_exclusive=0.0,
                    maximum=1.0,
                )
                max_tokens = _parse_int(
                    settings_payload.get(
                        "max_tokens", self.config.default_max_new_tokens
                    ),
                    "settings.max_tokens",
                    minimum=1,
                )
            else:
                raise ValueError("Field 'settings' must be an object if provided")

            spatial_refs = _parse_spatial_refs(payload.get("spatial_refs"))

            image_data = payload.get("image_url")
            if image_data is None:
                image: Optional[pyvips.Image] = None
            elif isinstance(image_data, str):
                image = load_vips_from_base64(image_data)
            else:
                raise ValueError("Field 'image_url' must be a string if provided")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        settings_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            engine = self.engine
            assert engine is not None
            if stream:
                segment_stream = await engine.submit_streaming(
                    request_context=SegmentRequest(
                        object=object_name,
                        image=image,
                        stream=True,
                        settings=SegmentSettings(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        ),
                        spatial_refs=spatial_refs,
                    ),
                    max_new_tokens=max_tokens,
                    skill="segment",
                    image=image,
                    temperature=temperature,
                    top_p=top_p,
                )

                async def event_generator():
                    latest_bbox = None
                    try:
                        async for update in segment_stream:
                            text = update.text
                            if not text:
                                continue
                            if text.startswith("__BBOX__"):
                                try:
                                    latest_bbox = json.loads(text[len("__BBOX__") :])
                                except Exception:
                                    latest_bbox = None
                                if latest_bbox:
                                    yield _sse_payload({"type": "bbox", "bbox": latest_bbox})
                                continue
                            # Path fragment is a valid suffix; client appends to previous chunks.
                            payload_chunk = {
                                "type": "path_delta",
                                "chunk": text,
                                "completed": False,
                                "token_index": update.token_index,
                            }
                            yield _sse_payload(payload_chunk)
                        result = await segment_stream.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.exception("Streaming segment errored")
                        raise exc

                    segment = (result.output.get("segments") or [{}])[0]
                    bbox_full = segment.get("bbox")
                    bbox_minmax = latest_bbox
                    if bbox_minmax is None and bbox_full and set(bbox_full.keys()) == {
                        "x_center",
                        "y_center",
                        "width",
                        "height",
                        "x_min",
                        "x_max",
                        "y_min",
                        "y_max",
                    }:
                        bbox_minmax = {
                            "x_min": bbox_full["x_min"],
                            "y_min": bbox_full["y_min"],
                            "x_max": bbox_full["x_max"],
                            "y_max": bbox_full["y_max"],
                        }

                    path = segment.get("svg_path") or ""
                    parse_error = segment.get("parse_error")
                    metrics_payload = _format_metrics(result.metrics)
                    payload_final = {
                        "type": "final",
                        "chunk": "",
                        "completed": True,
                        "request_id": str(result.request_id),
                        "finish_reason": result.finish_reason,
                        "path": path,
                        "bbox": bbox_minmax,
                        "metrics": metrics_payload,
                    }
                    if parse_error:
                        payload_final["parse_error"] = parse_error
                    yield _sse_payload(payload_final)

                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers=STREAM_HEADERS,
                )
            else:
                result = await engine.segment(
                    image=image,
                    object=object_name,
                    spatial_refs=spatial_refs,
                    settings=settings_dict,
                )
        except Exception as exc:  # pragma: no cover
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )

        segment = (result.output.get("segments") or [{}])[0]
        bbox = segment.get("bbox")
        path = segment.get("svg_path") or ""
        if bbox and set(bbox.keys()) == {"x_center", "y_center", "width", "height", "x_min", "x_max", "y_min", "y_max"}:
            bbox = {
                "x_min": bbox["x_min"],
                "y_min": bbox["y_min"],
                "x_max": bbox["x_max"],
                "y_max": bbox["y_max"],
            }

        metrics_payload = _format_metrics(result.metrics)
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "bbox": bbox,
            "path": path,
            "metrics": metrics_payload,
        }
        parse_error = segment.get("parse_error")
        if parse_error:
            response_payload["parse_error"] = parse_error

        return JSONResponse(response_payload)




def create_app(
    runtime_cfg: RuntimeConfig,
    *,
    default_max_new_tokens: int = 768,
    default_temperature: float = 0.2,
    default_top_p: float = 0.9,
) -> Starlette:
    """Create a Starlette application bound to the given runtime configuration."""

    if default_temperature < 0.0:
        raise ValueError("default_temperature must be non-negative")
    if not (0.0 < default_top_p <= 1.0):
        raise ValueError("default_top_p must be in the range (0, 1]")
    if default_max_new_tokens <= 0:
        raise ValueError("default_max_new_tokens must be positive")

    config = _ServerConfig(
        runtime_cfg=runtime_cfg,
        default_max_new_tokens=default_max_new_tokens,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
    )
    state = _ServerState(config)

    routes = [
        Route("/v1/query", state.handle_query, methods=["POST"]),
        Route("/v1/point", state.handle_point, methods=["POST"]),
        Route("/v1/detect", state.handle_detect, methods=["POST"]),
        Route("/v1/segment", state.handle_segment, methods=["POST"]),
        Route("/v1/caption", state.handle_caption, methods=["POST"]),
        Route("/healthz", state.handle_health, methods=["GET"]),
    ]

    app = Starlette(routes=routes, on_startup=[state.startup], on_shutdown=[state.shutdown])
    app.state.server_state = state
    return app


def _parse_required_str(payload: Dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Field '{field}' must be a string")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"Field '{field}' must be a non-empty string")
    return trimmed


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Field '{field}' must be a boolean")


def _parse_int(
    value: Any,
    field: str,
    *,
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
    field: str,
    *,
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


def _parse_spatial_refs(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Field 'spatial_refs' must be a list of coordinate lists")
    parsed: List[List[float]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)):
            raise ValueError(
                f"spatial_refs[{idx}] must be a list of floats"
            )
        if len(item) not in (2, 4):
            raise ValueError(
                f"spatial_refs[{idx}] must contain 2 (point) or 4 (bbox) values"
            )
        coord: List[float] = []
        for j, raw in enumerate(item):
            if not isinstance(raw, (int, float)):
                raise ValueError(
                    f"spatial_refs[{idx}][{j}] must be a number"
                )
            value_f = float(raw)
            if value_f < 0.0 or value_f > 1.0:
                raise ValueError(
                    f"spatial_refs[{idx}][{j}] must be within [0, 1]"
                )
            coord.append(value_f)
        if len(coord) == 4:
            x_min, y_min, x_max, y_max = coord
            if x_min > x_max or y_min > y_max:
                raise ValueError(
                    f"spatial_refs[{idx}] must be [x_min, y_min, x_max, y_max] with x_min<=x_max and y_min<=y_max"
                )
        parsed.append(coord)
    return parsed


__all__ = ["create_app"]
