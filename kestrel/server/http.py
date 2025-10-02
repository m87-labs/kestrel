"""ASGI application exposing the Kestrel inference engine over HTTP."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pyvips
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from kestrel.config import RuntimeConfig
from kestrel.engine import EngineMetrics, InferenceEngine
from kestrel.utils.image import load_vips_from_base64

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
    batch_timeout_s: float
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
        self.engine = await InferenceEngine.create(
            self.config.runtime_cfg, batch_timeout_s=self.config.batch_timeout_s
        )
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




def create_app(
    runtime_cfg: RuntimeConfig,
    *,
    batch_timeout_s: float = 0.02,
    default_max_new_tokens: int = 64,
    default_temperature: float = 0.0,
    default_top_p: float = 1.0,
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
        batch_timeout_s=batch_timeout_s,
        default_max_new_tokens=default_max_new_tokens,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
    )
    state = _ServerState(config)

    routes = [
        Route("/v1/query", state.handle_query, methods=["POST"]),
        Route("/v1/point", state.handle_point, methods=["POST"]),
        Route("/v1/detect", state.handle_detect, methods=["POST"]),
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


__all__ = ["create_app"]
