"""ASGI application exposing the Kestrel inference engine over HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pyvips
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.utils.image import load_vips_from_base64

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ServerConfig:
    runtime_cfg: RuntimeConfig
    batch_timeout_s: float
    default_max_new_tokens: int
    default_temperature: float
    default_top_p: float


class _ServerState:
    """Container that owns the inference engine shared by all requests."""

    __slots__ = ("config", "engine", "_decode_executor")

    def __init__(self, config: _ServerConfig) -> None:
        self.config = config
        self.engine: Optional[InferenceEngine] = None
        self._decode_executor: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks

    async def startup(self) -> None:
        if self.engine is not None:
            return
        logger.info("Starting inference engine")
        self.engine = await InferenceEngine.create(
            self.config.runtime_cfg, batch_timeout_s=self.config.batch_timeout_s
        )
        runtime = self.engine.runtime
        max_workers = max(1, getattr(runtime, "max_batch_size", 4))
        if self._decode_executor is not None:
            self._decode_executor.shutdown(wait=True)
        self._decode_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="kestrel-image"
        )
        logger.info("Inference engine ready")

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        logger.info("Shutting down inference engine")
        await self.engine.shutdown()
        self.engine = None
        executor = self._decode_executor
        self._decode_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

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

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return JSONResponse(
                {"error": "Field 'question' must be a non-empty string"},
                status_code=400,
            )
        question = question.strip()

        if payload.get("reasoning"):
            return JSONResponse(
                {"error": "Reasoning is not supported for this endpoint"},
                status_code=400,
            )
        if payload.get("stream"):
            return JSONResponse(
                {"error": "Streaming is not supported for this endpoint"},
                status_code=400,
            )

        image_url = payload.get("image_url")
        image: Optional[pyvips.Image] = None
        if image_url is not None:
            if not isinstance(image_url, str):
                return JSONResponse(
                    {"error": "Field 'image_url' must be a string"}, status_code=400
                )
            try:
                image = await self._decode_image(image_url)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

        settings = payload.get("settings")
        if settings is not None and not isinstance(settings, dict):
            return JSONResponse({"error": "Field 'settings' must be an object"}, status_code=400)
        settings_dict: Dict[str, Any] = settings or {}

        try:
            max_new_tokens = _coerce_int(
                payload.get("max_new_tokens"),
                default=self.config.default_max_new_tokens,
                field_name="max_new_tokens",
                minimum=1,
            )
            temperature = _coerce_float(
                settings_dict.get("temperature"),
                default=self.config.default_temperature,
                field_name="temperature",
                minimum=0.0,
            )
            top_p = _coerce_float(
                settings_dict.get("top_p"),
                default=self.config.default_top_p,
                field_name="top_p",
                minimum_exclusive=0.0,
                maximum=1.0,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        engine = self.engine
        assert engine is not None  # mypy narrow

        start_time = time.perf_counter()
        try:
            result = await engine.submit(
                question,
                max_new_tokens=max_new_tokens,
                image=image,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )
        metrics = result.metrics
        total_latency = time.perf_counter() - start_time
        response_payload = {
            "answer": result.text,
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "metrics": {
                "prompt_tokens": metrics.prompt_tokens,
                "decode_tokens": metrics.decode_tokens,
                "processing_latency_s": metrics.processing_latency_s,
                "ttft_s": metrics.ttft_s,
                "decode_latency_s": metrics.decode_latency_s,
                "decode_tokens_per_s": metrics.decode_tokens_per_s,
                "total_latency_s": total_latency,
            },
        }
        return JSONResponse(response_payload)

    async def _decode_image(self, data: str) -> pyvips.Image:
        """Run base64 decode and colour conversion off the event loop."""

        loop = asyncio.get_running_loop()
        executor = self._decode_executor
        return await loop.run_in_executor(executor, load_vips_from_base64, data)
def _coerce_int(
    value: Any,
    *,
    default: int,
    field_name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if value is None:
        result = default
    elif isinstance(value, bool):  # pragma: no cover - guard against bool subclassing int
        raise ValueError(f"Field '{field_name}' must be an integer")
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Field '{field_name}' must be an integer")
        result = int(value)
    else:
        raise ValueError(f"Field '{field_name}' must be an integer")

    if minimum is not None and result < minimum:
        raise ValueError(
            f"Field '{field_name}' must be >= {minimum}; received {result}"
        )
    if maximum is not None and result > maximum:
        raise ValueError(
            f"Field '{field_name}' must be <= {maximum}; received {result}"
        )
    return result


def _coerce_float(
    value: Any,
    *,
    default: float,
    field_name: str,
    minimum: Optional[float] = None,
    minimum_exclusive: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if value is None:
        result = default
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise ValueError(f"Field '{field_name}' must be a number")

    if minimum is not None and result < minimum:
        raise ValueError(
            f"Field '{field_name}' must be >= {minimum}; received {result}"
        )
    if minimum_exclusive is not None and result <= minimum_exclusive:
        raise ValueError(
            f"Field '{field_name}' must be > {minimum_exclusive}; received {result}"
        )
    if maximum is not None and result > maximum:
        raise ValueError(
            f"Field '{field_name}' must be <= {maximum}; received {result}"
        )
    return result


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
        Route("/healthz", state.handle_health, methods=["GET"]),
    ]

    app = Starlette(routes=routes, on_startup=[state.startup], on_shutdown=[state.shutdown])
    app.state.server_state = state
    return app


__all__ = ["create_app"]
