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
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.utils.image import load_vips_from_base64
from kestrel.skills.query import QueryDefaults, QuerySkill

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

        engine = self.engine
        assert engine is not None  # mypy narrow

        skill = engine.skills.resolve("query")
        assert isinstance(skill, QuerySkill)

        defaults = QueryDefaults(
            max_new_tokens=self.config.default_max_new_tokens,
            temperature=self.config.default_temperature,
            top_p=self.config.default_top_p,
        )

        try:
            invocation = skill.parse_http_payload(
                payload,
                image_decoder=load_vips_from_base64,
                defaults=defaults,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        start_time = time.perf_counter()
        try:
            result = await engine.query(
                invocation.request,
                max_new_tokens=invocation.max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Inference request failed")
            return JSONResponse(
                {"error": "Inference failed", "detail": str(exc)}, status_code=500
            )
        metrics = result.metrics
        total_latency = time.perf_counter() - start_time
        extras = result.extras
        answer = extras.get("answer", result.text)
        response_payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "answer": answer,
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
        if "reasoning" in extras:
            response_payload["reasoning"] = extras["reasoning"]
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
        Route("/healthz", state.handle_health, methods=["GET"]),
    ]

    app = Starlette(routes=routes, on_startup=[state.startup], on_shutdown=[state.shutdown])
    app.state.server_state = state
    return app


__all__ = ["create_app"]
