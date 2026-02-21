"""Triton Python backend model for Kestrel inference.

A single ``moondream`` Triton model handles all skills (query, caption,
detect, point).  The skill is selected per-request via the ``SKILL`` input
tensor.

The model creates one :class:`InferenceEngine` via :mod:`engine_singleton`
and dispatches work onto its asyncio event loop with
:func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Triton Python backend utilities — available at runtime inside the Triton
# container but not locally, so we import lazily to keep the module importable
# in non-Triton environments.
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None  # type: ignore[assignment]

STREAM_CHUNK_SIZE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_metrics(metrics: Any) -> Dict[str, float]:
    return {
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "prefill_time_ms": metrics.prefill_time_ms,
        "decode_time_ms": metrics.decode_time_ms,
        "ttft_ms": metrics.ttft_ms,
    }


def _get_optional_string(request: Any, name: str) -> Optional[str]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return tensor.as_numpy()[0].decode("utf-8")


def _get_required_string(request: Any, name: str) -> str:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise ValueError(f"Required input '{name}' is missing")
    value = tensor.as_numpy()[0].decode("utf-8")
    if not value.strip():
        raise ValueError(f"Input '{name}' must not be empty")
    return value


def _get_optional_bytes(request: Any, name: str) -> Optional[bytes]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return bytes(tensor.as_numpy()[0])


def _get_optional_float(request: Any, name: str) -> Optional[float]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return float(tensor.as_numpy()[0])


def _get_optional_int(request: Any, name: str) -> Optional[int]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return int(tensor.as_numpy()[0])


def _get_optional_bool(request: Any, name: str) -> Optional[bool]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    return bool(tensor.as_numpy()[0])


def _get_spatial_refs(request: Any) -> Optional[Sequence[Sequence[float]]]:
    tensor = pb_utils.get_input_tensor_by_name(request, "SPATIAL_REFS")
    if tensor is None:
        return None
    arr = tensor.as_numpy()
    if arr.size == 0:
        return None
    return arr.tolist()


def _text_response(data: Dict[str, Any]) -> Any:
    """Build an InferenceResponse with a TEXT_OUTPUT string tensor."""
    out_tensor = pb_utils.Tensor(
        "TEXT_OUTPUT",
        np.array([json.dumps(data)], dtype=np.object_),
    )
    return pb_utils.InferenceResponse(output_tensors=[out_tensor])


def _error_response(message: str) -> Any:
    """Build an error InferenceResponse."""
    return pb_utils.InferenceResponse(
        output_tensors=[],
        error=pb_utils.TritonError(message),
    )


_FINAL = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if pb_utils else 0


def _send(sender: Any, response: Any, final: bool = True) -> None:
    """Send a response, with COMPLETE_FINAL flag when *final* is True."""
    if final:
        sender.send(response, flags=_FINAL)
    else:
        sender.send(response)


# ---------------------------------------------------------------------------
# TritonPythonModel
# ---------------------------------------------------------------------------

class TritonPythonModel:
    """Triton Python backend model for Moondream."""

    def initialize(self, args: Dict[str, str]) -> None:
        from engine_singleton import get_or_create_engine

        model_name: str = args["model_name"]
        logger.info("Initialising Triton model %s", model_name)

        self.engine = get_or_create_engine()

    def execute(self, requests: List[Any]) -> None:
        """Dispatch each request on its own thread (decoupled mode)."""
        for request in requests:
            thread = threading.Thread(
                target=self._handle_request, args=(request,), daemon=True
            )
            thread.start()
        return None

    def finalize(self) -> None:
        from engine_singleton import release_engine

        release_engine()

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle_request(self, request: Any) -> None:
        response_sender = request.get_response_sender()
        try:
            self._dispatch(request, response_sender)
        except Exception as exc:
            logger.exception("Request failed")
            try:
                _send(response_sender, _error_response(str(exc)))
            except Exception:
                logger.exception("Failed to send error response")

    def _dispatch(self, request: Any, response_sender: Any) -> None:
        from engine_singleton import get_event_loop

        loop = get_event_loop()
        skill = _get_required_string(request, "SKILL").lower()

        if skill == "query":
            self._handle_query(request, response_sender, loop)
        elif skill == "caption":
            self._handle_caption(request, response_sender, loop)
        elif skill == "detect":
            self._handle_detect(request, response_sender, loop)
        elif skill == "point":
            self._handle_point(request, response_sender, loop)
        else:
            _send(response_sender, _error_response(f"Unknown skill: {skill}"))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _handle_query(self, request: Any, sender: Any, loop: asyncio.AbstractEventLoop) -> None:
        question = _get_required_string(request, "QUESTION")
        image = _get_optional_bytes(request, "IMAGE")
        temperature = _get_optional_float(request, "TEMPERATURE")
        top_p = _get_optional_float(request, "TOP_P")
        max_tokens = _get_optional_int(request, "MAX_TOKENS")
        reasoning = _get_optional_bool(request, "REASONING") or False
        stream = _get_optional_bool(request, "STREAM") or False
        spatial_refs = _get_spatial_refs(request)

        settings: Dict[str, Any] = {}
        if temperature is not None:
            settings["temperature"] = temperature
        if top_p is not None:
            settings["top_p"] = top_p
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens

        if stream and reasoning:
            _send(sender, _error_response(
                "Streaming is not supported when reasoning is enabled"
            ))
            return

        if stream:
            future = asyncio.run_coroutine_threadsafe(
                self.engine.query(
                    image=image,
                    question=question,
                    reasoning=reasoning,
                    spatial_refs=spatial_refs,
                    stream=True,
                    settings=settings or None,
                ),
                loop,
            )
            query_stream = future.result()

            async def _consume():
                chunk_buffer: list[str] = []
                tokens_emitted = 0
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
                        _send(sender, _text_response(payload), final=False)

                result = await query_stream.result()
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
                _send(sender, _text_response(payload))

            asyncio.run_coroutine_threadsafe(_consume(), loop).result()
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.engine.query(
                    image=image,
                    question=question,
                    reasoning=reasoning,
                    spatial_refs=spatial_refs,
                    stream=False,
                    settings=settings or None,
                ),
                loop,
            )
            result = future.result()
            metrics = _format_metrics(result.metrics)
            payload: Dict[str, Any] = {
                "request_id": str(result.request_id),
                "finish_reason": result.finish_reason,
                "answer": result.output.get("answer", ""),
                "metrics": metrics,
            }
            if "reasoning" in result.output:
                payload["reasoning"] = result.output["reasoning"]
            _send(sender, _text_response(payload))

    # ------------------------------------------------------------------
    # Caption
    # ------------------------------------------------------------------

    def _handle_caption(self, request: Any, sender: Any, loop: asyncio.AbstractEventLoop) -> None:
        image = _get_optional_bytes(request, "IMAGE")
        if image is None:
            _send(sender, _error_response("Required input 'IMAGE' is missing"))
            return
        length = _get_optional_string(request, "LENGTH") or "normal"
        temperature = _get_optional_float(request, "TEMPERATURE")
        top_p = _get_optional_float(request, "TOP_P")
        max_tokens = _get_optional_int(request, "MAX_TOKENS")
        stream = _get_optional_bool(request, "STREAM") or False

        settings: Dict[str, Any] = {}
        if temperature is not None:
            settings["temperature"] = temperature
        if top_p is not None:
            settings["top_p"] = top_p
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens

        if stream:
            future = asyncio.run_coroutine_threadsafe(
                self.engine.caption(
                    image=image,
                    length=length,
                    stream=True,
                    settings=settings or None,
                ),
                loop,
            )
            caption_stream = future.result()

            async def _consume():
                chunk_buffer: list[str] = []
                tokens_emitted = 0
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
                        _send(sender, _text_response(payload), final=False)

                result = await caption_stream.result()
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
                _send(sender, _text_response(payload))

            asyncio.run_coroutine_threadsafe(_consume(), loop).result()
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.engine.caption(
                    image=image,
                    length=length,
                    stream=False,
                    settings=settings or None,
                ),
                loop,
            )
            result = future.result()
            metrics = _format_metrics(result.metrics)
            payload = {
                "request_id": str(result.request_id),
                "finish_reason": result.finish_reason,
                "caption": result.output.get("caption", ""),
                "metrics": metrics,
            }
            _send(sender, _text_response(payload))

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def _handle_detect(self, request: Any, sender: Any, loop: asyncio.AbstractEventLoop) -> None:
        object_name = _get_required_string(request, "OBJECT")
        image = _get_optional_bytes(request, "IMAGE")
        max_objects = _get_optional_int(request, "MAX_OBJECTS") or 150

        settings: Dict[str, Any] = {"max_objects": max_objects}

        future = asyncio.run_coroutine_threadsafe(
            self.engine.detect(
                image=image,
                object=object_name,
                settings=settings,
            ),
            loop,
        )
        result = future.result()
        metrics = _format_metrics(result.metrics)
        payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "objects": result.output.get("objects"),
            "metrics": metrics,
        }
        _send(sender, _text_response(payload))

    # ------------------------------------------------------------------
    # Point
    # ------------------------------------------------------------------

    def _handle_point(self, request: Any, sender: Any, loop: asyncio.AbstractEventLoop) -> None:
        object_name = _get_required_string(request, "OBJECT")
        image = _get_optional_bytes(request, "IMAGE")
        max_objects = _get_optional_int(request, "MAX_OBJECTS") or 150

        settings: Dict[str, Any] = {"max_objects": max_objects}

        future = asyncio.run_coroutine_threadsafe(
            self.engine.point(
                image=image,
                object=object_name,
                settings=settings,
            ),
            loop,
        )
        result = future.result()
        metrics = _format_metrics(result.metrics)
        payload = {
            "request_id": str(result.request_id),
            "finish_reason": result.finish_reason,
            "points": result.output.get("points"),
            "metrics": metrics,
        }
        _send(sender, _text_response(payload))
