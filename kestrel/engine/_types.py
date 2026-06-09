"""Value types exchanged across the engine package.

Plain data — request/result containers, the streaming iterator, and the
executor handoff values (:class:`Completion`, :class:`TickResult`). Kept
in their own module so the executor and the kernel core can share them
without an import cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

import numpy as np

from kestrel.scheduler import GeneratedPrefix, StreamUpdate
from kestrel.models.moondream.runtime import Token
from kestrel.skills import SkillSpec, SkillState

@dataclass(slots=True)
class EngineMetrics:
    """Token counts and timing for a single request."""

    input_tokens: int
    output_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    ttft_ms: float
    cached_tokens: int = 0  # KV positions reused from prefix cache


@dataclass(slots=True)
class EngineResult:
    """Inference output returned to callers."""

    request_id: int
    tokens: List[Token]
    finish_reason: str
    metrics: EngineMetrics
    output: Dict[str, object]
    logprobs: Optional[List[float]] = None


@dataclass(slots=True)
class _StreamCompletion:
    result: Optional[EngineResult] = None
    error: Optional[BaseException] = None


_StreamQueueItem = Union[StreamUpdate, _StreamCompletion]
_StreamQueue = asyncio.Queue[_StreamQueueItem]


class EngineStream(AsyncIterator[StreamUpdate]):
    """Asynchronous iterator that yields incremental generation updates."""

    __slots__ = (
        "request_id",
        "_queue",
        "_result_future",
        "_final_result",
        "_error",
    )

    def __init__(
        self,
        request_id: int,
        queue: _StreamQueue,
        result_future: asyncio.Future[EngineResult],
    ) -> None:
        self.request_id = request_id
        self._queue = queue
        self._result_future = result_future
        self._final_result: Optional[EngineResult] = None
        self._error: Optional[BaseException] = None

    def __aiter__(self) -> "EngineStream":
        return self

    async def __anext__(self) -> StreamUpdate:
        while True:
            item = await self._queue.get()
            if isinstance(item, _StreamCompletion):
                if item.error is not None:
                    self._error = item.error
                    raise item.error
                if item.result is not None:
                    self._final_result = item.result
                raise StopAsyncIteration
            return item

    async def result(self) -> EngineResult:
        if self._final_result is not None:
            return self._final_result
        if self._error is not None:
            raise self._error
        result = await self._result_future
        self._final_result = result
        return result


@dataclass(slots=True)
class ModelStreamUpdate:
    """One model-defined update from a stateful streaming session."""

    session_id: int
    task: str
    output: Dict[str, object]


@dataclass(slots=True)
class _ModelStreamCompletion:
    result: Optional[EngineResult] = None
    error: Optional[BaseException] = None


_ModelStreamQueueItem = Union[ModelStreamUpdate, _ModelStreamCompletion]
_ModelStreamQueue = asyncio.Queue[_ModelStreamQueueItem]
_SendModelStreamChunk = Callable[[int, Dict[str, Any]], Awaitable[None]]
_CloseModelStream = Callable[[int], Awaitable[None]]


class ModelStream(AsyncIterator[ModelStreamUpdate]):
    """Asynchronous session for stateful model streaming.

    Unlike :class:`EngineStream`, which streams token deltas for one
    autoregressive request, ``ModelStream`` is caller-driven: the caller
    sends chunks/frames into a model-owned session and iterates over
    model-defined updates.
    """

    __slots__ = (
        "session_id",
        "task",
        "_queue",
        "_result_future",
        "_send_chunk",
        "_close_session",
        "_final_result",
        "_error",
        "_closed",
    )

    def __init__(
        self,
        *,
        session_id: int,
        task: str,
        queue: _ModelStreamQueue,
        result_future: asyncio.Future[EngineResult],
        send_chunk: _SendModelStreamChunk,
        close_session: _CloseModelStream,
    ) -> None:
        self.session_id = session_id
        self.task = task
        self._queue = queue
        self._result_future = result_future
        self._send_chunk = send_chunk
        self._close_session = close_session
        self._final_result: Optional[EngineResult] = None
        self._error: Optional[BaseException] = None
        self._closed = False

    async def send(self, **chunk: Any) -> None:
        """Append one model-defined chunk/frame to the session."""
        if self._closed:
            raise RuntimeError("model stream is closed")
        await self._send_chunk(self.session_id, dict(chunk))

    def updates(self) -> "ModelStream":
        """Return the async update iterator."""
        return self

    def __aiter__(self) -> "ModelStream":
        return self

    async def __anext__(self) -> ModelStreamUpdate:
        while True:
            item = await self._queue.get()
            if isinstance(item, _ModelStreamCompletion):
                if item.error is not None:
                    self._error = item.error
                    raise item.error
                if item.result is not None:
                    self._final_result = item.result
                raise StopAsyncIteration
            return item

    async def close(self) -> EngineResult:
        """Close the session and return its final result."""
        if not self._closed:
            self._closed = True
            await self._close_session(self.session_id)
        return await self.result()

    async def result(self) -> EngineResult:
        if self._final_result is not None:
            return self._final_result
        if self._error is not None:
            raise self._error
        result = await self._result_future
        self._final_result = result
        return result

    async def __aenter__(self) -> "ModelStream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            await self.close()
        elif not self._closed:
            self._closed = True
            await self._close_session(self.session_id)


@dataclass(slots=True)
class _AutoregressiveRequest:
    request_id: int
    prompt: str
    prompt_tokens: Sequence[Token]
    image: Optional[np.ndarray | bytes]
    image_hash: Optional[bytes]  # SHA256 hash for prefix caching
    max_new_tokens: int
    temperature: float
    top_p: float
    submitted_at: float
    future: asyncio.Future[EngineResult]
    stream_queue: Optional["_StreamQueue"]
    skill: SkillSpec
    request_context: object
    adapter: Optional[str] = None
    lora_slot: int = 0  # Always 0 here; scheduler assigns actual slot at admission
    return_logprobs: Optional[bool] = None
    generated_prefix: GeneratedPrefix = field(default_factory=GeneratedPrefix)
    suppress_next_token_ids: Optional[tuple[int, ...]] = None


@dataclass(slots=True)
class _ReadyAdmission:
    req: _AutoregressiveRequest
    crops: Any
    prefix_cache_hit: bool


def _hash_image(image: np.ndarray | bytes) -> bytes:
    """SHA-256 over the raw image input for prefix-cache keying."""
    raw = image.tobytes() if isinstance(image, np.ndarray) else image
    return hashlib.sha256(raw).digest()


class EngineRequest(Protocol):
    """The envelope the kernel needs to return an answer to a caller.

    Every execution shape submits a concrete request carrying its own
    lane-specific payload (``_AutoregressiveRequest`` holds prompt/tokens/skill
    for the autoregressive lane; the single-pass lane holds task/inputs).
    What the kernel's delivery path actually touches is only this common
    envelope: the identity, the future to resolve, the optional stream
    sink, and the finetune id. Typing ``Completion.request`` as this
    protocol keeps the kernel's return path independent of any one lane's
    request type — a new lane defines its own request satisfying this and
    nothing in delivery changes.

    ``adapter`` is the finetune id (``None`` when no finetune is
    selected); it is reported to telemetry and, for lanes that support
    finetunes, selects the weights.
    """

    request_id: int
    future: "asyncio.Future[EngineResult]"
    stream_queue: "Optional[_StreamQueue]"
    adapter: Optional[str]


@dataclass(slots=True)
class Completion:
    """A terminal result from an executor, for the kernel to deliver.

    Executors emit these as plain values and never touch the event loop
    or telemetry; the kernel maps each ``Completion`` to its effects
    (resolve the request future, record usage, finish the stream). This
    keeps executors pure-compute and directly testable.

    Exactly one of ``result`` / ``error`` is set.
    """

    request: EngineRequest
    result: Optional[EngineResult] = None
    error: Optional[BaseException] = None


@dataclass(frozen=True, slots=True)
class TickResult:
    """Immutable summary of one executor ``advance`` step."""

    progressed: bool = False
    completed: tuple[Completion, ...] = ()
    has_work: bool = False  # queued or in flight (gates shutdown exit)

