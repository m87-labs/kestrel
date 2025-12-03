"""Async coordination layer for Moondream inference.

The engine is the high-level entry point for clients. It owns:

- Lifecycle of the shared :class:`~kestrel.moondream.runtime.MoondreamRuntime`, including warmup and shutdown.
- A micro-batching worker that pulls pending requests, prepares image crops, and runs the scheduler.
- Skill orchestration — resolving the active :class:`~kestrel.skills.base.SkillSpec`, building prompt tokens when necessary, instantiating :class:`~kestrel.skills.base.SkillState` with skill-specific request contexts, and bridging streaming callbacks back to callers.
- Conversion between scheduler outputs (``SchedulerResult``) and user-facing ``EngineResult`` objects augmented with metrics and per-skill output payloads.

Relationship to other components:

- Receives raw prompts or structured skill requests from clients (CLI, HTTP, etc.).
- Uses :class:`GenerationScheduler` to multiplex work across the runtime while keeping the scheduler skill-agnostic.
- Delegates low-level execution to :class:`MoondreamRuntime` for prefill/decode and to :mod:`kestrel.moondream.vision` for optional image preprocessing.

Internal API overview:

- :meth:`InferenceEngine.create` / :meth:`InferenceEngine.shutdown`: manage runtime instantiation and cleanup.
- :meth:`InferenceEngine.submit` / :meth:`InferenceEngine.submit_streaming`: enqueue non-streaming or streaming requests.
- :meth:`InferenceEngine.query`: helper that mirrors ``moondream.query`` while internally materialising the skill request context.
- `_submit_request`: normalises parameters, resolves the skill, builds prompt tokens, and stashes the per-request context so the scheduler receives a fully initialised ``SkillState``.
- `_worker_loop`: background task that batches queued requests, invokes the scheduler, and delivers results or stream completions back to callers.

Callers provide raw questions/objects; the engine derives skill-specific contexts and validation before handing work to the scheduler.
"""


import asyncio
import itertools
import logging
import math
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
    Literal,
)

import numpy as np
import torch
import pyvips

from kestrel.config import RuntimeConfig
from kestrel.moondream.runtime import MoondreamRuntime
from kestrel.scheduler import (
    GenerationScheduler,
    GenerationRequest,
    SchedulerResult,
    StreamUpdate,
)
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.moondream.vision import compute_overlap_crops
from kestrel.skills import (
    CaptionSkill,
    DetectSkill,
    PointSkill,
    QuerySkill,
    SegmentSkill,
    SkillRegistry,
    SkillSpec,
    SkillState,
)
from kestrel.moondream.runtime import Token
from kestrel.skills.caption import CaptionRequest, CaptionSettings
from kestrel.skills.detect import DetectRequest, DetectSettings
from kestrel.skills.point import PointRequest, PointSettings
from kestrel.skills.query import QueryRequest, QuerySettings
from kestrel.skills.segment import SegmentRequest, SegmentSettings
from kestrel.moondream.lora import LoRA


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineMetrics:
    """Token counts and timing for a single request."""

    input_tokens: int
    output_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    ttft_ms: float


@dataclass(slots=True)
class EngineResult:
    """Inference output returned to callers."""

    request_id: int
    tokens: List[Token]
    finish_reason: str
    metrics: EngineMetrics
    output: Dict[str, object]


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
class _PendingRequest:
    request_id: int
    prompt: str
    prompt_tokens: torch.Tensor
    prompt_length: int
    image: Optional[pyvips.Image | np.ndarray]
    max_new_tokens: int
    temperature: float
    top_p: float
    submitted_at: float
    future: asyncio.Future[EngineResult]
    stream_queue: Optional["_StreamQueue"]
    skill: SkillSpec
    request_context: object


class InferenceEngine:
    """Orchestrates batched inference over a shared runtime and scheduler."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        *,
        skills: Optional[SkillRegistry] = None,
        lora: Optional[LoRA] = None,
    ) -> None:
        self._runtime_cfg = runtime_cfg
        self._lora = lora

        self._runtime: Optional[MoondreamRuntime] = None
        self._queue: asyncio.Queue[Optional[_PendingRequest]] = asyncio.Queue()
        self._scheduler_queue: queue.Queue[_PendingRequest | None] = queue.Queue()
        self._scheduler_event = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()  # set == running
        self._paused_flag = threading.Event()  # set == paused
        self._paused_event = threading.Event()  # acknowledgment for callers
        self._scheduler_thread: Optional[threading.Thread] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._request_ids = itertools.count()
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._image_executor: Optional[ThreadPoolExecutor] = None
        self._skills = skills or SkillRegistry(
            [
                QuerySkill(),
                PointSkill(),
                DetectSkill(),
                CaptionSkill(),
                SegmentSkill(),
            ]
        )
        self._default_max_new_tokens = 768
        self._default_temperature = 0.2
        self._default_top_p = 0.9

    @property
    def runtime(self) -> MoondreamRuntime:
        if self._runtime is None:
            raise RuntimeError("InferenceEngine has not been started")
        return self._runtime

    @property
    def skills(self) -> SkillRegistry:
        return self._skills

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def is_paused(self) -> bool:
        """Return True when the scheduler loop is currently paused."""

        return self._paused_flag.is_set()

    @classmethod
    async def create(
        cls,
        runtime_cfg: RuntimeConfig,
        *,
        skills: Optional[SkillRegistry] = None,
        lora: Optional[LoRA] = None,
    ) -> "InferenceEngine":
        engine = cls(runtime_cfg, skills=skills, lora=lora)
        await engine._initialize()
        return engine

    async def _initialize(self) -> None:
        if self._runtime is not None:
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._runtime = await loop.run_in_executor(
            None, lambda: MoondreamRuntime(self._runtime_cfg, lora=self._lora)
        )
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True)
        max_workers = max(1, self._runtime.max_batch_size)
        self._image_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="kestrel-img"
        )
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="kestrel-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        self._worker_task = asyncio.create_task(self._worker_loop())
        await self._warmup_query_pipeline()

    async def _warmup_query_pipeline(self) -> None:
        """Ensure the high-level query path is exercised before serving traffic."""

        try:
            await self.query(
                image=None,
                question="Warmup prompt.",
                reasoning=False,
                stream=False,
                settings={
                    "temperature": self._default_temperature,
                    "top_p": self._default_top_p,
                    "max_tokens": 1,
                },
            )
        except Exception:
            _LOGGER.exception("Warmup query pipeline failed")
            raise

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self._queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
        self._worker_task = None
        if self._scheduler_thread is not None:
            self._scheduler_event.set()
            self._scheduler_thread.join()
            self._scheduler_thread = None
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True)
            self._image_executor = None

    async def submit(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        image: Optional[pyvips.Image | np.ndarray] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> EngineResult:
        future, _ = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            image=image,
            temperature=temperature,
            top_p=top_p,
            stream_queue=None,
            skill=skill,
        )
        return await future

    @overload
    async def query(
        self,
        image: Optional[pyvips.Image | np.ndarray] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[True] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream": ...

    @overload
    async def query(
        self,
        image: Optional[pyvips.Image | np.ndarray] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult: ...

    @overload
    async def query(
        self,
        image: Optional[pyvips.Image | np.ndarray] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]: ...

    async def query(
        self,
        image: Optional[pyvips.Image | np.ndarray] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        if question is None:
            raise ValueError("question must be provided")
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must be a non-empty string")
        if spatial_refs is not None:
            raise ValueError("spatial_refs are not supported")

        temperature = self._default_temperature
        top_p = self._default_top_p
        max_tokens = self._default_max_new_tokens
        adapter: Optional[LoRA] = None
        if settings is not None:
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])
            if "max_tokens" in settings:
                max_tokens = int(settings["max_tokens"])
            if "adapter" in settings:
                maybe_adapter = settings["adapter"]
                if maybe_adapter is not None and not isinstance(maybe_adapter, LoRA):
                    raise TypeError("adapter must be a LoRA instance or None")
                adapter = maybe_adapter

        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        request = QueryRequest(
            question=normalized_question,
            image=image,
            reasoning=reasoning,
            stream=stream,
            settings=QuerySettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                adapter=adapter,
            ),
        )
        if stream:
            return await self.submit_streaming(
                request,
                max_new_tokens=max_tokens,
                image=image,
                temperature=temperature,
                top_p=top_p,
                skill="query",
            )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            skill="query",
        )

    async def point(
        self,
        image: Optional[pyvips.Image | np.ndarray],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
    ) -> EngineResult:
        normalized_object = object.strip()
        if not normalized_object:
            raise ValueError("object must be a non-empty string")

        max_tokens = self._default_max_new_tokens
        max_objects = None
        temperature = 0.0
        top_p = 1.0
        if settings is not None:
            if "max_objects" in settings:
                max_objects = max(1, int(settings["max_objects"]))
            if "max_tokens" in settings:
                max_tokens = int(settings["max_tokens"])
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])

        if max_objects is not None:
            max_tokens = max(2 * max_objects + 1, 2)

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        request = PointRequest(
            object=normalized_object,
            image=image,
            stream=False,
            settings=PointSettings(temperature=temperature, top_p=top_p),
        )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            skill="point",
        )

    @overload
    async def caption(
        self,
        image: pyvips.Image | np.ndarray,
        *,
        length: str = ...,
        stream: Literal[True],
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream": ...

    @overload
    async def caption(
        self,
        image: pyvips.Image | np.ndarray,
        *,
        length: str = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult: ...

    @overload
    async def caption(
        self,
        image: pyvips.Image | np.ndarray,
        *,
        length: str = ...,
        stream: bool = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> Union[EngineResult, EngineStream]: ...

    async def caption(
        self,
        image: pyvips.Image | np.ndarray,
        *,
        length: str = "normal",
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        if image is None:
            raise ValueError("image must be provided for captioning")
        normalized_length = length.strip().lower() or "normal"
        if normalized_length not in CaptionSkill.VALID_LENGTHS:
            valid = ", ".join(sorted(CaptionSkill.VALID_LENGTHS))
            raise ValueError(f"length must be one of: {valid}")

        temperature = self._default_temperature
        top_p = self._default_top_p
        max_tokens = self._default_max_new_tokens
        if settings is not None:
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])
            if "max_tokens" in settings:
                max_tokens = int(settings["max_tokens"])

        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        request = CaptionRequest(
            length=normalized_length,
            image=image,
            stream=stream,
            settings=CaptionSettings(temperature=temperature, top_p=top_p),
        )
        if stream:
            return await self.submit_streaming(
                request,
                max_new_tokens=max_tokens,
                image=image,
                temperature=temperature,
                top_p=top_p,
                skill="caption",
            )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            skill="caption",
        )

    async def detect(
        self,
        image: Optional[pyvips.Image | np.ndarray],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
    ) -> EngineResult:
        normalized_object = object.strip()
        if not normalized_object:
            raise ValueError("object must be a non-empty string")

        max_objects = 150
        temperature = 0.0
        top_p = 1.0
        if settings is not None:
            if "max_objects" in settings:
                max_objects = max(1, int(settings["max_objects"]))
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])

        # Each object consumes up to 3 tokens (x, y, size); allow one extra for EOS.
        max_tokens = max(3 * max_objects + 1, 3)

        request = DetectRequest(
            object=normalized_object,
            image=image,
            stream=False,
            settings=DetectSettings(temperature=temperature, top_p=top_p),
            max_objects=max_objects,
        )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            skill="detect",
        )

    async def segment(
        self,
        image: Optional[pyvips.Image | np.ndarray],
        object: str,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        settings: Optional[Mapping[str, object]] = None,
    ) -> EngineResult:
        normalized_object = object.strip()
        if not normalized_object:
            raise ValueError("object must be a non-empty string")

        temperature = 0.0
        top_p = 1.0
        max_tokens = self._default_max_new_tokens
        if settings is not None:
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])
            if "max_tokens" in settings:
                max_tokens = int(settings["max_tokens"])

        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        normalized_refs: Optional[List[Tuple[float, ...]]] = None
        if spatial_refs is not None:
            normalized_refs = []
            for idx, ref in enumerate(spatial_refs):
                if len(ref) not in (2, 4):
                    raise ValueError(
                        f"spatial_refs[{idx}] must contain 2 (point) or 4 (bbox) values"
                    )
                converted = [float(value) for value in ref]
                if not all(math.isfinite(value) for value in converted):
                    raise ValueError(
                        f"spatial_refs[{idx}] contains non-finite values"
                    )
                if not all(0.0 <= value <= 1.0 for value in converted):
                    raise ValueError(
                        f"spatial_refs[{idx}] values must be normalised to [0, 1]"
                    )
                if len(converted) == 4:
                    x_min, y_min, x_max, y_max = converted
                    if x_min > x_max or y_min > y_max:
                        raise ValueError(
                            f"spatial_refs[{idx}] bbox must satisfy x_min<=x_max and y_min<=y_max"
                        )
                normalized_refs.append(tuple(converted))

        request = SegmentRequest(
            object=normalized_object,
            image=image,
            stream=False,
            settings=SegmentSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
            spatial_refs=tuple(normalized_refs) if normalized_refs else None,
        )

        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            skill="segment",
        )

    async def submit_streaming(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        image: Optional[pyvips.Image | np.ndarray] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> EngineStream:
        queue: _StreamQueue = asyncio.Queue()
        future, request_id = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            image=image,
            temperature=temperature,
            top_p=top_p,
            stream_queue=queue,
            skill=skill,
        )
        return EngineStream(request_id=request_id, queue=queue, result_future=future)

    # ------------------------------------------------------------------
    # Control APIs

    def pause(self, *, timeout: Optional[float] = None) -> None:
        """Pause scheduler progress and wait until GPU work is drained.

        In-flight sequences remain allocated; new work is not admitted until
        ``resume`` is called. Returns only after the scheduler loop acknowledges
        the pause (or ``timeout`` elapses).
        """

        if self._shutdown:
            return
        self._run_gate.clear()
        self._paused_flag.set()
        self._paused_event.clear()
        self._scheduler_event.set()
        self._paused_event.wait(timeout)

    def resume(self) -> None:
        """Resume scheduler progress after a pause."""

        if self._shutdown:
            return
        self._paused_event.clear()
        self._paused_flag.clear()
        self._run_gate.set()
        self._scheduler_event.set()

    async def _submit_request(
        self,
        *,
        max_new_tokens: int,
        request_context: object,
        image: Optional[pyvips.Image | np.ndarray],
        temperature: Optional[float],
        top_p: Optional[float],
        stream_queue: Optional[_StreamQueue],
        skill: str,
    ) -> Tuple[asyncio.Future[EngineResult], int]:
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        await self._ensure_started()

        loop = asyncio.get_running_loop()
        req_id = next(self._request_ids)
        future: asyncio.Future[EngineResult] = loop.create_future()

        skill_spec = self._skills.resolve(skill)

        image_obj: Optional[pyvips.Image | np.ndarray] = None
        if image is not None:
            if self.runtime.image_prefix_length == 0:
                raise ValueError("Runtime does not support image inputs")
            if not isinstance(image, (pyvips.Image, np.ndarray)):
                raise TypeError("image must be a pyvips.Image or np.ndarray")
            image_obj = image

        prompt_str = self._extract_prompt_text(skill_spec, request_context)
        tokens = skill_spec.build_prompt_tokens(self.runtime, request_context)

        tokens_cpu = tokens.to(device="cpu", dtype=torch.long)
        payload = _PendingRequest(
            request_id=req_id,
            prompt=prompt_str,
            prompt_tokens=tokens_cpu,
            prompt_length=tokens_cpu.shape[1],
            image=image_obj,
            max_new_tokens=max_new_tokens,
            temperature=self._normalize_temperature(temperature),
            top_p=self._normalize_top_p(top_p),
            submitted_at=time.perf_counter(),
            future=future,
            stream_queue=stream_queue,
            skill=skill_spec,
            request_context=request_context,
        )
        await self._queue.put(payload)
        return future, req_id

    async def _ensure_started(self) -> None:
        if self._runtime is None:
            await self._initialize()

    async def _worker_loop(self) -> None:
        shutdown_error = RuntimeError("Engine shut down")

        while True:
            request = await self._queue.get()
            if request is None:
                self._scheduler_queue.put(None)
                self._scheduler_event.set()
                break
            self._scheduler_queue.put(request)
            self._scheduler_event.set()

        while not self._queue.empty():
            pending = self._queue.get_nowait()
            if pending is None:
                continue
            self._fail_request(pending, shutdown_error)

    def _normalize_temperature(self, value: Optional[float]) -> float:
        if value is None:
            return 0.0
        if value < 0.0:
            raise ValueError("temperature must be non-negative")
        return float(value)

    def _normalize_top_p(self, value: Optional[float]) -> float:
        if value is None:
            return 1.0
        top_p = float(value)
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        return top_p

    def _build_stream_callback(
        self, req: _PendingRequest
    ) -> Optional[Callable[[StreamUpdate], None]]:
        queue = req.stream_queue
        if queue is None:
            return None

        loop = self._loop
        assert loop is not None

        target_queue = queue
        target_loop = loop

        def _callback(update: StreamUpdate) -> None:
            target_loop.call_soon_threadsafe(target_queue.put_nowait, update)

        return _callback

    def _extract_adapter(self, request_context: object) -> Optional[LoRA]:
        if isinstance(request_context, QueryRequest):
            return request_context.settings.adapter
        return None

    def _complete_stream(
        self,
        req: _PendingRequest,
        *,
        result: Optional[EngineResult] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        queue = req.stream_queue
        if queue is None:
            return
        req.stream_queue = None
        completion = _StreamCompletion(result=result, error=error)
        self._loop.call_soon_threadsafe(queue.put_nowait, completion)

    def _fail_request(self, req: _PendingRequest, error: BaseException) -> None:
        future = req.future
        if future and not future.done():
            assert self._loop is not None
            self._loop.call_soon_threadsafe(future.set_exception, error)
        self._complete_stream(req, error=error)

    def _release_active_sequences(self) -> None:
        try:
            runtime_sequences = list(self.runtime.active_sequences.values())
        except Exception:  # pragma: no cover - defensive cleanup
            return
        for state in runtime_sequences:
            try:
                self.runtime.release_sequence(state)
            except Exception:
                pass

    def _extract_prompt_text(self, skill: SkillSpec, request_context: object) -> str:
        if isinstance(request_context, QueryRequest):
            return request_context.question
        if isinstance(request_context, PointRequest):
            return request_context.object
        if isinstance(request_context, DetectRequest):
            return request_context.object
        if isinstance(request_context, SegmentRequest):
            return request_context.object
        if isinstance(request_context, CaptionRequest):
            return request_context.length
        return str(request_context)

    def _scheduler_loop(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        torch.cuda.set_device(runtime.device)

        scheduler = GenerationScheduler(
            runtime,
            default_temperature=self._default_temperature,
            default_top_p=self._default_top_p,
            skill_registry=self._skills,
        )

        pending_crops: Dict[int, tuple[_PendingRequest, Future[OverlapCropOutput]]] = {}
        ready_crops: queue.Queue[int] = queue.Queue()
        active_requests: Dict[int, _PendingRequest] = {}
        shutdown_requested = False
        wake_event = self._scheduler_event
        run_gate = self._run_gate
        paused_flag = self._paused_flag
        paused_event = self._paused_event

        def admit_request(
            req: _PendingRequest, crops: Optional[OverlapCropOutput]
        ) -> None:
            try:
                generation_req, skill_state = self._build_generation_request(
                    runtime, req, crops
                )
            except Exception as exc:
                self._fail_request(req, exc)
                return
            scheduler.enqueue_request(generation_req, skill_state)
            active_requests[req.request_id] = req

        def handle_incoming(req: _PendingRequest) -> None:
            if req.image is None:
                admit_request(req, None)
                return
            executor = self._image_executor
            if executor is None:
                try:
                    crops = compute_overlap_crops(req.image, runtime.config.vision)
                except Exception as exc:
                    self._fail_request(req, exc)
                    return
                admit_request(req, crops)
                return
            try:
                future = executor.submit(
                    compute_overlap_crops, req.image, runtime.config.vision
                )
            except Exception as exc:
                self._fail_request(req, exc)
                return
            req_id = req.request_id
            pending_crops[req_id] = (req, future)

            def _on_crops_ready(fut: Future[OverlapCropOutput], rid: int = req_id) -> None:
                ready_crops.put(rid)
                wake_event.set()

            future.add_done_callback(_on_crops_ready)

        def promote_crops() -> bool:
            promoted = False
            while len(scheduler.waiting) < runtime.max_batch_size * 2:
                try:
                    rid = ready_crops.get_nowait()
                except queue.Empty:
                    break
                req_fut = pending_crops.pop(rid, None)
                if req_fut is None:
                    continue
                req, fut = req_fut
                try:
                    crops = fut.result()
                except Exception as exc:
                    self._fail_request(req, exc)
                    continue
                admit_request(req, crops)
                promoted = True
            return promoted

        def deliver_results(results: List[SchedulerResult]) -> None:
            if not results:
                return
            loop = self._loop
            assert loop is not None
            for result in results:
                req = active_requests.pop(result.request_id, None)
                if req is None:
                    _LOGGER.error(
                        "Scheduler produced unknown request_id %s",
                        result.request_id,
                    )
                    continue
                if result.finish_reason == "error" and "error" in result.output:
                    # Fail only the offending request while keeping the engine running.
                    self._fail_request(req, RuntimeError(result.output["error"]))
                    continue

                engine_result = self._to_engine_result(result)
                future = req.future
                if future and not future.done():
                    loop.call_soon_threadsafe(future.set_result, engine_result)
                self._complete_stream(req, result=engine_result)

        try:
            with torch.inference_mode():
                while True:
                    # If paused, wait until resumed or shutdown completes.
                    if paused_flag.is_set():
                        with runtime.graph_capture_lock:
                            torch.cuda.synchronize(runtime.device)
                        paused_event.set()
                        if (
                            shutdown_requested
                            and not scheduler.has_pending_work()
                            and not pending_crops
                            and not active_requests
                        ):
                            break
                        run_gate.wait(timeout=0.1)
                        continue

                    progressed = False
                    while True:
                        try:
                            item = self._scheduler_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            shutdown_requested = True
                            continue
                        handle_incoming(item)
                        progressed = True

                    if promote_crops():
                        progressed = True

                    try:
                        if scheduler.has_pending_work():
                            progressed = scheduler.advance() or progressed
                    except Exception as exc:
                        _LOGGER.exception("Scheduler advance failed", exc_info=exc)
                        self._fail_all_pending(active_requests, pending_crops, exc)
                        self._release_active_sequences()
                        return

                    results = scheduler.pop_completed()
                    if results:
                        deliver_results(results)
                        progressed = True

                    if (
                        shutdown_requested
                        and not scheduler.has_pending_work()
                        and not pending_crops
                        and not active_requests
                    ):
                        break

                    if not progressed:
                        if shutdown_requested:
                            break
                        wake_event.wait()
                        wake_event.clear()
        finally:
            while True:
                try:
                    ready_crops.get_nowait()
                except queue.Empty:
                    break
            self._fail_all_pending(active_requests, pending_crops)

    def _build_generation_request(
        self,
        runtime: MoondreamRuntime,
        req: _PendingRequest,
        image_crops: Optional[OverlapCropOutput],
    ) -> tuple[GenerationRequest, SkillState]:
        prompt_tokens = req.prompt_tokens
        stream_cb = self._build_stream_callback(req)
        image_length = (
            runtime.image_prefix_length
            if (req.image is not None or image_crops is not None)
            else 0
        )
        adapter = self._extract_adapter(req.request_context)
        request_obj = GenerationRequest(
            request_id=req.request_id,
            prompt=req.prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream_callback=stream_cb,
            image=req.image,
            image_crops=image_crops,
            image_length=image_length,
            submitted_at=req.submitted_at,
            skill=req.skill,
            request_context=req.request_context,
            adapter=adapter,
        )
        limit = runtime.max_seq_length
        target_total = request_obj.target_length
        if target_total > limit:
            raise ValueError(
                "Request length exceeds runtime max_seq_length: "
                f"needs {target_total} tokens but limit is {limit}."
            )
        skill_state = req.skill.create_state(
            runtime,
            request_obj,
            request_context=request_obj.request_context,
        )
        return request_obj, skill_state

    def _to_engine_result(self, result: SchedulerResult) -> EngineResult:
        sched_metrics = result.metrics
        prefill_time_ms = max(sched_metrics.prefill_time_ms, 0.0)
        decode_time_ms = max(sched_metrics.decode_time_ms, 0.0)
        ttft_ms = max(sched_metrics.ttft_ms, 0.0)
        metrics = EngineMetrics(
            input_tokens=sched_metrics.prompt_tokens,
            output_tokens=sched_metrics.decode_tokens,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            ttft_ms=ttft_ms,
        )
        return EngineResult(
            request_id=result.request_id,
            tokens=result.tokens,
            finish_reason=result.finish_reason,
            metrics=metrics,
            output=result.output,
        )

    def _fail_all_pending(
        self,
        active_requests: Dict[int, _PendingRequest],
        pending_crops: Dict[int, tuple[_PendingRequest, Future[OverlapCropOutput]]],
        error: Optional[BaseException] = None,
    ) -> None:
        exc = error or RuntimeError("Engine shut down")
        if active_requests:
            for req in list(active_requests.values()):
                self._fail_request(req, exc)
            active_requests.clear()
        if pending_crops:
            for req, future in list(pending_crops.values()):
                if future and not future.done():
                    future.cancel()
                self._fail_request(req, exc)
            pending_crops.clear()
        while True:
            try:
                pending = self._scheduler_queue.get_nowait()
            except queue.Empty:
                break
            if pending is None:
                continue
            self._fail_request(pending, exc)


__all__ = ["InferenceEngine", "EngineResult", "EngineMetrics", "EngineStream"]
