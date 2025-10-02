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

from __future__ import annotations

import asyncio
import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
    Literal,
)

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
    SkillRegistry,
    SkillSpec,
)
from kestrel.moondream.runtime import Token
from kestrel.skills.caption import CaptionRequest, CaptionSettings
from kestrel.skills.detect import DetectRequest, DetectSettings
from kestrel.skills.point import PointRequest, PointSettings
from kestrel.skills.query import QueryRequest, QuerySettings


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
    image: Optional[pyvips.Image]
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
        batch_timeout_s: float = 0.02,
        skills: Optional[SkillRegistry] = None,
    ) -> None:
        self._runtime_cfg = runtime_cfg
        self._batch_timeout_s = batch_timeout_s

        self._runtime: Optional[MoondreamRuntime] = None
        self._queue: asyncio.Queue[Optional[_PendingRequest]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._request_ids = itertools.count()
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._image_executor: Optional[ThreadPoolExecutor] = None
        self._skills = skills or SkillRegistry(
            [QuerySkill(), PointSkill(), DetectSkill(), CaptionSkill()]
        )
        self._default_max_new_tokens = 512
        self._default_temperature = 0.0
        self._default_top_p = 1.0

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

    @classmethod
    async def create(
        cls,
        runtime_cfg: RuntimeConfig,
        *,
        batch_timeout_s: float = 0.02,
        skills: Optional[SkillRegistry] = None,
    ) -> "InferenceEngine":
        engine = cls(runtime_cfg, batch_timeout_s=batch_timeout_s, skills=skills)
        await engine._initialize()
        return engine

    async def _initialize(self) -> None:
        if self._runtime is not None:
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._runtime = await loop.run_in_executor(
            None, MoondreamRuntime, self._runtime_cfg
        )
        await loop.run_in_executor(None, self._warmup)
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True)
        max_workers = max(1, self._runtime.max_batch_size)
        self._image_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="kestrel-img"
        )
        self._worker_task = asyncio.create_task(self._worker_loop())

    def _warmup(self) -> None:
        assert self._runtime is not None
        runtime = self._runtime
        prompt = "Warmup prompt."
        skill = self._skills.default
        if isinstance(skill, QuerySkill):
            warmup_request = QueryRequest(
                question=prompt,
                image=None,
                reasoning=False,
                stream=False,
                settings=QuerySettings(
                    temperature=self._default_temperature,
                    top_p=self._default_top_p,
                    max_tokens=1,
                ),
            )
        else:
            raise RuntimeError(
                "Warmup currently requires the default skill to be QuerySkill"
            )
        tokens = skill.build_prompt_tokens(runtime, warmup_request)
        state, logits = runtime.start_sequence(prompt_tokens=tokens, max_new_tokens=1)
        try:
            next_token = torch.argmax(logits, dim=-1)
            runtime.decode(state, next_token.view(-1))
        finally:
            runtime.release_sequence(state)

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self._queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
        self._worker_task = None
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True)
            self._image_executor = None

    async def submit(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        image: Optional[pyvips.Image] = None,
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
        image: Optional[pyvips.Image] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[True] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream":
        ...

    @overload
    async def query(
        self,
        image: Optional[pyvips.Image] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult:
        ...

    @overload
    async def query(
        self,
        image: Optional[pyvips.Image] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        ...

    async def query(
        self,
        image: Optional[pyvips.Image] = None,
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

        request = QueryRequest(
            question=normalized_question,
            image=image,
            reasoning=reasoning,
            stream=stream,
            settings=QuerySettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
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
        image: Optional[pyvips.Image],
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
        image: pyvips.Image,
        *,
        length: str = ...,
        stream: Literal[True],
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream":
        ...

    @overload
    async def caption(
        self,
        image: pyvips.Image,
        *,
        length: str = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult:
        ...

    @overload
    async def caption(
        self,
        image: pyvips.Image,
        *,
        length: str = ...,
        stream: bool = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> Union[EngineResult, EngineStream]:
        ...

    async def caption(
        self,
        image: pyvips.Image,
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
        image: Optional[pyvips.Image],
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

    async def submit_streaming(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        image: Optional[pyvips.Image] = None,
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

    async def _submit_request(
        self,
        *,
        max_new_tokens: int,
        request_context: object,
        image: Optional[pyvips.Image],
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

        image_obj: Optional[pyvips.Image] = None
        if image is not None:
            if self.runtime.image_prefix_length == 0:
                raise ValueError("Runtime does not support image inputs")
            if not isinstance(image, pyvips.Image):
                raise TypeError("image must be a pyvips.Image")
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
        assert self._runtime is not None
        max_batch = self._runtime.max_batch_size
        loop = asyncio.get_running_loop()

        while True:
            batch: List[_PendingRequest] = []

            request = await self._queue.get()
            if request is None:
                break
            batch.append(request)

            timeout = self._batch_timeout_s
            while len(batch) < max_batch:
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if req is None:
                    # Drain existing work before stopping.
                    self._shutdown = True
                    break
                batch.append(req)
            if not batch:
                continue

            try:
                results = await loop.run_in_executor(None, self._run_batch, batch)
            except Exception as exc:
                for req in batch:
                    future = req.future
                    if future and not future.done():
                        future.set_exception(exc)
                    self._complete_stream(req, error=exc)

                self._shutdown = True

                # Best-effort cleanup of any sequences that may have been admitted.
                try:
                    runtime_sequences = list(self.runtime.active_sequences.values())
                except Exception:  # pragma: no cover - defensive cleanup
                    runtime_sequences = []
                for state in runtime_sequences:
                    try:
                        self.runtime.release_sequence(state)
                    except Exception:
                        pass

                # Propagate the same failure to anything still queued.
                try:
                    while True:
                        pending = self._queue.get_nowait()
                        if pending is None:
                            continue
                        future = pending.future
                        if future and not future.done():
                            future.set_exception(exc)
                except asyncio.QueueEmpty:
                    pass

                break

            for req in batch:
                future = req.future
                try:
                    result = results[req.request_id]
                except KeyError:
                    error = RuntimeError(
                        f"Request {req.request_id} missing from scheduler results"
                    )
                    future.set_exception(error)
                    self._complete_stream(req, error=error)
                    continue
                sched_metrics = result.metrics
                prefill_time_ms = max(sched_metrics.prefill_time_s * 1000.0, 0.0)
                decode_time_ms = max(sched_metrics.decode_time_s * 1000.0, 0.0)
                ttft_ms = max(sched_metrics.ttft_s * 1000.0, 0.0)
                metrics = EngineMetrics(
                    input_tokens=sched_metrics.prompt_tokens,
                    output_tokens=sched_metrics.decode_tokens,
                    prefill_time_ms=prefill_time_ms,
                    decode_time_ms=decode_time_ms,
                    ttft_ms=ttft_ms,
                )
                engine_result = EngineResult(
                    request_id=req.request_id,
                    tokens=result.tokens,
                    finish_reason=result.finish_reason,
                    metrics=metrics,
                    output=result.output,
                )
                if not future.done():
                    future.set_result(engine_result)
                self._complete_stream(req, result=engine_result)

            if self._shutdown:
                break

        # Cancel any pending futures in the queue.
        while not self._queue.empty():
            pending = self._queue.get_nowait()
            if pending and pending.future and not pending.future.done():
                error = RuntimeError("Engine shut down")
                pending.future.set_exception(error)
                self._complete_stream(pending, error=error)

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
        loop = self._loop
        if queue is None or loop is None:
            return None

        target_queue = queue
        target_loop = loop

        def _callback(update: StreamUpdate) -> None:
            target_loop.call_soon_threadsafe(target_queue.put_nowait, update)

        return _callback

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
        queue.put_nowait(completion)

    def _extract_prompt_text(self, skill: SkillSpec, request_context: object) -> str:
        if isinstance(request_context, QueryRequest):
            return request_context.question
        if isinstance(request_context, PointRequest):
            return request_context.object
        if isinstance(request_context, DetectRequest):
            return request_context.object
        if isinstance(request_context, CaptionRequest):
            return request_context.length
        return str(request_context)

    def _run_batch(
        self, batch: Iterable[_PendingRequest]
    ) -> dict[int, SchedulerResult]:
        image_crops: dict[int, OverlapCropOutput] = {}
        if self._image_executor is not None:
            futures: List[tuple[int, Future[OverlapCropOutput]]] = []
            vision_config = self.runtime.config.vision
            for req in batch:
                if req.image is not None:
                    futures.append(
                        (
                            req.request_id,
                            self._image_executor.submit(
                                compute_overlap_crops, req.image, vision_config
                            ),
                        )
                    )
            for req_id, future in futures:
                image_crops[req_id] = future.result()

        scheduler = GenerationScheduler(
            self.runtime,
            default_temperature=0.0,
            default_top_p=1.0,
            skill_registry=self._skills,
        )
        runtime = self.runtime
        for req in batch:
            prompt_tokens = req.prompt_tokens.clone()
            stream_cb = self._build_stream_callback(req)
            crops = image_crops.get(req.request_id)
            image_length = (
                runtime.image_prefix_length
                if (req.image is not None or crops is not None)
                else 0
            )
            request_obj = GenerationRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_tokens=prompt_tokens,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stream_callback=stream_cb,
                image=req.image,
                image_crops=crops,
                image_length=image_length,
                submitted_at=req.submitted_at,
                skill=req.skill,
                request_context=req.request_context,
            )
            skill_state = req.skill.create_state(
                runtime,
                request_obj,
                request_context=request_obj.request_context,
            )
            scheduler.enqueue_request(request_obj, skill_state)
        results = scheduler.run()
        return {result.request_id: result for result in results}


__all__ = ["InferenceEngine", "EngineResult", "EngineMetrics", "EngineStream"]
