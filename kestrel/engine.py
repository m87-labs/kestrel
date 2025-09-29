"""Async coordination layer for Moondream inference.

The engine is the high-level entry point for clients. It owns:

- Lifecycle of the shared :class:`~kestrel.moondream.runtime.MoondreamRuntime`, including warmup and shutdown.
- A micro-batching worker that pulls pending requests, prepares image crops, and runs the scheduler.
- Skill orchestration — resolving the active :class:`~kestrel.skills.base.SkillSpec`, building prompt tokens when necessary, instantiating :class:`~kestrel.skills.base.SkillState` with skill-specific context, and bridging streaming callbacks back to callers.
- Conversion between scheduler outputs (``SchedulerResult``) and user-facing ``EngineResult`` objects augmented with metrics and per-skill extras.

Relationship to other components:

- Receives raw prompts or structured skill requests from clients (CLI, HTTP, etc.).
- Uses :class:`GenerationScheduler` to multiplex work across the runtime while keeping the scheduler skill-agnostic.
- Delegates low-level execution to :class:`MoondreamRuntime` for prefill/decode and to :mod:`kestrel.moondream.vision` for optional image preprocessing.

Internal API overview:

- :meth:`InferenceEngine.create` / :meth:`InferenceEngine.shutdown`: manage runtime instantiation and cleanup.
- :meth:`InferenceEngine.submit` / :meth:`InferenceEngine.submit_streaming`: enqueue non-streaming or streaming requests with optional pre-tokenised prompts.
- :meth:`InferenceEngine.query`: helper that wraps the default query skill with a validated :class:`QueryRequest`.
- `_submit_request`: normalises parameters, resolves the skill, builds prompt tokens when missing, and stashes a ``skill_context`` so the scheduler receives a fully initialised ``SkillState``.
- `_worker_loop`: background task that batches queued requests, invokes the scheduler, and delivers results or stream completions back to callers.

Clients should construct skill-specific request objects (e.g. ``QueryRequest``) whenever possible so the engine can enforce validation before work reaches the scheduler.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import pyvips

from kestrel.config import RuntimeConfig
from kestrel.moondream.runtime import MoondreamRuntime
from kestrel.scheduler import GenerationScheduler, GenerationRequest, SchedulerResult, StreamUpdate
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.moondream.vision import compute_overlap_crops
from kestrel.skills import QuerySkill, SkillRegistry, SkillSpec
from kestrel.skills.query import QueryRequest


@dataclass(slots=True)
class EngineMetrics:
    """Timing and token accounting for a single request."""

    prompt_tokens: int
    decode_tokens: int
    processing_latency_s: float
    ttft_s: float
    decode_latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.decode_tokens

    @property
    def decode_tokens_per_s(self) -> float:
        if self.decode_latency_s <= 0 or self.decode_tokens <= 0:
            return 0.0
        return self.decode_tokens / self.decode_latency_s


@dataclass(slots=True)
class EngineResult:
    """Inference output returned to callers."""

    request_id: int
    prompt: str
    text: str
    tokens: List[int]
    finish_reason: str
    metrics: EngineMetrics
    extras: Dict[str, object]


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
    skill_context: Optional[object]


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

        self._runtime: MoondreamRuntime | None = None
        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._request_ids = itertools.count()
        self._shutdown = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._image_executor: ThreadPoolExecutor | None = None
        self._skills = skills or SkillRegistry([QuerySkill()])

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
        self._runtime = await loop.run_in_executor(None, MoondreamRuntime, self._runtime_cfg)
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
        tokens = skill.build_prompt_tokens(runtime, prompt)
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
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[torch.Tensor] = None,
        image: Optional[pyvips.Image] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        skill: Optional[str | SkillSpec] = None,
        skill_context: Optional[object] = None,
    ) -> EngineResult:
        future, _ = await self._submit_request(
            prompt,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            stream_queue=None,
            skill=skill,
            skill_context=skill_context,
        )
        return await future

    async def query(
        self,
        request: QueryRequest,
        *,
        max_new_tokens: int,
    ) -> EngineResult:
        return await self.submit(
            request.question,
            max_new_tokens=max_new_tokens,
            image=request.image,
            temperature=request.settings.temperature,
            top_p=request.settings.top_p,
            skill="query",
            skill_context=request,
        )

    async def submit_streaming(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[torch.Tensor] = None,
        image: Optional[pyvips.Image] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        skill: Optional[str | SkillSpec] = None,
        skill_context: Optional[object] = None,
    ) -> EngineStream:
        queue: _StreamQueue = asyncio.Queue()
        future, request_id = await self._submit_request(
            prompt,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
            image=image,
            temperature=temperature,
            top_p=top_p,
            stream_queue=queue,
            skill=skill,
            skill_context=skill_context,
        )
        return EngineStream(request_id=request_id, queue=queue, result_future=future)

    async def _submit_request(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[torch.Tensor],
        image: Optional[pyvips.Image],
        temperature: Optional[float],
        top_p: Optional[float],
        stream_queue: Optional[_StreamQueue],
        skill: Optional[str | SkillSpec],
        skill_context: Optional[object],
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

        if prompt_tokens is None:
            tokens = skill_spec.build_prompt_tokens(
                self.runtime,
                prompt,
                image=image_obj,
                image_crops=None,
            )
        else:
            tokens = prompt_tokens

        tokens_cpu = tokens.to(device="cpu", dtype=torch.long)
        payload = _PendingRequest(
            request_id=req_id,
            prompt=prompt,
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
            skill_context=skill_context,
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
                metrics = EngineMetrics(
                    prompt_tokens=sched_metrics.prompt_tokens,
                    decode_tokens=sched_metrics.decode_tokens,
                    processing_latency_s=sched_metrics.processing_latency_s,
                    ttft_s=sched_metrics.ttft_s,
                    decode_latency_s=sched_metrics.decode_latency_s,
                )
                engine_result = EngineResult(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    text=result.text,
                    tokens=result.tokens,
                    finish_reason=result.finish_reason,
                    metrics=metrics,
                    extras=result.extras,
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

    def _run_batch(self, batch: Iterable[_PendingRequest]) -> dict[int, SchedulerResult]:
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
            image_length = runtime.image_prefix_length if (req.image is not None or crops is not None) else 0
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
                skill=req.skill,
            )
            skill_state = req.skill.create_state(runtime, request_obj, context=req.skill_context)
            scheduler.enqueue_request(request_obj, skill_state)
        results = scheduler.run()
        return {result.request_id: result for result in results}


__all__ = ["InferenceEngine", "EngineResult", "EngineMetrics", "EngineStream"]
