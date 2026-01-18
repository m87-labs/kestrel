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
import hashlib
import itertools
import logging
import queue
import threading
import time
from concurrent.futures import Future
import os
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
    RequestLifecycle,
    RequestPhase,
    SchedulerResult,
    StreamUpdate,
)
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.moondream.image_preprocessor import ImagePreprocessor
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
from kestrel.moondream.lora import AdapterProvider
from kestrel.utils.spatial_refs import normalize_spatial_refs


_LOGGER = logging.getLogger(__name__)


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
    prompt_tokens: Sequence[Token]
    image: Optional[pyvips.Image | np.ndarray]
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


class InferenceEngine:
    """Orchestrates batched inference over a shared runtime and scheduler."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        *,
        skills: Optional[SkillRegistry] = None,
        adapter_provider: Optional[AdapterProvider] = None,
    ) -> None:
        self._runtime_cfg = runtime_cfg
        self._adapter_provider = adapter_provider

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
        self._image_preprocessor = ImagePreprocessor()
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
        adapter_provider: Optional[AdapterProvider] = None,
    ) -> "InferenceEngine":
        # MOONDREAM_API_KEY is always required
        api_key = os.environ.get("MOONDREAM_API_KEY")
        if not api_key:
            raise ValueError("MOONDREAM_API_KEY environment variable is required.")

        # Auto-create provider if none provided
        if adapter_provider is None:
            from kestrel.cloud import MoondreamAdapterProvider
            from kestrel.moondream.config import load_config

            config = load_config()
            adapter_provider = MoondreamAdapterProvider(
                text_config=config.text,
                api_key=api_key,
                device=torch.device(runtime_cfg.device),
                dtype=runtime_cfg.resolved_dtype(),
            )

        engine = cls(runtime_cfg, skills=skills, adapter_provider=adapter_provider)
        await engine._initialize()
        return engine

    async def _initialize(self) -> None:
        if self._runtime is not None:
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        max_lora_rank = (
            self._adapter_provider.config()["max_lora_rank"]
            if self._adapter_provider is not None
            else None
        )
        self._runtime = await loop.run_in_executor(
            None, lambda: MoondreamRuntime(self._runtime_cfg, max_lora_rank=max_lora_rank)
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
            warmup_settings: Dict[str, object] = {
                "temperature": self._default_temperature,
                "top_p": self._default_top_p,
                "max_tokens": 1,
            }
            # Warmup uses slot 0 (no LoRA) - adapter-specific warmup is not required
            # since workspace tensors have fixed addresses.
            await self.query(
                image=None,
                question="Warmup prompt.",
                reasoning=False,
                stream=False,
                settings=warmup_settings,
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
        self._image_preprocessor.shutdown(wait=True)

    async def submit(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        adapter: Optional[str] = None,
        image: Optional[pyvips.Image | np.ndarray] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> EngineResult:
        future, _ = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            adapter=adapter,
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
        if spatial_refs and image is None:
            raise ValueError("spatial_refs can only be used with an image")

        temperature = self._default_temperature
        top_p = self._default_top_p
        max_tokens = self._default_max_new_tokens
        adapter: Optional[str] = None
        if settings is not None:
            if "temperature" in settings:
                temperature = float(settings["temperature"])
            if "top_p" in settings:
                top_p = float(settings["top_p"])
            if "max_tokens" in settings:
                max_tokens = int(settings["max_tokens"])
            if "adapter" in settings:
                maybe_adapter = settings["adapter"]
                if maybe_adapter is None:
                    adapter = None
                elif not isinstance(maybe_adapter, str):
                    raise TypeError("settings.adapter must be a string or None")
                else:
                    adapter = maybe_adapter

        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        normalized_refs = self._normalize_spatial_refs(spatial_refs)
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
            spatial_refs=normalized_refs,
        )
        if stream:
            return await self.submit_streaming(
                request,
                max_new_tokens=max_tokens,
                adapter=adapter,
                image=image,
                temperature=temperature,
                top_p=top_p,
                skill="query",
            )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            adapter=adapter,
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

        adapter = self._extract_adapter_id(settings)
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
            adapter=adapter,
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

        adapter = self._extract_adapter_id(settings)
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
                adapter=adapter,
                image=image,
                temperature=temperature,
                top_p=top_p,
                skill="caption",
            )
        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            adapter=adapter,
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

        adapter = self._extract_adapter_id(settings)
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
            adapter=adapter,
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

        adapter = self._extract_adapter_id(settings)
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

        normalized_refs = self._normalize_spatial_refs(spatial_refs)

        request = SegmentRequest(
            object=normalized_object,
            image=image,
            stream=False,
            settings=SegmentSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
            spatial_refs=normalized_refs,
        )

        return await self.submit(
            request,
            max_new_tokens=max_tokens,
            adapter=adapter,
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
        adapter: Optional[str] = None,
        image: Optional[pyvips.Image | np.ndarray] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> EngineStream:
        queue: _StreamQueue = asyncio.Queue()
        future, request_id = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            adapter=adapter,
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
        adapter: Optional[str],
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
        adapter_id = self._normalize_adapter_id(adapter)
        if self._adapter_provider is None and adapter_id is not None:
            raise NotImplementedError(
                "Adapter support requires an adapter_provider at engine creation."
            )

        image_obj: Optional[pyvips.Image | np.ndarray] = None
        if image is not None:
            if self.runtime.image_prefix_length == 0:
                raise ValueError("Runtime does not support image inputs")
            if not isinstance(image, (pyvips.Image, np.ndarray)):
                raise TypeError("image must be a pyvips.Image or np.ndarray")
            image_obj = image

        prompt_str = self._extract_prompt_text(skill_spec, request_context)
        tokens = list(skill_spec.build_prompt_tokens(self.runtime, request_context))
        payload = _PendingRequest(
            request_id=req_id,
            prompt=prompt_str,
            prompt_tokens=tokens,
            image=image_obj,
            image_hash=None,  # Computed in scheduler thread if prefix cache enabled
            max_new_tokens=max_new_tokens,
            temperature=self._normalize_temperature(temperature),
            top_p=self._normalize_top_p(top_p),
            submitted_at=time.perf_counter(),
            future=future,
            stream_queue=stream_queue,
            skill=skill_spec,
            request_context=request_context,
            adapter=adapter_id,
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

    def _normalize_adapter_id(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("adapter must be a non-empty string")
        return normalized

    def _normalize_spatial_refs(
        self, spatial_refs: Optional[Sequence[Sequence[float]]]
    ) -> Optional[Tuple[Tuple[float, ...], ...]]:
        return normalize_spatial_refs(spatial_refs)

    def _extract_adapter_id(
        self, settings: Optional[Mapping[str, object]]
    ) -> Optional[str]:
        if settings is None or "adapter" not in settings:
            return None
        raw = settings["adapter"]
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError("settings.adapter must be a string or None")
        return self._normalize_adapter_id(raw)

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

        adapter_provider = self._adapter_provider
        scheduler = GenerationScheduler(
            runtime,
            default_temperature=self._default_temperature,
            default_top_p=self._default_top_p,
            skill_registry=self._skills,
            adapter_provider=adapter_provider,
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
            req: _PendingRequest,
            crops: Optional[OverlapCropOutput],
            *,
            prefix_cache_hit: bool = False,
        ) -> None:
            try:
                generation_req, skill_state = self._build_generation_request(
                    runtime, req, crops
                )
            except Exception as exc:
                self._fail_request(req, exc)
                return
            crops_ready = (
                req.image is None or prefix_cache_hit or (crops is not None)
            )
            lora_slot_ready = req.adapter is None
            phase = (
                RequestPhase.READY_FOR_PREFILL
                if (crops_ready and lora_slot_ready)
                else RequestPhase.WAITING_RESOURCES
            )
            lifecycle = RequestLifecycle(
                request=generation_req,
                skill_state=skill_state,
                phase=phase,
                has_image=req.image is not None,
                crops_ready=crops_ready,
                lora_slot_ready=lora_slot_ready,
                prefix_cache_hit=prefix_cache_hit,
                submitted_at=req.submitted_at,
            )
            generation_req.lifecycle = lifecycle
            scheduler.enqueue_request(generation_req, skill_state)
            active_requests[req.request_id] = req

        def handle_incoming(req: _PendingRequest) -> None:
            if req.image is None:
                admit_request(req, None, prefix_cache_hit=False)
                return

            # Compute image hash and check prefix cache for early skip
            if runtime.prefix_cache is not None:
                # Hash using raw bytes (faster than PNG encoding)
                if isinstance(req.image, np.ndarray):
                    raw_bytes = req.image.tobytes()
                else:
                    raw_bytes = req.image.write_to_memory()
                image_hash = hashlib.sha256(raw_bytes).digest()
                req.image_hash = image_hash

                # Early cache lookup - skip crop computation if cache hit
                tokens_list = list(req.prompt_tokens)
                if runtime.check_prefix_cache(tokens_list, image_hash, req.adapter):
                    admit_request(req, None, prefix_cache_hit=True)  # No crops needed
                    return

            # Cache miss or prefix cache disabled - compute crops asynchronously
            try:
                future = self._image_preprocessor.submit(req.image, runtime.config.vision)
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
                admit_request(req, crops, prefix_cache_hit=False)
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
                        # Drain pipeline before pause - complete all in-flight work.
                        # Required per design doc §4.5a: callers may mutate runtime
                        # state while paused (e.g. rebuild CUDA graphs).
                        scheduler._drain_pipeline()
                        # Deliver any results from drained steps before pausing.
                        drained_results = scheduler.pop_completed()
                        if drained_results:
                            deliver_results(drained_results)
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
        adapter = req.adapter
        request_obj = GenerationRequest(
            request_id=req.request_id,
            prompt=req.prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream_callback=stream_cb,
            image=req.image,
            image_hash=req.image_hash,
            image_crops=image_crops,
            image_length=image_length,
            submitted_at=req.submitted_at,
            skill=req.skill,
            request_context=req.request_context,
            adapter=adapter,
            lora_slot=req.lora_slot,
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
            cached_tokens=sched_metrics.cached_tokens,
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
