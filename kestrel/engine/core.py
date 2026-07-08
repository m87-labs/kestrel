"""Async coordination layer for Moondream inference.

The engine is the high-level entry point for clients. It owns:

- Lifecycle of the shared :class:`~kestrel.models.moondream.runtime.MoondreamRuntime`, including warmup and shutdown.
- A micro-batching worker that pulls pending requests, prepares image crops, and runs the scheduler.
- Skill orchestration — resolving the active :class:`~kestrel.skills.base.SkillSpec`, building prompt tokens when necessary, instantiating :class:`~kestrel.skills.base.SkillState` with skill-specific request contexts, and bridging streaming callbacks back to callers.
- Conversion between scheduler outputs (``SchedulerResult``) and user-facing ``EngineResult`` objects augmented with metrics and per-skill output payloads.

Relationship to other components:

- Receives raw prompts or structured skill requests from clients (CLI, HTTP, etc.).
- Uses :class:`GenerationScheduler` to multiplex work across the runtime while keeping the scheduler skill-agnostic.
- Delegates low-level execution to :class:`MoondreamRuntime` for prefill/decode and to :mod:`kestrel.models.moondream.vision` for optional image preprocessing.

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
import queue
import threading
import time
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
    Literal,
)

import numpy as np
import torch

from dataclasses import replace

from kestrel_kernels import get_runtime
from kestrel.config import RuntimeConfig
from kestrel.device import make_stream, set_device, synchronize
from kestrel.runtime import (
    AutoregressiveRuntime,
    ExecutionShape,
    Runtime,
)
from kestrel.scheduler import (
    GeneratedPrefix,
    GenerationRequest,
    SchedulerResult,
    StreamUpdate,
)
from kestrel.skills import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    DecodeStep,
    SkillRegistry,
    SkillSpec,
    SkillState,
)
from kestrel.models.moondream.runtime import CoordToken, SizeToken, TextToken, Token
from kestrel.models.moondream.lora import AdapterProvider
from kestrel.photon import PhotonReporter

from kestrel.engine._types import (
    Completion,
    EngineMetrics,
    EngineRequest,
    EngineResult,
    EngineStream,
    ModelStream,
    ModelStreamUpdate,
    _AutoregressiveRequest,
    _ModelStreamCompletion,
    _ModelStreamQueue,
    _StreamCompletion,
    _StreamQueue,
    _StreamQueueItem,
    _StreamingChunk,
    _StreamingSessionRequest,
)
from kestrel.engine.executor import AutoregressiveExecutor
from kestrel.engine.single_pass import SinglePassExecutor, _SinglePassRequest
from kestrel.engine.streaming import StreamingExecutor
from kestrel.engine.handle import ModelHandle


_LOGGER = logging.getLogger(__name__)

# PR landing this warning + the kernel envelope it documents. Update if
# the canonical docs move.
_MPS_SAMPLER_DOCS_URL = "https://github.com/m87-labs/kestrel/pull/33"

_DEFAULT_API_BASE_URL = "https://api.moondream.ai"
_DOCS_URL = "https://moondream.ai/docs"
_ALLOWED_API_BASE_URLS = (
    "https://api-staging.moondream.ai",
    _DEFAULT_API_BASE_URL,
)

# How long the kernel parks between polls while a single-pass forward is
# in flight. A single forward is a few ms and its GPU completion event
# sets no host event, so the kernel re-checks on this cadence; ~1ms keeps
# completion latency low without busy-spinning a core.
_SINGLE_PASS_POLL_INTERVAL_S = 0.001


class InferenceEngine:
    """Orchestrates batched inference over a shared runtime and scheduler."""

    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        *,
        runtime: Optional[AutoregressiveRuntime] = None,
        skills: Optional[SkillRegistry] = None,
        adapter_provider: Optional[AdapterProvider] = None,
        api_key: Optional[str] = None,
        api_base_url: str = _DEFAULT_API_BASE_URL,
        models: Optional[Sequence[str]] = None,
        adapter_provider_uses_api_key: bool = False,
    ) -> None:
        if runtime is not None:
            cfg_device = runtime_cfg.resolved_device()
            if runtime.device != cfg_device:
                raise ValueError(
                    f"runtime.device ({runtime.device}) does not match "
                    f"runtime_cfg.resolved_device() ({cfg_device})"
                )
        self._runtime_cfg = runtime_cfg
        self._adapter_provider = adapter_provider
        self._adapter_provider_uses_api_key = adapter_provider_uses_api_key
        self._api_key = api_key
        self._api_base_url = api_base_url

        # Runtimes keyed by model id. The engine is a model-agnostic
        # kernel: it holds a registry rather than a single runtime, and
        # addresses runtimes by model id (data from ``runtime_cfg`` / the
        # model registry), never by execution shape and never by
        # interrogating the runtime object. ``_default_model`` is the id
        # the single-model surface (the ``runtime`` property and
        # ``query``/``caption``/...) targets; it is the foundation the
        # multi-model ``run`` surface builds on.
        #
        # An externally-built runtime is registered from construction;
        # otherwise ``_initialize`` builds one via the spec's ``runtime``
        # factory on first ``create`` and registers it under the same id.
        # The engine owns the lifecycle of every runtime it holds and
        # tears them down on ``shutdown``.
        #
        self._default_model: str = runtime_cfg.model
        # Every model this engine co-hosts. The default (``runtime_cfg.model``,
        # autoregressive) is always hosted and is what the single-model verbs
        # target; ``models`` adds co-hosted models (e.g. single-pass), each
        # built from its registered spec at startup. Order-preserving with the
        # default first; deduplicated.
        model_ids: list[str] = [self._default_model]
        for model_id in models or ():
            if model_id not in model_ids:
                model_ids.append(model_id)
        self._model_ids: list[str] = model_ids
        # Keyed by model id, mixed execution shapes: the default is
        # autoregressive, co-hosted models may be single-pass. The kernel
        # addresses runtimes by id and routes by ``execution_shape``, never by
        # interrogating the object.
        self._runtimes: Dict[str, Runtime] = {}
        self._kv_pool = runtime.kv_pool if runtime is not None else None
        if runtime is not None:
            self._runtimes[self._default_model] = runtime
        runtime_device = (
            runtime.device if runtime is not None else runtime_cfg.resolved_device()
        )
        self._compute_stream = (
            cast(Any, runtime).compute_stream
            if runtime is not None
            else make_stream(runtime_device)
        )
        # ``_initialized`` flips at the very end of ``_initialize`` so
        # any partial failure (warmup, photon validation) leaves the
        # engine retry-able. ``_init_task`` is the asyncio task that's
        # currently running ``_initialize``; concurrent callers await
        # it instead of racing each other, and the warmup pipeline
        # (which loops back through ``query()`` → ``_ensure_started``)
        # detects via ``asyncio.current_task() is self._init_task``
        # that it's already inside init and bails without recursing.
        self._initialized = False
        self._init_task: Optional[asyncio.Task[None]] = None
        self._queue: asyncio.Queue[Optional[_AutoregressiveRequest]] = asyncio.Queue()
        self._scheduler_queue: queue.Queue[_AutoregressiveRequest | None] = queue.Queue()
        # Single-pass ingress: (model_id, request) tagged so the kernel
        # routes to that model's single-pass lane. Separate from the AR
        # queue, which has no model tag (it always targets the default
        # autoregressive model).
        self._single_pass_queue: "queue.Queue[tuple[str, _SinglePassRequest]]" = (
            queue.Queue()
        )
        # Stateful streaming ingress: starts and chunks are tagged by model
        # id so the kernel routes them to that model's streaming lane.
        self._streaming_start_queue: (
            "queue.Queue[tuple[str, _StreamingSessionRequest]]"
        ) = queue.Queue()
        self._streaming_chunk_queue: "queue.Queue[tuple[str, _StreamingChunk]]" = (
            queue.Queue()
        )
        self._model_stream_models: Dict[int, str] = {}
        self._model_stream_queues: Dict[int, _ModelStreamQueue] = {}
        self._scheduler_event = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()  # set == running
        self._paused_flag = threading.Event()  # set == paused
        self._paused_event = threading.Event()  # acknowledgment for callers
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_error: Optional[BaseException] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._photon_reporter: Optional[PhotonReporter] = None
        self._request_ids = itertools.count()
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Skills are owned by the model (the runtime supplies them via
        # runtime.skills()). ``skills=`` is an optional override, mainly
        # for tests; when None the default model's registry is used.
        self._skills_override = skills
        # AR serving defaults live with the AR skills (sampling config, not
        # kernel config); the engine only needs them to seed the warmup
        # query. Sourced from the skill-layer constants, not redefined here.
        self._default_max_new_tokens = AR_DEFAULT_MAX_NEW_TOKENS
        self._default_temperature = AR_DEFAULT_TEMPERATURE
        self._default_top_p = AR_DEFAULT_TOP_P

    @property
    def runtime(self) -> AutoregressiveRuntime:
        runtime = self._runtimes.get(self._default_model)
        if runtime is None:
            raise RuntimeError("InferenceEngine has not been started")
        # The default model is always autoregressive; co-hosted models are
        # reached via ``run`` / their single-pass lane, never this property.
        return cast(AutoregressiveRuntime, runtime)

    def _skill_registry(self) -> SkillRegistry:
        """The active skill registry for the default model.

        Resolution order: the test ``_skills_override``; else the built
        runtime's declared skills; else the default model's
        :class:`~kestrel.models.registry.ModelSpec` skills. The spec path is
        what lets input validation and ``tasks`` work *before* startup —
        skills are static model metadata, so resolving them must not require
        loading model weights.
        """
        if self._skills_override is not None:
            return self._skills_override
        runtime = self._runtimes.get(self._default_model)
        if runtime is not None:
            return runtime.skills()
        from kestrel.models.registry import get_spec

        return get_spec(self._default_model).skills()

    @property
    def skills(self) -> SkillRegistry:
        return self._skill_registry()

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
        runtime: Optional[AutoregressiveRuntime] = None,
        skills: Optional[SkillRegistry] = None,
        adapter_provider: Optional[AdapterProvider] = None,
        api_key: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
    ) -> "InferenceEngine":
        # Auto-create provider if none provided. Photon telemetry itself does
        # not require auth; cloud-backed finetune inference does.
        if api_key is None:
            api_key = os.environ.get("MOONDREAM_API_KEY")
        api_key = PhotonReporter._normalize_api_key(api_key)
        api_base_url = os.environ.get("MOONDREAM_API_BASE_URL", _DEFAULT_API_BASE_URL)
        api_base_url = api_base_url.rstrip("/")
        if api_base_url not in _ALLOWED_API_BASE_URLS:
            api_base_url = _DEFAULT_API_BASE_URL
        adapter_provider_uses_api_key = False
        if adapter_provider is None and api_key:
            if PhotonReporter._is_api_key_header_safe(api_key):
                from kestrel.cloud import MoondreamAdapterProvider
                from kestrel.models.moondream.config import load_config

                config = load_config()
                adapter_provider = MoondreamAdapterProvider(
                    text_config=config.text,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    device=torch.device(runtime_cfg.device),
                    dtype=runtime_cfg.resolved_dtype(),
                )
                adapter_provider_uses_api_key = True
            else:
                _LOGGER.warning(
                    "MOONDREAM_API_KEY has non-ASCII or whitespace characters. "
                    "Finetune inference is disabled. See %s",
                    _DOCS_URL,
                )
        elif adapter_provider is None:
            _LOGGER.warning(
                "MOONDREAM_API_KEY is not set. Finetune inference is disabled. "
                "Set MOONDREAM_API_KEY to enable finetune inference."
            )

        engine = cls(
            runtime_cfg,
            runtime=runtime,
            skills=skills,
            adapter_provider=adapter_provider,
            api_key=api_key,
            api_base_url=api_base_url,
            models=models,
            adapter_provider_uses_api_key=adapter_provider_uses_api_key,
        )
        await engine._initialize()
        return engine

    async def _initialize(self) -> None:
        if self._initialized:
            return
        self._scheduler_error = None
        loop = asyncio.get_running_loop()
        self._loop = loop
        try:
            from kestrel.model_download import probe_supported_model_configs

            threading.Thread(
                target=probe_supported_model_configs,
                name="kestrel-hf-config-probe",
                daemon=True,
            ).start()
        except Exception:
            pass
        reporter: Optional[PhotonReporter] = None
        if self._photon_reporter is None:
            reporter = PhotonReporter(
                self._runtime_cfg,
                self._runtime_cfg.resolved_device(),
                api_key=self._api_key,
                api_base_url=self._api_base_url,
            )
            auth_available = await reporter.validate_api_key()
            if not auth_available and self._adapter_provider_uses_api_key:
                self._disable_cloud_adapter_provider()
        if any(model_id not in self._runtimes for model_id in self._model_ids):
            max_lora_rank = (
                self._adapter_provider.config()["max_lora_rank"]
                if self._adapter_provider is not None
                else None
            )
            if max_lora_rank is not None:
                if not get_runtime().moe.supports_lora(self._runtime_cfg.device):
                    _LOGGER.warning(
                        "MoE LoRA adapters are unavailable in this kernel runtime. "
                        "Disabling LoRA — base model inference will still work, but "
                        "adapter requests will be rejected. Contact contact@moondream.ai "
                        "if LoRA support is needed on your platform."
                    )
                    max_lora_rank = None
            # Build every configured model's runtime from its spec, off the
            # event loop (weight loading is blocking). The default is
            # autoregressive; co-hosted models may be single-pass.
            await loop.run_in_executor(
                None, self._build_configured_runtimes, max_lora_rank
            )
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="kestrel-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
        # Scheduler + worker are now up. The warmup pipeline below
        # loops back through ``query()`` → ``_ensure_started``; the
        # ``_init_task is current_task`` check there keeps that
        # recursive call from re-entering this body.
        await self._warmup_query_pipeline()
        if reporter is not None:
            reporter.start()
            self._photon_reporter = reporter
        # Flip the guard last so a failure partway through (runtime
        # construction, warmup, photon) leaves the engine retry-able
        # rather than wedged in a half-initialized state.
        self._initialized = True
        # Drop the reference to the completed init task so its
        # captured locals can be garbage-collected.
        self._init_task = None

    def _disable_cloud_adapter_provider(self) -> None:
        if self._adapter_provider is None:
            return
        _LOGGER.warning(
            "Finetune inference is disabled because no valid MOONDREAM_API_KEY "
            "is available."
        )
        self._adapter_provider = None
        self._adapter_provider_uses_api_key = False

    def _build_configured_runtimes(self, max_lora_rank: Optional[int]) -> None:
        """Build each configured model's runtime from its registered spec.

        Synchronous (called via ``run_in_executor`` during init) and
        idempotent: models already in ``_runtimes`` (e.g. an injected
        default) are skipped. The default model is autoregressive;
        co-hosted models may be single-pass.
        """
        for model_id in self._model_ids:
            if model_id in self._runtimes:
                continue
            self._runtimes[model_id] = self._build_runtime(model_id, max_lora_rank)

    def _build_runtime(self, model_id: str, max_lora_rank: Optional[int]) -> Runtime:
        """Construct one model's runtime via its ``ModelSpec.runtime`` factory.

        The default model reuses ``runtime_cfg`` (already resolved for it).
        A co-hosted model gets a per-model config — its own ``model`` id, and
        ``model_path``/``tokenizer_path`` reset so it resolves for that model
        (a single-pass spec declares no weight file, so its path stays
        ``None`` and the factory owns loading). Runtime factories receive
        common engine-owned resources such as the compute stream and KV pool.
        ``max_lora_rank`` is the default autoregressive model's concern,
        supplied by the adapter provider only for that model.
        """
        from kestrel.models import get_spec

        spec = get_spec(model_id)
        kwargs: dict[str, Any] = {
            "compute_stream": self._compute_stream,
            "kv_pool": self._shared_kv_pool(),
        }
        if model_id == self._default_model:
            return spec.runtime(
                self._runtime_cfg,
                max_lora_rank=max_lora_rank,
                **kwargs,
            )
        cfg = replace(
            self._runtime_cfg,
            model=model_id,
            model_path=None,
            tokenizer_path=None,
        )
        return spec.runtime(cfg, **kwargs)

    def _shared_kv_pool(self):
        if self._kv_pool is None:
            from kestrel.kv_cache import KVMemoryPool

            self._kv_pool = KVMemoryPool(device=self._runtime_cfg.resolved_device())
        return self._kv_pool

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
        if self._photon_reporter is not None:
            try:
                await self._photon_reporter.shutdown()
            except Exception:
                _LOGGER.exception("Failed to stop Photon reporter")
            finally:
                self._photon_reporter = None
        for runtime in self._runtimes.values():
            runtime.shutdown()

    async def _run_skill(
        self,
        skill_name: str,
        *,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
        stream: bool = False,
    ) -> "EngineResult | EngineStream":
        """Validate + build via the model's skill, then submit.

        The skill (model-owned) validates the raw ``prompt``/``settings``
        and assembles its request + sampling params; the engine adds the
        decode-pipeline concerns it owns (adapter, logprobs, generated
        prefix, suppressed tokens) and submits. The kernel never builds or
        inspects a model's request type.
        """
        built = self._skill_registry().resolve(skill_name).build_request(
            image, prompt, settings
        )
        adapter = self._extract_adapter_id(settings)
        return_logprobs = self._extract_logprobs(settings)
        generated_prefix = self._extract_generated_prefix(settings)
        suppress_next_token_ids = self._extract_suppress_next_token_ids(settings)
        # A skill may carry media it pulled out of its own prompt (e.g. an
        # image inside chat messages); prefer it over the top-level argument.
        effective_image = built.image if built.image is not None else image
        submit_fn = self.submit_streaming if stream else self.submit
        return await submit_fn(
            built.request_context,
            max_new_tokens=built.max_new_tokens,
            adapter=adapter,
            image=effective_image,
            temperature=built.temperature,
            top_p=built.top_p,
            _logprobs=return_logprobs,
            _generated_prefix=generated_prefix,
            _suppress_next_token_ids=suppress_next_token_ids,
            skill=skill_name,
        )

    async def submit(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        adapter: Optional[str] = None,
        image: Optional[np.ndarray | bytes] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        _logprobs: Optional[bool] = None,
        _generated_prefix: Optional[object] = None,
        _suppress_next_token_ids: Optional[Sequence[int]] = None,
    ) -> EngineResult:
        generated_prefix = self._normalize_generated_prefix(
            _generated_prefix,
            "_generated_prefix",
        )
        suppress_next_token_ids = self._normalize_suppress_next_token_ids(
            _suppress_next_token_ids,
            field_name="_suppress_next_token_ids",
        )
        future, _ = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            adapter=adapter,
            image=image,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=_logprobs,
            generated_prefix=generated_prefix,
            suppress_next_token_ids=suppress_next_token_ids,
            stream_queue=None,
            skill=skill,
        )
        return await future

    @overload
    async def query(
        self,
        image: Optional[np.ndarray | bytes] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[True] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream": ...

    @overload
    async def query(
        self,
        image: Optional[np.ndarray | bytes] = ...,
        question: Optional[str] = ...,
        reasoning: bool = ...,
        spatial_refs: Optional[Sequence[Sequence[float]]] = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult: ...

    @overload
    async def query(
        self,
        image: Optional[np.ndarray | bytes] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]: ...

    async def query(
        self,
        image: Optional[np.ndarray | bytes] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        return await self._run_skill(
            "query",
            image=image,
            prompt={
                "question": question,
                "reasoning": reasoning,
                "stream": stream,
                "spatial_refs": spatial_refs,
            },
            settings=settings,
            stream=stream,
        )

    async def chat(
        self,
        messages: Optional[Sequence[Mapping[str, object]]] = None,
        *,
        reasoning: Optional[bool] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        # Omit ``reasoning`` from the prompt when the caller doesn't set it, so
        # ``ChatSkill.build_request`` falls back to the skill's
        # ``default_reasoning`` (e.g. MoondreamChatSkill defaults it on) —
        # matching ``ModelHandle.chat()``, which preserves the model default by
        # not passing the key.
        prompt: dict[str, object] = {"messages": messages, "stream": stream}
        if reasoning is not None:
            prompt["reasoning"] = reasoning
        return await self._run_skill(
            "chat",
            image=None,
            prompt=prompt,
            settings=settings,
            stream=stream,
        )

    async def point(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
    ) -> EngineResult:
        return await self._run_skill(
            "point",
            image=image,
            prompt={"object": object, "spatial_refs": spatial_refs},
            settings=settings,
        )

    @overload
    async def caption(
        self,
        image: np.ndarray | bytes,
        *,
        length: str = ...,
        stream: Literal[True],
        settings: Optional[Mapping[str, object]] = ...,
    ) -> "EngineStream": ...

    @overload
    async def caption(
        self,
        image: np.ndarray | bytes,
        *,
        length: str = ...,
        stream: Literal[False] = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> EngineResult: ...

    @overload
    async def caption(
        self,
        image: np.ndarray | bytes,
        *,
        length: str = ...,
        stream: bool = ...,
        settings: Optional[Mapping[str, object]] = ...,
    ) -> Union[EngineResult, EngineStream]: ...

    async def caption(
        self,
        image: np.ndarray | bytes,
        *,
        length: str = "normal",
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> Union[EngineResult, EngineStream]:
        return await self._run_skill(
            "caption",
            image=image,
            prompt={"length": length, "stream": stream},
            settings=settings,
            stream=stream,
        )

    async def detect(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
    ) -> EngineResult:
        return await self._run_skill(
            "detect",
            image=image,
            prompt={"object": object},
            settings=settings,
        )

    async def segment(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        settings: Optional[Mapping[str, object]] = None,
    ) -> EngineResult:
        return await self._run_skill(
            "segment",
            image=image,
            prompt={"object": object, "spatial_refs": spatial_refs},
            settings=settings,
        )

    async def submit_streaming(
        self,
        request_context: object,
        *,
        max_new_tokens: int,
        skill: str,
        adapter: Optional[str] = None,
        image: Optional[np.ndarray | bytes] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        _logprobs: Optional[bool] = None,
        _generated_prefix: Optional[object] = None,
        _suppress_next_token_ids: Optional[Sequence[int]] = None,
    ) -> EngineStream:
        queue: _StreamQueue = asyncio.Queue()
        generated_prefix = self._normalize_generated_prefix(
            _generated_prefix,
            "_generated_prefix",
        )
        suppress_next_token_ids = self._normalize_suppress_next_token_ids(
            _suppress_next_token_ids,
            field_name="_suppress_next_token_ids",
        )
        future, request_id = await self._submit_request(
            max_new_tokens=max_new_tokens,
            request_context=request_context,
            adapter=adapter,
            image=image,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=_logprobs,
            generated_prefix=generated_prefix,
            suppress_next_token_ids=suppress_next_token_ids,
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
        synchronize(self.runtime.device)

    def resume(self) -> None:
        """Resume scheduler progress after a pause."""

        if self._shutdown:
            return
        synchronize(self.runtime.device)
        self._paused_event.clear()
        self._paused_flag.clear()
        self._run_gate.set()
        self._scheduler_event.set()

    async def run(self, model: str, task: str, inputs: Any) -> EngineResult:
        """Run a single-pass ``task`` on ``model`` and await its result.

        The low-level multi-model entry point: routes to the model's
        single-pass lane, which runs one ``forward`` and returns the
        structured result. (Autoregressive models keep using ``submit`` /
        the typed verbs.)
        """
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        await self._ensure_started()

        runtime = self._runtimes.get(model)
        if runtime is None:
            raise ValueError(f"Unknown model {model!r}")
        if runtime.execution_shape is not ExecutionShape.SINGLE_PASS:
            raise ValueError(
                f"Model {model!r} is not a single-pass model "
                f"(execution_shape={runtime.execution_shape.value})"
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[EngineResult] = loop.create_future()
        req = _SinglePassRequest(
            request_id=next(self._request_ids),
            future=future,
            task=task,
            inputs=inputs,
            submitted_at=time.perf_counter(),
        )
        self._raise_if_scheduler_failed()
        self._single_pass_queue.put((model, req))
        self._scheduler_event.set()
        if self._scheduler_error is not None:
            self._fail_all_pending(self._scheduler_failed_error())
        return await future

    async def stream(
        self,
        model: str,
        task: str,
        initial_inputs: Mapping[str, object],
    ) -> ModelStream:
        """Start a stateful streaming ``task`` on ``model``.

        This is distinct from autoregressive response streaming: the
        caller supplies chunks/frames after the initial prompt, and a
        streaming runtime carries model-owned state across those chunks.
        The returned ``ModelStream`` is caller-driven: callers ``send``
        chunks/frames and iterate over model-defined updates.
        """
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        await self._ensure_started()

        runtime = self._runtimes.get(model)
        if runtime is None:
            raise ValueError(f"Unknown model {model!r}")
        if runtime.execution_shape is not ExecutionShape.STREAMING:
            raise ValueError(
                f"Model {model!r} is not a streaming model "
                f"(execution_shape={runtime.execution_shape.value})"
            )
        if task not in tuple(runtime.tasks()):
            supported = ", ".join(tuple(runtime.tasks())) or "none"
            raise ValueError(
                f"Model {model!r} does not support {task!r} "
                f"(supports: {supported})"
            )

        loop = asyncio.get_running_loop()
        session_id = next(self._request_ids)
        queue: _ModelStreamQueue = asyncio.Queue()
        future: asyncio.Future[EngineResult] = loop.create_future()
        req = _StreamingSessionRequest(
            request_id=session_id,
            future=future,
            task=task,
            initial_inputs=dict(initial_inputs),
            submitted_at=time.perf_counter(),
            model_stream_queue=queue,
        )
        self._model_stream_models[session_id] = model
        self._model_stream_queues[session_id] = queue
        self._streaming_start_queue.put((model, req))
        self._scheduler_event.set()
        return ModelStream(
            session_id=session_id,
            task=task,
            queue=queue,
            result_future=future,
            send_chunk=self._send_model_stream_chunk,
            close_session=self._close_model_stream,
        )

    async def _send_model_stream_chunk(
        self,
        session_id: int,
        chunk: Dict[str, Any],
    ) -> None:
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        model = self._model_stream_models.get(session_id)
        if model is None:
            raise RuntimeError(f"Unknown or closed model stream {session_id}")
        self._streaming_chunk_queue.put(
            (model, _StreamingChunk(session_id=session_id, inputs=chunk))
        )
        self._scheduler_event.set()

    async def _close_model_stream(self, session_id: int) -> None:
        model = self._model_stream_models.get(session_id)
        if model is None:
            return
        self._streaming_chunk_queue.put(
            (model, _StreamingChunk(session_id=session_id, close=True))
        )
        self._scheduler_event.set()

    def model(self, model_id: Optional[str] = None) -> "ModelHandle":
        """Return a :class:`ModelHandle` bound to ``model_id``.

        The primary surface: bind a model once, then call its capability
        methods (``query`` / ``segment_masks`` / ...) or ``run``. Defaults
        to the configured default model.

        Validates against the *configured* models, so a handle can be taken
        before the engine is started (runtimes are built lazily in
        ``_initialize``); the handle's async methods trigger startup. Raises
        if the id isn't a configured model.
        """
        target = model_id or self._default_model
        if target not in self._configured_models():
            known = ", ".join(sorted(self._configured_models())) or "none"
            raise ValueError(f"Unknown model {target!r} (configured: {known})")
        return ModelHandle(self, target)

    def _configured_models(self) -> set[str]:
        """Model ids this engine serves — built or pending build.

        Built runtimes plus every configured model id (the default and any
        co-hosted ``models``), so model() works both before and after
        ``_initialize`` populates ``_runtimes``.
        """
        return set(self._runtimes) | set(self._model_ids)

    def _tasks_for(self, model_id: str) -> tuple[str, ...]:
        """Capability names a registered model serves.

        The default model reports exactly what its autoregressive verbs
        will run — ``_skill_registry()`` (the test override, the built
        runtime's skills, or the spec's). This keeps reported capabilities
        consistent with execution, works before startup, and does not
        require ``tasks()`` on the autoregressive runtime protocol. Every
        other non-autoregressive model advertises its own ``runtime.tasks()``
        once built.
        """
        if model_id == self._default_model:
            return self._skill_registry().names()
        runtime = self._runtimes.get(model_id)
        if runtime is not None:
            return tuple(runtime.tasks())
        raise ValueError(f"Unknown model {model_id!r}")

    async def _submit_request(
        self,
        *,
        max_new_tokens: int,
        request_context: object,
        adapter: Optional[str],
        image: Optional[np.ndarray | bytes],
        temperature: Optional[float],
        top_p: Optional[float],
        return_logprobs: Optional[bool],
        generated_prefix: GeneratedPrefix,
        suppress_next_token_ids: Optional[tuple[int, ...]],
        stream_queue: Optional[_StreamQueue],
        skill: str,
    ) -> Tuple[asyncio.Future[EngineResult], int]:
        if self._shutdown:
            raise RuntimeError("InferenceEngine is shut down")
        await self._ensure_started()

        loop = asyncio.get_running_loop()
        req_id = next(self._request_ids)
        future: asyncio.Future[EngineResult] = loop.create_future()

        skill_spec = self._skill_registry().resolve(skill)
        adapter_id = self._normalize_adapter_id(adapter)
        if self._adapter_provider is None and adapter_id is not None:
            _LOGGER.warning(
                "Adapter %r was requested, but finetune inference is disabled. "
                "Set MOONDREAM_API_KEY or pass an adapter_provider to enable it.",
                adapter_id,
            )
            raise NotImplementedError(
                "Finetune inference is disabled because no authenticated "
                "adapter provider is available."
            )

        image_obj = None
        if image is not None:
            if self.runtime.image_prefix_length == 0:
                raise ValueError("Runtime does not support image inputs")
            # A skill may pass several images (an ordered list, e.g. chat with
            # multiple images); the runtime that accepts them unpacks the list.
            items = image if isinstance(image, (list, tuple)) else [image]
            if not items or not all(isinstance(one, (np.ndarray, bytes)) for one in items):
                raise TypeError("image must be an np.ndarray/bytes, or a list of them")
            image_obj = image

        prompt_str = skill_spec.prompt_text(request_context)
        tokens = list(skill_spec.build_prompt_tokens(self.runtime, request_context))
        norm_temperature = self._normalize_temperature(temperature)
        norm_top_p = self._normalize_top_p(top_p)
        self._warn_if_outside_mps_sampler_envelope(norm_temperature, norm_top_p)
        self._validate_generated_prefix_for_request(
            generated_prefix,
            max_new_tokens=max_new_tokens,
            return_logprobs=return_logprobs,
            streaming=stream_queue is not None,
        )
        payload = _AutoregressiveRequest(
            request_id=req_id,
            prompt=prompt_str,
            prompt_tokens=tokens,
            image=image_obj,
            image_hash=None,  # Computed in scheduler thread if prefix cache enabled
            max_new_tokens=max_new_tokens,
            temperature=norm_temperature,
            top_p=norm_top_p,
            submitted_at=time.perf_counter(),
            future=future,
            stream_queue=stream_queue,
            skill=skill_spec,
            request_context=request_context,
            adapter=adapter_id,
            return_logprobs=return_logprobs,
            generated_prefix=generated_prefix,
            suppress_next_token_ids=suppress_next_token_ids,
        )
        await self._queue.put(payload)
        return future, req_id

    async def _ensure_started(self) -> None:
        self._raise_if_scheduler_failed()
        if self._initialized:
            return

        # If init is already in flight on the current task (warmup
        # pipeline calling back through ``query()``), bail without
        # awaiting — awaiting our own task deadlocks.
        current = asyncio.current_task()
        if self._init_task is not None and self._init_task is current:
            return

        # If another caller has init in flight, wait for it.
        if self._init_task is not None and not self._init_task.done():
            await asyncio.shield(self._init_task)
            self._raise_if_scheduler_failed()
            if self._initialized:
                return
            # Init failed; fall through to retry below.

        if self._init_task is None or self._init_task.done():
            # Run ``_initialize`` as a Task so concurrent callers can
            # wait on the same object (and ``current_task() is
            # self._init_task`` catches the warmup recursion above).
            self._init_task = asyncio.create_task(self._initialize())
        await asyncio.shield(self._init_task)
        self._raise_if_scheduler_failed()

    def _scheduler_failed_error(self) -> RuntimeError:
        exc = RuntimeError("InferenceEngine scheduler is not running")
        exc.__cause__ = self._scheduler_error
        return exc

    def _raise_if_scheduler_failed(self) -> None:
        if self._scheduler_error is not None:
            raise self._scheduler_failed_error()

    async def _worker_loop(self) -> None:
        shutdown_error = RuntimeError("Engine shut down")

        while True:
            request = await self._queue.get()
            if request is None:
                self._scheduler_queue.put(None)
                self._scheduler_event.set()
                break
            if self._scheduler_error is not None:
                self._fail_request(request, self._scheduler_failed_error())
                continue
            self._scheduler_queue.put(request)
            self._scheduler_event.set()
            if self._scheduler_error is not None:
                self._fail_all_pending(self._scheduler_failed_error())

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

    # Settings outside this envelope cause the MPS fused sampler to
    # silently bias outputs (top-K=64 may not span the nucleus). Logged
    # once per (temperature, top_p) tuple seen so a chat session with
    # a warm prompt doesn't spam.
    _MPS_SAMPLER_TOP_P_MAX = 0.95
    _MPS_SAMPLER_TEMP_MAX = 1.0
    _mps_sampler_warned: set[Tuple[float, float]] = set()

    def _warn_if_outside_mps_sampler_envelope(
        self, temperature: float, top_p: float,
    ) -> None:
        # _submit_request awaits _ensure_started before calling us, so
        # the default runtime is set by here.
        if self.runtime.device.type != "mps":
            return
        # Greedy decoding (``temperature == 0``) hits the scheduler's
        # ``torch.argmax`` short-circuit and never reaches the fused
        # MPS sampler — top_p is irrelevant. ``detect`` / ``point`` /
        # ``segment`` defaults sit here, so warning would be noise.
        if temperature <= 0.0:
            return
        if (
            temperature <= self._MPS_SAMPLER_TEMP_MAX
            and top_p <= self._MPS_SAMPLER_TOP_P_MAX
        ):
            return
        key = (temperature, top_p)
        if key in self._mps_sampler_warned:
            return
        self._mps_sampler_warned.add(key)
        _LOGGER.warning(
            "MPS sampler may produce biased outputs for "
            "temperature=%.3g, top_p=%.3g. The fused Metal kernel only "
            "inspects the top-K=64 candidates and is calibrated for "
            "top_p ≤ %.2f / temperature ≤ %.1f; settings outside that "
            "envelope can sample from a truncated nucleus. See %s for "
            "details.",
            temperature, top_p,
            self._MPS_SAMPLER_TOP_P_MAX, self._MPS_SAMPLER_TEMP_MAX,
            _MPS_SAMPLER_DOCS_URL,
        )

    def _normalize_adapter_id(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("adapter must be a non-empty string")
        return normalized

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

    def _extract_logprobs(
        self, settings: Optional[Mapping[str, object]]
    ) -> Optional[bool]:
        if settings is None or "_logprobs" not in settings:
            return None
        raw = settings["_logprobs"]
        if raw is None:
            return None
        if not isinstance(raw, bool):
            raise TypeError("settings._logprobs must be a bool or None")
        return raw

    def _extract_generated_prefix(
        self, settings: Optional[Mapping[str, object]]
    ) -> GeneratedPrefix:
        if settings is None or "_generated_prefix" not in settings:
            return GeneratedPrefix()
        return self._normalize_generated_prefix(
            settings["_generated_prefix"],
            "settings._generated_prefix",
        )

    def _normalize_generated_prefix(
        self,
        raw: Optional[object],
        field_name: str,
    ) -> GeneratedPrefix:
        if raw is None:
            return GeneratedPrefix()
        if isinstance(raw, GeneratedPrefix):
            raw_tokens = raw.tokens
            raw_logprobs = raw.logprobs
        else:
            if not isinstance(raw, Mapping):
                raise TypeError(f"{field_name} must be a mapping or None")

            allowed_keys = {"tokens", "logprobs"}
            extra_keys = set(raw) - allowed_keys
            if extra_keys:
                names = ", ".join(sorted(str(key) for key in extra_keys))
                raise TypeError(f"{field_name} has unsupported key(s): {names}")

            if "tokens" not in raw:
                raise ValueError(f"{field_name}.tokens is required")
            raw_tokens = raw["tokens"]
            raw_logprobs = raw.get("logprobs")

        tokens = self._normalize_generated_prefix_tokens(
            raw_tokens,
            f"{field_name}.tokens",
        )
        logprobs = self._normalize_generated_prefix_logprobs(
            raw_logprobs,
            f"{field_name}.logprobs",
            expected_length=len(tokens),
        )
        return GeneratedPrefix(tokens=tokens, logprobs=logprobs)

    def _normalize_generated_prefix_tokens(
        self,
        raw: object,
        field_name: str,
    ) -> tuple[Token, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"{field_name} must be a sequence of generated Token objects"
            )
        tokens = tuple(raw)
        for token in tokens:
            if not isinstance(token, (TextToken, CoordToken, SizeToken)):
                raise TypeError(
                    f"{field_name} must contain only generated Token objects"
                )
        return tokens

    def _normalize_generated_prefix_logprobs(
        self,
        raw: object,
        field_name: str,
        *,
        expected_length: int,
    ) -> Optional[tuple[float, ...]]:
        if raw is None:
            return None
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise TypeError(f"{field_name} must be a sequence of floats or None")
        logprobs = tuple(float(value) for value in raw)
        if len(logprobs) != expected_length:
            raise ValueError(
                f"{field_name} must have the same length as generated prefix tokens"
            )
        return logprobs

    def _validate_generated_prefix_for_request(
        self,
        generated_prefix: GeneratedPrefix,
        *,
        max_new_tokens: int,
        return_logprobs: Optional[bool],
        streaming: bool,
    ) -> None:
        if not generated_prefix.tokens:
            return
        if streaming:
            raise ValueError("settings._generated_prefix is not supported with streaming")
        if len(generated_prefix.tokens) >= max_new_tokens:
            raise ValueError(
                "settings._generated_prefix.tokens must be shorter than max_tokens"
            )
        if return_logprobs is True and generated_prefix.logprobs is None:
            raise ValueError(
                "settings._generated_prefix.logprobs is required when "
                "settings._logprobs is true"
            )

        eos_id = self.runtime.prompt_template.eos_id
        for token in generated_prefix.tokens:
            if isinstance(token, TextToken) and token.token_id == eos_id:
                raise ValueError(
                    "settings._generated_prefix.tokens must not contain EOS"
                )

    def _extract_suppress_next_token_ids(
        self, settings: Optional[Mapping[str, object]]
    ) -> Optional[tuple[int, ...]]:
        if settings is None or "_suppress_next_token_ids" not in settings:
            return None
        return self._normalize_suppress_next_token_ids(
            settings["_suppress_next_token_ids"],
            field_name="settings._suppress_next_token_ids",
        )

    def _normalize_suppress_next_token_ids(
        self,
        raw: object,
        *,
        field_name: str,
    ) -> Optional[tuple[int, ...]]:
        if raw is None:
            return None
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"{field_name} must be a sequence of non-negative token ids or None"
            )

        seen: set[int] = set()
        token_ids: list[int] = []
        for idx, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name}[{idx}] must be an int token id")
            if value < 0:
                raise ValueError(f"{field_name}[{idx}] must be non-negative")
            if value not in seen:
                seen.add(value)
                token_ids.append(value)
        if not token_ids:
            return None
        return tuple(token_ids)

    def _build_stream_callback(
        self, req: _AutoregressiveRequest
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
        req: EngineRequest,
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

    def _complete_model_stream(
        self,
        req: EngineRequest,
        *,
        result: Optional[EngineResult] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        queue = getattr(req, "model_stream_queue", None)
        if queue is None:
            return
        req.model_stream_queue = None
        self._model_stream_models.pop(req.request_id, None)
        self._model_stream_queues.pop(req.request_id, None)
        completion = _ModelStreamCompletion(result=result, error=error)
        self._loop.call_soon_threadsafe(queue.put_nowait, completion)

    def _deliver_model_stream_update(
        self,
        update: ModelStreamUpdate,
    ) -> None:
        queue = self._model_stream_queues.get(update.session_id)
        if queue is None:
            return
        self._loop.call_soon_threadsafe(queue.put_nowait, update)

    def _fail_request(self, req: EngineRequest, error: BaseException) -> None:
        future = req.future
        if future and not future.done():
            assert self._loop is not None
            self._loop.call_soon_threadsafe(future.set_exception, error)
        self._complete_stream(req, error=error)
        self._complete_model_stream(req, error=error)
        if self._photon_reporter is not None:
            try:
                self._photon_reporter.record_error(finetune=req.adapter)
            except Exception:
                _LOGGER.exception("Failed to record Photon error telemetry")


    def _scheduler_loop(self) -> None:
        try:
            runtime = self._runtimes.get(self._default_model)
            if runtime is None:
                raise RuntimeError(
                    f"No runtime registered for default model {self._default_model!r}"
                )
            set_device(runtime.device)
            # The default (autoregressive) lane. Its pipeline owns the device's
            # CUDA-graph capture, so pause/drain is handled specially below.
            ar_executor = AutoregressiveExecutor(
                runtime,
                compute_stream=self._compute_stream,
                skills=self._skill_registry(),
                adapter_provider=self._adapter_provider,
                build_generation_request=self._build_generation_request,
                to_engine_result=self._to_engine_result,
                wake_event=self._scheduler_event,
            )
            # One single-pass lane per registered single-pass model. The kernel
            # folds advance() over every lane; single-pass forwards interleave
            # with autoregressive decode on the shared stream.
            single_pass: Dict[str, SinglePassExecutor] = {
                name: SinglePassExecutor(rt, compute_stream=self._compute_stream)
                for name, rt in self._runtimes.items()
                if rt.execution_shape is ExecutionShape.SINGLE_PASS
            }
            streaming: Dict[str, StreamingExecutor] = {
                name: StreamingExecutor(rt, compute_stream=self._compute_stream)
                for name, rt in self._runtimes.items()
                if rt.execution_shape is ExecutionShape.STREAMING
            }
        except Exception as exc:
            self._scheduler_error = exc
            _LOGGER.exception("Scheduler startup failed", exc_info=exc)
            self._fail_all_pending(self._scheduler_failed_error())
            return

        shutdown_requested = False
        wake_event = self._scheduler_event
        run_gate = self._run_gate
        paused_flag = self._paused_flag
        paused_event = self._paused_event

        def deliver(completions: tuple[Completion, ...]) -> None:
            loop = self._loop
            assert loop is not None
            for c in completions:
                req = c.request
                if c.error is not None:
                    self._fail_request(req, c.error)
                    continue
                engine_result = c.result
                assert engine_result is not None
                if self._photon_reporter is not None:
                    try:
                        self._photon_reporter.record_success(
                            finetune=req.adapter,
                            input_tokens=engine_result.metrics.input_tokens,
                            output_tokens=engine_result.metrics.output_tokens,
                        )
                    except Exception:
                        _LOGGER.exception("Failed to record Photon usage telemetry")
                future = req.future
                if future and not future.done():
                    loop.call_soon_threadsafe(future.set_result, engine_result)
                self._complete_stream(req, result=engine_result)
                self._complete_model_stream(req, result=engine_result)

        def deliver_model_stream_updates(
            updates: tuple[ModelStreamUpdate, ...]
        ) -> None:
            for update in updates:
                self._deliver_model_stream_update(update)

        def any_work() -> bool:
            return ar_executor.has_work or any(
                sp.has_work for sp in single_pass.values()
            ) or any(stream.has_work for stream in streaming.values())

        try:
            with torch.inference_mode():
                while True:
                    # If paused, wait until resumed or shutdown completes.
                    if paused_flag.is_set():
                        # Drain in-flight work before pause — callers may
                        # mutate runtime state while paused (e.g. rebuild
                        # CUDA graphs). Only the AR lane has graph state;
                        # single-pass forwards are one-shot, so draining the
                        # AR pipeline is sufficient.
                        deliver(ar_executor.drain())
                        with runtime.graph_capture_lock:
                            synchronize(runtime.device)
                        paused_event.set()
                        if shutdown_requested and not any_work():
                            break
                        run_gate.wait(timeout=0.1)
                        continue

                    progressed = False
                    # AR ingress (untagged — always the default model).
                    while True:
                        try:
                            item = self._scheduler_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            shutdown_requested = True
                            continue
                        ar_executor.submit(item)
                        progressed = True
                    # Single-pass ingress (model-tagged).
                    while True:
                        try:
                            model, req = self._single_pass_queue.get_nowait()
                        except queue.Empty:
                            break
                        lane = single_pass.get(model)
                        if lane is None:  # pragma: no cover - guarded in run()
                            self._fail_request(
                                req, ValueError(f"No single-pass lane for {model!r}")
                            )
                        else:
                            lane.submit(req)
                        progressed = True
                    # Streaming session starts (model-tagged).
                    while True:
                        try:
                            model, req = self._streaming_start_queue.get_nowait()
                        except queue.Empty:
                            break
                        lane = streaming.get(model)
                        if lane is None:  # pragma: no cover - guarded in stream()
                            self._fail_request(
                                req, ValueError(f"No streaming lane for {model!r}")
                            )
                        else:
                            lane.submit(req)
                        progressed = True
                    # Streaming chunks / close commands (model-tagged).
                    while True:
                        try:
                            model, chunk = self._streaming_chunk_queue.get_nowait()
                        except queue.Empty:
                            break
                        lane = streaming.get(model)
                        if lane is not None:
                            lane.submit_chunk(chunk)
                        progressed = True

                    # Advance the autoregressive lane. Its pipeline is the
                    # shared device state, so a failure here is fatal to the
                    # kernel — tear everything down.
                    try:
                        tick = ar_executor.advance()
                    except Exception as exc:
                        self._scheduler_error = exc
                        _LOGGER.exception("Autoregressive advance failed", exc_info=exc)
                        deliver(ar_executor.shutdown(exc))
                        for lane in single_pass.values():
                            deliver(lane.shutdown(exc))
                        for lane in streaming.values():
                            deliver(lane.shutdown(exc))
                        self._fail_all_pending(exc)
                        return
                    if tick.completed:
                        deliver(tick.completed)
                    progressed = tick.progressed or progressed

                    # Advance each single-pass lane in isolation: a single
                    # forward blowing up must fail only that lane's in-flight
                    # work, never take down the kernel or the other lanes.
                    for name, lane in single_pass.items():
                        try:
                            sp_tick = lane.advance()
                        except Exception as exc:
                            _LOGGER.exception(
                                "Single-pass advance failed for %s", name, exc_info=exc
                            )
                            deliver(lane.shutdown(exc))
                            progressed = True
                            continue
                        if sp_tick.completed:
                            deliver(sp_tick.completed)
                        progressed = sp_tick.progressed or progressed

                    # Advance each streaming lane in isolation; a runtime
                    # session failure should fail only that lane/session.
                    for name, lane in streaming.items():
                        try:
                            stream_tick = lane.advance()
                        except Exception as exc:
                            _LOGGER.exception(
                                "Streaming advance failed for %s", name, exc_info=exc
                            )
                            deliver(lane.shutdown(exc))
                            progressed = True
                            continue
                        if stream_tick.model_stream_updates:
                            deliver_model_stream_updates(
                                stream_tick.model_stream_updates
                            )
                        if stream_tick.completed:
                            deliver(stream_tick.completed)
                        progressed = stream_tick.progressed or progressed

                    if shutdown_requested and not any_work():
                        break

                    if not progressed:
                        if shutdown_requested:
                            break
                        # A launched single-pass forward signals completion
                        # via a CUDA event, which sets no host event — so if
                        # any lane has in-flight event-backed work, poll on a
                        # short timeout instead of blocking, or the kernel
                        # could sleep past the GPU finishing and leave
                        # engine.run() unresolved until an unrelated request
                        # happens to wake the loop.
                        if any(
                            sp.has_in_flight for sp in single_pass.values()
                        ) or any(
                            stream.has_in_flight for stream in streaming.values()
                        ):
                            wake_event.wait(timeout=_SINGLE_PASS_POLL_INTERVAL_S)
                        else:
                            wake_event.wait()
                        wake_event.clear()
        finally:
            deliver(ar_executor.shutdown())
            for lane in single_pass.values():
                deliver(lane.shutdown())
            for lane in streaming.values():
                deliver(lane.shutdown())
            self._fail_all_pending()

    def _build_generation_request(
        self,
        runtime: AutoregressiveRuntime,
        req: _AutoregressiveRequest,
        image_crops: Any,
    ) -> tuple[GenerationRequest, SkillState]:
        prompt_tokens = req.prompt_tokens
        stream_cb = self._build_stream_callback(req)
        if req.image is None and image_crops is None:
            image_length = 0
        elif isinstance(req.image, (list, tuple)):
            # Multi-image chat: each image expands one ImageMarker token — which
            # is already counted in prompt_length — into an image_prefix_length
            # patch block, so it adds image_prefix_length - 1 KV positions per
            # image. (target_length = prompt_length + image_length + max_new.)
            image_length = len(req.image) * (runtime.image_prefix_length - 1)
        else:
            image_length = runtime.image_prefix_length
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
            return_logprobs=req.return_logprobs,
            generated_prefix=req.generated_prefix,
            suppress_next_token_ids=req.suppress_next_token_ids,
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
        for token in request_obj.generated_prefix.tokens:
            skill_state.consume_step(
                runtime,
                DecodeStep(token=token, position=skill_state.token_count),
            )
        self._validate_suppress_next_token_ids(
            runtime,
            request_obj,
            skill_state,
        )
        return request_obj, skill_state

    def _validate_suppress_next_token_ids(
        self,
        runtime: AutoregressiveRuntime,
        request: GenerationRequest,
        skill_state: SkillState,
    ) -> None:
        suppress = request.suppress_next_token_ids
        if not suppress:
            return

        vocab_size = int(runtime.vocab_size)
        for token_id in suppress:
            if token_id >= vocab_size:
                raise ValueError(
                    "_suppress_next_token_ids contains token id "
                    f"{token_id}, but vocab size is {vocab_size}"
                )

        allowed = skill_state.allowed_token_ids(runtime)
        skill_suppressed = skill_state.suppressed_token_ids(runtime) or ()
        if allowed:
            remaining = (
                set(int(token_id) for token_id in allowed)
                - set(int(token_id) for token_id in skill_suppressed)
                - set(suppress)
            )
            if not remaining:
                raise ValueError(
                    "_suppress_next_token_ids removed every allowed next token"
                )
        else:
            banned = set(int(token_id) for token_id in skill_suppressed)
            banned.update(suppress)
            if len(banned) >= vocab_size:
                raise ValueError("_suppress_next_token_ids removed every next token")

    def _to_engine_result(self, result: SchedulerResult) -> EngineResult:
        sched_metrics = result.metrics
        prefill_time_ms = max(sched_metrics.prefill_time_ms, 0.0)
        decode_time_ms = max(sched_metrics.decode_time_ms, 0.0)
        ttft_ms = max(sched_metrics.ttft_ms, 0.0)
        request_time_ms = max(sched_metrics.request_time_ms, 0.0)
        metrics = EngineMetrics(
            input_tokens=sched_metrics.prompt_tokens,
            output_tokens=sched_metrics.decode_tokens,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            ttft_ms=ttft_ms,
            request_time_ms=request_time_ms,
            cached_tokens=sched_metrics.cached_tokens,
        )
        return EngineResult(
            request_id=result.request_id,
            tokens=result.tokens,
            finish_reason=result.finish_reason,
            metrics=metrics,
            output=result.output,
            logprobs=result.logprobs,
        )

    def _fail_all_pending(
        self,
        error: Optional[BaseException] = None,
    ) -> None:
        """Fail any requests still sitting in the ingress queue.

        In-flight requests are owned by the executor (failed via its
        ``shutdown``); this drains only what the kernel hasn't handed off
        yet.
        """
        exc = error or RuntimeError("Engine shut down")
        while True:
            try:
                pending = self._scheduler_queue.get_nowait()
            except queue.Empty:
                break
            if pending is None:
                continue
            self._fail_request(pending, exc)
        while True:
            try:
                _model, req = self._single_pass_queue.get_nowait()
            except queue.Empty:
                break
            self._fail_request(req, exc)
        while True:
            try:
                _model, req = self._streaming_start_queue.get_nowait()
            except queue.Empty:
                break
            self._fail_request(req, exc)
        while True:
            try:
                self._streaming_chunk_queue.get_nowait()
            except queue.Empty:
                break
