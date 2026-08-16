"""Execution lanes: the Executor protocol and the autoregressive lane.

An executor is a kernel-side component wrapping one driver behind a
uniform face (``submit`` / ``advance`` -> :class:`TickResult` /
``shutdown``). The kernel loop folds ``advance`` over its executors and
performs the delivery effects for the :class:`Completion` values they
emit; executors themselves never touch the event loop.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from kestrel.runtime import AutoregressiveRuntime
from kestrel.scheduler import (
    GenerationScheduler,
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
    SchedulerResult,
)
from kestrel.skills import SkillRegistry, SkillState
from kestrel.models.moondream.lora import AdapterProvider

from kestrel.engine._types import (
    Completion,
    EngineResult,
    TickResult,
    _AutoregressiveRequest,
    _ReadyAdmission,
    _hash_image,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingAdmission:
    req: _AutoregressiveRequest
    crops_future: Future[Any] | None
    encoder_input_future: Future[Any] | None
    prefix_cache_hit: bool


class _AdmissionCoordinator:
    def __init__(
        self,
        runtime: AutoregressiveRuntime,
        wake_event: threading.Event,
        fail_request: Callable[[_AutoregressiveRequest, BaseException], None],
    ) -> None:
        self._runtime = runtime
        self._wake_event = wake_event
        self._fail_request = fail_request
        # Both preprocessing payloads are opaque. Image crops expand decoder
        # positions; an encoder input conditions an encoder-decoder model without
        # changing decoder length. A request becomes admissible only when every
        # preprocessing future it owns is ready.
        self._pending: Dict[int, _PendingAdmission] = {}
        self._ready: queue.Queue[int] = queue.Queue()

    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, req: _AutoregressiveRequest) -> Optional[_ReadyAdmission]:
        crops_future: Future[Any] | None = None
        encoder_input_future: Future[Any] | None = None
        prefix_cache_hit = False
        try:
            if isinstance(req.image, (list, tuple)):
                # Multi-image chat: the single-image prefix cache and
                # overlap-crop precompute don't apply. Decode/validate each
                # element up front so a bad image fails only this request.
                from kestrel.utils.image import decode_to_srgb

                req.image = tuple(decode_to_srgb(im) for im in req.image)
            elif req.image is not None:
                # Encoder-conditioned prompts are never eligible for decoder
                # prefix reuse: the cache key does not include encoder identity.
                if req.encoder_input is None and self._runtime.prefix_cache is not None:
                    req.image_hash = _hash_image(req.image)
                    prefill_tokens = list(req.prompt_tokens) + list(
                        req.generated_prefix.tokens
                    )
                    prefix_cache_hit = self._runtime.check_prefix_cache(
                        prefill_tokens, req.image_hash, req.adapter
                    )
                if not prefix_cache_hit:
                    crops_future = self._runtime.preprocess_image_async(req.image)

            if req.encoder_input is not None:
                encoder_input_future = self._runtime.preprocess_encoder_input_async(
                    req.encoder_input
                )
        except Exception as exc:
            if crops_future is not None and not crops_future.done():
                crops_future.cancel()
            if encoder_input_future is not None and not encoder_input_future.done():
                encoder_input_future.cancel()
            self._fail_request(req, exc)
            return None

        if crops_future is None and encoder_input_future is None:
            return _ReadyAdmission(
                req=req,
                crops=None,
                encoder_input=None,
                prefix_cache_hit=prefix_cache_hit,
            )

        req_id = req.request_id
        self._pending[req_id] = _PendingAdmission(
            req=req,
            crops_future=crops_future,
            encoder_input_future=encoder_input_future,
            prefix_cache_hit=prefix_cache_hit,
        )
        for future in (crops_future, encoder_input_future):
            if future is not None:
                future.add_done_callback(
                    lambda _future, rid=req_id: self._on_ready(rid)
                )
        return None

    def take_ready(self) -> Optional[_ReadyAdmission]:
        while True:
            try:
                req_id = self._ready.get_nowait()
            except queue.Empty:
                return None

            pending = self._pending.get(req_id)
            if pending is None:
                continue
            futures = (pending.crops_future, pending.encoder_input_future)
            failed: BaseException | None = None
            for future in futures:
                if future is None or not future.done():
                    continue
                try:
                    failed = future.exception()
                except BaseException as exc:
                    failed = exc
                if failed is not None:
                    break
            if failed is not None:
                self._pending.pop(req_id, None)
                for future in futures:
                    if future is not None and not future.done():
                        future.cancel()
                self._fail_request(pending.req, failed)
                continue
            if any(future is not None and not future.done() for future in futures):
                continue

            self._pending.pop(req_id, None)
            try:
                crops = (
                    pending.crops_future.result()
                    if pending.crops_future is not None
                    else None
                )
                encoder_input = (
                    pending.encoder_input_future.result()
                    if pending.encoder_input_future is not None
                    else None
                )
            except Exception as exc:
                for future in futures:
                    if future is not None and not future.done():
                        future.cancel()
                self._fail_request(pending.req, exc)
                continue
            # The raw input is no longer needed once preprocessing succeeds.
            # Keep only the model-owned prepared payload throughout generation.
            pending.req.encoder_input = None
            return _ReadyAdmission(
                req=pending.req,
                crops=crops,
                encoder_input=encoder_input,
                prefix_cache_hit=pending.prefix_cache_hit,
            )

    def fail_all(self, error: Optional[BaseException] = None) -> None:
        exc = error or RuntimeError("Engine shut down")
        for pending in list(self._pending.values()):
            for future in (pending.crops_future, pending.encoder_input_future):
                if future is not None and not future.done():
                    future.cancel()
            self._fail_request(pending.req, exc)
        self._pending.clear()
        self._drain_ready_notifications()

    def _on_ready(self, request_id: int) -> None:
        self._ready.put(request_id)
        self._wake_event.set()

    def _drain_ready_notifications(self) -> None:
        while True:
            try:
                self._ready.get_nowait()
            except queue.Empty:
                break


class Executor(Protocol):
    """A kernel-side lane wrapping one driver behind a uniform face.

    The kernel loop folds ``advance`` over its executors without knowing
    the execution shape. ``submit`` runs on the event-loop thread
    (thread-safe ingress); ``advance`` / ``shutdown`` run on the kernel
    thread. ``advance`` returns an immutable :class:`TickResult`; the
    kernel performs the effects for any ``completed`` entries.
    """

    def submit(self, request: "_AutoregressiveRequest") -> None: ...

    def advance(self) -> TickResult: ...

    def shutdown(self, error: Optional[BaseException] = ...) -> None: ...


class AutoregressiveExecutor:
    """Executor lane wrapping the autoregressive prefill/decode scheduler.

    Owns the :class:`GenerationScheduler`, the image-crop admission
    coordinator, and the in-flight request map. The pipelined-decode
    internals (``PipelineState``, ``launch_forward_async``,
    ``commit_step``, ping-pong slots) are untouched — this is a
    lift-and-wrap of the engine's former scheduler loop into the uniform
    :class:`Executor` face, with delivery turned into :class:`Completion`
    values the kernel acts on.
    """

    def __init__(
        self,
        runtime: AutoregressiveRuntime,
        *,
        compute_stream: Any,
        skills: "SkillRegistry",
        adapter_provider: Optional[AdapterProvider],
        build_generation_request: Callable[
            [AutoregressiveRuntime, "_AutoregressiveRequest", Any, Any],
            "tuple[GenerationRequest, SkillState]",
        ],
        to_engine_result: Callable[[SchedulerResult], EngineResult],
        wake_event: threading.Event,
    ) -> None:
        self._runtime = runtime
        self._build_generation_request = build_generation_request
        self._to_engine_result = to_engine_result
        self._scheduler = GenerationScheduler(
            runtime,
            compute_stream=compute_stream,
            skill_registry=skills,
            adapter_provider=adapter_provider,
        )
        # Admission wakes the kernel loop when async crop work completes.
        self._admission = _AdmissionCoordinator(
            runtime=runtime,
            wake_event=wake_event,
            fail_request=self._fail_via_admission,
        )
        self._active: Dict[int, _AutoregressiveRequest] = {}
        self._admission_capacity = max(
            runtime.max_batch_slots,
            runtime.max_batch_size * 4,
        )
        # Admission-time failures surface as completions the kernel delivers.
        self._admission_failures: List[Completion] = []

    # -- ingress (event-loop thread) ----------------------------------

    def submit(self, request: _AutoregressiveRequest) -> None:
        if (
            request.max_new_tokens > 0
            and self._runtime.sampling_hooks.sample_greedy is not None
        ):
            error: str | None = None
            # Spell this as the accepted relation so NaN cannot slip through:
            # both ``nan > 0`` and ``nan <= 0`` are false.
            if not request.temperature <= 0.0:
                error = "custom greedy sampling requires greedy requests"
            elif request.return_logprobs is True:
                error = "custom greedy sampling does not support token logprobs"
            if error is not None:
                self._fail_via_admission(request, ValueError(error))
                return
        ready = self._admission.submit(request)
        if ready is not None:
            self._admit_ready(ready)

    # -- step (kernel thread) -----------------------------------------

    @property
    def has_work(self) -> bool:
        """Queued, admitting, or in-flight work remains (read-only)."""
        return (
            self._scheduler.has_pending_work()
            or self._admission.has_pending()
            or bool(self._active)
            or bool(self._admission_failures)
        )

    @property
    def has_ingress_capacity(self) -> bool:
        return (
            len(self._active) + self._admission.pending_count
            < self._admission_capacity
        )

    def advance(self) -> TickResult:
        scheduler = self._scheduler
        progressed = self._promote_ready()

        if scheduler.has_pending_work():
            progressed = scheduler.advance() or progressed

        new = self._collect()

        # Drain the admission-failure buffer LAST. _admission_failures is
        # the durable home for not-yet-delivered failures; clearing it
        # only here (after the work above that can raise) means that if
        # scheduler.advance() raises, the buffer stays intact and the
        # kernel's shutdown(exc) path still delivers those callers'
        # completions instead of leaving their futures unresolved.
        completed = self._admission_failures + new
        self._admission_failures = []
        progressed = progressed or bool(completed)

        return TickResult(
            progressed=progressed,
            completed=tuple(completed),
            has_work=self.has_work,
        )

    def drain(self) -> tuple[Completion, ...]:
        """Complete in-flight pipeline work (used before a pause)."""
        self._scheduler._drain_pipeline()
        return tuple(self._collect())

    def shutdown(self, error: Optional[BaseException] = None) -> tuple[Completion, ...]:
        exc = error or RuntimeError("Engine shut down")
        # Fail in-flight admission first: fail_all() synchronously routes
        # any request still in async preprocessing through
        # _fail_via_admission, which appends to _admission_failures — so
        # collect that list *after*, or those requests' futures never get
        # resolved and callers hang.
        self._admission.fail_all(exc)
        for req in self._active.values():
            self._admission_failures.append(Completion(request=req, error=exc))
        self._active.clear()
        completions = self._admission_failures
        self._admission_failures = []
        self._release_active_sequences()
        return tuple(completions)

    # -- internals ----------------------------------------------------

    def _fail_via_admission(
        self, req: _AutoregressiveRequest, error: BaseException
    ) -> None:
        self._admission_failures.append(Completion(request=req, error=error))

    def _admit_ready(self, ready: _ReadyAdmission) -> None:
        req = ready.req
        try:
            generation_req, skill_state = self._build_generation_request(
                self._runtime, req, ready.crops, ready.encoder_input
            )
        except Exception as exc:
            self._admission_failures.append(Completion(request=req, error=exc))
            return
        crops_ready = (
            req.image is None
            # Multi-image chat crops each image inline (no overlap precompute).
            or isinstance(req.image, (list, tuple))
            or ready.prefix_cache_hit
            or (ready.crops is not None)
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
            prefix_cache_hit=ready.prefix_cache_hit,
            submitted_at=req.submitted_at,
        )
        generation_req.lifecycle = lifecycle
        self._scheduler.enqueue_request(generation_req, skill_state)
        self._active[req.request_id] = req

    def _promote_ready(self) -> bool:
        promoted = False
        while len(self._scheduler.waiting) < self._admission_capacity:
            ready = self._admission.take_ready()
            if ready is None:
                break
            self._admit_ready(ready)
            promoted = True
        return promoted

    def _collect(self) -> List[Completion]:
        """Drain finished scheduler results into Completion values."""
        completions: List[Completion] = []
        for result in self._scheduler.pop_completed():
            completion = self._completion_for(result)
            if completion is not None:
                completions.append(completion)
        return completions

    def _completion_for(self, result: SchedulerResult) -> Optional[Completion]:
        req = self._active.pop(result.request_id, None)
        if req is None:
            _LOGGER.error(
                "Scheduler produced unknown request_id %s", result.request_id
            )
            return None
        if result.finish_reason == "error" and "error" in result.output:
            return Completion(
                request=req, error=RuntimeError(result.output["error"])
            )
        return Completion(request=req, result=self._to_engine_result(result))

    def _release_active_sequences(self) -> None:
        try:
            runtime_sequences = list(self._runtime.active_sequences.values())
        except Exception:  # pragma: no cover - defensive cleanup
            return
        # On a spec runtime the rows registered in ``active_sequences`` are
        # decoder-owned: a fixed, persistently-reserved pool backing the
        # captured verify/draft CUDA graphs. ``runtime.release_sequence``
        # would erase those pages out from under the graphs and never call
        # ``decoder.retire``, so reclaim each row through the decoder the
        # same way ``GenerationScheduler._retire_spec_row`` does (the single
        # spec-path retire contract). The adapter slot a spec admit acquired
        # is not part of that pool, so release it here too -- the spec path
        # never runs through ``release_sequence`` -> ``release_adapter_slot``
        # (``release_adapter_slot`` is a no-op for ``lora_slot == 0``).
        spec = getattr(self._runtime, "spec", None)
        decoder = getattr(spec, "decoder", None) if spec is not None else None
        for state in runtime_sequences:
            try:
                if decoder is not None:
                    decoder.retire(state)
                    self._runtime.active_sequences.pop(state.batch_idx, None)
                    self._runtime.release_adapter_slot(
                        getattr(state, "lora_slot", 0)
                    )
                else:
                    self._runtime.release_sequence(state)
            except Exception:
                pass
