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
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

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
        # Image preprocessing payloads are opaque — the runtime decides
        # what they contain (Moondream's ``OverlapCropOutput``,
        # Gemma 4's pixel_values bundle, etc.) and threads them back
        # into ``launch_prepared_batch``.
        self._pending_crops: Dict[
            int, tuple[_AutoregressiveRequest, "Future[Any]"]
        ] = {}
        self._ready_crops: queue.Queue[int] = queue.Queue()

    def has_pending(self) -> bool:
        return bool(self._pending_crops)

    def submit(self, req: _AutoregressiveRequest) -> Optional[_ReadyAdmission]:
        if req.image is None:
            return _ReadyAdmission(req=req, crops=None, prefix_cache_hit=False)

        if self._runtime.prefix_cache is not None:
            req.image_hash = _hash_image(req.image)
            prefill_tokens = list(req.prompt_tokens) + list(req.generated_prefix.tokens)
            if self._runtime.check_prefix_cache(
                prefill_tokens, req.image_hash, req.adapter
            ):
                return _ReadyAdmission(req=req, crops=None, prefix_cache_hit=True)

        try:
            future = self._runtime.preprocess_image_async(req.image)
        except Exception as exc:
            self._fail_request(req, exc)
            return None

        req_id = req.request_id
        self._pending_crops[req_id] = (req, future)
        future.add_done_callback(
            lambda _future, rid=req_id: self._on_crops_ready(rid)
        )
        return None

    def take_ready(self) -> Optional[_ReadyAdmission]:
        while True:
            try:
                req_id = self._ready_crops.get_nowait()
            except queue.Empty:
                return None

            pending = self._pending_crops.pop(req_id, None)
            if pending is None:
                continue

            req, future = pending
            try:
                crops = future.result()
            except Exception as exc:
                self._fail_request(req, exc)
                continue
            return _ReadyAdmission(req=req, crops=crops, prefix_cache_hit=False)

    def fail_all(self, error: Optional[BaseException] = None) -> None:
        exc = error or RuntimeError("Engine shut down")
        for req, future in list(self._pending_crops.values()):
            if future and not future.done():
                future.cancel()
            self._fail_request(req, exc)
        self._pending_crops.clear()
        self._drain_ready_notifications()

    def _on_crops_ready(self, request_id: int) -> None:
        self._ready_crops.put(request_id)
        self._wake_event.set()

    def _drain_ready_notifications(self) -> None:
        while True:
            try:
                self._ready_crops.get_nowait()
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
        skills: "SkillRegistry",
        adapter_provider: Optional[AdapterProvider],
        build_generation_request: Callable[
            [AutoregressiveRuntime, "_AutoregressiveRequest", Any],
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
        # Admission-time failures surface as completions the kernel delivers.
        self._admission_failures: List[Completion] = []

    # -- ingress (event-loop thread) ----------------------------------

    def submit(self, request: _AutoregressiveRequest) -> None:
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
                self._runtime, req, ready.crops
            )
        except Exception as exc:
            self._admission_failures.append(Completion(request=req, error=exc))
            return
        crops_ready = (
            req.image is None or ready.prefix_cache_hit or (ready.crops is not None)
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
        cap = self._runtime.max_batch_size * 4
        while len(self._scheduler.waiting) < cap:
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
        for state in runtime_sequences:
            try:
                self._runtime.release_sequence(state)
            except Exception:
                pass


