"""The single-pass executor lane.

A single-pass driver (:class:`~kestrel.runtime.SinglePassRuntime`) fulfils a
same-task request cohort with one ``forward(task, inputs)``. This executor owns
cohort formation, completion events, and per-request result delivery.

It presents the same uniform :class:`Executor` face the kernel folds over
(``submit`` / ``advance`` -> :class:`TickResult` / ``shutdown``) and emits
:class:`Completion` values; like the autoregressive lane, it never touches
the event loop.

One forward is in flight by default; each forward may contain up to the
runtime's declared ``batch_capacity``.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from kestrel.device import make_event, stream_context
from kestrel.runtime import SinglePassRuntime

from kestrel.engine._types import (
    Completion,
    EngineMetrics,
    EngineResult,
    TickResult,
    _StreamQueue,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SinglePassRequest:
    """A single-pass request the engine is tracking.

    Satisfies the :class:`~kestrel.engine._types.EngineRequest` envelope
    (``request_id`` / ``future`` / ``stream_queue`` / ``adapter``) the
    kernel delivers to, plus this lane's payload: the ``task`` name and
    its ``inputs``. ``adapter`` is ``None`` until finetune support lands
    for single-pass models; ``stream_queue`` is ``None`` until partial
    output (e.g. streamed masks) is supported.
    """

    request_id: int
    future: "asyncio.Future[EngineResult]"
    task: str
    inputs: Any
    submitted_at: float
    adapter: Optional[str] = None
    stream_queue: "Optional[_StreamQueue]" = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


def _single_pass_result(
    request_id: int, output: Any, finish_reason: str = "stop"
) -> EngineResult:
    """Wrap a driver forward's output as an EngineResult.

    Single-pass tasks produce structured output (e.g. masks + scores),
    not tokens. We deliberately reuse the one ``EngineResult`` type so the
    kernel delivers every lane through a single path (a second result type
    would force the delivery code to branch); the token fields are
    zero-filled because they don't apply.

    KNOWN GAP (usage metering): the zero token counts mean a single-pass
    request is counted by Photon (request_count++) but contributes no
    billable token usage. Token-based billing therefore undercounts
    single-pass work. Picking the right unit for single-pass (images /
    pixels / forwards) and wiring it into telemetry is deferred — tracked
    for when single-pass models are actually metered.
    """
    return EngineResult(
        request_id=request_id,
        tokens=[],
        finish_reason=finish_reason,
        metrics=EngineMetrics(
            input_tokens=0,
            output_tokens=0,
            prefill_time_ms=0.0,
            decode_time_ms=0.0,
            ttft_ms=0.0,
        ),
        output=output if isinstance(output, dict) else {"result": output},
    )


def _cancelled_completion(request: _SinglePassRequest) -> Completion:
    return Completion(
        request=request,
        result=_single_pass_result(request.request_id, {}, "cancelled"),
    )


@dataclass(slots=True)
class _InFlight:
    """A forward whose kernels are enqueued and whose result is pending.

    ``error`` is set instead of ``outputs``/``done_event`` when the
    ``forward`` call raised at launch; it surfaces as an error completion
    on the next collect, keeping launch failures on the same path as
    results.
    """

    requests: tuple[_SinglePassRequest, ...]
    outputs: tuple[Any, ...]
    done_event: Any  # torch.cuda.Event | NoopEvent | None
    error: Optional[BaseException] = None


class SinglePassExecutor:
    """Executor lane driving single-forward requests with async collect."""

    def __init__(
        self,
        runtime: SinglePassRuntime,
        *,
        compute_stream: Any,
        max_in_flight: int = 1,
    ) -> None:
        self._runtime = runtime
        self._device = runtime.device
        self._stream = compute_stream
        self._max_in_flight = max_in_flight
        if runtime.batch_capacity < 1:
            raise ValueError("single-pass batch_capacity must be positive")
        self._batch_capacity = runtime.batch_capacity
        self._queue: "queue.Queue[_SinglePassRequest]" = queue.Queue()
        self._deferred: _SinglePassRequest | None = None
        self._in_flight: List[_InFlight] = []

    # -- ingress (event-loop thread) ----------------------------------

    def submit(self, request: _SinglePassRequest) -> None:
        self._queue.put(request)

    # -- step (kernel thread) -----------------------------------------

    @property
    def has_work(self) -> bool:
        return (
            bool(self._in_flight)
            or self._deferred is not None
            or not self._queue.empty()
        )

    @property
    def has_in_flight(self) -> bool:
        """A launched forward is awaiting its GPU completion event.

        Distinct from ``has_work``: queued requests wake the kernel via the
        submit event, but a pending GPU event sets no host event, so the
        kernel must keep polling (timed wait, not block) while this holds.
        """
        return bool(self._in_flight)

    def advance(self) -> TickResult:
        progressed, completed = self._launch()
        completed.extend(self._collect())
        progressed = progressed or bool(completed)
        return TickResult(
            progressed=progressed,
            completed=tuple(completed),
            has_work=self.has_work,
        )

    def drain(self) -> tuple[Completion, ...]:
        """Settle terminal work while leaving ordinary queued work paused."""

        completed: List[Completion] = []
        if self._deferred is not None and self._deferred.cancel_event.is_set():
            completed.append(_cancelled_completion(self._deferred))
            self._deferred = None
        deferred = []
        for _ in range(self._queue.qsize()):
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request.cancel_event.is_set():
                completed.append(_cancelled_completion(request))
            else:
                deferred.append(request)
        for request in deferred:
            self._queue.put(request)
        completed.extend(self._collect())
        return tuple(completed)

    def shutdown(self, error: Optional[BaseException] = None) -> tuple[Completion, ...]:
        exc = error or RuntimeError("Engine shut down")
        completions: List[Completion] = [
            Completion(request=request, error=exc)
            for in_flight in self._in_flight
            for request in in_flight.requests
        ]
        self._in_flight = []
        if self._deferred is not None:
            completions.append(Completion(request=self._deferred, error=exc))
            self._deferred = None
        while True:
            try:
                req = self._queue.get_nowait()
            except queue.Empty:
                break
            completions.append(Completion(request=req, error=exc))
        return tuple(completions)

    # -- internals ----------------------------------------------------

    def _launch(self) -> tuple[bool, List[Completion]]:
        """Start forwards until the in-flight pool is full or the queue drains."""
        progressed = False
        completed: List[Completion] = []
        while len(self._in_flight) < self._max_in_flight:
            try:
                requests = self._take_batch()
            except queue.Empty:
                break
            pending = []
            for request in requests:
                if request.cancel_event.is_set():
                    completed.append(_cancelled_completion(request))
                    progressed = True
                else:
                    pending.append(request)
            if pending:
                progressed = self._launch_batch(pending) or progressed
        return progressed, completed

    def _take_batch(self) -> tuple[_SinglePassRequest, ...]:
        if self._deferred is None:
            first = self._queue.get_nowait()
        else:
            first, self._deferred = self._deferred, None
        requests = [first]
        while len(requests) < self._batch_capacity:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request.task != first.task:
                self._deferred = request
                break
            requests.append(request)
        return tuple(requests)

    def _launch_batch(self, requests: Sequence[_SinglePassRequest]) -> bool:
        try:
            with stream_context(self._stream):
                outputs = tuple(
                    self._runtime.forward(
                        requests[0].task,
                        tuple(request.inputs for request in requests),
                    )
                )
                if len(outputs) != len(requests):
                    raise ValueError(
                        "single-pass forward returned "
                        f"{len(outputs)} results for {len(requests)} requests"
                    )
                done_event = make_event(self._device)
                done_event.record()
        except Exception as exc:
            self._in_flight.append(
                _InFlight(
                    requests=tuple(requests),
                    outputs=(),
                    done_event=None,
                    error=exc,
                )
            )
            return True
        self._in_flight.append(
            _InFlight(
                requests=tuple(requests),
                outputs=outputs,
                done_event=done_event,
                error=None,
            )
        )
        return True

    def _collect(self) -> List[Completion]:
        """Emit completions for any in-flight forward that has finished."""
        if not self._in_flight:
            return []
        still: List[_InFlight] = []
        completed: List[Completion] = []
        for f in self._in_flight:
            if f.error is not None:
                completed.extend(
                    Completion(request=request, error=f.error) for request in f.requests
                )
            elif f.done_event.query():
                completed.extend(
                    (
                        _cancelled_completion(request)
                        if request.cancel_event.is_set()
                        else Completion(request=request, error=output)
                        if isinstance(output, BaseException)
                        else Completion(
                            request=request,
                            result=_single_pass_result(request.request_id, output),
                        )
                    )
                    for request, output in zip(f.requests, f.outputs, strict=True)
                )
            else:
                still.append(f)
        self._in_flight = still
        return completed


__all__ = ["SinglePassExecutor", "_SinglePassRequest"]
