"""The single-pass executor lane.

A single-pass driver (:class:`~kestrel.runtime.SinglePassRuntime`) fulfils
a request with one ``forward(task, inputs)`` that enqueues kernels and
returns result tensors *without* a host sync. This executor owns the
pipeline around that call — a slot pool, a completion event per in-flight
forward, and result delivery — so a single-pass forward overlaps
autoregressive decode on the shared stream (async launch + deferred
collect, the autoregressive pipeline's trick generalized to a second
lane).

It presents the same uniform :class:`Executor` face the kernel folds over
(``submit`` / ``advance`` -> :class:`TickResult` / ``shutdown``) and emits
:class:`Completion` values; like the autoregressive lane, it never touches
the event loop.

First cut: one in-flight forward (``max_in_flight=1``). Adding slots or
batching is a parameter change here, not a new abstraction.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import dataclass
from typing import Any, List, Optional

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


def _single_pass_result(request_id: int, output: Any) -> EngineResult:
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
        finish_reason="stop",
        metrics=EngineMetrics(
            input_tokens=0,
            output_tokens=0,
            prefill_time_ms=0.0,
            decode_time_ms=0.0,
            ttft_ms=0.0,
        ),
        output=output if isinstance(output, dict) else {"result": output},
    )


@dataclass(slots=True)
class _InFlight:
    """A forward whose kernels are enqueued and whose result is pending.

    ``error`` is set instead of ``output``/``done_event`` when the
    ``forward`` call raised at launch; it surfaces as an error completion
    on the next collect, keeping launch failures on the same path as
    results.
    """

    request: _SinglePassRequest
    output: Any
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
        self._queue: "queue.Queue[_SinglePassRequest]" = queue.Queue()
        self._in_flight: List[_InFlight] = []

    # -- ingress (event-loop thread) ----------------------------------

    def submit(self, request: _SinglePassRequest) -> None:
        self._queue.put(request)

    # -- step (kernel thread) -----------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self._in_flight) or not self._queue.empty()

    @property
    def has_in_flight(self) -> bool:
        """A launched forward is awaiting its GPU completion event.

        Distinct from ``has_work``: queued requests wake the kernel via the
        submit event, but a pending GPU event sets no host event, so the
        kernel must keep polling (timed wait, not block) while this holds.
        """
        return bool(self._in_flight)

    def advance(self) -> TickResult:
        progressed = self._launch()
        completed = self._collect()
        progressed = progressed or bool(completed)
        return TickResult(
            progressed=progressed,
            completed=tuple(completed),
            has_work=self.has_work,
        )

    def shutdown(self, error: Optional[BaseException] = None) -> tuple[Completion, ...]:
        exc = error or RuntimeError("Engine shut down")
        completions: List[Completion] = [
            Completion(request=f.request, error=exc) for f in self._in_flight
        ]
        self._in_flight = []
        while True:
            try:
                req = self._queue.get_nowait()
            except queue.Empty:
                break
            completions.append(Completion(request=req, error=exc))
        return tuple(completions)

    # -- internals ----------------------------------------------------

    def _launch(self) -> bool:
        """Start forwards until the in-flight pool is full or the queue drains."""
        launched = False
        while len(self._in_flight) < self._max_in_flight:
            try:
                req = self._queue.get_nowait()
            except queue.Empty:
                break
            launched = self._launch_one(req) or launched
        return launched

    def _launch_one(self, req: _SinglePassRequest) -> bool:
        try:
            with stream_context(self._stream):
                output = self._runtime.forward(req.task, req.inputs)
                done_event = make_event(self._device)
                done_event.record()
        except Exception as exc:
            # Launch-time failure: hold the error so it surfaces as a
            # completion on the next collect, uniform with normal results.
            self._in_flight.append(
                _InFlight(request=req, output=None, done_event=None, error=exc)
            )
            return True
        self._in_flight.append(
            _InFlight(request=req, output=output, done_event=done_event, error=None)
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
                completed.append(Completion(request=f.request, error=f.error))
            elif f.done_event.query():
                completed.append(
                    Completion(
                        request=f.request,
                        result=_single_pass_result(f.request.request_id, f.output),
                    )
                )
            else:
                still.append(f)
        self._in_flight = still
        return completed


__all__ = ["SinglePassExecutor", "_SinglePassRequest"]
