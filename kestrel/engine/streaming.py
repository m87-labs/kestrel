"""The stateful streaming executor lane."""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kestrel.device import make_event, stream_context
from kestrel.runtime import StreamingRuntime

from kestrel.engine._types import (
    Completion,
    EngineMetrics,
    EngineResult,
    ModelStreamUpdate,
    TickResult,
    _StreamingChunk,
    _StreamingSessionRequest,
)

_LOGGER = logging.getLogger(__name__)


def _stream_output(output: Any) -> Dict[str, object]:
    return output if isinstance(output, dict) else {"result": output}


def _stream_result(request_id: int, output: Any | None = None) -> EngineResult:
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
        output=_stream_output({"closed": True} if output is None else output),
    )


@dataclass(slots=True)
class _Session:
    request: _StreamingSessionRequest
    state: Any
    pending: List[_StreamingChunk] = field(default_factory=list)


@dataclass(slots=True)
class _PendingStart:
    request: _StreamingSessionRequest
    state: Any
    done_event: Any
    error: Optional[BaseException] = None


@dataclass(slots=True)
class _PendingStep:
    session_id: int
    request: _StreamingSessionRequest
    output: Any
    next_state: Any
    done_event: Any
    error: Optional[BaseException] = None


class StreamingExecutor:
    """Executor lane for stateful runtime sessions.

    First cut: one in-flight start/step globally. The session table and
    per-session queues are already explicit, so adding slots or batching
    later changes policy rather than the API shape.
    """

    def __init__(
        self,
        runtime: StreamingRuntime,
        *,
        compute_stream: Any,
        max_in_flight: int = 1,
    ) -> None:
        self._runtime = runtime
        self._device = runtime.device
        self._stream = compute_stream
        self._max_in_flight = max_in_flight
        self._starts: "queue.Queue[_StreamingSessionRequest]" = queue.Queue()
        self._sessions: Dict[int, _Session] = {}
        self._pending_chunks: Dict[int, List[_StreamingChunk]] = {}
        self._starting: List[_PendingStart] = []
        self._in_flight: List[_PendingStep] = []

    # -- ingress (event-loop thread) ----------------------------------

    def submit(self, request: _StreamingSessionRequest) -> None:
        self._starts.put(request)

    def submit_chunk(self, chunk: _StreamingChunk) -> None:
        session = self._sessions.get(chunk.session_id)
        if session is not None:
            session.pending.append(chunk)
            return
        self._pending_chunks.setdefault(chunk.session_id, []).append(chunk)

    # -- step (kernel thread) -----------------------------------------

    @property
    def has_work(self) -> bool:
        return (
            not self._starts.empty()
            or bool(self._starting)
            or bool(self._in_flight)
            or any(session.pending for session in self._sessions.values())
            or bool(self._pending_chunks)
        )

    @property
    def has_in_flight(self) -> bool:
        return bool(self._starting) or bool(self._in_flight)

    def advance(self) -> TickResult:
        progressed = self._launch_starts()
        start_completions = self._collect_starts()
        progressed = progressed or bool(start_completions)
        progressed = self._launch_steps() or progressed
        updates, completions = self._collect_steps()
        progressed = progressed or bool(updates) or bool(completions)

        return TickResult(
            progressed=progressed,
            completed=tuple(start_completions + completions),
            model_stream_updates=tuple(updates),
            has_work=self.has_work,
        )

    def shutdown(self, error: Optional[BaseException] = None) -> tuple[Completion, ...]:
        exc = error or RuntimeError("Engine shut down")
        completions: List[Completion] = []
        for start in self._starting:
            completions.append(Completion(request=start.request, error=exc))
        self._starting = []
        completed_request_ids: set[int] = set()
        for step in self._in_flight:
            session = self._sessions.pop(step.session_id, None)
            if session is not None:
                self._finish_session(session)
            request_id = step.request.request_id
            if request_id not in completed_request_ids:
                completions.append(Completion(request=step.request, error=exc))
                completed_request_ids.add(request_id)
        self._in_flight = []
        for session in self._sessions.values():
            self._finish_session(session)
            completions.append(Completion(request=session.request, error=exc))
        self._sessions.clear()
        while True:
            try:
                req = self._starts.get_nowait()
            except queue.Empty:
                break
            completions.append(Completion(request=req, error=exc))
        self._pending_chunks.clear()
        return tuple(completions)

    # -- launch / collect ---------------------------------------------

    def _launch_starts(self) -> bool:
        launched = False
        while self._capacity_available():
            try:
                req = self._starts.get_nowait()
            except queue.Empty:
                break
            launched = self._launch_start(req) or launched
        return launched

    def _launch_start(self, req: _StreamingSessionRequest) -> bool:
        try:
            with stream_context(self._stream):
                state = self._runtime.start(req.task, req.initial_inputs)
                done_event = make_event(self._device)
                done_event.record()
        except Exception as exc:
            self._starting.append(
                _PendingStart(request=req, state=None, done_event=None, error=exc)
            )
            return True
        self._starting.append(
            _PendingStart(request=req, state=state, done_event=done_event)
        )
        return True

    def _collect_starts(self) -> List[Completion]:
        if not self._starting:
            return []
        still: List[_PendingStart] = []
        completed: List[Completion] = []
        for start in self._starting:
            if start.error is not None:
                self._pending_chunks.pop(start.request.request_id, None)
                completed.append(Completion(request=start.request, error=start.error))
            elif start.done_event.query():
                session = _Session(request=start.request, state=start.state)
                session.pending.extend(
                    self._pending_chunks.pop(start.request.request_id, [])
                )
                self._sessions[start.request.request_id] = session
            else:
                still.append(start)
        self._starting = still
        return completed

    def _launch_steps(self) -> bool:
        launched = False
        while self._capacity_available():
            next_item = self._next_ready_chunk()
            if next_item is None:
                break
            session_id, session, chunk = next_item
            if chunk.close:
                self._finish_session(session)
                self._sessions.pop(session_id, None)
                self._in_flight.append(
                    _PendingStep(
                        session_id=session_id,
                        request=session.request,
                        output={"closed": True},
                        next_state=None,
                        done_event=make_event(self._device),
                    )
                )
                self._in_flight[-1].done_event.record()
                launched = True
                continue
            launched = self._launch_step(session_id, session, chunk) or launched
        return launched

    def _launch_step(
        self,
        session_id: int,
        session: _Session,
        chunk: _StreamingChunk,
    ) -> bool:
        try:
            with stream_context(self._stream):
                output = self._runtime.step(session.state, chunk.inputs)
                update, next_state = self._split_step_output(output, session.state)
                done_event = make_event(self._device)
                done_event.record()
        except Exception as exc:
            self._in_flight.append(
                _PendingStep(
                    session_id=session_id,
                    request=session.request,
                    output=None,
                    next_state=session.state,
                    done_event=None,
                    error=exc,
                )
            )
            return True
        self._in_flight.append(
            _PendingStep(
                session_id=session_id,
                request=session.request,
                output=update,
                next_state=next_state,
                done_event=done_event,
            )
        )
        return True

    def _collect_steps(self) -> tuple[List[ModelStreamUpdate], List[Completion]]:
        if not self._in_flight:
            return [], []
        still: List[_PendingStep] = []
        updates: List[ModelStreamUpdate] = []
        completed: List[Completion] = []
        for step in self._in_flight:
            if step.error is not None:
                session = self._sessions.pop(step.session_id, None)
                if session is not None:
                    self._finish_session(session)
                completed.append(Completion(request=step.request, error=step.error))
            elif step.done_event.query():
                if step.next_state is None:
                    completed.append(
                        Completion(
                            request=step.request,
                            result=_stream_result(
                                step.request.request_id,
                                step.output,
                            ),
                        )
                    )
                else:
                    session = self._sessions.get(step.session_id)
                    if session is not None:
                        session.state = step.next_state
                    updates.append(
                        ModelStreamUpdate(
                            session_id=step.session_id,
                            task=step.request.task,
                            output=_stream_output(step.output),
                        )
                    )
            else:
                still.append(step)
        self._in_flight = still
        return updates, completed

    # -- helpers ------------------------------------------------------

    def _capacity_available(self) -> bool:
        return len(self._starting) + len(self._in_flight) < self._max_in_flight

    def _next_ready_chunk(self) -> Optional[tuple[int, _Session, _StreamingChunk]]:
        for session_id, session in list(self._sessions.items()):
            if not session.pending:
                continue
            return session_id, session, session.pending.pop(0)
        return None

    def _finish_session(self, session: _Session) -> None:
        try:
            self._runtime.finish(session.state)
        except Exception:
            _LOGGER.exception("Streaming runtime finish failed")

    def _split_step_output(self, output: Any, current_state: Any) -> tuple[Any, Any]:
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        return output, current_state


__all__ = ["StreamingExecutor", "_StreamingChunk", "_StreamingSessionRequest"]
