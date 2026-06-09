"""StreamingExecutor: stateful start/chunk/close execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from kestrel.engine import StreamingExecutor
from kestrel.engine._types import _ModelStreamQueue, _StreamingChunk
from kestrel.engine.streaming import _StreamingSessionRequest
from kestrel.runtime import ExecutionShape


class _StubStreamingRuntime:
    def __init__(self) -> None:
        self.model_name = "tracker"
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.STREAMING
        self.calls: list[tuple[str, Any]] = []
        self.finished: list[Any] = []

    def tasks(self) -> tuple[str, ...]:
        return ("point",)

    def start(self, task: str, inputs: Any) -> dict[str, Any]:
        self.calls.append(("start", task, inputs))
        if task == "boom-start":
            raise ValueError("start failed")
        return {"step": 0, "seed": inputs}

    def step(self, session: Any, inputs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append(("step", session, inputs))
        if inputs.get("boom"):
            raise ValueError("step failed")
        next_session = dict(session)
        next_session["step"] += 1
        return {"points": inputs["frame"], "step": next_session["step"]}, next_session

    def finish(self, session: Any) -> None:
        self.finished.append(session)

    def shutdown(self) -> None:
        pass


def _req(request_id: int, task: str = "point") -> _StreamingSessionRequest:
    loop = asyncio.new_event_loop()
    try:
        future: asyncio.Future = loop.create_future()
        queue: _ModelStreamQueue = asyncio.Queue()
    finally:
        loop.close()
    return _StreamingSessionRequest(
        request_id=request_id,
        future=future,
        task=task,
        initial_inputs={"points": [[0.5, 0.5]]},
        submitted_at=0.0,
        model_stream_queue=queue,
    )


def test_streaming_executor_yields_updates_and_close_completion() -> None:
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    req = _req(1)

    ex.submit(req)
    tick0 = ex.advance()
    assert tick0.completed == ()
    assert tick0.model_stream_updates == ()
    assert ex.has_work is False

    ex.submit_chunk(_StreamingChunk(session_id=1, inputs={"frame": [[0.1, 0.2]]}))
    tick1 = ex.advance()
    assert tick1.completed == ()
    assert len(tick1.model_stream_updates) == 1
    assert tick1.model_stream_updates[0].output == {
        "points": [[0.1, 0.2]],
        "step": 1,
    }

    ex.submit_chunk(_StreamingChunk(session_id=1, close=True))
    tick2 = ex.advance()
    assert tick2.model_stream_updates == ()
    assert len(tick2.completed) == 1
    assert tick2.completed[0].result is not None
    assert tick2.completed[0].result.output == {"closed": True}
    assert runtime.finished == [{"step": 1, "seed": {"points": [[0.5, 0.5]]}}]
    assert ex.has_work is False


def test_chunks_sent_before_start_finishes_are_buffered() -> None:
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    req = _req(2)

    ex.submit(req)
    ex.submit_chunk(_StreamingChunk(session_id=2, inputs={"frame": [[0.3, 0.4]]}))

    tick = ex.advance()

    assert len(tick.model_stream_updates) == 1
    assert tick.model_stream_updates[0].session_id == 2
    assert tick.model_stream_updates[0].output["points"] == [[0.3, 0.4]]


def test_step_error_completes_session_with_error() -> None:
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    req = _req(3)

    ex.submit(req)
    ex.advance()
    ex.submit_chunk(_StreamingChunk(session_id=3, inputs={"boom": True}))
    tick = ex.advance()

    assert len(tick.completed) == 1
    assert isinstance(tick.completed[0].error, ValueError)
    assert runtime.finished == [{"step": 0, "seed": {"points": [[0.5, 0.5]]}}]
    assert ex.has_work is False


def test_start_error_drops_buffered_chunks() -> None:
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    req = _req(6, task="boom-start")

    ex.submit(req)
    ex.submit_chunk(_StreamingChunk(session_id=6, inputs={"frame": [[0.1, 0.2]]}))
    tick = ex.advance()

    assert len(tick.completed) == 1
    assert isinstance(tick.completed[0].error, ValueError)
    assert ex.has_work is False


def test_shutdown_fails_starting_and_active_sessions() -> None:
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    active = _req(4)
    queued = _req(5)

    ex.submit(active)
    ex.advance()
    ex.submit(queued)

    completions = ex.shutdown(RuntimeError("stop"))
    ids = sorted(c.request.request_id for c in completions)

    assert ids == [4, 5]
    assert all(isinstance(c.error, RuntimeError) for c in completions)
    assert ex.has_work is False


def test_shutdown_deduplicates_in_flight_session_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEvent:
        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def record(self) -> None:
            pass

        def query(self) -> bool:
            return self.ready

    events = [_FakeEvent(True), _FakeEvent(False)]

    def fake_make_event(device: torch.device) -> _FakeEvent:
        return events.pop(0)

    monkeypatch.setattr("kestrel.engine.streaming.make_event", fake_make_event)
    runtime = _StubStreamingRuntime()
    ex = StreamingExecutor(runtime, compute_stream=None)
    req = _req(7)

    ex.submit(req)
    ex.submit_chunk(_StreamingChunk(session_id=7, inputs={"frame": [[0.7, 0.8]]}))
    tick = ex.advance()
    completions = ex.shutdown(RuntimeError("stop"))

    assert tick.completed == ()
    assert tick.model_stream_updates == ()
    assert [c.request.request_id for c in completions] == [7]
    assert all(isinstance(c.error, RuntimeError) for c in completions)
    assert runtime.finished == [{"step": 0, "seed": {"points": [[0.5, 0.5]]}}]
    assert ex.has_work is False
