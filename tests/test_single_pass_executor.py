"""SinglePassExecutor: async launch + deferred collect, value-based delivery.

Drives the executor directly with a stub single-pass driver on CPU
(where make_event() is a NoopEvent that reports done immediately). Pins
the launch/collect contract, error handling, and shutdown — the kernel
integration is covered separately by the engine e2e test.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import torch

from kestrel.engine import SinglePassExecutor
import kestrel.engine.single_pass as single_pass_mod
from kestrel.engine.single_pass import _SinglePassRequest
from kestrel.runtime import ExecutionShape


class _StubDriver:
    """Single-pass driver whose forward() echoes (task, inputs)."""

    def __init__(self) -> None:
        self.model_name = "stub-sp"
        # CPU device -> make_event() returns a NoopEvent whose query()
        # reports done immediately, so collect() fires on the next tick.
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self.calls: list[tuple[str, Any]] = []

    def forward(self, task: str, inputs: Any) -> Any:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("forward failed")
        return {"task": task, "inputs": inputs}

    def shutdown(self) -> None:
        pass


def _req(request_id: int, task: str, inputs: Any) -> _SinglePassRequest:
    loop = asyncio.new_event_loop()
    try:
        fut: asyncio.Future = loop.create_future()
    finally:
        loop.close()
    return _SinglePassRequest(
        request_id=request_id,
        future=fut,
        task=task,
        inputs=inputs,
        submitted_at=0.0,
    )


def test_forward_result_becomes_completion() -> None:
    ex = SinglePassExecutor(_StubDriver(), compute_stream=None)
    ex.submit(_req(1, "segment", {"points": [[1, 2]]}))

    tick = ex.advance()  # launch + collect (NoopEvent done immediately)

    assert tick.progressed is True
    assert len(tick.completed) == 1
    c = tick.completed[0]
    assert c.error is None
    assert c.result is not None
    assert c.result.output == {"task": "segment", "inputs": {"points": [[1, 2]]}}
    assert ex.has_work is False


def test_forward_error_becomes_error_completion() -> None:
    ex = SinglePassExecutor(_StubDriver(), compute_stream=None)
    ex.submit(_req(2, "boom", {}))

    tick = ex.advance()

    assert len(tick.completed) == 1
    assert tick.completed[0].result is None
    assert isinstance(tick.completed[0].error, ValueError)
    assert ex.has_work is False


def test_one_in_flight_at_a_time() -> None:
    """Default max_in_flight=1: a second job waits until the first frees."""
    driver = _StubDriver()
    ex = SinglePassExecutor(driver, compute_stream=None, max_in_flight=1)
    ex.submit(_req(3, "a", 1))
    ex.submit(_req(4, "b", 2))

    # First advance launches+collects job 3 (NoopEvent completes at once),
    # leaving job 4 queued.
    tick1 = ex.advance()
    assert [c.request.request_id for c in tick1.completed] == [3]
    assert ex.has_work is True  # job 4 still queued

    tick2 = ex.advance()
    assert [c.request.request_id for c in tick2.completed] == [4]
    assert ex.has_work is False


def test_idle_executor_reports_no_work() -> None:
    ex = SinglePassExecutor(_StubDriver(), compute_stream=None)
    tick = ex.advance()
    assert tick.completed == ()
    assert tick.has_work is False
    assert ex.has_work is False


class _PendingEvent:
    """Fake completion event: reports not-done for the first ``n`` polls.

    Lets a CPU test exercise the deferred-collect path that NoopEvent
    (always done) can't reach.
    """

    def __init__(self, not_done_polls: int) -> None:
        self._remaining = not_done_polls

    def record(self, *_a: Any, **_k: Any) -> None:
        pass

    def query(self) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True


def test_forward_stays_in_flight_until_event_fires(monkeypatch) -> None:
    """The result is held back until its completion event reports done."""
    event = _PendingEvent(not_done_polls=2)
    monkeypatch.setattr(single_pass_mod, "make_event", lambda device: event)

    ex = SinglePassExecutor(_StubDriver(), compute_stream=None)
    ex.submit(_req(7, "segment", {"k": "v"}))

    # Tick 1: forward launched, but the event reports not-done — nothing
    # delivered yet, work still pending.
    tick1 = ex.advance()
    assert tick1.completed == ()
    assert tick1.progressed is True  # a launch is progress
    assert ex.has_work is True
    assert len(ex._in_flight) == 1

    # Tick 2: event still not done — still held.
    tick2 = ex.advance()
    assert tick2.completed == ()
    assert ex.has_work is True

    # Tick 3: event fires — the result is finally delivered.
    tick3 = ex.advance()
    assert [c.request.request_id for c in tick3.completed] == [7]
    assert tick3.completed[0].result is not None
    assert ex.has_work is False


def test_shutdown_fails_queued_and_in_flight() -> None:
    driver = _StubDriver()
    ex = SinglePassExecutor(driver, compute_stream=None, max_in_flight=1)
    ex.submit(_req(5, "a", 1))
    ex.submit(_req(6, "b", 2))
    # Launch job 5 into the in-flight slot without collecting it: a real
    # CUDA event would still be pending here.
    ex._launch()
    assert len(ex._in_flight) == 1

    completions = ex.shutdown(RuntimeError("stop"))

    ids = sorted(c.request.request_id for c in completions)
    assert ids == [5, 6]  # both the in-flight and the queued one
    assert all(isinstance(c.error, RuntimeError) for c in completions)
    assert ex.has_work is False
