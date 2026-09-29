"""SinglePassExecutor: async launch + deferred collect, value-based delivery.

Drives the executor directly with a stub single-pass driver on CPU
(where make_event() is a NoopEvent that reports done immediately). Pins
the launch/collect contract, error handling, and shutdown — the kernel
integration is covered separately by the engine e2e test.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
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
        self.batch_capacity = 2
        self.calls: list[tuple[str, Any]] = []

    def forward(self, task: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("forward failed")
        return tuple(
            ValueError("invalid input")
            if value == "bad"
            else {"task": task, "inputs": value}
            for value in inputs
        )

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
    """Default max_in_flight=1: a second task waits until the first frees."""
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


def test_same_task_requests_share_one_forward() -> None:
    driver = _StubDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    ex.submit(_req(8, "segment", 1))
    ex.submit(_req(9, "segment", 2))

    tick = ex.advance()

    assert driver.calls == [("segment", (1, 2))]
    assert [completion.result.output for completion in tick.completed] == [
        {"task": "segment", "inputs": 1},
        {"task": "segment", "inputs": 2},
    ]


def test_one_invalid_input_does_not_fail_its_batch() -> None:
    ex = SinglePassExecutor(_StubDriver(), compute_stream=None)
    ex.submit(_req(10, "segment", "bad"))
    ex.submit(_req(11, "segment", "valid"))

    tick = ex.advance()

    assert isinstance(tick.completed[0].error, ValueError)
    assert tick.completed[1].error is None
    assert tick.completed[1].result.output["inputs"] == "valid"


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


class _PipelinedDriver(_StubDriver):
    """Driver that splits its forward: launch enqueues, collect reads back."""

    pipelined = True

    def __init__(self) -> None:
        super().__init__()
        self.collected: list[Any] = []

    def forward(self, task: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        raise AssertionError("a pipelined driver is driven by launch/collect")

    def launch(self, task: str, inputs: tuple[Any, ...]) -> Any:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("launch failed")
        return (task, inputs)

    def collect(self, batch: Any) -> tuple[Any, ...]:
        task, inputs = batch
        self.collected.append(batch)
        if task == "kaboom":
            raise ValueError("collect failed")
        if task == "short":
            return ()
        return tuple({"task": task, "inputs": value} for value in inputs)


def test_plain_runtime_keeps_one_forward_in_flight() -> None:
    assert SinglePassExecutor(_StubDriver(), compute_stream=None)._max_in_flight == 1


def test_pipelined_runtime_keeps_two_batches_in_flight(monkeypatch) -> None:
    """launch returns with its device work merely enqueued, so the executor
    starts the next cohort instead of waiting for the first to come back."""
    monkeypatch.setattr(
        single_pass_mod, "make_event", lambda device: _PendingEvent(not_done_polls=4)
    )
    driver = _PipelinedDriver()  # batch_capacity 2
    ex = SinglePassExecutor(driver, compute_stream=None)
    assert ex._max_in_flight == 2
    for request_id in range(6):
        ex.submit(_req(request_id, "a", request_id))

    tick = ex.advance()

    assert [inputs for _task, inputs in driver.calls] == [(0, 1), (2, 3)]
    assert driver.collected == []  # neither event has fired
    assert tick.completed == ()
    assert tick.progressed is True


def test_explicit_in_flight_limit_overrides_the_pipelined_default() -> None:
    ex = SinglePassExecutor(_PipelinedDriver(), compute_stream=None, max_in_flight=1)
    assert ex._max_in_flight == 1


def test_rejects_a_non_positive_in_flight_limit() -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        SinglePassExecutor(_StubDriver(), compute_stream=None, max_in_flight=0)


def test_pipelined_collect_follows_launch_order(monkeypatch) -> None:
    """The second cohort's event fires first; its results still wait.

    The first cohort's event is polled twice per tick -- once by ``_launch``,
    deciding whether a second cohort is worth starting, and once by
    ``_collect`` -- so two not-done polls hold it for one tick.
    """
    events = iter([_PendingEvent(not_done_polls=2), _PendingEvent(not_done_polls=0)])
    monkeypatch.setattr(single_pass_mod, "make_event", lambda device: next(events))
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    for request_id in range(4):
        ex.submit(_req(request_id, "a", request_id))

    assert ex.advance().completed == ()  # first cohort's event still pending
    assert driver.collected == []

    tick = ex.advance()

    assert [c.request.request_id for c in tick.completed] == [0, 1, 2, 3]
    assert [inputs for _task, inputs in driver.collected] == [(0, 1), (2, 3)]


def test_pipelined_launch_failure_fails_only_its_cohort() -> None:
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    ex.submit(_req(1, "boom", 1))
    ex.submit(_req(2, "a", 2))

    first, second = ex.advance(), ex.advance()

    assert [c.request.request_id for c in first.completed] == [1]
    assert isinstance(first.completed[0].error, ValueError)
    assert [c.request.request_id for c in second.completed] == [2]
    assert second.completed[0].error is None
    assert second.completed[0].result.output == {"task": "a", "inputs": 2}


def test_pipelined_collect_failure_fails_only_its_cohort() -> None:
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    ex.submit(_req(1, "kaboom", 1))
    ex.submit(_req(2, "a", 2))

    first, second = ex.advance(), ex.advance()

    assert [c.request.request_id for c in first.completed] == [1]
    assert str(first.completed[0].error) == "collect failed"
    assert [c.request.request_id for c in second.completed] == [2]
    assert second.completed[0].error is None


def test_a_launch_that_already_finished_does_not_start_the_next_cohort() -> None:
    """``max_in_flight`` is a ceiling, not a target.

    A pipelined ``launch`` only defers when the cohort lets it: not on CPU or
    MPS, where the completion event is a stand-in that reads as fired the
    moment it is recorded, and not for a cohort it had to finish in place.
    Starting a second cohort behind one that is already done buys no overlap
    and runs a whole forward before delivering results that are ready.
    """
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    for request_id in range(4):
        ex.submit(_req(request_id, "a", request_id))

    first = ex.advance()

    assert [inputs for _task, inputs in driver.calls] == [(0, 1)]
    assert [c.request.request_id for c in first.completed] == [0, 1]

    second = ex.advance()

    assert [inputs for _task, inputs in driver.calls] == [(0, 1), (2, 3)]
    assert [c.request.request_id for c in second.completed] == [2, 3]


def test_pipelined_collect_must_answer_every_request() -> None:
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    ex.submit(_req(1, "short", 1))

    tick = ex.advance()

    assert "0 results for 1 requests" in str(tick.completed[0].error)


def test_shutdown_fails_a_launched_pipelined_cohort(monkeypatch) -> None:
    monkeypatch.setattr(
        single_pass_mod, "make_event", lambda device: _PendingEvent(not_done_polls=4)
    )
    driver = _PipelinedDriver()
    ex = SinglePassExecutor(driver, compute_stream=None)
    ex.submit(_req(1, "a", 1))
    ex._launch()

    completions = ex.shutdown(RuntimeError("stop"))

    assert [c.request.request_id for c in completions] == [1]
    assert isinstance(completions[0].error, RuntimeError)
    assert driver.collected == []  # a cohort torn down is never read back


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
