"""The AR executor maps scheduler output to Completion values.

`AutoregressiveExecutor` wraps the generation scheduler + admission and
presents the uniform `Executor` face (submit / advance -> TickResult /
shutdown). These tests pin the new mapping logic — scheduler results and
in-flight requests becoming `Completion`s the kernel delivers — without a
GPU: the scheduler and admission are stubbed, since their internals are
covered by tests/scheduler/.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

from kestrel.engine import (
    AutoregressiveExecutor,
    Completion,
    EngineResult,
    EngineMetrics,
    _AutoregressiveRequest,
)
from kestrel.scheduler import SchedulerResult
from kestrel.scheduler.types import RequestMetrics


def _pending(request_id: int) -> _AutoregressiveRequest:
    loop = asyncio.new_event_loop()
    try:
        fut: asyncio.Future = loop.create_future()
    finally:
        loop.close()
    return _AutoregressiveRequest(
        request_id=request_id,
        prompt="p",
        prompt_tokens=[],
        image=None,
        image_hash=None,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        submitted_at=0.0,
        future=fut,
        stream_queue=None,
        skill=SimpleNamespace(name="stub"),
        request_context=object(),
    )


def _executor() -> AutoregressiveExecutor:
    """An executor with the scheduler + admission replaced by stubs."""
    ex = object.__new__(AutoregressiveExecutor)
    ex._runtime = SimpleNamespace(
        max_batch_size=1,
        active_sequences={},
        release_sequence=lambda state: None,
    )
    ex._to_engine_result = _fake_to_engine_result
    ex._active = {}
    ex._admission_failures = []
    ex._scheduler = SimpleNamespace(
        _completed=[],
        waiting=[],
        has_pending_work=lambda: bool(ex._scheduler._completed),
        advance=lambda: False,
        pop_completed=lambda: [
            ex._scheduler._completed.pop(0) for _ in list(ex._scheduler._completed)
        ],
        enqueue_request=lambda req, state: None,
    )
    ex._admission = SimpleNamespace(
        has_pending=lambda: False,
        take_ready=lambda: None,
        fail_all=lambda exc: None,
    )
    return ex


def _fake_to_engine_result(result: SchedulerResult) -> EngineResult:
    return EngineResult(
        request_id=result.request_id,
        tokens=result.tokens,
        finish_reason=result.finish_reason,
        metrics=EngineMetrics(
            input_tokens=0,
            output_tokens=0,
            prefill_time_ms=0.0,
            decode_time_ms=0.0,
            ttft_ms=0.0,
        ),
        output=result.output,
    )


def _sched_result(request_id: int, *, error: str | None = None) -> SchedulerResult:
    return SchedulerResult(
        request_id=request_id,
        tokens=[],
        finish_reason="error" if error else "stop",
        metrics=RequestMetrics(
            prompt_tokens=0,
            decode_tokens=0,
            prefill_time_ms=0.0,
            ttft_ms=0.0,
            decode_time_ms=0.0,
        ),
        output={"error": error} if error else {"answer": "hi"},
    )


def test_completed_result_becomes_success_completion() -> None:
    ex = _executor()
    req = _pending(1)
    ex._active[1] = req
    ex._scheduler._completed.append(_sched_result(1))

    tick = ex.advance()

    assert tick.progressed is True
    assert len(tick.completed) == 1
    completion = tick.completed[0]
    assert completion.request is req
    assert completion.error is None
    assert completion.result is not None and completion.result.request_id == 1
    # The request left the in-flight map.
    assert 1 not in ex._active


def test_error_result_becomes_error_completion() -> None:
    ex = _executor()
    req = _pending(2)
    ex._active[2] = req
    ex._scheduler._completed.append(_sched_result(2, error="boom"))

    tick = ex.advance()

    assert len(tick.completed) == 1
    completion = tick.completed[0]
    assert completion.result is None
    assert isinstance(completion.error, RuntimeError)
    assert "boom" in str(completion.error)


def test_unknown_request_id_is_dropped() -> None:
    ex = _executor()
    ex._scheduler._completed.append(_sched_result(999))

    tick = ex.advance()

    assert tick.completed == ()


def test_idle_executor_reports_no_work() -> None:
    ex = _executor()
    tick = ex.advance()
    assert tick.completed == ()
    assert tick.has_work is False
    assert ex.has_work is False


def test_shutdown_fails_in_flight_requests() -> None:
    ex = _executor()
    req = _pending(3)
    ex._active[3] = req

    completions = ex.shutdown(RuntimeError("stop"))

    assert len(completions) == 1
    assert completions[0].request is req
    assert isinstance(completions[0].error, RuntimeError)
    assert not ex._active


def test_admission_failure_surfaces_as_completion() -> None:
    ex = _executor()
    req = _pending(4)
    # Simulate admission deciding the request can't proceed.
    ex._fail_via_admission(req, ValueError("bad image"))
    assert ex.has_work is True  # a pending failure counts as work

    tick = ex.advance()

    assert len(tick.completed) == 1
    assert isinstance(tick.completed[0].error, ValueError)
    assert ex.has_work is False


def test_advance_raising_preserves_queued_admission_failures() -> None:
    """Regression: a buffered admission failure must survive advance() raising.

    advance() must not move _admission_failures into a local that's lost
    if scheduler.advance() raises mid-tick. The kernel responds to the
    exception by calling shutdown(exc); that must still see the buffered
    failure and return it, or the failed-admission caller hangs.
    """
    ex = _executor()
    failed = _pending(6)
    ex._fail_via_admission(failed, ValueError("bad image"))

    def boom() -> bool:
        raise RuntimeError("advance blew up")

    ex._scheduler.has_pending_work = lambda: True
    ex._scheduler.advance = boom

    # Mirror the kernel loop: advance() raises, then shutdown(exc) runs.
    import pytest

    with pytest.raises(RuntimeError, match="advance blew up"):
        ex.advance()

    completions = ex.shutdown(RuntimeError("stop"))
    assert failed in [c.request for c in completions]
    assert isinstance(
        next(c for c in completions if c.request is failed).error, ValueError
    )


def test_shutdown_returns_requests_failed_during_fail_all() -> None:
    """Regression: a request still in async preprocessing at shutdown.

    The real admission coordinator's fail_all() synchronously routes
    in-flight preprocessing requests through _fail_via_admission, which
    appends to _admission_failures. shutdown() must collect that list
    *after* fail_all() runs, or those requests' completions are dropped
    and their callers hang forever.
    """
    ex = _executor()
    stuck = _pending(5)
    # Mimic the real coordinator: fail_all routes the stuck request back
    # through the executor's admission-failure path.
    ex._admission.fail_all = lambda exc: ex._fail_via_admission(stuck, exc)

    completions = ex.shutdown(RuntimeError("stop"))

    assert [c.request for c in completions] == [stuck]
    assert isinstance(completions[0].error, RuntimeError)
    assert ex._admission_failures == []
