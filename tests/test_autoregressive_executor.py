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
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from kestrel.engine import (
    AutoregressiveExecutor,
    EngineResult,
    EngineMetrics,
    _AutoregressiveRequest,
)
from kestrel.runtime.sampling import SamplingHooks
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
        max_batch_slots=2,
        active_sequences={},
        release_sequence=lambda state: None,
        sampling_hooks=SamplingHooks(),
    )
    ex._admission_capacity = 4
    ex._to_engine_result = _fake_to_engine_result
    ex._active = {}
    ex._admission_completions = []
    ex._scheduler = SimpleNamespace(
        _completed=[],
        waiting=[],
        _drain_pipeline=lambda: None,
        has_pending_work=lambda: bool(ex._scheduler._completed),
        advance=lambda: False,
        pop_completed=lambda: [
            ex._scheduler._completed.pop(0) for _ in list(ex._scheduler._completed)
        ],
        enqueue_request=lambda req, state: None,
    )
    ex._admission = SimpleNamespace(
        has_pending=lambda: False,
        pending_count=0,
        pop_cancelled=lambda: [],
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


def test_ingress_capacity_counts_active_and_preprocessing_requests() -> None:
    ex = _executor()
    ex._active = {request_id: _pending(request_id) for request_id in range(3)}

    assert ex.has_ingress_capacity is True

    ex._admission.pending_count = 1
    assert ex.has_ingress_capacity is False


def test_ready_request_does_not_wait_for_preprocessing_siblings() -> None:
    ex = _executor()
    ex._runtime.max_batch_size = 4
    waiting = ex._scheduler.waiting
    ready = [object()]
    calls = 0

    def take_ready():
        nonlocal calls
        calls += 1
        if not ready:
            return None
        return ready.pop(0)

    # One sibling is still preprocessing. Its completion must not delay the
    # request that is already ready.
    ex._admission.pending_count = 1
    ex._admission.take_ready = take_ready
    ex._admit_ready = waiting.append
    launched = []
    ex._scheduler.has_pending_work = lambda: bool(waiting)
    ex._scheduler.advance = lambda: launched.append(len(waiting)) or True

    ex.advance()

    assert launched == [1]
    assert calls == 2


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


def test_pause_drain_settles_cancelled_admission() -> None:
    ex = _executor()
    request = _pending(5)
    request.cancel_event.set()
    ex._admission.pop_cancelled = lambda: [request]

    (completion,) = ex.drain()

    assert completion.request is request
    assert completion.result is not None
    assert completion.result.finish_reason == "cancelled"
    assert ex._admission_completions == []


@pytest.mark.parametrize(
    ("configure", "message"),
    [
        (lambda request: setattr(request, "return_logprobs", True), "logprobs"),
        (lambda request: setattr(request, "temperature", 0.5), "greedy requests"),
        (
            lambda request: setattr(request, "temperature", float("nan")),
            "greedy requests",
        ),
    ],
)
def test_unsupported_custom_greedy_request_fails_before_admission(
    configure: Callable[[_AutoregressiveRequest], None],
    message: str,
) -> None:
    ex = _executor()
    ex._runtime.sampling_hooks = SamplingHooks(
        sample_greedy=lambda logits, out, **_kwargs: out
    )
    admitted: list[_AutoregressiveRequest] = []
    ex._admission.submit = lambda request: admitted.append(request)
    rejected = _pending(5)
    configure(rejected)
    supported = _pending(6)

    ex.submit(rejected)
    ex.submit(supported)

    assert admitted == [supported]
    tick = ex.advance()
    assert len(tick.completed) == 1
    assert tick.completed[0].request is rejected
    assert isinstance(tick.completed[0].error, ValueError)
    assert message in str(tick.completed[0].error)


def test_advance_raising_preserves_queued_admission_failures() -> None:
    """Regression: a buffered admission failure must survive advance() raising.

    advance() must not move admission completions into a local that's lost
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
    appends to the admission completion buffer. shutdown() must collect that list
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
    assert ex._admission_completions == []


def test_shutdown_retires_spec_rows_through_the_decoder() -> None:
    """Regression: spec rows must be retired via ``decoder.retire`` on shutdown.

    On a spec runtime the rows in ``active_sequences`` are decoder-owned: a
    persistently-reserved pool backing captured CUDA graphs. The generic
    cleanup must NOT call ``runtime.release_sequence`` for them (that erases the
    reserved pages and skips the decoder), but route through the decoder's
    retire the same way ``GenerationScheduler._retire_spec_row`` does -- freeing
    the decoder-owned row (so it is reusable) and the admit-acquired adapter
    slot. Mirrors the L836 admit site that registers the spec row here.
    """
    retired: list[Any] = []
    released_slots: list[int] = []
    released_via_release_sequence: list[Any] = []
    free_rows: set[int] = set()

    def _retire(state: Any) -> None:
        retired.append(state)
        # Model the decoder reclaiming its pool row: the freed row index
        # becomes reusable for a future admit.
        free_rows.add(state.batch_idx)

    def _release_sequence(st: Any) -> None:
        released_via_release_sequence.append(st)

    decoder = SimpleNamespace(retire=_retire)
    state = SimpleNamespace(batch_idx=7, lora_slot=3)

    ex = _executor()
    ex._runtime = SimpleNamespace(
        max_batch_size=1,
        active_sequences={state.batch_idx: state},
        spec=SimpleNamespace(decoder=decoder),
        release_adapter_slot=lambda slot: released_slots.append(slot),
        # If the generic non-spec path is taken for a spec row this records it;
        # the assertion below proves it never runs.
        release_sequence=_release_sequence,
    )

    completions = ex.shutdown(RuntimeError("stop"))

    assert completions == ()
    # The decoder retired the row (not the generic release_sequence path).
    assert retired == [state]
    assert released_via_release_sequence == []
    # Adapter slot the spec admit acquired was released.
    assert released_slots == [state.lora_slot]
    # The row is freed from the registry and reusable (decoder reclaimed it).
    assert state.batch_idx not in ex._runtime.active_sequences
    assert state.batch_idx in free_rows


def test_shutdown_retires_base_model_spec_row_with_zero_slot() -> None:
    """A base-model spec row (``lora_slot == 0``) still retires through the
    decoder; ``release_adapter_slot(0)`` is the documented no-op."""
    retired: list[Any] = []
    released_slots: list[int] = []
    state = SimpleNamespace(batch_idx=2, lora_slot=0)

    def _must_not_run(st: Any) -> None:
        raise AssertionError("release_sequence must not run for a spec row")

    ex = _executor()
    ex._runtime = SimpleNamespace(
        max_batch_size=1,
        active_sequences={state.batch_idx: state},
        spec=SimpleNamespace(decoder=SimpleNamespace(retire=retired.append)),
        release_adapter_slot=lambda slot: released_slots.append(slot),
        release_sequence=_must_not_run,
    )

    ex.shutdown(RuntimeError("stop"))

    assert retired == [state]
    assert released_slots == [0]
    assert state.batch_idx not in ex._runtime.active_sequences
