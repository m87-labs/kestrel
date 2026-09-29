"""Terminal request graphs release by reference counting, not cyclic GC."""

from collections import deque
from contextlib import contextmanager
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from kestrel.runtime import SequenceState, TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle, RequestPhase
from kestrel.scheduler.queues import RunningQueue
from tests.scheduler._fake_runtime import FakeRuntime
from tests.scheduler.test_spec_decode_path import (
    _FakeDecoder, _RecordingState, _enqueue, _make_scheduler, _spec_runtime,
)


@contextmanager
def reference_counting_only():
    # A lifetime proof only: runtime and performance tests retain default GC.
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def lifecycle_fixture():
    runtime = FakeRuntime(device="cpu")
    request = GenerationRequest(
        request_id=7, prompt="p", prompt_tokens=[TextToken(1)],
        max_new_tokens=4, skill=object(), request_context=object())
    skill = _RecordingState(request)
    skill.append_token(TextToken(10))
    state = SequenceState(batch_idx=0, length=2, max_length=5, prompt_length=1)
    runtime.active_sequences[0] = state
    lifecycle = RequestLifecycle(request=request, skill_state=skill, sequence_state=state)
    request.lifecycle = lifecycle
    request.skill_state = skill
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._completed = deque()
    scheduler._eos_token_ids = (999,)
    return scheduler, request


@pytest.mark.parametrize("reason", ["stop", "length", "cancelled"])
def test_terminal_result_preserves_retained_request_and_output(reason):
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        request_ref, lifecycle_ref = weakref.ref(request), weakref.ref(lifecycle)
        scheduler._finalize_sequence(lifecycle, reason)
        lifecycle.scheduler_detached()
        result, = scheduler.pop_completed()
        assert request.lifecycle is lifecycle
        assert lifecycle.phase == RequestPhase.COMPLETED
        assert lifecycle.request is None
        assert request.skill_state.request is None
        assert result.tokens == [TextToken(10)]
        assert result.metrics.prompt_tokens == 1
        assert lifecycle.total_length == 2
        assert lifecycle.build_metrics(decode_tokens=1) == result.metrics
        with pytest.raises(RuntimeError, match="terminal request detachment"):
            lifecycle.stage_token(scheduler.runtime, TextToken(99))
        del request
        assert request_ref() is None
        del lifecycle
        assert lifecycle_ref() is None
        assert result.tokens == [TextToken(10)]
        assert result.finish_reason == reason


def test_materialized_zombie_keeps_request_until_physical_retirement():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        lifecycle.inflight_refs = 1
        request_ref = weakref.ref(request)
        scheduler._finalize_sequence(lifecycle, "stop")
        lifecycle.scheduler_detached()
        result, = scheduler.pop_completed()
        del request
        assert request_ref() is not None
        assert lifecycle.phase == RequestPhase.FINALIZING
        with pytest.raises(RuntimeError, match="in-flight"):
            lifecycle.resources_retired()
        assert request_ref() is not None
        lifecycle.inflight_refs = 0
        lifecycle.transition(RequestPhase.COMPLETED)
        scheduler._release_sequence(lifecycle)
        assert request_ref() is None
        assert result.tokens == [TextToken(10)]


def test_spec_terminal_retirement_breaks_both_request_backedges():
    with reference_counting_only():
        decoder = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12, 999]]})
        scheduler = _make_scheduler(_spec_runtime(decoder, eos_id=999))
        request = _enqueue(scheduler, 0, prompt_len=3, max_new=20)
        request_ref = weakref.ref(request)
        lifecycle = request.lifecycle
        scheduler._spec_admit()
        while scheduler._spec_decode_step():
            pass
        result, = scheduler.pop_completed()
        assert decoder.retired == [0]
        assert lifecycle.phase == RequestPhase.COMPLETED
        del request
        assert request_ref() is None
        assert result.tokens == [TextToken(11), TextToken(12), TextToken(999)]


def test_proven_uninstalled_failure_releases_request_after_metrics():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        scheduler.runtime.active_sequences.clear()
        lifecycle = request.lifecycle
        lifecycle.sequence_state = None
        request_ref = weakref.ref(request)
        scheduler._fail_request_early(request, ValueError("invalid admission"), resources_retired=True)
        result, = scheduler.pop_completed()
        assert result.metrics.decode_tokens == 0
        assert result.output == {"error": "invalid admission"}
        assert lifecycle.build_metrics(decode_tokens=0, cached_tokens=0) == result.metrics
        assert lifecycle.total_length == 2
        del request
        assert request_ref() is None


def test_failed_retirement_does_not_weaken_request():
    scheduler, request = lifecycle_fixture()
    lifecycle = request.lifecycle
    def fail(_state):
        raise RuntimeError("release failed")
    scheduler.runtime.release_sequence = fail
    with pytest.raises(RuntimeError, match="release failed"):
        scheduler._finalize_sequence(lifecycle, "stop")
    assert lifecycle.request is request
    assert request.skill_state.request is request
    assert not lifecycle._resources_retired
    assert not lifecycle._result_materialized


def test_shutdown_keeps_unresolved_inflight_request_ownership():
    from tests.test_autoregressive_executor import _executor, _pending
    scheduler, request = lifecycle_fixture()
    lifecycle = request.lifecycle
    lifecycle.inflight_refs = 1
    executor = _executor()
    executor._runtime = scheduler.runtime
    executor._active[7] = _pending(7)
    completion, = executor.shutdown(RuntimeError("shutdown"))
    assert completion.error is not None
    assert lifecycle.request is request
    assert request.skill_state.request is request
    assert not lifecycle._resources_retired


@reference_counting_only()
def test_cancelled_future_does_not_discard_retained_result_or_stream_tokens():
    from tests.test_autoregressive_executor import _executor, _pending
    scheduler, request = lifecycle_fixture()
    streamed = []
    request.stream_callback = streamed.append
    lifecycle = request.lifecycle
    lifecycle.stage_token(scheduler.runtime, TextToken(12))
    executor = _executor()
    executor._scheduler = scheduler
    pending = _pending(7)
    pending.future.cancel()
    executor._active[7] = pending
    scheduler._finalize_sequence(lifecycle, "length")
    lifecycle.scheduler_detached()
    completion, = executor._collect()
    assert completion.result.tokens == [TextToken(10), TextToken(12)]
    assert pending.future.cancelled()
    assert streamed[0].token == TextToken(12)
    request_ref = weakref.ref(request)
    del request
    assert request_ref() is None
    assert completion.result.tokens == [TextToken(10), TextToken(12)]


def test_later_running_row_finishes_without_external_request_owner():
    with reference_counting_only():
        scheduler, first_request = lifecycle_fixture()
        _, second_request = lifecycle_fixture()
        first, second = first_request.lifecycle, second_request.lifecycle
        second.state.batch_idx = 1
        second.request.max_new_tokens = 2
        second.inflight_refs = 1
        scheduler.runtime.active_sequences[1] = second.state
        scheduler.running = RunningQueue()
        scheduler.running.extend((first, second))
        scheduler.runtime.decode_slots = [SimpleNamespace(meta=SimpleNamespace(
            batch_idx=SimpleNamespace(cpu=torch.tensor([1]))))]
        scheduler._materialize_tokens = lambda *_args: [TextToken(12)]
        step = SimpleNamespace(kind="decode", slot_id=0, sequences=[second],
            transfer=SimpleNamespace(wait=lambda: (torch.tensor([12]), None)),
            payload=SimpleNamespace(pending_write_covered_by_transfer=True, runtime_step=None))
        first_ref, second_ref = weakref.ref(first_request), weakref.ref(second_request)
        del first_request, second_request
        assert first_ref() is not None and second_ref() is not None
        scheduler.commit_step(step)
        assert list(scheduler.running) == [first]
        assert first_ref() is not None
        assert second_ref() is None
        result, = scheduler.pop_completed()
        assert result.tokens == [TextToken(10), TextToken(12)]


def test_result_and_resource_completion_do_not_detach_queued_owner():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        reference = weakref.ref(request)
        scheduler.running = RunningQueue()
        scheduler.running.push(lifecycle)
        scheduler._finalize_sequence(lifecycle, "stop")
        del request
        assert reference() is not None and lifecycle.request is reference()
        assert lifecycle.skill_state.request is reference()
        scheduler.running.remove(lifecycle)
        lifecycle.scheduler_detached()
        assert reference() is None and lifecycle.request is None
        assert lifecycle.skill_state.request is None


def test_finalization_error_preserves_materialized_error_and_retires_owners():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        reference = weakref.ref(request)

        def fail(*args, **kwargs):
            raise ValueError("invalid terminal tokens")

        lifecycle.skill_state.finalize = fail
        scheduler._finalize_sequence(lifecycle, "stop")
        lifecycle.scheduler_detached()
        result, = scheduler.pop_completed()
        assert result.finish_reason == "error" and not result.tokens
        assert result.output == {"error": "invalid terminal tokens"}
        del request
        assert reference() is None
        assert lifecycle.request is None and lifecycle.skill_state.request is None


def test_retained_tokens_logprobs_and_metrics_survive_request_teardown():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        request.return_logprobs = True
        lifecycle.logprobs.append(-0.25)
        request.image_length = 7
        before_length = lifecycle.total_length
        scheduler._finalize_sequence(lifecycle, "length")
        lifecycle.scheduler_detached()
        result, = scheduler.pop_completed()
        assert result.logprobs == [-0.25]
        assert lifecycle.logprobs == [-0.25]
        assert lifecycle.total_length == before_length
        del request
        assert result.tokens == list(lifecycle.skill_state.tokens)
        assert lifecycle.build_metrics(decode_tokens=1) == result.metrics


def test_unproven_early_failure_keeps_both_request_owners():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        reference = weakref.ref(request)
        scheduler._fail_request_early(request, ValueError("admission failed"))
        lifecycle.scheduler_detached()
        del request
        assert reference() is not None
        assert lifecycle.request is reference()
        assert lifecycle.skill_state.request is reference()
        assert lifecycle._result_materialized and not lifecycle._resources_retired


def test_stream_callback_failure_keeps_unfinished_request_owners():
    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle

        def fail(_update):
            raise RuntimeError("stream consumer failed")

        request.stream_callback = fail
        with pytest.raises(RuntimeError, match="stream consumer failed"):
            lifecycle.stage_token(scheduler.runtime, TextToken(12))
        assert lifecycle.request is request and lifecycle.skill_state.request is request
        assert not lifecycle._result_materialized and not lifecycle._resources_retired
        assert list(lifecycle.skill_state.tokens) == [TextToken(10), TextToken(12)]


def test_shutdown_after_terminal_completion_does_not_restore_request_cycle():
    from tests.test_autoregressive_executor import _executor

    with reference_counting_only():
        scheduler, request = lifecycle_fixture()
        lifecycle = request.lifecycle
        reference = weakref.ref(request)
        scheduler._finalize_sequence(lifecycle, "length")
        lifecycle.scheduler_detached()
        result, = scheduler.pop_completed()
        executor = _executor()
        executor._runtime = scheduler.runtime
        assert executor.shutdown() == ()
        del request
        assert reference() is None
        assert lifecycle.request is None and lifecycle.skill_state.request is None
        assert result.tokens == [TextToken(10)]
