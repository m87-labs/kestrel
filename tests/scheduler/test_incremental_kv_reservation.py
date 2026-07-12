from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pytest

from kestrel.runtime import SequenceState, TextToken
from kestrel.scheduler.pipeline import PipelineState
from kestrel.scheduler.queues import RequestQueue, RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import (
    GeneratedPrefix,
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)
from kestrel.skills import SkillFinalizeResult

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass
class _SkillStateStub:
    tokens: list[object] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def allowed_token_ids(self, runtime: object) -> None:
        return None

    def suppressed_token_ids(self, runtime: object) -> None:
        return None

    def stop_token_ids(self, runtime: object) -> None:
        return None

    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        return SkillFinalizeResult(text="", tokens=list(self.tokens), output={})


def _running_lifecycle(
    *,
    request_id: int,
    batch_idx: int,
    length: int = 4,
    max_length: int = 10,
    inflight_refs: int = 0,
    tokens: list[TextToken] | None = None,
    return_logprobs: bool = False,
) -> RequestLifecycle:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=16,
        skill=object(),
        request_context=object(),
        return_logprobs=True if return_logprobs else None,
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(tokens=tokens or [TextToken(10)]),
        sequence_state=SequenceState(
            batch_idx=batch_idx,
            length=length,
            max_length=max_length,
            prompt_length=1,
        ),
        phase=RequestPhase.RUNNING,
        packed_pending_ready=True,
        inflight_refs=inflight_refs,
    )
    request.lifecycle = lifecycle
    return lifecycle


def _waiting_request(request_id: int) -> GenerationRequest:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=16,
        skill=object(),
        request_context=object(),
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(),
        phase=RequestPhase.READY_FOR_PREFILL,
        crops_ready=True,
        lora_slot_ready=True,
    )
    request.lifecycle = lifecycle
    return request


def _scheduler(runtime: FakeRuntime) -> GenerationScheduler:
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler.waiting = RequestQueue()
    scheduler.running = RunningQueue()
    scheduler._completed = deque()
    scheduler._compute_stream = None
    scheduler._pipeline = PipelineState()
    scheduler._last_deferred_request_id = None
    scheduler._preempted_request_ids = set()
    scheduler._decode_kv_recovery_pending = False
    scheduler._decode_kv_recovery_request_id = None
    scheduler._decode_kv_recovery_progressed = False
    scheduler._decode_kv_recovery_deferred = False
    scheduler._pending_spec = None
    return scheduler


def test_schedule_decode_step_preempts_later_row_when_growth_fails() -> None:
    runtime = FakeRuntime(max_batch_size=2)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=1, batch_idx=0)
    victim = _running_lifecycle(
        request_id=2,
        batch_idx=1,
        tokens=[TextToken(20), TextToken(21)],
    )
    victim_state = victim.state
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = victim_state
    scheduler.running.push(blocked)
    scheduler.running.push(victim)

    fail_blocked_once = True

    def expand_once_after_preemption(
        state: SequenceState, tokens: int = 1
    ) -> bool:
        nonlocal fail_blocked_once
        runtime.expand_kv_reservation_calls.append((state, tokens))
        if state is blocked.state and fail_blocked_once:
            fail_blocked_once = False
            return False
        return True

    runtime.expand_kv_reservation = expand_once_after_preemption

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [blocked]
    assert not blocked.finished
    assert list(scheduler.running) == [blocked]
    assert list(scheduler.waiting) == [victim.request]
    assert scheduler._preempted_request_ids == {2}
    assert not scheduler._completed
    assert victim.phase == RequestPhase.READY_FOR_PREFILL
    assert victim.sequence_state is None
    assert not victim.packed_pending_ready
    assert victim.request.generated_prefix.tokens == (TextToken(20), TextToken(21))
    assert victim.request.remaining_new_tokens == 14
    assert runtime.released_sequences == [victim_state]
    assert runtime.retained_prefixes == []
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1), (0, 1)]


def test_schedule_decode_step_preempts_multiple_rows_until_growth_succeeds() -> None:
    runtime = FakeRuntime(max_batch_size=3)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=1, batch_idx=0)
    victim_a = _running_lifecycle(request_id=2, batch_idx=1)
    victim_b = _running_lifecycle(request_id=3, batch_idx=2)
    victim_a_state = victim_a.state
    victim_b_state = victim_b.state
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = victim_a_state
    runtime.active_sequences[2] = victim_b_state
    scheduler.running.push(blocked)
    scheduler.running.push(victim_a)
    scheduler.running.push(victim_b)

    attempts = 0

    def expand_after_two_preemptions(
        state: SequenceState, tokens: int = 1
    ) -> bool:
        nonlocal attempts
        runtime.expand_kv_reservation_calls.append((state, tokens))
        if state is blocked.state:
            attempts += 1
            return attempts >= 3
        return True

    runtime.expand_kv_reservation = expand_after_two_preemptions

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [blocked]
    assert list(scheduler.running) == [blocked]
    assert list(scheduler.waiting) == [victim_a.request, victim_b.request]
    assert scheduler._preempted_request_ids == {2, 3}
    assert runtime.released_sequences == [victim_b_state, victim_a_state]
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1), (0, 1), (0, 1)]


def test_decode_recovery_pending_survives_unrelated_ready_row() -> None:
    runtime = FakeRuntime(max_batch_size=1, max_batch_slots=3)
    scheduler = _scheduler(runtime)
    ready = _running_lifecycle(request_id=1, batch_idx=0, length=4, max_length=10)
    blocked = _running_lifecycle(request_id=2, batch_idx=1, length=4, max_length=10)
    runtime.active_sequences[0] = ready.state
    runtime.active_sequences[1] = blocked.state
    scheduler.running.push(ready)
    scheduler.running.push(blocked)
    scheduler._decode_kv_recovery_pending = True
    scheduler._decode_kv_recovery_request_id = blocked.request.request_id

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [ready]
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_request_id == blocked.request.request_id

    fresh = _waiting_request(request_id=10)
    scheduler.waiting.push(fresh)

    batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=1,
    )

    assert batch == []


def test_schedule_decode_step_preempts_idle_row_before_waiting_for_inflight() -> None:
    runtime = FakeRuntime(max_batch_size=3)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=1, batch_idx=0)
    idle_victim = _running_lifecycle(request_id=2, batch_idx=1)
    inflight = _running_lifecycle(request_id=3, batch_idx=2, inflight_refs=1)
    idle_victim_state = idle_victim.state
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = idle_victim_state
    runtime.active_sequences[2] = inflight.state
    scheduler.running.push(blocked)
    scheduler.running.push(idle_victim)
    scheduler.running.push(inflight)

    fail_blocked_once = True

    def expand_once_after_preemption(
        state: SequenceState, tokens: int = 1
    ) -> bool:
        nonlocal fail_blocked_once
        runtime.expand_kv_reservation_calls.append((state, tokens))
        if state is blocked.state and fail_blocked_once:
            fail_blocked_once = False
            return False
        return True

    runtime.expand_kv_reservation = expand_once_after_preemption

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [blocked, inflight]
    assert list(scheduler.running) == [blocked, inflight]
    assert list(scheduler.waiting) == [idle_victim.request]
    assert scheduler._preempted_request_ids == {2}
    assert not scheduler._decode_kv_recovery_deferred
    assert runtime.released_sequences == [idle_victim_state]


def test_advance_requeues_and_request_can_complete_after_recompute() -> None:
    runtime = FakeRuntime()
    runtime.kv_reservation_failures.add(0)
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(
        request_id=7,
        batch_idx=0,
        tokens=[TextToken(20)],
    )
    original_state = seq.state
    runtime.active_sequences[0] = original_state
    scheduler.running.push(seq)

    progressed = GenerationScheduler.advance(scheduler)

    assert progressed is True
    assert list(scheduler.running) == []
    assert list(scheduler.waiting) == [seq.request]
    assert seq.phase == RequestPhase.READY_FOR_PREFILL
    assert not seq.finished
    assert not scheduler._completed
    assert runtime.released_sequences == [original_state]

    candidate = GenerationScheduler._make_prefill_candidate(scheduler, seq.request)
    assert candidate is not None
    assert candidate.request is seq.request
    assert candidate.classification.prompt_length == len(seq.request.prefill_tokens)

    recompute_state = SequenceState(
        batch_idx=1,
        length=len(seq.request.prefill_tokens),
        max_length=seq.request.target_length,
        prompt_length=len(seq.request.prefill_tokens),
    )
    scheduler.waiting.remove(seq.request)
    scheduler._preempted_request_ids.discard(seq.request.request_id)
    seq.sequence_state = recompute_state
    seq.packed_pending_ready = True
    seq.transition(RequestPhase.RUNNING)
    runtime.active_sequences[1] = recompute_state
    scheduler.running.push(seq)

    GenerationScheduler._finalize_sequence(scheduler, seq, "length")

    completed = GenerationScheduler.pop_completed(scheduler)
    assert len(completed) == 1
    assert completed[0].request_id == seq.request.request_id
    assert completed[0].finish_reason == "length"
    assert completed[0].output == {}
    assert not scheduler._completed
    assert runtime.released_sequences == [original_state, recompute_state]


def test_schedule_decode_step_does_not_preempt_selected_active_row() -> None:
    runtime = FakeRuntime(max_batch_size=2)
    scheduler = _scheduler(runtime)
    selected_newer = _running_lifecycle(request_id=2, batch_idx=0)
    blocked = _running_lifecycle(request_id=1, batch_idx=1)
    selected_state = selected_newer.state
    blocked_state = blocked.state
    runtime.active_sequences[0] = selected_state
    runtime.active_sequences[1] = blocked_state
    scheduler.running.push(selected_newer)
    scheduler.running.push(blocked)

    def fail_blocked_growth(state: SequenceState, tokens: int = 1) -> bool:
        runtime.expand_kv_reservation_calls.append((state, tokens))
        return state is not blocked_state

    runtime.expand_kv_reservation = fail_blocked_growth

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [selected_newer]
    assert list(scheduler.running) == [selected_newer]
    assert list(scheduler.waiting) == [blocked.request]
    assert blocked.phase == RequestPhase.READY_FOR_PREFILL
    assert selected_newer.phase == RequestPhase.RUNNING
    assert scheduler._preempted_request_ids == {blocked.request.request_id}
    assert runtime.released_sequences == [blocked_state]
    assert selected_state in runtime.active_sequences.values()
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1), (1, 1)]


def test_preempt_request_preserves_logprobs() -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(
        request_id=5,
        batch_idx=0,
        tokens=[TextToken(20), TextToken(21)],
        return_logprobs=True,
    )
    seq.logprobs.extend([-0.5, -0.25])
    state = seq.state
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)

    GenerationScheduler._preempt_request(scheduler, seq)

    assert seq.request.generated_prefix.tokens == (TextToken(20), TextToken(21))
    assert seq.request.generated_prefix.logprobs == (-0.5, -0.25)
    assert seq.request.initial_generated_prefix_length == 0
    assert seq.kv_preemptions == 1
    assert seq.build_metrics(decode_tokens=2).kv_preemptions == 1
    assert runtime.released_sequences == [state]


def test_preempt_request_keeps_base_request_lora_ready() -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(request_id=5, batch_idx=0)
    seq.lora_slot_ready = True
    state = seq.state
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)

    GenerationScheduler._preempt_request(scheduler, seq)

    assert seq.request.adapter is None
    assert seq.request.lora_slot == 0
    assert seq.lora_slot_ready is True
    assert seq.phase == RequestPhase.READY_FOR_PREFILL
    assert list(scheduler.waiting) == [seq.request]


def test_preempt_request_extends_caller_generated_prefix() -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    request = GenerationRequest(
        request_id=6,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=6,
        skill=object(),
        request_context=object(),
        return_logprobs=True,
        generated_prefix=GeneratedPrefix(
            tokens=(TextToken(10),),
            logprobs=(-1.0,),
        ),
    )
    state = SequenceState(
        batch_idx=0,
        length=3,
        max_length=8,
        prompt_length=1,
    )
    seq = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(
            tokens=[TextToken(10), TextToken(20), TextToken(21)]
        ),
        sequence_state=state,
        phase=RequestPhase.RUNNING,
        packed_pending_ready=True,
    )
    request.lifecycle = seq
    seq.logprobs.extend([-0.5, -0.25])
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)

    GenerationScheduler._preempt_request(scheduler, seq)

    assert seq.request.generated_prefix.tokens == (
        TextToken(10),
        TextToken(20),
        TextToken(21),
    )
    assert seq.request.generated_prefix.logprobs == (-1.0, -0.5, -0.25)
    assert seq.request.initial_generated_prefix_length == 1
    assert seq.request.remaining_new_tokens == 3
    assert seq.build_metrics(decode_tokens=2).prompt_tokens == 2
    assert runtime.released_sequences == [state]


def test_preempt_request_does_not_rearm_one_shot_suppression() -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(
        request_id=5,
        batch_idx=0,
        tokens=[TextToken(20), TextToken(21)],
    )
    seq.request.suppress_next_token_ids = (99,)
    state = seq.state
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)

    GenerationScheduler._preempt_request(scheduler, seq)

    _allowed, _suppressed, _restrict, suppress_rows, _greedy, _logprobs = (
        GenerationScheduler._build_mask_spec(scheduler, [seq])
    )
    assert suppress_rows == []


def test_preempted_request_metrics_keep_original_prompt_tokens() -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(
        request_id=5,
        batch_idx=0,
        tokens=[TextToken(20), TextToken(21)],
    )
    state = seq.state
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)

    GenerationScheduler._preempt_request(scheduler, seq)
    seq.sequence_state = SequenceState(
        batch_idx=1,
        length=4,
        max_length=10,
        prompt_length=len(seq.request.prefill_tokens),
    )

    metrics = seq.build_metrics(decode_tokens=2)

    assert metrics.prompt_tokens == seq.request.prompt_length

def test_preempted_request_metrics_accumulate_preemption_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(
        request_id=5,
        batch_idx=0,
        tokens=[TextToken(20), TextToken(21)],
    )
    seq.submitted_at = 90.0
    seq.prefill_started_at = 100.0
    seq.prefill_completed_at = 110.0
    seq.first_token_time = 111.0
    state = seq.state
    runtime.active_sequences[0] = state
    scheduler.running.push(seq)
    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.time.perf_counter",
        lambda: 130.0,
    )

    GenerationScheduler._preempt_request(scheduler, seq)
    seq.sequence_state = SequenceState(
        batch_idx=1,
        length=4,
        max_length=10,
        prompt_length=len(seq.request.prefill_tokens),
    )
    seq.prefill_started_at = 140.0
    seq.prefill_completed_at = 145.0
    seq.completed_at = 160.0

    metrics = seq.build_metrics(decode_tokens=2)

    assert metrics.prefill_time_ms == 15000.0
    assert metrics.decode_time_ms == 35000.0
    assert metrics.ttft_ms == 21000.0


def test_schedule_decode_step_self_preempts_newer_row_so_older_row_can_run() -> None:
    runtime = FakeRuntime(max_batch_size=1)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=2, batch_idx=0)
    older = _running_lifecycle(request_id=1, batch_idx=1)
    blocked_state = blocked.state
    older_state = older.state
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = older_state
    scheduler.running.push(blocked)
    scheduler.running.push(older)

    older_released = False
    release_sequence = runtime.release_sequence

    def track_older_release(state: SequenceState) -> None:
        nonlocal older_released
        if state is older_state:
            older_released = True
        release_sequence(state)

    def grow_blocked_only_after_older_released(
        state: SequenceState,
        tokens: int = 1,
    ) -> bool:
        runtime.expand_kv_reservation_calls.append((state, tokens))
        if state is blocked_state:
            return older_released
        return True

    runtime.release_sequence = track_older_release
    runtime.expand_kv_reservation = grow_blocked_only_after_older_released

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [older]
    assert list(scheduler.running) == [older]
    assert list(scheduler.waiting) == [blocked.request]
    assert not blocked.finished
    assert not blocked.finalized
    assert blocked.phase == RequestPhase.READY_FOR_PREFILL
    assert blocked.sequence_state is None
    assert blocked.kv_preemptions == 1
    assert scheduler._preempted_request_ids == {blocked.request.request_id}
    assert not scheduler._completed
    assert not older_released
    assert runtime.released_sequences == [blocked_state]
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_request_id == blocked.request.request_id
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1), (1, 1)]


def test_schedule_decode_step_skips_growth_when_row_already_has_capacity() -> None:
    runtime = FakeRuntime(max_batch_size=1)
    scheduler = _scheduler(runtime)
    seq = _running_lifecycle(request_id=1, batch_idx=0, length=4, max_length=8)
    runtime.page_table.capacity[0] = 5
    runtime.active_sequences[0] = seq.state
    scheduler.running.push(seq)

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [seq]
    assert runtime.expand_kv_reservation_calls == []


def test_decode_recovery_pending_survives_unrelated_row_with_capacity() -> None:
    runtime = FakeRuntime(max_batch_size=1, max_batch_slots=3)
    scheduler = _scheduler(runtime)
    ready = _running_lifecycle(request_id=1, batch_idx=0, length=4, max_length=10)
    blocked = _running_lifecycle(request_id=2, batch_idx=1, length=4, max_length=10)
    runtime.active_sequences[0] = ready.state
    runtime.active_sequences[1] = blocked.state
    runtime.page_table.capacity[0] = 8
    runtime.page_table.capacity[1] = 4
    scheduler.running.push(ready)
    scheduler.running.push(blocked)
    scheduler._decode_kv_recovery_pending = True
    scheduler._decode_kv_recovery_request_id = blocked.request.request_id

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is not None
    assert plan.sequences == [ready]
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_request_id == blocked.request.request_id

    fresh = _waiting_request(request_id=10)
    scheduler.waiting.push(fresh)

    batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=1,
    )

    assert batch == []


def test_schedule_decode_step_defers_instead_of_preempting_older_rows() -> None:
    runtime = FakeRuntime(max_batch_size=1)
    runtime.kv_reservation_failures.add(0)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=2, batch_idx=0, inflight_refs=1)
    older = _running_lifecycle(request_id=1, batch_idx=1)
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = older.state
    scheduler.running.push(blocked)
    scheduler.running.push(older)

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is None
    assert list(scheduler.running) == [blocked, older]
    assert list(scheduler.waiting) == []
    assert not blocked.finished
    assert blocked.phase == RequestPhase.RUNNING
    assert older.phase == RequestPhase.RUNNING
    assert scheduler._preempted_request_ids == set()
    assert runtime.released_sequences == []
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_deferred
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1)]


def test_schedule_decode_step_waits_for_inflight_rows_instead_of_failing() -> None:
    runtime = FakeRuntime(max_batch_size=2)
    runtime.kv_reservation_failures.add(0)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=1, batch_idx=0)
    inflight = _running_lifecycle(request_id=2, batch_idx=1, inflight_refs=1)
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = inflight.state
    scheduler.running.push(blocked)
    scheduler.running.push(inflight)

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is None
    assert not blocked.finished
    assert not blocked.finalized
    assert blocked.phase == RequestPhase.RUNNING
    assert list(scheduler.running) == [blocked, inflight]
    assert list(scheduler.waiting) == []
    assert not scheduler._completed
    assert runtime.released_sequences == []
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_deferred


def test_schedule_decode_step_does_not_preempt_uncommitted_prefill_row() -> None:
    runtime = FakeRuntime(max_batch_size=2)
    runtime.kv_reservation_failures.add(0)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=1, batch_idx=0)
    prefill_pending = _running_lifecycle(request_id=2, batch_idx=1)
    prefill_pending.uncommitted_prefill_token = True
    runtime.active_sequences[0] = blocked.state
    runtime.active_sequences[1] = prefill_pending.state
    scheduler.running.push(blocked)
    scheduler.running.push(prefill_pending)
    scheduler._pipeline.batch_queue.append(object())

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is None
    assert list(scheduler.running) == [blocked, prefill_pending]
    assert list(scheduler.waiting) == []
    assert prefill_pending.phase == RequestPhase.RUNNING
    assert prefill_pending.sequence_state is not None
    assert prefill_pending.uncommitted_prefill_token
    assert runtime.released_sequences == []
    assert scheduler._decode_kv_recovery_pending
    assert scheduler._decode_kv_recovery_deferred


def test_schedule_decode_step_self_preempts_only_ready_row_when_growth_fails() -> None:
    runtime = FakeRuntime()
    runtime.kv_reservation_failures.add(0)
    scheduler = _scheduler(runtime)
    blocked = _running_lifecycle(request_id=7, batch_idx=0)
    state = blocked.state
    runtime.active_sequences[0] = state
    scheduler.running.push(blocked)

    plan = GenerationScheduler.schedule_decode_step(scheduler)

    assert plan is None
    assert not blocked.finished
    assert not blocked.finalized
    assert blocked.phase == RequestPhase.READY_FOR_PREFILL
    assert list(scheduler.running) == []
    assert list(scheduler.waiting) == [blocked.request]
    assert scheduler._preempted_request_ids == {blocked.request.request_id}
    assert blocked.kv_preemptions == 1
    assert not scheduler._completed
    assert runtime.released_sequences == [state]
    assert scheduler._decode_kv_recovery_pending
    assert [
        (state.batch_idx, tokens)
        for state, tokens in runtime.expand_kv_reservation_calls
    ] == [(0, 1)]


def test_select_prefill_batch_prioritizes_preempted_requests() -> None:
    runtime = FakeRuntime(max_batch_size=2, max_batch_slots=3)
    scheduler = _scheduler(runtime)
    fresh = _waiting_request(request_id=10)
    preempted = _waiting_request(request_id=11)
    scheduler.waiting.push(fresh)
    scheduler.waiting.push(preempted)
    scheduler._preempted_request_ids = {preempted.request_id}

    batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=2,
    )

    assert [candidate.request.request_id for candidate in batch] == [11]


def test_select_prefill_batch_blocks_fresh_requests_during_decode_kv_recovery() -> None:
    runtime = FakeRuntime(max_batch_size=2, max_batch_slots=3)
    scheduler = _scheduler(runtime)
    fresh = _waiting_request(request_id=10)
    scheduler.waiting.push(fresh)
    scheduler._decode_kv_recovery_pending = True

    batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=2,
    )

    assert batch == []


def test_prefill_retry_keeps_preempted_request_and_blocks_fresh_request() -> None:
    runtime = FakeRuntime(prepare_exc=RuntimeError("Cannot reserve KV pages"))
    scheduler = _scheduler(runtime)
    preempted = _waiting_request(request_id=10)
    fresh = _waiting_request(request_id=11)
    scheduler.waiting.push(preempted)
    scheduler.waiting.push(fresh)
    scheduler._preempted_request_ids = {preempted.request_id}
    scheduler._decode_kv_recovery_pending = True
    pipeline = PipelineState()

    progressed = GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert progressed is False
    assert list(scheduler.waiting) == [preempted, fresh]
    assert scheduler._preempted_request_ids == {preempted.request_id}
    assert scheduler._completed == deque()
    assert preempted.lifecycle.phase == RequestPhase.READY_FOR_PREFILL
    assert fresh.lifecycle.phase == RequestPhase.READY_FOR_PREFILL
    assert len(runtime.prepare_calls) == 1
    assert runtime.prepare_calls[0]["prompt_tokens"] == preempted.prefill_tokens
    assert len(runtime.released_prefill_slots) == 1


def test_recovery_blocks_fresh_prefill_while_preempted_request_waits_resources() -> None:
    runtime = FakeRuntime(max_batch_size=2, max_batch_slots=3)
    scheduler = _scheduler(runtime)
    preempted = _waiting_request(request_id=10)
    preempted.lifecycle.has_image = True
    preempted.lifecycle.crops_ready = False
    preempted.lifecycle.transition(RequestPhase.WAITING_RESOURCES)
    fresh = _waiting_request(request_id=11)
    scheduler.waiting.push(preempted)
    scheduler.waiting.push(fresh)
    scheduler._preempted_request_ids = {preempted.request_id}
    scheduler._decode_kv_recovery_pending = True

    blocked_batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=2,
    )

    assert blocked_batch == []
    assert list(scheduler.waiting) == [preempted, fresh]
    assert not scheduler._completed

    preempted.lifecycle.crops_ready = True
    preempted.lifecycle.transition(RequestPhase.READY_FOR_PREFILL)

    resumed_batch = GenerationScheduler._select_prefill_batch(
        scheduler,
        capacity_remaining=2,
    )

    assert [candidate.request for candidate in resumed_batch] == [preempted]


def test_no_progress_fallback_respects_reclaimable_prefill_budget() -> None:
    runtime = FakeRuntime(max_batch_size=1, max_batch_slots=2)
    runtime.decode_reserve_tokens = 16
    runtime.page_table.pages_available = 0
    runtime.prefill_budget = lambda: (17, 1)
    scheduler = _scheduler(runtime)
    request = _waiting_request(request_id=12)
    scheduler.waiting.push(request)
    scheduler._select_prefill_batch = lambda capacity_remaining, **kwargs: []

    progressed = GenerationScheduler.advance(scheduler)

    assert progressed is False
    assert runtime.classify_calls == [[TextToken(1)]]
    assert list(scheduler.waiting) == [request]


def test_no_progress_fallback_ignores_lora_resource_wait() -> None:
    runtime = FakeRuntime(max_batch_size=1, max_batch_slots=2)
    runtime.prefill_budget = lambda: (0, 1)
    scheduler = _scheduler(runtime)
    request = _waiting_request(request_id=13)
    request.lifecycle.lora_slot_ready = False
    request.lifecycle.transition(RequestPhase.WAITING_RESOURCES)
    scheduler.waiting.push(request)
    scheduler._select_prefill_batch = lambda capacity_remaining, **kwargs: []

    progressed = GenerationScheduler.advance(scheduler)

    assert progressed is False
    assert runtime.classify_calls == []
    assert list(scheduler.waiting) == [request]
