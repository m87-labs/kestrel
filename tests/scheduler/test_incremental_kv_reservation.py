from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pytest

from kestrel.runtime import SequenceState, TextToken
from kestrel.scheduler.queues import RequestQueue, RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import (
    GeneratedPrefix,
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)

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
    scheduler._last_deferred_request_id = None
    scheduler._preempted_request_ids = set()
    return scheduler


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
    assert runtime.released_sequences == [state]


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
