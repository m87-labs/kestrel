from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from kestrel.moondream.runtime import PrefillClassification, TextToken
from kestrel.scheduler.pipeline import PipelineState
from kestrel.scheduler.queues import RequestQueue, RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler, _PrefillCandidate
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle, RequestPhase

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass
class _SkillStateStub:
    token_count: int = 0
    tokens: list[object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = []


def _make_request(*, request_id: int = 1, max_new_tokens: int = 8) -> GenerationRequest:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=max_new_tokens,
        skill=object(),
        request_context=object(),
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(),
        phase=RequestPhase.READY_FOR_PREFILL,
        has_image=False,
        crops_ready=True,
        lora_slot_ready=False,
        submitted_at=0.0,
    )
    request.lifecycle = lifecycle
    return request


def _make_candidate(request: GenerationRequest) -> _PrefillCandidate:
    return _PrefillCandidate(
        request=request,
        classification=PrefillClassification(
            prompt_length=request.prompt_length,
            skip_positions=0,
            can_reuse=False,
            use_prefix_attn=False,
        ),
        reserve_length=request.target_length,
        pages_needed=1,
        cohort_key=None,
    )


def _make_scheduler(
    request: GenerationRequest,
    runtime: FakeRuntime,
) -> GenerationScheduler:
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler.waiting = RequestQueue()
    scheduler.waiting.push(request)
    scheduler.running = RunningQueue()
    scheduler._completed = deque()
    scheduler._select_prefill_batch = lambda capacity_remaining: [_make_candidate(request)]
    return scheduler


@pytest.mark.parametrize("failure_stage", ["adapter", "prepare"])
def test_launch_prefill_step_dequeues_requests_that_fail_to_bind(
    failure_stage: str,
) -> None:
    request = _make_request()
    runtime = FakeRuntime(
        prepare_exc=RuntimeError("prepare failed") if failure_stage == "prepare" else None
    )
    scheduler = _make_scheduler(request, runtime)
    pipeline = PipelineState()

    if failure_stage == "adapter":
        scheduler._acquire_adapter_slot = lambda adapter_id: (_ for _ in ()).throw(
            RuntimeError("adapter failed")
        )
    else:
        request.lifecycle.lora_slot_ready = True
        scheduler._acquire_adapter_slot = lambda adapter_id: 0

    progressed = GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert progressed is True
    assert len(scheduler.waiting) == 0
    assert len(scheduler._completed) == 1
    assert scheduler._completed[0].request_id == request.request_id
    assert request.lifecycle.phase == RequestPhase.COMPLETED
    assert len(runtime.released_prefill_slots) == 1
