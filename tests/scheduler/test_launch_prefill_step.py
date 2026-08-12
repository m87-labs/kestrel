from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kestrel.models.moondream.runtime import PrefillClassification, TextToken
from kestrel.runtime import SequenceState
from kestrel.scheduler.pipeline import PipelineState
from kestrel.scheduler.queues import RequestQueue, RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler, _PrefillCandidate
from kestrel.scheduler.types import (
    GeneratedPrefix,
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)
from kestrel.skills import SkillRegistry

from tests.scheduler._fake_runtime import FakeRuntime


class _EmptyEosRuntime(FakeRuntime):
    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return ()


@dataclass
class _SkillStateStub:
    token_count: int = 0
    tokens: list[object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = []


def _make_request(
    *,
    request_id: int = 1,
    max_new_tokens: int = 8,
    generated_prefix_tokens: list[TextToken] | None = None,
    encoder_input: object | None = None,
) -> GenerationRequest:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=max_new_tokens,
        skill=object(),
        request_context=object(),
        generated_prefix=GeneratedPrefix(tokens=tuple(generated_prefix_tokens or [])),
        encoder_input=encoder_input,
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
    scheduler._select_prefill_batch = lambda capacity_remaining: [
        _make_candidate(request)
    ]
    return scheduler


def test_scheduler_requires_uniform_sampling_hooks() -> None:
    runtime = FakeRuntime(device="cpu")
    del runtime.sampling_hooks

    with pytest.raises(TypeError, match="must define sampling_hooks"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


def test_scheduler_requires_nonempty_runtime_eos_ids() -> None:
    runtime = _EmptyEosRuntime(device="cpu")

    with pytest.raises(ValueError, match="eos_token_ids must not be empty"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


def test_make_prefill_candidate_classifies_prompt_and_generated_prefix() -> None:
    request = _make_request(generated_prefix_tokens=[TextToken(10), TextToken(11)])
    runtime = FakeRuntime()
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    candidate = GenerationScheduler._make_prefill_candidate(scheduler, request)

    assert candidate is not None
    assert runtime.classify_calls == [[TextToken(1), TextToken(10), TextToken(11)]]
    assert candidate.reserve_length == request.target_length


def test_make_prefill_candidate_never_reuses_encoder_conditioned_prompt() -> None:
    request = _make_request(encoder_input=object())
    request.encoder_input = None
    runtime = FakeRuntime()
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    candidate = GenerationScheduler._make_prefill_candidate(scheduler, request)

    assert candidate is not None
    assert runtime.classify_calls == []
    assert candidate.classification.skip_positions == 0
    assert candidate.classification.can_reuse is False
    assert candidate.reserve_length == request.target_length


def test_launch_prefill_step_prefills_generated_prefix_then_remaining_tokens() -> None:
    encoder_input = object()
    request = _make_request(
        max_new_tokens=5,
        generated_prefix_tokens=[TextToken(10), TextToken(11)],
        encoder_input=encoder_input,
    )
    request.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(prepare_exc=RuntimeError("prepare failed"))
    scheduler = _make_scheduler(request, runtime)
    scheduler._acquire_adapter_slot = lambda adapter_id: 0
    pipeline = PipelineState()

    GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert len(runtime.prepare_calls) == 1
    assert runtime.prepare_calls[0]["prompt_tokens"] == [
        TextToken(1),
        TextToken(10),
        TextToken(11),
    ]
    assert runtime.prepare_calls[0]["max_new_tokens"] == 3
    assert runtime.prepare_calls[0]["encoder_input"] is encoder_input


@pytest.mark.parametrize("failure_stage", ["adapter", "prepare"])
def test_launch_prefill_step_dequeues_requests_that_fail_to_bind(
    failure_stage: str,
) -> None:
    request = _make_request()
    runtime = FakeRuntime(
        prepare_exc=RuntimeError("prepare failed")
        if failure_stage == "prepare"
        else None
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
    assert request.encoder_input is None
    assert len(runtime.released_prefill_slots) == 1


def test_terminal_bound_failure_releases_adapter_from_deferred_bind() -> None:
    request = _make_request()
    # A previous attempt acquired the adapter, then deferred before launch.
    # This attempt inherits the scheduler-owned reference without acquiring it.
    request.adapter = "ft-1"
    request.lora_slot = 7
    request.lifecycle.lora_slot_ready = True
    request.lifecycle.skill_state.on_prefill = lambda _runtime: (
        (_ for _ in ()).throw(RuntimeError("on_prefill failed"))
    )
    prepared = SimpleNamespace(
        state=SequenceState(batch_idx=1, length=1, max_length=9, lora_slot=7),
        use_prefix_attn=False,
    )
    runtime = FakeRuntime(prepare_result=prepared)
    scheduler = _make_scheduler(request, runtime)
    scheduler._compute_stream = None
    staging = object()
    scheduler._acquire_prefill_staging = lambda: staging
    scheduler._release_prefill_staging = lambda _staging: None
    scheduler._stage_prefill_sampling_params = lambda *_args: None

    progressed = GenerationScheduler._launch_prefill_step(
        scheduler,
        PipelineState(),
    )

    assert progressed is True
    assert runtime.aborted_prepared == [prepared]
    assert runtime.released_adapter_slots == [7]
    assert request.lora_slot == 0
    assert request.lifecycle.lora_slot_ready is False
    assert request.lifecycle.phase == RequestPhase.COMPLETED


def test_terminal_prepare_failure_releases_adapter_from_deferred_bind() -> None:
    request = _make_request()
    request.adapter = "ft-1"
    request.lora_slot = 11
    request.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(prepare_exc=ValueError("invalid checkpoint state"))
    scheduler = _make_scheduler(request, runtime)

    progressed = GenerationScheduler._launch_prefill_step(
        scheduler,
        PipelineState(),
    )

    assert progressed is True
    assert runtime.released_adapter_slots == [11]
    assert request.lora_slot == 0
    assert request.lifecycle.lora_slot_ready is False
    assert request.lifecycle.phase == RequestPhase.COMPLETED


def test_retryable_prepare_failure_preserves_deferred_adapter() -> None:
    request = _make_request()
    request.adapter = "ft-1"
    request.lora_slot = 13
    request.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(prepare_exc=RuntimeError("Cannot reserve requested pages"))
    scheduler = _make_scheduler(request, runtime)

    progressed = GenerationScheduler._launch_prefill_step(
        scheduler,
        PipelineState(),
    )

    assert progressed is False
    assert runtime.released_adapter_slots == []
    assert request.lora_slot == 13
    assert request.lifecycle.lora_slot_ready is True
    assert list(scheduler.waiting) == [request]


def test_advance_rejects_only_request_that_cannot_fit_kv_cache() -> None:
    oversized = _make_request(request_id=42, max_new_tokens=8)
    admissible = _make_request(request_id=43, max_new_tokens=2)
    scheduler = _make_scheduler(oversized, FakeRuntime(max_seq_length=4))
    scheduler._compute_stream = None
    scheduler.waiting.push(admissible)
    scheduler._pipeline = PipelineState()
    scheduler._pending_spec = None
    scheduler._launch_prefill_step = lambda pipeline: False
    scheduler.schedule_decode_step = lambda: None

    progressed = GenerationScheduler.advance(scheduler)

    assert progressed is True
    assert list(scheduler.waiting) == [admissible]
    assert oversized.lifecycle.phase == RequestPhase.COMPLETED
    assert admissible.lifecycle.phase == RequestPhase.READY_FOR_PREFILL
    completed = scheduler.pop_completed()
    assert len(completed) == 1
    assert completed[0].request_id == oversized.request_id
    assert completed[0].finish_reason == "error"
    assert completed[0].output == {
        "error": ("Insufficient KV cache capacity for request 42 (needs 9 tokens).")
    }
