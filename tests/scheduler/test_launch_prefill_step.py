from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

import pytest

from kestrel.models.moondream.runtime import PrefillClassification, TextToken
from kestrel.scheduler.pipeline import PipelineState
from kestrel.scheduler.queues import RequestQueue, RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler, _PrefillCandidate
from kestrel.scheduler.types import (
    GeneratedPrefix,
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)

from tests.scheduler._fake_runtime import FakeRuntime


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
) -> GenerationRequest:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=max_new_tokens,
        skill=object(),
        request_context=object(),
        generated_prefix=GeneratedPrefix(tokens=tuple(generated_prefix_tokens or [])),
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


def _make_candidate(
    request: GenerationRequest,
    *,
    can_reuse: bool = False,
) -> _PrefillCandidate:
    return _PrefillCandidate(
        request=request,
        classification=PrefillClassification(
            prompt_length=request.prompt_length,
            skip_positions=1 if can_reuse else 0,
            can_reuse=can_reuse,
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
    scheduler._preempted_request_ids = set()
    scheduler._decode_kv_recovery_request_id = None
    scheduler._select_prefill_batch = lambda capacity_remaining, **kwargs: [
        _make_candidate(request)
    ]
    return scheduler


def test_make_prefill_candidate_classifies_prompt_and_generated_prefix() -> None:
    request = _make_request(generated_prefix_tokens=[TextToken(10), TextToken(11)])
    runtime = FakeRuntime()
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    candidate = GenerationScheduler._make_prefill_candidate(scheduler, request)

    assert candidate is not None
    assert runtime.classify_calls == [[TextToken(1), TextToken(10), TextToken(11)]]
    assert candidate.reserve_length == request.target_length


def test_make_prefill_candidate_uses_initial_reserve_for_large_cap() -> None:
    request = _make_request(max_new_tokens=100)
    runtime = FakeRuntime()
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    candidate = GenerationScheduler._make_prefill_candidate(scheduler, request)

    assert candidate is not None
    assert (
        candidate.reserve_length
        == request.prompt_length + runtime.decode_reserve_tokens
    )
    assert candidate.reserve_length < request.target_length


def test_make_prefill_candidate_rejects_initial_reserve_below_prompt() -> None:
    request = _make_request()
    runtime = FakeRuntime()
    runtime.initial_reserve_length = lambda prompt, max_length: prompt - 1
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    with pytest.raises(AssertionError, match="at least the prompt"):
        GenerationScheduler._make_prefill_candidate(scheduler, request)


def test_make_prefill_candidate_rejects_initial_reserve_above_row_max() -> None:
    request = _make_request()
    runtime = FakeRuntime()
    runtime.initial_reserve_length = lambda prompt, max_length: max_length + 1
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    with pytest.raises(AssertionError, match="row max length"):
        GenerationScheduler._make_prefill_candidate(scheduler, request)


def test_launch_prefill_step_prefills_generated_prefix_then_remaining_tokens() -> None:
    request = _make_request(
        max_new_tokens=5,
        generated_prefix_tokens=[TextToken(10), TextToken(11)],
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


def test_launch_prefill_step_materializes_crops_when_image_cache_not_reused() -> None:
    request = _make_request()
    image = object()
    crops = object()
    request.image = image
    request.image_hash = b"image-hash"
    request.lifecycle.has_image = True
    request.lifecycle.crops_ready = True
    request.lifecycle.prefix_cache_hit = True
    request.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(prepare_exc=RuntimeError("prepare failed"))
    preprocess_calls: list[object] = []

    def preprocess(image_arg: object) -> Future[object]:
        preprocess_calls.append(image_arg)
        future: Future[object] = Future()
        future.set_result(crops)
        return future

    runtime.preprocess_image_async = preprocess  # type: ignore[method-assign]
    scheduler = _make_scheduler(request, runtime)
    scheduler._acquire_adapter_slot = lambda adapter_id: 0
    pipeline = PipelineState()

    GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert preprocess_calls == [image]
    assert request.image_crops is crops
    assert runtime.prepare_calls[0]["image_crops"] is crops


def test_launch_prefill_step_skips_crop_work_when_image_cache_reused() -> None:
    request = _make_request()
    image = object()
    request.image = image
    request.image_hash = b"image-hash"
    request.lifecycle.has_image = True
    request.lifecycle.crops_ready = True
    request.lifecycle.prefix_cache_hit = True
    request.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(prepare_exc=RuntimeError("prepare failed"))
    preprocess_calls: list[object] = []

    def preprocess(image_arg: object) -> Future[object]:
        preprocess_calls.append(image_arg)
        future: Future[object] = Future()
        future.set_result(object())
        return future

    runtime.preprocess_image_async = preprocess  # type: ignore[method-assign]
    scheduler = _make_scheduler(request, runtime)
    scheduler._select_prefill_batch = lambda capacity_remaining, **kwargs: [
        _make_candidate(request, can_reuse=True)
    ]
    scheduler._acquire_adapter_slot = lambda adapter_id: 0
    pipeline = PipelineState()

    GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert preprocess_calls == []
    assert request.image_crops is None
    assert runtime.prepare_calls[0]["image_crops"] is None


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


def test_launch_prefill_step_defers_when_lora_slots_exhausted() -> None:
    request = _make_request()
    request.adapter = "ft-1"
    runtime = FakeRuntime()
    scheduler = _make_scheduler(request, runtime)
    scheduler._acquire_adapter_slot = lambda adapter_id: (_ for _ in ()).throw(
        RuntimeError("Out of LoRA slots: all slots are in use.")
    )
    pipeline = PipelineState()

    progressed = GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert progressed is False
    assert list(scheduler.waiting) == [request]
    assert not scheduler._completed
    assert request.lifecycle.phase == RequestPhase.WAITING_RESOURCES
    assert request.lifecycle.lora_slot_ready is False
    assert request.lora_slot == 0
    assert len(runtime.prepare_calls) == 0
    assert len(runtime.released_prefill_slots) == 1


def test_launch_prefill_step_allows_base_request_past_exhausted_lora() -> None:
    adapter = _make_request(request_id=1)
    adapter.adapter = "ft-1"
    base = _make_request(request_id=2)
    base.lifecycle.lora_slot_ready = True
    runtime = FakeRuntime(
        max_batch_size=2,
        max_batch_slots=3,
        prepare_exc=RuntimeError("prepare failed"),
    )
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler.waiting = RequestQueue()
    scheduler.waiting.push(adapter)
    scheduler.waiting.push(base)
    scheduler.running = RunningQueue()
    scheduler._completed = deque()
    scheduler._preempted_request_ids = set()
    scheduler._decode_kv_recovery_pending = False
    scheduler._decode_kv_recovery_request_id = None
    scheduler._compute_stream = None
    scheduler._acquire_adapter_slot = lambda adapter_id: (_ for _ in ()).throw(
        RuntimeError("Out of LoRA slots: all slots are in use.")
    )
    pipeline = PipelineState()

    progressed = GenerationScheduler._launch_prefill_step(scheduler, pipeline)

    assert progressed is True
    assert list(scheduler.waiting) == [adapter]
    assert adapter.lifecycle.phase == RequestPhase.WAITING_RESOURCES
    assert runtime.prepare_calls[0]["prompt_tokens"] == base.prefill_tokens
    assert scheduler._completed[0].request_id == base.request_id
