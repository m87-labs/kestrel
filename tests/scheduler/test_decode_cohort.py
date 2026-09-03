from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from kestrel.runtime import SequenceState, TextToken
from kestrel.runtime.decode_slot import DecodeMetaBuffers
from kestrel.runtime.sampling import SamplingHooks
from kestrel.scheduler.queues import RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle, StepPlan

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass
class _SkillStateStub:
    tokens: list[object] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def _make_lifecycle(request_id: int) -> RequestLifecycle:
    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=8,
        skill=object(),
        request_context=object(),
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(),
        sequence_state=SequenceState(
            batch_idx=request_id,
            length=1,
            max_length=9,
            prompt_length=1,
        ),
        packed_pending_ready=True,
    )
    request.lifecycle = lifecycle
    return lifecycle


def test_decode_cohort_stays_stable_until_a_member_retires() -> None:
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = FakeRuntime(max_batch_size=2, max_batch_slots=4)
    scheduler.running = RunningQueue()
    sequences = [_make_lifecycle(request_id) for request_id in (1, 2, 3)]
    scheduler.running.extend(sequences)

    assert scheduler.schedule_decode_step().sequences == sequences[:2]
    assert scheduler.schedule_decode_step().sequences == sequences[:2]

    scheduler.running.remove(sequences[0])
    assert scheduler.schedule_decode_step().sequences == sequences[1:]


def test_resident_tail_does_not_replace_temporarily_blocked_cohort_member() -> None:
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = FakeRuntime(max_batch_size=2, max_batch_slots=4)
    scheduler.running = RunningQueue()
    sequences = [_make_lifecycle(request_id) for request_id in (1, 2, 3)]
    scheduler.running.extend(sequences)
    sequences[0].inflight_refs = 2

    plan = scheduler.schedule_decode_step()

    assert plan.sequences == [sequences[1]]


def test_decode_launch_retains_maximum_staged_position() -> None:
    scheduler = object.__new__(GenerationScheduler)
    runtime = FakeRuntime(max_batch_size=2, max_batch_slots=4)
    slot = SimpleNamespace(
        slot_id=0,
        meta=DecodeMetaBuffers(
            max_batch_slots=4,
            device=torch.device("cpu"),
            pin_memory=False,
        ),
        decode_token_ids=torch.empty(4, dtype=torch.long),
    )
    runtime.decode_slots[0] = slot
    scheduler.runtime = runtime
    scheduler._pending_token_ids = torch.arange(4, dtype=torch.long)
    scheduler._hooks = SamplingHooks()
    sequences = [_make_lifecycle(request_id) for request_id in (1, 2)]
    sequences[0].state.length = 7
    sequences[1].state.length = 3

    scheduler._launch_forward_on_stream(StepPlan(sequences), slot_id=0)

    assert slot.meta.input_pos.cpu[:2].tolist() == [7, 3]
    assert slot.meta.max_input_pos == 7
