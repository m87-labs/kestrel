from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kestrel.runtime import SequenceState, TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass
class _SkillStateStub:
    tokens: list[object] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def test_finalize_sequence_retains_prefix_before_release() -> None:
    runtime = FakeRuntime()
    state = SequenceState(
        batch_idx=0,
        length=2,
        max_length=4,
        prompt_length=1,
    )
    runtime.active_sequences[state.batch_idx] = state

    request = GenerationRequest(
        request_id=7,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=object(),
        request_context=object(),
        image_hash=b"0123456789abcdef",
        adapter="adapter-a",
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(tokens=[TextToken(10), TextToken(11)]),
        sequence_state=state,
    )
    request.lifecycle = lifecycle

    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._completed = deque()
    scheduler._build_result = lambda seq: object()

    GenerationScheduler._finalize_sequence(scheduler, lifecycle, "stop")

    assert len(runtime.retained_prefixes) == 1
    retain_call = runtime.retained_prefixes[0]
    assert retain_call["state"] is state
    assert retain_call["generated_tokens"] == [TextToken(10), TextToken(11)]
    assert retain_call["adapter_id"] == "adapter-a"
    assert retain_call["image_hash"] == b"0123456789abcdef"
    assert runtime.released_sequences == [state]
