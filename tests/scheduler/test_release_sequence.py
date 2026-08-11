from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Sequence

import pytest

from kestrel.runtime import SequenceState, TextToken, Token
from kestrel.scheduler.pipeline import (
    LaunchHandle,
    PendingCommit,
    PrefillLaunch,
    PrefillPendingCommit,
)
from kestrel.scheduler.queues import RunningQueue
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import GeneratedPrefix, GenerationRequest, RequestLifecycle

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass
class _SkillStateStub:
    tokens: list[object] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


class _FailingRetainRuntime(FakeRuntime):
    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None:
        raise RuntimeError("retain failed")


def _make_lifecycle(
    runtime: FakeRuntime,
    *,
    request_id: int = 7,
    encoder_input: object | None = None,
) -> RequestLifecycle:
    state = SequenceState(
        batch_idx=0,
        length=2,
        max_length=4,
        prompt_length=1,
    )
    runtime.active_sequences[state.batch_idx] = state

    request = GenerationRequest(
        request_id=request_id,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=object(),
        request_context=object(),
        image_hash=b"0123456789abcdef",
        adapter="adapter-a",
        encoder_input=encoder_input,
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=_SkillStateStub(tokens=[TextToken(10), TextToken(11)]),
        sequence_state=state,
    )
    request.lifecycle = lifecycle
    return lifecycle


def test_finalize_sequence_retains_prefix_before_release() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime)
    state = lifecycle.state

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


def test_release_sequence_retains_only_decoded_suffix_after_generated_prefix() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime)
    lifecycle.state.length = 4
    lifecycle.state.prompt_length = 2
    lifecycle.request.generated_prefix = GeneratedPrefix(tokens=(TextToken(10),))
    lifecycle.skill_state.tokens = [TextToken(10), TextToken(11), TextToken(12)]

    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    GenerationScheduler._release_sequence(scheduler, lifecycle)

    assert len(runtime.retained_prefixes) == 1
    retain_call = runtime.retained_prefixes[0]
    assert retain_call["generated_tokens"] == [TextToken(11), TextToken(12)]
    assert runtime.released_sequences == [lifecycle.state]


def test_release_sequence_does_not_cache_encoder_conditioned_prompt() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime, encoder_input=object())
    # The large prepared payload is released after prefill, while the semantic
    # marker must continue to disable decoder-prefix retention.
    lifecycle.request.encoder_input = None
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    GenerationScheduler._release_sequence(scheduler, lifecycle)

    assert runtime.retained_prefixes == []
    assert runtime.released_sequences == [lifecycle.state]


def test_release_sequence_releases_and_propagates_when_retention_fails() -> None:
    runtime = _FailingRetainRuntime()
    lifecycle = _make_lifecycle(runtime)

    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    with pytest.raises(RuntimeError, match="retain failed"):
        GenerationScheduler._release_sequence(scheduler, lifecycle)

    assert runtime.released_sequences == [lifecycle.state]
    assert lifecycle.state.batch_idx not in runtime.active_sequences


def test_failed_prepared_cleanup_releases_installed_and_aborts_uninstalled() -> None:
    runtime = FakeRuntime()
    installed = SequenceState(batch_idx=0, length=1, max_length=2)
    uninstalled = SequenceState(batch_idx=1, length=1, max_length=2)
    runtime.active_sequences[installed.batch_idx] = installed
    installed_prepared = SimpleNamespace(state=installed)
    uninstalled_prepared = SimpleNamespace(state=uninstalled)
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime

    scheduler._retire_failed_prepared_sequences(
        [installed_prepared, uninstalled_prepared]
    )

    assert runtime.released_sequences == [installed]
    assert runtime.aborted_prepared == [uninstalled_prepared]
    assert runtime.active_sequences == {}


def test_failed_prepared_cleanup_returns_adapter_owned_before_install() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime)
    runtime.active_sequences.clear()
    lifecycle.request.lora_slot = 7
    lifecycle.lora_slot_ready = True
    prepared = SimpleNamespace(state=lifecycle.state)
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler.running = RunningQueue()
    scheduler.running.push(lifecycle)

    scheduler._retire_failed_prepared_sequences(
        [prepared],
        sequences=[lifecycle],
        abort_adapter_slots=[7],
    )

    assert runtime.aborted_prepared == [prepared]
    assert runtime.released_adapter_slots == [7]
    assert len(scheduler.running) == 0
    assert lifecycle.request.lora_slot == 0
    assert lifecycle.lora_slot_ready is False


def test_prefill_finalize_failure_retires_adapter_and_slot() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime)
    runtime.active_sequences.clear()
    lifecycle.request.lora_slot = 9
    lifecycle.lora_slot_ready = True
    prepared = SimpleNamespace(state=lifecycle.state)
    staging = object()
    released_staging: list[object] = []
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._compute_stream = None
    scheduler.running = RunningQueue()
    scheduler._release_prefill_staging = released_staging.append
    scheduler._finalize_prefill = lambda _handle: (_ for _ in ()).throw(
        RuntimeError("sampling failed")
    )
    slot = runtime.prefill_slots[0]
    handle = LaunchHandle(
        kind="prefill",
        sequences=[lifecycle],
        payload=PrefillLaunch(
            staging=staging,
            slot_id=0,
            logits=object(),
            prepared_sequences=[prepared],
            prefill_slot=slot,
            abort_adapter_slots=(9,),
        ),
    )

    with pytest.raises(RuntimeError, match="sampling failed"):
        scheduler.finalize_sampling(handle)

    assert runtime.aborted_prepared == [prepared]
    assert runtime.released_adapter_slots == [9]
    assert released_staging == [staging]
    assert runtime.released_prefill_slots == [slot]
    assert lifecycle.request.lora_slot == 0
    assert lifecycle.lora_slot_ready is False


def test_prefill_commit_failure_fences_before_returning_adapter() -> None:
    runtime = FakeRuntime()
    lifecycle = _make_lifecycle(runtime)
    runtime.active_sequences.clear()
    lifecycle.request.lora_slot = 12
    lifecycle.lora_slot_ready = True
    prepared = SimpleNamespace(state=lifecycle.state)
    staging = object()
    cleanup_order: list[str] = []

    class _ComputeStream:
        def synchronize(self) -> None:
            cleanup_order.append("fence")

    def abort_prepared(value: object) -> None:
        cleanup_order.append("abort")
        runtime.aborted_prepared.append(value)

    runtime.abort_prepared_sequence = abort_prepared  # type: ignore[method-assign]
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._compute_stream = _ComputeStream()
    scheduler.running = RunningQueue()
    scheduler.running.push(lifecycle)
    scheduler._release_prefill_staging = lambda _staging: None
    slot = runtime.prefill_slots[0]
    transfer = SimpleNamespace(
        wait=lambda: (_ for _ in ()).throw(RuntimeError("transfer failed"))
    )
    step = PendingCommit(
        kind="prefill",
        sequences=[lifecycle],
        transfer=transfer,
        payload=PrefillPendingCommit(
            staging=staging,
            slot_id=0,
            prepared_sequences=[prepared],
            prefill_slot=slot,
            abort_adapter_slots=(12,),
        ),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        scheduler._commit_prefill(step)

    assert cleanup_order == ["fence", "abort"]
    assert runtime.released_adapter_slots == [12]
    assert len(scheduler.running) == 0
    assert runtime.released_prefill_slots == [slot]


def test_prefill_commit_drops_prepared_encoder_input() -> None:
    runtime = FakeRuntime()
    encoder_input = object()
    lifecycle = _make_lifecycle(runtime, encoder_input=encoder_input)
    prepared = SimpleNamespace(state=lifecycle.state)
    staging = object()
    released_staging: list[object] = []
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._release_prefill_staging = released_staging.append
    scheduler._materialize_tokens = lambda *_args, **_kwargs: [TextToken(9)]
    slot = runtime.prefill_slots[0]
    transfer = SimpleNamespace(wait=lambda: (object(), None))
    step = PendingCommit(
        kind="prefill",
        sequences=[lifecycle],
        transfer=transfer,
        payload=PrefillPendingCommit(
            staging=staging,
            slot_id=0,
            prepared_sequences=[prepared],
            prefill_slot=slot,
        ),
    )

    tokens, logprobs = scheduler._commit_prefill(step)

    assert tokens == [TextToken(9)]
    assert logprobs is None
    assert lifecycle.request.encoder_input is None
    assert lifecycle.request.has_encoder_input is True
    assert runtime.finalized_prepared == [prepared]
    assert released_staging == [staging]
    assert runtime.released_prefill_slots == [slot]
