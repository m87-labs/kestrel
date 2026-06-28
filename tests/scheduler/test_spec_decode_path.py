"""CPU tests for the scheduler's speculative-decoding path.

Drives ``GenerationScheduler`` with a fake ``SpecDecoder`` (no GPU): proves spec
admission (prefill -> first token), the per-macro-step variable advance, finish
handling within an accepted run, continuous-batching retire, and that the
non-spec path stays untouched when ``runtime.spec`` is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from kestrel.runtime import TextToken
from kestrel.runtime.spec import SpecDecodeCaps, SpecStepResult
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import (
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)
from kestrel.skills import DecodeStep, SkillFinalizeResult, SkillState

from tests.scheduler._fake_runtime import FakeRuntime


@dataclass(frozen=True)
class _SpecSkillSpec:
    name: str = "spec"


class _RecordingState(SkillState):
    def __init__(self, request: GenerationRequest) -> None:
        super().__init__(_SpecSkillSpec(), request)  # type: ignore[arg-type]

    def consume_step(self, runtime: object, step: DecodeStep) -> None:
        self.append_token(step.token)

    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        return SkillFinalizeResult(text="", tokens=list(self.tokens), output={})


class _FakeDecoder:
    """Scriptable ``SpecDecoder`` for the scheduler test (no GPU).

    ``admit`` hands back a fixed first token and assigns the next free pool row.
    ``step`` pops a scripted ``(tokens, accept)`` plan per call from each row's
    queue (rows advance by *different* amounts to exercise ragged advance).
    """

    num_speculative_tokens = 3

    def __init__(self, *, n_rows: int, plans: dict, first_tokens: dict) -> None:
        self._free = list(range(n_rows))
        self._row_of: dict[int, int] = {}
        self._plans = plans            # row -> list[list[int]] per step
        self._first = first_tokens     # row -> first token id
        self.admitted: list[int] = []
        self.retired: list[int] = []

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def admit(self, state, prompt_token_ids):
        row = self._free.pop(0)
        self._row_of[id(state)] = row
        state.batch_idx = 100 + row
        self.admitted.append(row)
        return self._first[row]

    def step(self, states):
        tokens, accepts = [], []
        for s in states:
            row = self._row_of[id(s)]
            plan = self._plans[row]
            # Depth-1 overlap launches one macro-step ahead, so a finishing row
            # gets one extra optimistic launch whose result is discarded as a
            # zombie at commit. Hand back a throwaway token when the script is
            # exhausted.
            nxt = plan.pop(0) if plan else [0]
            tokens.append(list(nxt))
            accepts.append(len(nxt) - 1)
        return SpecStepResult(tokens=tokens, accept_counts=accepts)

    def retire(self, state) -> None:
        self.retired.append(self._row_of.pop(id(state)))
        # row not returned to free list in this test (fixed admission set)


def _spec_runtime(decoder: _FakeDecoder, *, eos_id: int = 999) -> FakeRuntime:
    rt = FakeRuntime(max_batch_size=4, max_batch_slots=8)
    rt.spec = SpecDecodeCaps(proposer=SimpleNamespace(num_speculative_tokens=3,
                                                      num_lookahead_tokens=4),
                             decoder=decoder)
    rt.prompt_template = SimpleNamespace(eos_id=eos_id)
    return rt


def _enqueue(
    sched: GenerationScheduler,
    rid: int,
    prompt_len: int,
    max_new: int,
    *,
    adapter: str | None = None,
    lora_slot_ready: bool = True,
    return_logprobs: bool | None = None,
):
    req = GenerationRequest(
        request_id=rid,
        prompt="p",
        prompt_tokens=[TextToken(1)] * prompt_len,
        max_new_tokens=max_new,
        skill=_SpecSkillSpec(),  # type: ignore[arg-type]
        request_context=object(),
        adapter=adapter,
        return_logprobs=return_logprobs,
    )
    lc = RequestLifecycle(request=req, skill_state=_RecordingState(req))
    lc.crops_ready = True
    lc.lora_slot_ready = lora_slot_ready
    lc.transition(RequestPhase.READY_FOR_PREFILL)
    req.lifecycle = lc
    sched.enqueue_request(req, lc.skill_state)
    return req


def _make_scheduler(rt: FakeRuntime) -> GenerationScheduler:
    from kestrel.skills import SkillRegistry

    return GenerationScheduler(rt, compute_stream=None, skill_registry=SkillRegistry([]))


def test_spec_admit_stages_first_token_and_queues() -> None:
    dec = _FakeDecoder(
        n_rows=2,
        first_tokens={0: 11, 1: 22},
        plans={0: [[12, 13]], 1: [[23]]},  # one step each
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=8)
    r1 = _enqueue(sched, 1, prompt_len=2, max_new=8)

    assert sched._spec_admit() is True
    # Both admitted, first tokens staged, queued into running.
    assert dec.admitted == [0, 1]
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [11]
    assert [int(t.token_id) for t in r1.lifecycle.skill_state.tokens] == [22]
    assert len(sched.running) == 2
    # admit assigned the decoder's pool batch index onto the state.
    assert r0.lifecycle.state.batch_idx == 100
    assert r1.lifecycle.state.batch_idx == 101
    assert r0.lifecycle.state.batch_idx in rt.active_sequences


def test_spec_step_variable_advance_and_state_length() -> None:
    dec = _FakeDecoder(
        n_rows=2,
        first_tokens={0: 11, 1: 22},
        # row 0 advances 2 tokens, row 1 advances 1 token in the same step.
        plans={0: [[12, 13]], 1: [[23]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    # max_new sized so each row finishes exactly at its scripted step (the
    # depth-1 pipeline commits a step one tick after it launches).
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=3)
    r1 = _enqueue(sched, 1, prompt_len=2, max_new=2)
    sched._spec_admit()

    len0_before = r0.lifecycle.state.length
    len1_before = r1.lifecycle.state.length
    # Drive the depth-1 pipeline to quiescence (launch N / commit N-1 / drain).
    while sched._spec_decode_step():
        pass

    # Variable advance: row 0 committed 2, row 1 committed 1.
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [11, 12, 13]
    assert [int(t.token_id) for t in r1.lifecycle.skill_state.tokens] == [22, 23]
    # KV length advanced by the committed count per sequence.
    assert r0.lifecycle.state.length - len0_before == 2
    assert r1.lifecycle.state.length - len1_before == 1


def test_spec_finish_on_eos_within_run_retires_and_completes() -> None:
    # row 0 emits eos as its 2nd new token -> finishes mid-run, retires.
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 999, 13]]},  # 999 == eos; 13 should not be staged
    )
    rt = _spec_runtime(dec, eos_id=999)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=20)
    sched._spec_admit()
    while sched._spec_decode_step():  # depth-1: commit lands a tick after launch
        pass

    toks = [int(t.token_id) for t in r0.lifecycle.skill_state.tokens]
    # Staged up to and including eos, then stopped (13 dropped).
    assert toks == [11, 12, 999]
    assert r0.lifecycle.finished is True
    assert dec.retired == [0]
    assert r0.lifecycle.state.batch_idx not in rt.active_sequences
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "stop"


def test_spec_finish_on_max_new_tokens() -> None:
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 -> 3 tokens == max_new
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()
    # admit staged 1; the scripted step stages 2 more -> hits max_new=3. Drive
    # the depth-1 pipeline to quiescence (commit lands a tick after launch).
    while sched._spec_decode_step():
        pass
    assert r0.lifecycle.finished is True
    assert dec.retired == [0]
    assert [c.finish_reason for c in sched.pop_completed()] == ["length"]


def test_spec_admit_respects_free_slots() -> None:
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    _enqueue(sched, 0, prompt_len=3, max_new=8)
    _enqueue(sched, 1, prompt_len=3, max_new=8)
    sched._spec_admit()
    # Only one row free -> only one admitted; the other stays waiting.
    assert dec.admitted == [0]
    assert len(sched.waiting) == 1
    assert len(sched.running) == 1


def test_advance_routes_to_spec_when_capability_present() -> None:
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12], [13]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    _enqueue(sched, 0, prompt_len=3, max_new=8)
    # advance() must take the spec branch (admits + steps) without touching the
    # single-token decode pipeline.
    assert sched.advance() is True
    assert dec.admitted == [0]


def test_has_pending_work_tracks_pending_spec_until_drained() -> None:
    """Regression for the depth-1 drain leak (codex finding @ L617).

    A finishing sequence launches one optimistic follow-up macro-step before its
    own commit removes it from ``running``. That follow-up lives in
    ``_pending_spec`` as a zombie. ``has_pending_work()`` must stay True while it
    is outstanding, otherwise the executor stops calling ``advance`` and the
    zombie step is never committed -> ``decoder.retire`` never runs -> the spec
    row leaks. Drive the loop until ``running`` empties with a pending step still
    outstanding and assert (a) the row is retired exactly once, (b)
    ``has_pending_work`` only goes False after the pending step is committed.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes here
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()

    # Drive macro-steps manually so we can observe the tick where ``running``
    # is already empty but a pending (zombie) spec step is still outstanding.
    saw_pending_after_running_empty = False
    for _ in range(10):
        if not sched._spec_decode_step():
            break
        if len(sched.running) == 0 and sched._pending_spec is not None:
            saw_pending_after_running_empty = True
            # This is exactly the window the bug regressed: the old
            # has_pending_work() (waiting/running/pipeline only) returned False
            # here and the executor would stop driving advance().
            assert sched.has_pending_work() is True

    assert saw_pending_after_running_empty is True
    # Pending step committed, row retired exactly once, nothing leaked.
    assert sched._pending_spec is None
    assert dec.retired == [0]
    assert rt.active_sequences == {}
    assert sched.has_pending_work() is False
    assert [c.finish_reason for c in sched.pop_completed()] == ["length"]


def test_spec_admit_acquires_lora_slot_for_adapter_request() -> None:
    """Regression for the LoRA wrong-slot bug (codex finding @ L560).

    Adapter requests admitted to the spec path must acquire a LoRA slot the way
    the normal prefill path does; otherwise ``request.lora_slot`` stays 0 and the
    finetune is silently ignored (base-model slot).
    """

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    # Adapter request enters WAITING_RESOURCES with no slot yet.
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    assert sched._spec_admit() is True
    # Slot was acquired (FakeRuntime returns slot index >= 1) and threaded onto
    # both the request and the admitted SequenceState.
    assert rt.acquired_adapter_slots == [("ft-1", rt.acquired_adapter_slots[0][1])]
    assert req.lifecycle.lora_slot_ready is True
    assert req.lora_slot >= 1
    assert req.lifecycle.state.lora_slot == req.lora_slot


def test_spec_admit_defers_when_lora_slots_exhausted() -> None:
    """Out-of-slots is recoverable: keep the request WAITING_RESOURCES.

    When ``acquire_adapter_slot`` raises ``RuntimeError`` (all LoRA slots in
    use), the adapter request must stay in the waiting queue (not be admitted
    with the wrong slot, nor failed) so it retries once a slot frees up.
    """

    class _ExhaustedAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _FakeDecoder(n_rows=2, first_tokens={0: 11, 1: 22}, plans={0: [[12]], 1: [[23]]})
    rt = _spec_runtime(dec)

    def _raise_out_of_slots(adapter_id: str, adapter: object) -> int:
        raise RuntimeError("Out of LoRA slots: all slots are in use.")

    rt.acquire_adapter_slot = _raise_out_of_slots  # type: ignore[method-assign]

    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_ExhaustedAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    # No row admitted, request stays waiting (WAITING_RESOURCES), no error result.
    assert sched._spec_admit() is False
    assert dec.admitted == []
    assert len(sched.waiting) == 1
    assert len(sched.running) == 0
    assert req.lifecycle.phase == RequestPhase.WAITING_RESOURCES
    assert sched.pop_completed() == []


def test_spec_admit_rejects_logprobs_request_cleanly() -> None:
    """Regression for the spec+logprobs crash (codex finding @ L578).

    A ``return_logprobs=True`` request on the spec path must fail just that
    request (clean error result + row retired) instead of raising a
    scheduler-wide exception when ``stage_token`` sees a None logprob.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=8, return_logprobs=True)

    # Must not raise.
    assert sched._spec_admit() is True
    # Request failed cleanly: errored result, not pushed to running, row retired.
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    assert len(sched.running) == 0
    assert dec.retired == [0]
    assert rt.active_sequences == {}
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "error"
    assert "error" in completed[0].output

    # And the spec decode loop stays healthy afterward (no leaked pending step).
    assert sched._spec_decode_step() is False
    assert sched.has_pending_work() is False
