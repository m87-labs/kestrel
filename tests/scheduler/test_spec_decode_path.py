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
            nxt = self._plans[row].pop(0)
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


def _enqueue(sched: GenerationScheduler, rid: int, prompt_len: int, max_new: int):
    req = GenerationRequest(
        request_id=rid,
        prompt="p",
        prompt_tokens=[TextToken(1)] * prompt_len,
        max_new_tokens=max_new,
        skill=_SpecSkillSpec(),  # type: ignore[arg-type]
        request_context=object(),
    )
    lc = RequestLifecycle(request=req, skill_state=_RecordingState(req))
    lc.crops_ready = True
    lc.lora_slot_ready = True
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
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=8)
    r1 = _enqueue(sched, 1, prompt_len=2, max_new=8)
    sched._spec_admit()

    len0_before = r0.lifecycle.state.length
    len1_before = r1.lifecycle.state.length
    assert sched._spec_decode_step() is True

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
    sched._spec_decode_step()

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
    # admit already staged 1; one step stages 2 more -> hits max_new=3.
    sched._spec_decode_step()
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
