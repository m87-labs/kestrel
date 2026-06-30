"""CPU tests for the scheduler's speculative-decoding path.

Drives ``GenerationScheduler`` with a fake ``SpecDecoder`` (no GPU): proves spec
admission (prefill -> first token), the per-macro-step variable advance, finish
handling within an accepted run, continuous-batching retire, and that the
non-spec path stays untouched when ``runtime.spec`` is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kestrel.runtime import TextToken
from kestrel.runtime.tokens import CoordToken, ImageMarker, SizeToken
from kestrel.runtime.spec import SpecDecodeCaps, SpecSideValues, SpecStepResult
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

    ``admit`` hands back a fixed first token and assigns the next free pool row,
    recording the image / skill-mask / sampling kwargs the scheduler forwards so
    tests can assert the spec path passes a request's full context through (no
    text-only/unconstrained fallback). ``step`` pops a scripted ``tokens`` plan
    per call from each row's queue (rows advance by *different* amounts to
    exercise ragged advance) and, when ``emit_logprobs`` is set, returns a
    parallel per-committed-token logprob list.
    """

    num_speculative_tokens = 3

    def __init__(
        self,
        *,
        n_rows: int,
        plans: dict,
        first_tokens: dict,
        emit_logprobs: bool = False,
        side_values: object | None = None,
        first_logprobs: dict | None = None,
        admit_side_values: object | None = None,
    ) -> None:
        self._free = list(range(n_rows))
        self._row_of: dict[int, int] = {}
        self._plans = plans            # row -> list[list[int]] per step
        self._first = first_tokens     # row -> first token id
        self._emit_logprobs = emit_logprobs
        self._side_values = side_values
        # row -> first-token logprob ``admit`` returns when the request wants
        # logprobs (defaults to the non-spec greedy-bonus convention 0.0).
        self._first_logprobs = first_logprobs or {}
        # Optional per-admit side-values the decoder attaches to the state so
        # the scheduler types the admit token via the runtime hook.
        self._admit_side_values = admit_side_values
        self.admitted: list[int] = []
        self.retired: list[int] = []
        # admit kwargs recorded per row.
        self.admit_kwargs: dict[int, dict] = {}
        # Per-``step`` skill masks the scheduler forwards (one entry per call,
        # each parallel to that call's ``states``). Lets a test assert the
        # stateful skill mask is RECOMPUTED per macro-step rather than reusing
        # the stale admit-time snapshot.

        self.step_allowed: list = []
        self.step_suppressed: list = []
        # Per-``step`` commit caps the scheduler forwards (one entry per call,
        # each parallel to that call's ``states``). Lets a test assert the
        # scheduler caps a STATEFUL-masked row to a single committed token.
        self.step_commit_caps: list = []
        # Per-``step`` row ids the scheduler launches (one entry per call, the
        # decoder rows backing that call's ``states``). Lets a test assert a row
        # finalized by the prior commit is NOT launched into another macro-step.
        self.step_rows: list = []

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def admit(
        self,
        state,
        prompt_token_ids,
        *,
        image=None,
        image_crops=None,
        allowed_token_ids=None,
        suppressed_token_ids=None,
        suppress_next_token_ids=None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        row = self._free.pop(0)
        self._row_of[id(state)] = row
        state.batch_idx = 100 + row
        if self._admit_side_values is not None:
            state.admit_side_values = self._admit_side_values
        self.admitted.append(row)
        want_logprobs = getattr(state, "return_logprobs", None) is True
        self.admit_kwargs[row] = {
            # The *typed* prompt tokens the scheduler forwards (the non-spec
            # ``prepare_sequence`` contract): asserted to survive admission whole,
            # not be stripped to ``int(t.token_id)`` (which would raise on
            # ImageMarker/Coord/Size tokens that lack ``token_id``).
            "prompt_tokens": list(prompt_token_ids),
            "image": image,
            "image_crops": image_crops,
            "allowed_token_ids": allowed_token_ids,
            "suppressed_token_ids": suppressed_token_ids,
            "suppress_next_token_ids": suppress_next_token_ids,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": getattr(state, "return_logprobs", None),
        }
        # New contract: ``admit`` returns ``(first_token_id, first_logprob)``.
        # ``first_logprob`` is the real selected-token logprob when the request
        # wants logprobs, else ``None``.
        first_logprob = (
            self._first_logprobs.get(row, 0.0) if want_logprobs else None
        )
        return self._first[row], first_logprob


    def step(
        self,
        states,
        *,
        allowed_token_ids=None,
        suppressed_token_ids=None,
        commit_caps=None,
    ):
        # Record the per-step masks the scheduler recomputes + forwards.
        self.step_allowed.append(
            list(allowed_token_ids) if allowed_token_ids is not None else None
        )
        self.step_suppressed.append(
            list(suppressed_token_ids)
            if suppressed_token_ids is not None
            else None
        )
        self.step_commit_caps.append(
            list(commit_caps) if commit_caps is not None else None
        )
        self.step_rows.append([self._row_of[id(s)] for s in states])
        tokens, accepts, logprobs = [], [], []
        for idx, s in enumerate(states):
            row = self._row_of[id(s)]
            plan = self._plans[row]
            # Depth-1 overlap launches one macro-step ahead, so a finishing row
            # gets one extra optimistic launch whose result is discarded as a
            # zombie at commit. Hand back a throwaway token when the script is
            # exhausted.
            nxt = plan.pop(0) if plan else [0]
            # Honor the scheduler's per-row commit cap exactly as the real
            # decoder must: truncate the accepted run to ``commit_caps[idx]``
            # tokens (a stateful-masked row is capped to 1) and push the dropped
            # tail back onto the script so it is committed on later steps -- the
            # KV/pool advance stays consistent with the (truncated) committed run.
            cap = None if commit_caps is None else commit_caps[idx]
            if cap is not None and len(nxt) > cap:
                plan.insert(0, nxt[cap:])
                nxt = nxt[:cap]
            tokens.append(list(nxt))
            accepts.append(len(nxt) - 1)
            # Scripted logprob per committed token: -(token_id) keeps it
            # deterministic and easy to assert.
            logprobs.append([-float(t) for t in nxt])
        return SpecStepResult(
            tokens=tokens,
            accept_counts=accepts,
            logprobs=logprobs if self._emit_logprobs else None,
            side_values=self._side_values,
        )

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
    temperature: float = 0.0,
    top_p: float = 1.0,
    image: object | None = None,
    image_crops: object | None = None,
    image_length: int = 0,
    skill_state: SkillState | None = None,
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
        temperature=temperature,
        top_p=top_p,
        image=image,
        image_crops=image_crops,
        image_length=image_length,
    )
    lc = RequestLifecycle(
        request=req, skill_state=skill_state or _RecordingState(req)
    )
    lc.has_image = image is not None or image_crops is not None
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


def test_spec_admit_caps_running_batch_to_max_batch_size() -> None:
    """Regression: cap the live spec batch to ``runtime.max_batch_size``.

    The decoder's pool can expose MORE free rows than ``max_batch_size`` -- it
    is sized from ``max_batch_slots == max_batch_size + 2`` for transient-prefill
    headroom (here ``_spec_runtime`` uses ``max_batch_size=4`` /
    ``max_batch_slots=8``, and the decoder pool has 8 rows). If ``_spec_admit``
    admitted ``while decoder.free_slots > 0`` with no batch bound, it would queue
    all 8 running rows, and ``_spec_decode_step`` would then snapshot every one
    into a single ``decoder.step`` call -- overrunning the verify/draft graphs
    and staging buffers captured for ``max_batch_size`` states (the non-spec path
    caps the same way via ``_cap_decode_dispatch``).

    Enqueue 6 launchable requests against an 8-row pool and assert:

    * admission stops at ``max_batch_size`` (4) even though 8 rows are free and 6
      requests are launchable -- the surplus stays WAITING; and
    * across the whole depth-1 run, NO ``decoder.step`` call is ever handed more
      than ``max_batch_size`` states (``_FakeDecoder.step_rows`` records each
      call's launched rows).
    """
    n = 8  # pool rows -- intentionally > max_batch_size (4)
    dec = _FakeDecoder(
        n_rows=n,
        first_tokens={r: 10 + r for r in range(n)},
        # One token per step for a few steps so ``decoder.step`` actually fires
        # with the full admitted batch before any row finishes.
        plans={r: [[100 + r], [200 + r], [300 + r]] for r in range(n)},
    )
    rt = _spec_runtime(dec)
    assert rt.max_batch_size == 4 and dec.free_slots == 8
    sched = _make_scheduler(rt)
    for rid in range(6):  # more launchable requests than the batch cap
        _enqueue(sched, rid, prompt_len=3, max_new=8)

    sched._spec_admit()
    # Capped at max_batch_size: only 4 rows admitted, the other 2 stay waiting,
    # and free pool rows remain (admission was bound by the batch, not the pool).
    assert len(dec.admitted) == rt.max_batch_size
    assert dec.admitted == [0, 1, 2, 3]
    assert len(sched.running) == rt.max_batch_size
    assert len(sched.waiting) == 2
    assert dec.free_slots == n - rt.max_batch_size  # pool not over-subscribed

    # Drive the depth-1 pipeline; every ``decoder.step`` must stay within the
    # captured batch size for the whole run.
    for _ in range(8):
        if not sched._spec_decode_step():
            break
    assert dec.step_rows  # at least one macro-step actually launched
    assert all(len(rows) <= rt.max_batch_size for rows in dec.step_rows)


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

    A launched-but-uncommitted macro-step lives in ``_pending_spec`` (depth-1
    overlap launches step N+1 while committing step N). ``has_pending_work()``
    must stay True while such a step is outstanding, otherwise the executor stops
    calling ``advance`` and the pending step is never committed -> the spec row
    leaks. The ``_pending_spec is not None`` term in ``has_pending_work`` is the
    guard for exactly that.

    NOTE on the L997 fix: a row finalized by the commit (it hit ``max_new_tokens``)
    is now dropped from the next launch and retired SYNCHRONOUSLY -- its reserved
    ref is decremented and the row retired inside the same ``_spec_decode_step``
    -- rather than launched into one more optimistic ``decoder.step`` and retired
    a tick later. So a *finishing* row no longer produces a ``running``-empty /
    ``_pending_spec``-set window (that window only existed because the finished
    row was launched into a throwaway step -- the exact unsafe relaunch L997
    fixes). This drives a CONTINUING (multi-step) sequence -- whose follow-up step
    is genuinely in flight across ticks -- to assert the live invariant
    (``_pending_spec`` set => ``has_pending_work`` True), then lets it finish and
    asserts the finishing row retires exactly once, ends ``COMPLETED``, and leaves
    ``has_pending_work`` False, with no leak.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        # A continuing run: one token per step across several steps, then EOS, so a
        # real follow-up step sits in ``_pending_spec`` while the row keeps running.
        plans={0: [[12], [13], [14], [999]]},  # 999 == eos -> finishes
    )
    rt = _spec_runtime(dec, eos_id=999)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=20)
    sched._spec_admit()

    # While the row is running with a launched-but-uncommitted step in flight,
    # ``has_pending_work`` must report True via the ``_pending_spec`` term.
    saw_pending_while_running = False
    for _ in range(20):
        if not sched._spec_decode_step():
            break
        if sched._pending_spec is not None:
            # The live L617 invariant: an outstanding pending step keeps work
            # pending so the executor keeps driving ``advance`` until it commits.
            assert sched.has_pending_work() is True
            if len(sched.running) > 0:
                saw_pending_while_running = True
        # L997: a finishing row retires synchronously, so a ``running``-empty tick
        # never leaves an optimistic follow-up step dangling in ``_pending_spec``.
        if len(sched.running) == 0:
            assert sched._pending_spec is None

    # The pending-while-running window (the one the L617 guard protects) occurred.
    assert saw_pending_while_running is True
    # Drained: pending committed, row retired exactly once, COMPLETED, no leak.
    assert sched._pending_spec is None
    assert dec.retired == [0]
    assert dec.retired.count(0) == 1
    assert r0.lifecycle.phase == RequestPhase.COMPLETED
    assert rt.active_sequences == {}
    assert sched.has_pending_work() is False
    assert [c.finish_reason for c in sched.pop_completed()] == ["stop"]


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


def test_spec_admit_and_step_stage_logprobs() -> None:
    """``return_logprobs`` is served on the spec path (no clean-reject fallback).

    The spec macro-step now supplies per-committed-token logprobs
    (``SpecStepResult.logprobs``), so a ``return_logprobs=True`` request is
    admitted and its logprobs are staged through ``stage_token`` like the
    non-spec single-token path -- not rejected. ``admit`` also learns the request
    wants logprobs (via ``state.return_logprobs``) so ``step`` knows to gather
    them.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes here
        emit_logprobs=True,
        # admit returns the REAL first-token logprob (not the 0.0 greedy
        # placeholder) for a return_logprobs request.
        first_logprobs={0: -0.5},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=3, return_logprobs=True)

    # Admission succeeds (not rejected) and tells the decoder logprobs are wanted.
    assert sched._spec_admit() is True
    assert dec.admitted == [0]
    assert dec.admit_kwargs[0]["return_logprobs"] is True
    assert req.lifecycle.finished is False
    assert len(sched.running) == 1

    # Drive the depth-1 pipeline to quiescence; the scripted step commits
    # tokens 12, 13 with logprobs -12.0, -13.0.
    while sched._spec_decode_step():
        pass

    assert [int(t.token_id) for t in req.lifecycle.skill_state.tokens] == [11, 12, 13]
    # First (bonus) token carries the REAL admit logprob (-0.5), and the
    # macro-step supplies the real logprobs for the committed tokens.
    assert req.lifecycle.logprobs == [-0.5, -12.0, -13.0]
    completed = sched.pop_completed()
    assert completed[0].logprobs == [-0.5, -12.0, -13.0]


def test_spec_admit_returns_real_first_token_logprob() -> None:
    """Regression for codex finding: real admit logprob, not a 0.0 placeholder.

    For a non-greedy ``return_logprobs`` request the first (bonus) token sampled
    by ``decoder.admit`` must be staged with the sampler's selected-token logprob
    (the value ``admit`` now returns), not the 0.0 greedy approximation. Asserts
    the staged first logprob is exactly the real value the decoder returned.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12]]},
        emit_logprobs=True,
        first_logprobs={0: -1.25},  # real selected-token logprob from admit
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    # temperature > 0: the case where 0.0 would be wrong (non-greedy default).
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, return_logprobs=True, temperature=0.7
    )

    assert sched._spec_admit() is True
    # The first staged logprob is the real value, not the 0.0 placeholder.
    assert req.lifecycle.logprobs == [-1.25]


def test_spec_admit_no_logprob_when_not_requested() -> None:
    """``admit`` returns ``None`` logprob (and none is staged) without a request."""
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=8)  # return_logprobs unset

    assert sched._spec_admit() is True
    # No logprobs requested -> none collected (staging would raise if a stray
    # 0.0 were threaded for a non-logprobs request).
    assert req.lifecycle.logprobs == []


def test_spec_spatial_logprob_folds_head_logprob_into_staged() -> None:
    """P2 regression: the scheduler stages the spatial-head-augmented logprob.

    When a macro-step emits ``SpecSideValues`` (spatial runtime) and the request
    wants logprobs, the scheduler must hand the spec decoder's *vocab* logprobs to
    ``materialize_spec_tokens`` and stage the value the hook returns -- which a
    spatial runtime augments with the coord/size head logprob in place. This is
    the wiring the P2 fixed: previously the hook got ``token_logprobs=None`` and
    the scheduler staged only the vocab logprob (under-reporting spatial tokens,
    including the admit token, which routes through the same hook).

    A fake spatial hook here adds a fixed ``-0.5`` head delta to every position
    (standing in for the real coord/size head logprob) and records the vocab
    logprobs it received. We assert: (a) the hook was handed the spec decoder's
    vocab logprobs (parallel to the committed ids), and (b) every staged logprob
    -- the admit token AND each committed token -- is the vocab value PLUS the
    head delta (not the bare vocab value).
    """
    from kestrel.runtime.sampling import SamplingHooks

    HEAD_DELTA = -0.5
    side_values = object()
    received: list[list[float]] = []

    @dataclass
    class _TypedToken:
        token_id: int

    def materialize_spec_tokens(
        token_ids_cpu, sequences, batch_idx, sv, token_logprobs=None
    ):
        ids = [int(t) for t in token_ids_cpu.view(-1).tolist()]
        if token_logprobs is not None:
            received.append(list(token_logprobs))
            # Fold a per-position "spatial head" logprob in place, exactly as the
            # real Moondream hook folds the coord/size head logprob.
            for k in range(len(token_logprobs)):
                token_logprobs[k] += HEAD_DELTA
        return [_TypedToken(token_id=t) for t in ids]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes here
        emit_logprobs=True,
        first_logprobs={0: -0.5},  # real admit vocab logprob
        side_values=side_values,
        admit_side_values=side_values,  # route the admit token through the hook
    )
    rt = _spec_runtime(dec)
    rt.sampling_hooks = SamplingHooks(
        materialize_spec_tokens=materialize_spec_tokens
    )
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=3, return_logprobs=True)

    assert sched._spec_admit() is True
    while sched._spec_decode_step():
        pass

    # The hook was handed the spec decoder's vocab logprobs: the admit token
    # (-0.5) on its own, then the macro-step's committed-token vocab logprobs
    # (-12.0, -13.0 == -(token_id)).
    assert [-0.5] in received
    assert [-12.0, -13.0] in received
    # Every staged logprob is vocab + head delta (folded by the hook), not the
    # bare vocab value the spec decoder gathered.
    assert req.lifecycle.logprobs == pytest.approx(
        [-0.5 + HEAD_DELTA, -12.0 + HEAD_DELTA, -13.0 + HEAD_DELTA]
    )
    completed = sched.pop_completed()
    assert completed[0].logprobs == pytest.approx(
        [-0.5 + HEAD_DELTA, -12.0 + HEAD_DELTA, -13.0 + HEAD_DELTA]
    )


def test_spec_admit_applies_one_shot_suppression() -> None:
    """Regression for codex finding: one-shot suppression reaches the admit sample.

    ``GenerationRequest.suppress_next_token_ids`` is a one-shot blacklist the
    non-spec path applies to a request's *first* generated token. ``admit``
    samples that exact token, so the scheduler must forward the suppression as
    ``suppress_next_token_ids`` -- otherwise a suppressed id can be emitted at
    admit and the one-shot window closes before ``step`` could apply it.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=8)
    req.suppress_next_token_ids = (7, 8, 9)

    assert sched._spec_admit() is True
    # The one-shot suppression was forwarded into admit (the first sample).
    assert tuple(dec.admit_kwargs[0]["suppress_next_token_ids"]) == (7, 8, 9)


def test_spec_admit_no_one_shot_suppression_past_prefix() -> None:
    """One-shot suppression is *not* forwarded once past a request's prefix.

    Mirrors the non-spec gate (``token_count == generated_prefix_length``): a
    resumed request that already generated its prefix is not on its first
    generated token, so the one-shot suppression must not re-fire at admit.
    """

    class _PrefixState(_RecordingState):
        # Pretend the request already generated one prefix token.
        @property
        def token_count(self) -> int:  # type: ignore[override]
            return 1

    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=8)
    req.suppress_next_token_ids = (7, 8, 9)
    prefix_state = _PrefixState(req)
    req.lifecycle.skill_state = prefix_state
    req.skill_state = prefix_state

    assert sched._spec_admit() is True
    # token_count (1) != generated_prefix_length (0) -> suppression not applied.
    assert dec.admit_kwargs[0]["suppress_next_token_ids"] is None


def test_spec_admit_types_admit_token_via_materialize_hook() -> None:
    """Regression for codex finding: type the admit token through the spec hook.

    On a spatial runtime the first generated id can be coord/size, so the admit
    (first) token must be typed through the runtime ``materialize_spec_tokens``
    hook using ``admit``'s :class:`SpecSideValues` -- not hard-coded to
    ``TextToken``, and not via the non-spec ``materialize_tokens`` hook (whose
    ``(slot, batch_size)`` handle shape ``SpecSideValues`` does not match). Stubs
    the spec hook to wrap the admit id and asserts (a) it ran for the admit
    token, (b) it received ``admit``'s side-values, (c) the staged first token is
    the typed token, (d) the non-spec hook was never invoked with the
    side-values.
    """
    admit_side_values = object()
    calls: list[dict] = []
    nonspec_calls: list[object] = []

    @dataclass
    class _TypedToken:
        token_id: int

    def materialize_spec_tokens(
        token_ids_cpu, sequences, batch_idx, side_values, token_logprobs=None
    ):
        ids = [int(t) for t in token_ids_cpu.view(-1).tolist()]
        calls.append({"side_values": side_values, "ids": ids})
        return [_TypedToken(token_id=int(t)) for t in ids]

    def materialize_tokens(token_ids_cpu, sequences, batch_idx, step_handle):
        # The spec side-values must never reach the non-spec hook.
        nonspec_calls.append(step_handle)
        return [_TypedToken(token_id=int(t)) for t in token_ids_cpu.view(-1).tolist()]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12]]},
        admit_side_values=admit_side_values,
    )
    rt = _spec_runtime(dec)
    from kestrel.runtime.sampling import SamplingHooks

    # Hooks are read at scheduler construction, so install before _make_scheduler.
    rt.sampling_hooks = SamplingHooks(
        materialize_tokens=materialize_tokens,
        materialize_spec_tokens=materialize_spec_tokens,
    )
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=8)

    assert sched._spec_admit() is True
    # The spec hook typed the admit id (11) and was handed admit's side-values.
    admit_call = next(c for c in calls if c["ids"] == [11])
    assert admit_call["side_values"] is admit_side_values
    # The non-spec hook was never called with the SpecSideValues.
    assert admit_side_values not in nonspec_calls
    staged = req.lifecycle.skill_state.tokens
    assert isinstance(staged[0], _TypedToken)
    assert int(staged[0].token_id) == 11


def test_spec_admit_fails_on_non_slot_adapter_error() -> None:
    """Regression for codex finding: a non-slot adapter error fails, not defers.

    Only the recoverable out-of-LoRA-slots ``RuntimeError`` should defer the
    request back to WAITING_RESOURCES. A different ``RuntimeError`` (e.g. a CUDA
    load/copy failure) is unrecoverable: the request must FAIL cleanly rather
    than loop forever in the waiting queue.
    """

    class _BrokenAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)

    def _raise_load_failure(adapter_id: str, adapter: object) -> int:
        # A genuine load/copy failure, NOT the out-of-slots signal.
        raise RuntimeError("CUDA error: out of memory while loading adapter slot")

    rt.acquire_adapter_slot = _raise_load_failure  # type: ignore[method-assign]

    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_BrokenAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    # The request is removed from waiting and failed (error result), not deferred.
    assert sched._spec_admit() is True
    assert dec.admitted == []
    assert len(sched.waiting) == 0
    assert len(sched.running) == 0
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "error"


def test_spec_admit_forwards_image_mask_and_sampling_params() -> None:
    """The spec path passes a request's full context into ``admit`` (no fallback).

    Regression for the three newer codex findings (image @ L615, masks/sampling
    @ L761): image requests must prefill *with the image*, skill masks must
    constrain the drafter+verify, and temperature/top_p must reach the decoder --
    not be silently dropped to a text-only / unconstrained / greedy path.
    """

    class _MaskedState(_RecordingState):
        def allowed_token_ids(self, runtime: object):
            return [12, 13]

        def suppressed_token_ids(self, runtime: object):
            return [99]

    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    img = object()
    # Build the request, then bind a masked skill-state to it (the state needs
    # the real request so its mask methods see the right context).
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, temperature=0.7, top_p=0.9, image=img
    )
    masked_state = _MaskedState(req)
    req.lifecycle.skill_state = masked_state
    req.skill_state = masked_state

    assert sched._spec_admit() is True
    kw = dec.admit_kwargs[0]
    assert kw["image"] is img
    assert list(kw["allowed_token_ids"]) == [12, 13]
    assert list(kw["suppressed_token_ids"]) == [99]
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9


def test_spec_admit_forwards_image_crops_for_multicrop_request() -> None:
    """The spec path forwards a request's multi-crop tiles into ``admit``.

    Regression for the codex finding @ scheduler.py:757: ``_spec_admit`` forwarded
    ``request.image`` but NOT ``request.image_crops`` -- the high-res overlap crop
    tiles Moondream tiles a large image into. The non-spec ``prepare_sequence``
    path forwards BOTH (the vision encoder reads ``image_crops`` as its ``overlap``
    so the crop tiles -- not just the global/thumbnail image -- are encoded into
    the KV prefix). Dropping ``image_crops`` on the spec path gives a multi-crop
    request an incomplete image prefill and diverges from the non-spec output.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    img = object()
    crops = object()  # stand-in for the OverlapCropOutput the runtime tiles
    _enqueue(sched, 0, prompt_len=3, max_new=8, image=img, image_crops=crops)

    assert sched._spec_admit() is True
    kw = dec.admit_kwargs[0]
    # Both the image AND its crop tiles must reach ``admit`` -- exactly what the
    # non-spec ``prepare_sequence`` forwards. Forwarding ``image`` alone (the
    # pre-fix behaviour) would prefill only the global image for a multi-crop
    # request.
    assert kw["image"] is img
    assert kw["image_crops"] is crops


def test_spec_commit_routes_committed_ids_through_materialize_hook() -> None:
    """Committed ids are typed via the runtime ``materialize_spec_tokens`` hook.

    Regression for typed-token finding @ L792 + SpecSideValues finding @ L836:
    the spec commit must route the committed ids (+ ``SpecStepResult.side_values``)
    through the *spec-aware* runtime hook so a spatial runtime types coord/size
    ids -- not hardcode ``TextToken``, and not via the non-spec
    ``materialize_tokens`` hook (whose ``(slot, batch_size)`` handle shape a
    ``SpecSideValues`` does not match). Stubs the spec hook to wrap every
    committed id and asserts the macro-step's ``side_values`` reached it and the
    non-spec hook was never handed the side-values.
    """
    sentinel_side_values = object()
    calls: list[dict] = []
    nonspec_calls: list[object] = []

    @dataclass
    class _TypedToken:
        token_id: int

    def materialize_spec_tokens(
        token_ids_cpu, sequences, batch_idx, side_values, token_logprobs=None
    ):
        ids = [int(t) for t in token_ids_cpu.view(-1).tolist()]
        calls.append({"side_values": side_values, "ids": ids})
        return [_TypedToken(token_id=int(t)) for t in ids]

    def materialize_tokens(token_ids_cpu, sequences, batch_idx, step_handle):
        nonspec_calls.append(step_handle)
        return [_TypedToken(token_id=int(t)) for t in token_ids_cpu.view(-1).tolist()]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes here
        side_values=sentinel_side_values,
    )
    rt = _spec_runtime(dec)
    from kestrel.runtime.sampling import SamplingHooks

    # Hooks are read at scheduler construction, so install before _make_scheduler.
    rt.sampling_hooks = SamplingHooks(
        materialize_tokens=materialize_tokens,
        materialize_spec_tokens=materialize_spec_tokens,
    )
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()
    while sched._spec_decode_step():
        pass

    # The spec hook typed the committed ids (12, 13) and was handed the
    # macro-step's side_values.
    real = next(c for c in calls if c["ids"] == [12, 13])
    assert real["side_values"] is sentinel_side_values
    # The SpecSideValues was never passed into the non-spec hook.
    assert sentinel_side_values not in nonspec_calls
    staged = req.lifecycle.skill_state.tokens
    # token 11 is the admit bonus (TextToken); 12/13 come from the typed spec hook.
    assert isinstance(staged[1], _TypedToken)
    assert isinstance(staged[2], _TypedToken)
    assert [int(t.token_id) for t in staged] == [11, 12, 13]


def test_spec_admit_preserves_typed_prefill_tokens() -> None:
    """Round-2 codex finding @ L619: typed prefill tokens survive admission.

    A launchable prefill can contain non-text tokens -- multi-image chat prompts
    carry ``ImageMarker`` tokens, and a resumed request's generated prefix can
    carry ``CoordToken`` / ``SizeToken`` -- none of which expose ``token_id``.
    The old ``prompt_ids = [int(t.token_id) for t in prefill_tokens]`` ran
    *before* the per-request ``try`` and raised ``AttributeError``, aborting the
    whole scheduler. The fix forwards the full typed list (like the non-spec
    ``prepare_sequence`` path). Asserts admission succeeds and the decoder
    receives the typed tokens whole (not stripped / dropped).
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    img = object()
    req = _enqueue(sched, 0, prompt_len=1, max_new=8, image=img)
    # Inject a mixed typed prefill: BOS text + image marker + a generated-prefix
    # spatial pair. The base ``_enqueue`` already added one TextToken; replace
    # the prompt with the multimodal layout.
    typed_prompt = [
        TextToken(1),
        ImageMarker(index=0),
        CoordToken(pos=0.25),
        SizeToken(width=0.1, height=0.2),
        TextToken(2),
    ]
    req.prompt_tokens = typed_prompt

    # Must NOT raise (the bug raised AttributeError here, aborting the scheduler).
    assert sched._spec_admit() is True
    assert dec.admitted == [0]
    # The decoder got the *typed* tokens, unstripped and in order.
    forwarded = dec.admit_kwargs[0]["prompt_tokens"]
    assert forwarded == typed_prompt
    # Non-text tokens survived (the regression dropped them via int(t.token_id)).
    assert any(isinstance(t, ImageMarker) for t in forwarded)
    assert any(isinstance(t, CoordToken) for t in forwarded)
    assert any(isinstance(t, SizeToken) for t in forwarded)
    # The state's token count reflects the typed list length (markers count 1).
    assert req.lifecycle.state.prompt_length == len(typed_prompt)


def test_spec_adapter_request_releases_lora_slot_on_retire() -> None:
    """Round-2 codex finding @ L1725: a retiring spec row frees its LoRA slot.

    ``_spec_admit`` acquires a runtime LoRA slot for an adapter request, but the
    spec completion path retires the pool row via ``decoder.retire`` and skips
    ``runtime.release_sequence`` (which is where the non-spec path drops the
    adapter ref). Without an explicit release the slot leaks until the adapter
    pool starves. Asserts every acquired slot is released after the row retires
    (slot count returns to baseline -- no leak).
    """

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes + retires
    )
    rt = _spec_runtime(dec)
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=3, adapter="ft-1", lora_slot_ready=False
    )
    sched._spec_admit()
    acquired_slot = req.lora_slot
    assert acquired_slot >= 1  # a real adapter slot was acquired
    assert rt.released_adapter_slots == []  # not yet released

    # Drive the row to completion (it finishes at max_new=3 and retires).
    while sched._spec_decode_step():
        pass

    assert req.lifecycle.finished is True
    assert dec.retired == [0]
    # The acquired slot was released exactly once -> count back to baseline.
    assert rt.released_adapter_slots == [acquired_slot]
    assert rt.released_adapter_slots.count(acquired_slot) == 1
    assert req.lifecycle.state.batch_idx not in rt.active_sequences


def test_spec_admit_finish_releases_lora_slot() -> None:
    """LoRA slot is released even when the request finishes at admit time.

    If the admit bonus token already finishes the request (e.g. immediate EOS),
    ``_spec_admit`` retires the row inline. That path must also release the
    adapter slot (it goes through the same ``_retire_spec_row`` helper).
    """

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    # First token is EOS -> request finishes right after admit.
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 999}, plans={0: [[0]]})
    rt = _spec_runtime(dec, eos_id=999)
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    assert sched._spec_admit() is True
    acquired_slot = req.lora_slot
    assert acquired_slot >= 1
    assert req.lifecycle.finished is True
    assert dec.retired == [0]
    # Slot released on the admit-finish path too (no leak).
    assert rt.released_adapter_slots == [acquired_slot]
    assert req.lifecycle.state.batch_idx not in rt.active_sequences


class _RaisingAdmitDecoder(_FakeDecoder):
    """``SpecDecoder`` whose ``admit`` reserves a pool row, sets ``batch_idx``,
    then RAISES -- emulating a prefill that fails mid-flight (e.g. an image
    preprocessing / CUDA error) *after* the decoder has already claimed a free
    row for the sequence. Unlike the base ``_FakeDecoder``, ``retire`` returns
    the row to the free pool so a test can assert the failed admission did not
    leak a slot (``free_slots`` returns to baseline).
    """

    def admit(self, state, prompt_token_ids, **kwargs):
        # Reserve a row + assign batch_idx exactly like a real admit does before
        # its prefill work runs, so the failure path has a row to leak.
        row = self._free.pop(0)
        self._row_of[id(state)] = row
        state.batch_idx = 100 + row
        self.admitted.append(row)
        raise RuntimeError("CUDA error: image prefill failed mid-admit")

    def retire(self, state) -> None:
        row = self._row_of.pop(id(state))
        self.retired.append(row)
        # Return the row to the free pool so the test can prove no leak.
        self._free.append(row)


def test_spec_admit_failure_retires_row_no_leak() -> None:
    """P2 codex finding @ scheduler.py:776: a failed ``decoder.admit`` retires
    the reserved spec row so ``free_slots`` does not leak.

    ``admit`` reserves a free pool row and assigns ``state.batch_idx`` BEFORE its
    prefill image/CUDA work, so a mid-admit failure leaves the row reserved even
    though no token was staged. The request has already left ``waiting`` and
    ``lifecycle.sequence_state`` is never set, so no finish/zombie path retires
    it -- the row would leak permanently and repeated failures would drain
    ``decoder.free_slots`` and stall unrelated spec requests. Asserts the failed
    admission (a) fails the request cleanly, (b) retires the reserved row, and
    (c) leaves ``free_slots`` back at baseline (no leak), and that a SUBSEQUENT
    request can still be admitted into the reclaimed row.
    """
    dec = _RaisingAdmitDecoder(
        n_rows=1, first_tokens={0: 11}, plans={0: [[12]]}
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    assert dec.free_slots == 1  # baseline: one row free

    req = _enqueue(sched, 0, prompt_len=3, max_new=8)

    assert sched._spec_admit() is True
    # The row was reserved (admit started) then retired on failure.
    assert dec.admitted == [0]
    assert dec.retired == [0]
    # No leak: the reserved row was returned to the pool.
    assert dec.free_slots == 1
    # The request failed cleanly (error result), not deferred/queued.
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    assert len(sched.waiting) == 0
    assert len(sched.running) == 0
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "error"
    # ``sequence_state`` is intentionally never set on the admit-failure path
    # (it is only assigned AFTER a successful admit) -- which is exactly why no
    # finish/zombie path would have retired the row without this fix. The
    # reserved row left no dangling ``active_sequences`` entry.
    assert req.lifecycle.sequence_state is None
    assert rt.active_sequences == {}

    # The reclaimed row is genuinely reusable: a fresh request admits into it
    # rather than starving on a permanently-drained pool.
    dec.admit = _FakeDecoder.admit.__get__(dec, _RaisingAdmitDecoder)  # stop raising
    req2 = _enqueue(sched, 1, prompt_len=2, max_new=8)
    assert sched._spec_admit() is True
    assert req2.lifecycle.finished is False
    assert req2.lifecycle.state.batch_idx in rt.active_sequences
    assert dec.free_slots == 0  # the single row is now in use by req2


def test_spec_admit_failure_releases_lora_slot_and_retires_row() -> None:
    """P2 codex finding @ scheduler.py:776 (adapter variant): a failed
    ``decoder.admit`` releases BOTH the reserved spec row and the LoRA slot.

    For an adapter request, ``_spec_admit`` acquires a LoRA slot *and* (via
    ``admit``) reserves a pool row before the prefill can fail. On failure both
    must be released exactly once -- the row via ``decoder.retire`` and the
    adapter slot via ``release_adapter_slot`` -- without double-releasing the
    slot (the cleanup is deliberately NOT ``_retire_spec_row``, which would
    release the slot a second time). Asserts the row is retired, the acquired
    slot is released exactly once, and the request fails cleanly.
    """

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _RaisingAdmitDecoder(
        n_rows=1, first_tokens={0: 11}, plans={0: [[12]]}
    )
    rt = _spec_runtime(dec)
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    assert sched._spec_admit() is True
    # The slot was acquired for the adapter, then released on the failure path.
    assert len(rt.acquired_adapter_slots) == 1
    acquired_slot = len(rt.acquired_adapter_slots)  # FakeRuntime numbers from 1
    assert rt.released_adapter_slots == [acquired_slot]
    assert rt.released_adapter_slots.count(acquired_slot) == 1  # not double-released
    # The reserved row was retired -> no leak.
    assert dec.admitted == [0]
    assert dec.retired == [0]
    assert dec.free_slots == 1
    # request.lora_slot was zeroed so a later path cannot re-release it.
    assert req.lora_slot == 0
    assert req.lifecycle.lora_slot_ready is False
    # The request failed cleanly.
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    assert req.lifecycle.sequence_state is None
    assert rt.active_sequences == {}


def test_spec_pre_admit_hook_failure_releases_lora_slot_and_fails_cleanly() -> None:
    """P2 codex finding @ scheduler.py:781: a pre-admit hook failure stays on the
    admit-failure cleanup path (LoRA slot released, request failed cleanly).

    The skill ``on_prefill`` hook (and the ``allowed_token_ids`` /
    ``suppressed_token_ids`` mask queries) run AFTER the request has left
    ``waiting``, transitioned to ``PREFILLING``, and acquired its LoRA slot, but
    BEFORE ``decoder.admit``. If that hook raises while it sits outside the
    ``try``/cleanup block, the request is dropped from ``waiting`` stuck in
    ``PREFILLING`` with its LoRA slot leaked (``sequence_state`` is never set, so
    no finish/zombie path ever releases it) and the scheduler later crashes on
    it. With the hook inside the same ``try`` as ``admit``, the existing
    admit-failure cleanup releases the slot and fails the request cleanly.
    ``decoder.admit`` never runs, so no pool row is reserved (``batch_idx`` stays
    at its ``-1`` sentinel) and nothing is retired. Asserts the slot is released
    exactly once, ``admit``/``retire`` are never called (no row leak), and the
    request ends as a clean error -- not stuck PREFILLING.
    """

    class _RaisingOnPrefillState(_RecordingState):
        def on_prefill(self, runtime: object) -> None:  # type: ignore[override]
            raise RuntimeError("skill on_prefill failed before admit")

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    assert dec.free_slots == 1  # baseline: one row free
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    req = GenerationRequest(
        request_id=0,
        prompt="p",
        prompt_tokens=[TextToken(1)] * 3,
        max_new_tokens=8,
        skill=_SpecSkillSpec(),  # type: ignore[arg-type]
        request_context=object(),
        adapter="ft-1",
    )
    lc = RequestLifecycle(request=req, skill_state=_RaisingOnPrefillState(req))
    lc.lora_slot_ready = False
    lc.transition(RequestPhase.READY_FOR_PREFILL)
    req.lifecycle = lc
    sched.enqueue_request(req, lc.skill_state)

    assert sched._spec_admit() is True
    # ``admit`` was never reached (the hook raised first) -> no pool row was ever
    # reserved, so none was admitted and none retired, and ``free_slots`` is
    # untouched (no leak).
    assert dec.admitted == []
    assert dec.retired == []
    assert dec.free_slots == 1
    # The LoRA slot acquired before the hook ran is released exactly once on the
    # cleanup path (not leaked, not double-released).
    assert len(rt.acquired_adapter_slots) == 1
    acquired_slot = len(rt.acquired_adapter_slots)  # FakeRuntime numbers from 1
    assert rt.released_adapter_slots == [acquired_slot]
    assert rt.released_adapter_slots.count(acquired_slot) == 1
    # request.lora_slot was zeroed so no later path can re-release it.
    assert req.lora_slot == 0
    assert req.lifecycle.lora_slot_ready is False
    # The request failed cleanly -- it is NOT left stuck in PREFILLING, and no
    # sequence_state / active row was registered for it.
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    assert req.lifecycle.phase == RequestPhase.COMPLETED
    assert req.lifecycle.sequence_state is None
    assert rt.active_sequences == {}
    # It was removed from the waiting queue (the scheduler removes it before the
    # hook runs) and never entered the running set.
    assert req not in sched.waiting
    assert len(sched.running) == 0


class _AdmitSideValuesDecoder(_FakeDecoder):
    """``SpecDecoder`` whose ``admit`` SUCCEEDS -- reserving a pool row, assigning
    ``batch_idx``, and attaching spatial ``admit_side_values`` to the state -- but
    whose row is returned to the free pool by ``retire`` so a test can prove the
    post-admit cleanup did not leak the row.

    This drives the failure site that the base ``_RaisingAdmitDecoder`` does NOT:
    ``decoder.admit`` returns cleanly, the request is registered in
    ``active_sequences`` and ``sequence_state`` is set, and the failure happens
    LATER when the scheduler types the admit token via ``_materialize_spec_tokens``
    (a spatial ``admit_side_values`` with no ``materialize_spec_tokens`` hook fails
    fast). That materialization runs OUTSIDE the original admit ``try``/cleanup, so
    without the fix it leaks the spec row + LoRA slot + ``active_sequences`` entry
    and strands the request.
    """

    def retire(self, state) -> None:
        row = self._row_of.pop(id(state))
        self.retired.append(row)
        # Return the row to the free pool so the test can prove no leak.
        self._free.append(row)


def test_spec_admit_token_materialize_failure_retires_row_no_leak() -> None:
    """P2 codex finding @ scheduler.py:950: a raising admit-token
    materialization retires the spec row and frees the LoRA slot, no leak.

    The admit-token typing/staging path runs AFTER ``decoder.admit`` has reserved
    the pool row, set ``sequence_state``, and registered the state in
    ``active_sequences`` -- but BEFORE the request is pushed to ``running`` and
    while it has no finish/zombie path. It also sits OUTSIDE the admit ``try`` that
    retires the row + releases the LoRA slot. If ``_materialize_spec_tokens``
    raises here (a spatial admit token whose runtime emits ``admit_side_values``
    but exposes no ``materialize_spec_tokens`` hook -> fail-fast ``RuntimeError``),
    an unguarded raise would leak the spec pool row + adapter slot +
    ``active_sequences`` entry and strand the request with no scheduler path to
    finish. Asserts the bad spatial admit (a) retires the reserved row, (b)
    releases the LoRA slot exactly once, (c) fails the request cleanly (error
    result, COMPLETED -- not stuck PREFILLING), (d) leaves no ``active_sequences``
    leak and ``free_slots`` back at baseline, and (e) lets a SUBSEQUENT request
    admit into the reclaimed row.
    """

    class _FakeAdapterProvider:
        def config(self) -> dict:
            return {"max_lora_rank": 16}

        def get(self, adapter: str) -> object:
            return object()

    # ``admit`` succeeds and attaches spatial side-values; the runtime exposes NO
    # ``materialize_spec_tokens`` hook (the default ``_spec_runtime`` SamplingHooks
    # are empty), so typing the admit token fails fast inside ``_spec_admit``.
    admit_side_values = object()
    dec = _AdmitSideValuesDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12]]},
        admit_side_values=admit_side_values,
    )
    rt = _spec_runtime(dec)
    assert dec.free_slots == 1  # baseline: one row free
    sched = GenerationScheduler(
        rt,
        compute_stream=None,
        skill_registry=__import__(
            "kestrel.skills", fromlist=["SkillRegistry"]
        ).SkillRegistry([]),
        adapter_provider=_FakeAdapterProvider(),
    )
    req = _enqueue(
        sched, 0, prompt_len=3, max_new=8, adapter="ft-1", lora_slot_ready=False
    )

    # The scheduler does not crash out -- it cleans up and reports progress.
    assert sched._spec_admit() is True
    # ``admit`` reserved the row; the admit-token materialization then failed and
    # the row was retired -> no leak (free pool back at baseline).
    assert dec.admitted == [0]
    assert dec.retired == [0]
    assert dec.free_slots == 1
    # The LoRA slot acquired for the adapter request was released exactly once
    # (not leaked, not double-released) and zeroed so no later path re-releases it.
    assert len(rt.acquired_adapter_slots) == 1
    acquired_slot = len(rt.acquired_adapter_slots)  # FakeRuntime numbers from 1
    assert rt.released_adapter_slots == [acquired_slot]
    assert rt.released_adapter_slots.count(acquired_slot) == 1
    assert req.lora_slot == 0
    assert req.lifecycle.lora_slot_ready is False
    # The request failed cleanly -- COMPLETED, error result, not stuck PREFILLING,
    # never pushed to running, and no token staged.
    assert req.lifecycle.finished is True
    assert req.lifecycle.finish_reason == "error"
    assert req.lifecycle.phase == RequestPhase.COMPLETED
    assert len(req.lifecycle.skill_state.tokens) == 0
    assert len(sched.running) == 0
    assert len(sched.waiting) == 0
    # No dangling active row: the state registered just before the failed
    # materialization was popped on cleanup.
    assert rt.active_sequences == {}
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "error"

    # The reclaimed row is genuinely reusable: a fresh (non-spatial) request admits
    # into it rather than starving on a permanently-drained pool. Clear the admit
    # side-values so the next admit token types as a plain TextToken (no hook
    # needed) and succeeds.
    dec._admit_side_values = None
    req2 = _enqueue(sched, 1, prompt_len=2, max_new=8)
    assert sched._spec_admit() is True
    assert req2.lifecycle.finished is False
    assert req2.lifecycle.state.batch_idx in rt.active_sequences
    assert dec.free_slots == 0  # the single row is now in use by req2


def test_spec_side_values_fails_fast_when_no_spec_hook() -> None:
    """Spatial ``SpecSideValues`` without a spec hook must FAIL FAST, not degrade.

    P2: ``side_values`` is produced ONLY when the runtime decoded spatial values
    that must be typed into ``CoordToken`` / ``SizeToken`` from the target's
    per-position final hidden (the detect/point spec path). If the runtime exposes
    only the non-spec ``materialize_tokens`` hook (whose handle is the
    ``post_sample`` ``(slot, batch_size)`` aux value, which a ``SpecSideValues``
    does not match), the scheduler must NOT (a) feed the side-values into that hook
    (``cannot unpack`` / wrong layout) NOR (b) silently materialise the committed
    ids as plain ``TextToken``s -- that would DROP the decoded coordinates/sizes
    and corrupt the result. The project goal is full coverage / no silent
    fallback, so the scheduler raises a clear ``RuntimeError`` instead. Asserts
    (a) it raises, (b) the message names ``materialize_spec_tokens``, (c) the
    non-spec hook was never handed the side-values.
    """
    import torch

    real_side_values = SpecSideValues(
        hidden=torch.zeros(1, 3, 4),
        temperatures=torch.zeros(1),
        top_ps=torch.ones(1),
    )
    nonspec_handles: list[object] = []

    def materialize_tokens(token_ids_cpu, sequences, batch_idx, step_handle):
        # Emulate the Moondream hook's unpack so a SpecSideValues would blow up
        # here if the scheduler ever routed it through this non-spec hook.
        nonspec_handles.append(step_handle)
        if step_handle is not None:
            _slot, _batch_size = step_handle  # would raise on SpecSideValues
        return [TextToken(token_id=int(t)) for t in token_ids_cpu.view(-1).tolist()]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes here
        side_values=real_side_values,
    )
    rt = _spec_runtime(dec)
    from kestrel.runtime.sampling import SamplingHooks

    # Only the NON-spec hook is installed; no materialize_spec_tokens.
    rt.sampling_hooks = SamplingHooks(materialize_tokens=materialize_tokens)
    sched = _make_scheduler(rt)
    _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()
    # The macro-step commit must fail fast rather than silently drop coord/size.
    with pytest.raises(RuntimeError, match="materialize_spec_tokens"):
        while sched._spec_decode_step():
            pass

    # The non-spec hook was never handed the SpecSideValues (no misuse).
    assert real_side_values not in nonspec_handles


def _synthetic_spatial_tables(coord_bins: int, size_bins: int):
    """An MLP ``SpatialDecodeTables`` whose argmax bins are read straight out of
    the hidden state, with a pure-PyTorch decode (CPU-runnable, no custom kernels).

    Lays the hidden out as ``[coord one-hot (coord_bins) | width one-hot (size_bins)
    | height one-hot (size_bins)]``. The MLP path (``F.gelu`` + ``F.linear`` only,
    unlike the linear path which calls a bf16-only LayerNorm CUDA kernel) is built
    so ``argmax`` over each logit group recovers exactly the one-hot index set in
    that group's hidden slice: ``fc1`` is a scaled identity (so gelu maps the hot
    position to a large positive value and zeros to ~0), and ``fc2`` selects each
    group's hidden dims. The greedy branch of ``compute_spatial_values`` uses
    ``torch.argmax`` only, so the decoded coord/size values are deterministic and
    the whole thing runs on CPU.
    """
    import torch
    import torch.nn as nn

    from kestrel.models.moondream.region import SpatialDecodeTables

    hidden_dim = coord_bins + 2 * size_bins
    inner = hidden_dim
    scale = 50.0

    def _decoder(select_rows: int, offset: int) -> nn.ModuleDict:
        fc1 = nn.Linear(hidden_dim, inner)
        with torch.no_grad():
            fc1.weight.copy_(torch.eye(inner, hidden_dim) * scale)
            fc1.bias.zero_()
        fc2 = nn.Linear(inner, select_rows)
        with torch.no_grad():
            sel = torch.zeros(select_rows, inner)
            for r in range(select_rows):
                sel[r, offset + r] = 1.0
            fc2.weight.copy_(sel)
            fc2.bias.zero_()
        return nn.ModuleDict({"fc1": fc1, "fc2": fc2})

    coord_decoder = _decoder(coord_bins, offset=0)
    size_decoder = _decoder(2 * size_bins, offset=coord_bins)

    coord_value_lut = torch.linspace(0.0, 1.0, coord_bins, dtype=torch.float32)
    size_exponents = torch.linspace(-10.0, 0.0, size_bins, dtype=torch.float32)
    size_value_lut = torch.exp2(size_exponents)
    tables = SpatialDecodeTables(
        coord_value_lut=coord_value_lut,
        size_value_lut=size_value_lut,
        coord_logits_dim=coord_bins,
        coord_decoder=coord_decoder,
        size_decoder=size_decoder,
    )
    return tables, hidden_dim


def _encode_spatial_hidden(coord_bins, size_bins, hidden_dim, *, coord, width, height):
    """One-hot a (coord_bin, width_bin, height_bin) triple into a hidden vector that
    ``_synthetic_spatial_tables`` decodes back to those bins."""
    import torch

    h = torch.zeros(hidden_dim, dtype=torch.float32)
    h[coord] = 1.0
    h[coord_bins + width] = 1.0
    h[coord_bins + size_bins + height] = 1.0
    return h


def test_moondream_materialize_spec_tokens_types_coord_size() -> None:
    """The REAL ``MoondreamRuntime.materialize_spec_tokens`` types a ragged spec
    macro-step's committed ids into ``CoordToken`` / ``SizeToken`` (P2 full
    coverage).

    Exercises the actual production hook (not a stub) on CPU: builds the hook via
    ``MoondreamRuntime.sampling_hooks`` over a minimal stand-in that supplies only
    the attributes the property reads (``config.tokenizer.coord_id``/``size_id``,
    ``spatial_tables``, the pending-pool buffers + copy stream + spatial RNG used by
    the *other* closures), then drives it with the scheduler's exact
    ``materialize_spec_tokens`` contract: a flat (sequences-major, contiguous-
    per-sequence) committed-id batch + per-token ``batch_idx`` + a
    :class:`SpecSideValues` whose ``hidden[i, j]`` is the committed-position final
    hidden. Two ragged sequences mix ``coord_id`` / ``size_id`` / a plain text id,
    and we assert each id is typed correctly with the value decoded from its OWN
    hidden position (proving per-position routing, not a single shared value).
    """
    import torch

    from kestrel.models.moondream.runtime import (
        CoordToken,
        MoondreamRuntime,
        SizeToken,
        TextToken,
    )
    from kestrel.runtime.spec import SpecSideValues

    coord_bins, size_bins = 11, 8
    tables, hidden_dim = _synthetic_spatial_tables(coord_bins, size_bins)
    coord_id, size_id = 5000, 5001

    # Minimal stand-in carrying exactly what ``sampling_hooks`` reads.
    stub = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(coord_id=coord_id, size_id=size_id)
        ),
        spatial_tables=tables,
        _pending_coord_values=torch.zeros(8, 1),
        _pending_size_values=torch.zeros(8, 2),
        _copy_stream=None,
        _spatial_rng=None,  # greedy path never samples, so the RNG is unused
    )
    hooks = MoondreamRuntime.sampling_hooks.func(stub)
    assert hooks.materialize_spec_tokens is not None

    # Two sequences (greedy: temperature 0). Seq A commits 3 tokens
    # [coord, size, text]; seq B commits 1 token [coord]. Flat layout is
    # sequences-major + contiguous per sequence (the scheduler's contract).
    seq_a = SimpleNamespace(
        request=SimpleNamespace(temperature=0.0, top_p=1.0),
        state=SimpleNamespace(batch_idx=101),
    )
    seq_b = SimpleNamespace(
        request=SimpleNamespace(temperature=0.0, top_p=1.0),
        state=SimpleNamespace(batch_idx=102),
    )
    sequences = [seq_a, seq_b]

    text_id = 42
    flat_ids = [coord_id, size_id, text_id, coord_id]
    batch_idx = torch.tensor([101, 101, 101, 102], dtype=torch.long)
    token_ids_cpu = torch.tensor(flat_ids, dtype=torch.long)

    # Per-committed-position target bins -> expected decoded values.
    a_coord_bin = 7   # coord value 7/10 = 0.7
    a_w_bin, a_h_bin = 2, 5
    b_coord_bin = 3   # coord value 3/10 = 0.3
    # K+1 hidden slots per sequence; only the committed leading positions matter.
    K1 = 4
    hidden = torch.zeros(2, K1, hidden_dim, dtype=torch.float32)
    # Seq A committed positions 0,1,2 (coord, size, text); position 2 is text so
    # its decoded coord/size are ignored -- still set a sentinel to prove text ids
    # are NOT mis-typed even when their hidden would decode to a valid bin.
    hidden[0, 0] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=a_coord_bin, width=0, height=0
    )
    hidden[0, 1] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=0, width=a_w_bin, height=a_h_bin
    )
    hidden[0, 2] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=9, width=7, height=7
    )
    # Seq B committed position 0 (coord).
    hidden[1, 0] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=b_coord_bin, width=0, height=0
    )

    side_values = SpecSideValues(
        hidden=hidden,
        temperatures=torch.zeros(2, dtype=torch.float32),
        top_ps=torch.ones(2, dtype=torch.float32),
        counts=None,  # force the batch_idx-derived run-length path
    )

    tokens = hooks.materialize_spec_tokens(
        token_ids_cpu, sequences, batch_idx, side_values
    )

    assert len(tokens) == 4
    # Seq A: coord -> CoordToken(0.7); size -> SizeToken(2^(...)); text -> TextToken.
    assert isinstance(tokens[0], CoordToken)
    assert tokens[0].pos == pytest.approx(a_coord_bin / (coord_bins - 1))
    assert isinstance(tokens[1], SizeToken)
    assert tokens[1].width == pytest.approx(float(tables.size_value_lut[a_w_bin]))
    assert tokens[1].height == pytest.approx(float(tables.size_value_lut[a_h_bin]))
    assert isinstance(tokens[2], TextToken)
    assert tokens[2].token_id == text_id
    # Seq B: its coord uses seq B's OWN hidden row (0.3), not seq A's (0.7).
    assert isinstance(tokens[3], CoordToken)
    assert tokens[3].pos == pytest.approx(b_coord_bin / (coord_bins - 1))


def test_moondream_materialize_spec_tokens_folds_spatial_head_logprob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 regression: a spatial spec token requesting logprobs reports the
    coord/size head logprob *added* to the vocab logprob, not the vocab logprob
    alone -- matching the non-spec ``post_sample`` convention.

    The spec decoder only gathers the vocab-token logprob (it never runs the
    spatial head), so the scheduler hands ``materialize_spec_tokens`` the flat
    per-committed-position vocab logprobs and the hook must fold each spatial
    position's coord/size head logprob into them in place (exactly as the non-spec
    path's ``compute_spatial_values`` does). We patch ``sample_step_from_logits``
    (the spatial sampler) to emit scripted per-head logprobs -- the same hermetic
    technique ``tests/scheduler/test_logprobs.py`` uses for the non-spec path -- so
    the expected combined value is exact and CPU-deterministic. We assert:

    * a ``coord_id`` position: ``vocab + coord_head`` logprob,
    * a ``size_id`` position: ``vocab + width_head + height_head`` logprob,
    * a plain text position: vocab logprob unchanged (no spatial head),
    * the hook returns the same per-token combined values to the scheduler.
    """
    import torch

    from kestrel.models.moondream.runtime import (
        CoordToken,
        MoondreamRuntime,
        SizeToken,
        TextToken,
    )
    from kestrel.runtime.spec import SpecSideValues

    coord_bins, size_bins = 11, 8
    tables, hidden_dim = _synthetic_spatial_tables(coord_bins, size_bins)
    coord_id, size_id = 5000, 5001

    stub = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(coord_id=coord_id, size_id=size_id)
        ),
        spatial_tables=tables,
        _pending_coord_values=torch.zeros(8, 1),
        _pending_size_values=torch.zeros(8, 2),
        _copy_stream=None,
        _spatial_rng=None,
    )
    hooks = MoondreamRuntime.sampling_hooks.func(stub)

    # One sequence committing 3 tokens: [coord, size, text]. Greedy (temp 0), but
    # ``token_logprobs`` is set so ``compute_spatial_values`` still runs the
    # sampler (and thus the spatial-head logprob fold) rather than the pure-argmax
    # branch -- which is exactly the path that was passing ``token_logprobs=None``.
    seq = SimpleNamespace(
        request=SimpleNamespace(temperature=0.0, top_p=1.0),
        state=SimpleNamespace(batch_idx=300),
    )
    text_id = 42
    flat_ids = [coord_id, size_id, text_id]
    token_ids_cpu = torch.tensor(flat_ids, dtype=torch.long)
    batch_idx = torch.tensor([300, 300, 300], dtype=torch.long)
    total = len(flat_ids)

    coord_bin, w_bin, h_bin = 7, 2, 5
    hidden = torch.zeros(1, total, hidden_dim, dtype=torch.float32)
    hidden[0, 0] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=coord_bin, width=0, height=0
    )
    hidden[0, 1] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=0, width=w_bin, height=h_bin
    )
    # Text position: hidden would decode to a valid bin, but the head logprob must
    # NOT be added (it is not a coord/size id).
    hidden[0, 2] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=9, width=7, height=7
    )

    side_values = SpecSideValues(
        hidden=hidden,
        temperatures=torch.zeros(1, dtype=torch.float32),
        top_ps=torch.ones(1, dtype=torch.float32),
        counts=None,
    )

    # Scripted spatial-head logprobs. ``compute_spatial_values`` calls the sampler
    # twice: first coord (batch == total), then size (batch == total*2, the
    # width rows then the height rows stacked). Return the argmax bins (so decoded
    # values stay consistent with the one-hot hidden) and write known logprobs.
    coord_head = [-0.10, -0.20, -0.30]               # per position
    width_head = [-0.40, -0.50, -0.60]               # per position
    height_head = [-0.70, -0.80, -0.90]              # per position

    def fake_sample_step_from_logits(
        logits, temperatures, top_p, *, generator=None, logprobs_out=None
    ):
        bins = torch.argmax(logits, dim=-1)
        if logits.shape[0] == total:  # coord head
            if logprobs_out is not None:
                logprobs_out.copy_(torch.tensor(coord_head, dtype=torch.float32))
        else:  # size head: [width(total) | height(total)]
            if logprobs_out is not None:
                logprobs_out.copy_(
                    torch.tensor(width_head + height_head, dtype=torch.float32)
                )
        return bins

    monkeypatch.setattr(
        "kestrel.scheduler.spatial.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    # Vocab logprobs the spec decoder staged (parallel to ``flat_ids``). The hook
    # mutates this list in place to fold in the spatial head logprobs.
    vocab_logprobs = [-1.0, -2.0, -3.0]
    token_logprobs = list(vocab_logprobs)

    tokens = hooks.materialize_spec_tokens(
        token_ids_cpu, [seq], batch_idx, side_values, token_logprobs
    )

    # Token typing is unaffected by the logprob threading.
    assert isinstance(tokens[0], CoordToken)
    assert isinstance(tokens[1], SizeToken)
    assert isinstance(tokens[2], TextToken)
    assert tokens[2].token_id == text_id

    # coord: vocab + coord_head; size: vocab + width_head + height_head; text: vocab.
    expected = [
        vocab_logprobs[0] + coord_head[0],
        vocab_logprobs[1] + width_head[1] + height_head[1],
        vocab_logprobs[2],  # text id -- no spatial head added
    ]
    assert token_logprobs == pytest.approx(expected)
    # The spatial positions are strictly more negative than the vocab-only value
    # (under-reporting would have left them at the vocab logprob).
    assert token_logprobs[0] < vocab_logprobs[0]
    assert token_logprobs[1] < vocab_logprobs[1]
    assert token_logprobs[2] == pytest.approx(vocab_logprobs[2])


def test_moondream_materialize_spec_tokens_honors_explicit_counts() -> None:
    """The hook uses ``SpecSideValues.counts`` when the producer supplies them.

    Mirrors the explicit-``counts`` admit path (single position per sequence); the
    decoded coord must come from each sequence's own ``hidden[i, 0]``.
    """
    import torch

    from kestrel.models.moondream.runtime import (
        CoordToken,
        MoondreamRuntime,
    )
    from kestrel.runtime.spec import SpecSideValues

    coord_bins, size_bins = 11, 8
    tables, hidden_dim = _synthetic_spatial_tables(coord_bins, size_bins)
    coord_id, size_id = 5000, 5001
    stub = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(coord_id=coord_id, size_id=size_id)
        ),
        spatial_tables=tables,
        _pending_coord_values=torch.zeros(8, 1),
        _pending_size_values=torch.zeros(8, 2),
        _copy_stream=None,
        _spatial_rng=None,
    )
    hooks = MoondreamRuntime.sampling_hooks.func(stub)

    seqs = [
        SimpleNamespace(
            request=SimpleNamespace(temperature=0.0, top_p=1.0),
            state=SimpleNamespace(batch_idx=200),
        ),
        SimpleNamespace(
            request=SimpleNamespace(temperature=0.0, top_p=1.0),
            state=SimpleNamespace(batch_idx=201),
        ),
    ]
    hidden = torch.zeros(2, 1, hidden_dim, dtype=torch.float32)
    hidden[0, 0] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=4, width=0, height=0
    )
    hidden[1, 0] = _encode_spatial_hidden(
        coord_bins, size_bins, hidden_dim, coord=8, width=0, height=0
    )
    side_values = SpecSideValues(
        hidden=hidden,
        temperatures=torch.zeros(2, dtype=torch.float32),
        top_ps=torch.ones(2, dtype=torch.float32),
        counts=[1, 1],
    )
    token_ids_cpu = torch.tensor([coord_id, coord_id], dtype=torch.long)
    batch_idx = torch.tensor([200, 201], dtype=torch.long)

    tokens = hooks.materialize_spec_tokens(
        token_ids_cpu, seqs, batch_idx, side_values
    )
    assert isinstance(tokens[0], CoordToken)
    assert tokens[0].pos == pytest.approx(4 / (coord_bins - 1))
    assert isinstance(tokens[1], CoordToken)
    assert tokens[1].pos == pytest.approx(8 / (coord_bins - 1))


def test_spec_admit_image_sequence_state_includes_image_kv_length() -> None:
    """Regression for the image-prompt length under-report (codex finding @ L646).

    For an image request the spec ``SequenceState`` must record the *KV* prompt
    length -- the typed-token count PLUS the image KV prefix -- exactly like the
    non-spec ``prepare_sequence`` (which expands a single-image prefix /
    ``ImageMarker``s into the KV prompt length). Initializing ``length`` /
    ``prompt_length`` from the typed token count alone (leaving ``image_length``
    at 0) made ``build_metrics`` under-report prompt tokens by the image prefix
    and skewed ``output_length`` (``length - prompt_length``). Assert the state
    carries ``len(prompt_tokens) + request.image_length`` and ``image_length``.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    prompt_len = 3
    image_kv = 729  # a single-image prefix (image_prefix_length) worth of KV
    req = _enqueue(
        sched, 0, prompt_len=prompt_len, max_new=8, image=object(),
        image_length=image_kv,
    )

    assert sched._spec_admit() is True
    state = req.lifecycle.state
    # KV prompt length = typed token count + image KV prefix (matches the
    # non-spec prepare_sequence single-image layout: len(tokens) + image_kv).
    assert state.length == prompt_len + image_kv
    assert state.prompt_length == prompt_len + image_kv
    assert state.image_length == image_kv
    # build_metrics reports the full (image-inclusive) prompt length, not the
    # bare typed-token count.
    metrics = req.lifecycle.build_metrics(decode_tokens=0)
    assert metrics.prompt_tokens == prompt_len + image_kv


def test_spec_admit_text_sequence_state_length_unchanged() -> None:
    """Text-only spec admission is unaffected by the image-KV length fix.

    With ``image_length == 0`` the state length must remain the bare typed-token
    count (no regression to the non-image path).
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=4, max_new=8)

    assert sched._spec_admit() is True
    state = req.lifecycle.state
    assert state.length == 4
    assert state.prompt_length == 4
    assert state.image_length == 0


def test_spec_admit_zero_token_request_admits_and_samples_nothing() -> None:
    """Regression for the 0-token spec admit (codex finding @ L699).

    ``decoder.admit`` prefills *and* samples/stages token0, but a request with
    ``max_new_tokens == 0`` must not sample/stream any token (the non-spec path
    has a dedicated zero-length branch that finalizes as ``"length"`` without
    sampling). A 0-token request routed through the spec runtime must therefore
    admit nothing, consume no spec pool row, stage no token, and complete
    immediately with finish_reason ``"length"`` and zero output tokens.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=0)

    progressed = sched._spec_admit()

    assert progressed is True
    # No spec row consumed (admit never called), nothing retired, no active row.
    assert dec.admitted == []
    assert dec.retired == []
    assert rt.active_sequences == {}
    # Not queued into running. A metrics-only ``SequenceState`` is attached so
    # ``build_metrics`` reports the KV prompt length (no spec pool row backs it:
    # ``batch_idx == -1`` and ``_release_sequence`` early-returns for spec).
    assert len(sched.running) == 0
    assert len(sched.waiting) == 0
    assert req.lifecycle.sequence_state is not None
    assert req.lifecycle.sequence_state.batch_idx == -1
    # No token sampled/staged.
    assert list(req.lifecycle.skill_state.tokens) == []
    # Completed immediately as a length-capped (0-token) result.
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "length"
    assert list(completed[0].tokens) == []
    # Text-only zero-token request (image_length == 0): the recorded KV prompt
    # length equals request.prompt_length, so prompt_tokens is unchanged.
    assert completed[0].metrics.prompt_tokens == req.prompt_length


def test_spec_admit_zero_token_image_request_reports_image_prompt_tokens() -> None:
    """Regression for the 0-token spec image-prompt metrics under-report (P3).

    The ``max_new_tokens <= 0`` spec fast-path finalizes WITHOUT calling
    ``decoder.admit`` (no pool row), so it does not go through the regular spec
    admit path that records ``prompt_length = len(prompt_tokens) + image_length``.
    Earlier it left ``sequence_state`` unset, so ``build_metrics`` fell back to
    the TEXT-ONLY ``request.prompt_length`` and under-reported ``prompt_tokens``
    by the image KV prefix -- even though the request carried image context. The
    fast-path now attaches a metrics-only ``SequenceState`` whose KV prompt
    length includes ``image_length`` (matching the regular spec path and the
    non-spec 0-length path, whose ``prepare_sequence`` state also includes the
    image prefix). Assert a zero-token IMAGE request reports
    ``prompt_tokens == prompt_len + image_length``.
    """
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    prompt_len = 3
    image_kv = 729  # a single-image prefix (image_prefix_length) worth of KV
    req = _enqueue(
        sched, 0, prompt_len=prompt_len, max_new=0, image=object(),
        image_length=image_kv,
    )

    assert sched._spec_admit() is True
    # No spec row consumed (admit never called), nothing retired, no active row.
    assert dec.admitted == []
    assert dec.retired == []
    assert rt.active_sequences == {}
    assert len(sched.running) == 0
    assert len(sched.waiting) == 0
    # Metrics-only state carries the image-inclusive KV prompt length; no pool
    # row backs it (batch_idx == -1).
    state = req.lifecycle.sequence_state
    assert state is not None
    assert state.batch_idx == -1
    assert state.prompt_length == prompt_len + image_kv
    assert state.image_length == image_kv
    # Completed immediately as a length-capped (0-token) result.
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "length"
    assert list(completed[0].tokens) == []
    # build_metrics reports the FULL (image-inclusive) prompt length, matching
    # the regular prefill path -- not the bare text-only request.prompt_length.
    assert completed[0].metrics.prompt_tokens == prompt_len + image_kv
    assert completed[0].metrics.prompt_tokens != req.prompt_length


def test_spec_drain_pipeline_commits_pending_spec_before_pause() -> None:
    """Regression for the pause/drain leak of in-flight spec work (finding @ L548).

    The engine PAUSE path calls ``Executor.drain`` -> ``_drain_pipeline`` before
    mutating runtime/graph state. ``_spec_decode_step`` launches a macro-step
    (stored in ``_pending_spec``) one tick before committing it, so a pause can
    land with a spec result still uncommitted. ``_drain_pipeline`` must drain that
    pending spec work (commit + retire) before returning -- otherwise the runtime
    is mutated under an active, uncommitted spec row and the completion/retire is
    deferred to resume. Launch a macro-step, leave it pending, then assert
    ``_drain_pipeline`` flushes it.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes in the run
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()

    # One macro-step: depth-1 means this LAUNCHES the step and leaves it pending
    # (uncommitted) -- exactly the window a pause can land in.
    assert sched._spec_decode_step() is True
    assert sched._pending_spec is not None
    assert req.lifecycle.finished is False  # not yet committed

    # The pause path's drain entry point must flush the pending spec step.
    sched._drain_pipeline()

    assert sched._pending_spec is None
    assert req.lifecycle.finished is True
    assert dec.retired == [0]
    assert rt.active_sequences == {}
    # Drained work surfaced its completion (committed tokens 11,12,13).
    assert [int(t.token_id) for t in req.lifecycle.skill_state.tokens] == [11, 12, 13]
    completed = sched.pop_completed()
    assert [c.request_id for c in completed] == [0]
    assert completed[0].finish_reason == "length"


def test_spec_drain_pipeline_noop_when_no_pending_spec() -> None:
    """``_drain_pipeline`` is a safe no-op when no spec step is outstanding."""
    dec = _FakeDecoder(n_rows=1, first_tokens={0: 11}, plans={0: [[12]]})
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    _enqueue(sched, 0, prompt_len=3, max_new=8)
    sched._spec_admit()
    assert sched._pending_spec is None

    # No launched-but-uncommitted step: drain must not raise or launch new work.
    sched._drain_pipeline()
    assert sched._pending_spec is None
    assert dec.retired == []
def test_spec_step_recomputes_stateful_skill_mask_per_step() -> None:
    """Regression for codex P1 @ L727: stateful skill masks refresh per step.

    For a STATEFUL constrained skill the allowed-token set evolves per committed
    token (point toggles ``[coord, eos]`` <-> ``[coord]`` after each coordinate;
    detect cycles x->y->size). The spec path used to snapshot the mask ONCE at
    ``admit`` and reuse it for every later position, so ``decoder.step`` -- which
    only saw ``SequenceState`` -- could never see the mask change after a
    ``consume_step``, allowing e.g. EOS where ``y`` is required.

    This drives a toggling skill state (mirroring point: allowed set flips every
    committed token) through several macro-steps with ONE token committed per
    step (so at most one stateful transition happens per committed run -- the
    regime where a single per-step mask is exact, matching the non-spec
    one-token-per-step refresh) and asserts the mask the scheduler hands to each
    ``decoder.step`` is RECOMPUTED from the live skill state -- it alternates
    with the state instead of being frozen at the admit-time value.
    """
    COORD, EOS = 7, 8

    class _TogglingState(_RecordingState):
        def __init__(self, request: GenerationRequest) -> None:
            super().__init__(request)
            self._awaiting_y = False

        def consume_step(self, runtime: object, step: DecodeStep) -> None:
            # Mirror PointSkillState: every committed token flips the stage.
            self.append_token(step.token)
            self._awaiting_y = not self._awaiting_y

        def allowed_token_ids(self, runtime: object):
            # awaiting y -> only a coordinate may follow; else coord or EOS.
            return [COORD] if self._awaiting_y else [COORD, EOS]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 19},
        # One token committed per macro-step -> one stateful transition per run.
        plans={0: [[20], [21], [22], [23]]},
        # EOS exists in the skill mask but is never the scripted token, so the
        # request runs long enough to observe several mask refreshes.
    )
    rt = _spec_runtime(dec, eos_id=EOS)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=20)
    toggling = _TogglingState(req)
    req.lifecycle.skill_state = toggling
    req.skill_state = toggling

    assert sched._spec_admit() is True
    # At admit the state was ``_awaiting_y=False`` -> the admit-time (stale)
    # snapshot allows [COORD, EOS]. The bug reused exactly this for all later
    # positions. ``admit`` staged token 19, flipping the state to awaiting_y.
    assert list(dec.admit_kwargs[0]["allowed_token_ids"]) == [COORD, EOS]

    # Drive four macro-steps so each commits one scripted token and flips state.
    for _ in range(4):
        sched._spec_decode_step()

    # Each ``step`` got a single-row mask (the one active sequence). Pull the
    # per-call allowed set for that row.
    per_step = [call[0] for call in dec.step_allowed]

    # The mask handed to step 0 reflects the POST-admit state (awaiting_y=True)
    # -> [COORD] -- already DIFFERENT from the admit-time [COORD, EOS], proving
    # it is recomputed, not the frozen admit snapshot.
    assert per_step[0] == [COORD]
    assert per_step[0] != list(dec.admit_kwargs[0]["allowed_token_ids"])
    # ... and it tracks the toggling skill state across subsequent steps: each
    # committed token flips the allowed set, so it alternates per macro-step.

    assert per_step[:4] == [[COORD], [COORD, EOS], [COORD], [COORD, EOS]]


def test_spec_step_caps_stateful_mask_to_one_committed_token_per_step() -> None:
    """Regression for codex P1 @ L958: cap stateful masks within a macro-step.

    The L727 fix re-queries the per-row skill mask each macro-step, but a single
    ``decoder.step`` mask still covers a *variable* committed run: ``_commit_spec``
    can stage MULTIPLE accepted tokens before the next ``consume_step`` refreshes
    the mask. For a per-token-STATEFUL skill (detect cycles x->y->size; point
    toggles coord/eos) a >1-token run would verify its 2nd..Nth positions under
    the stale 1st-position mask -- e.g. a detect run could accept a ``coord``
    where ``size`` is required, or suppress the required ``size_id``.

    The scheduler now sends ``commit_caps[i] = 1`` for a stateful-masked row, so
    that row commits exactly one token per macro-step (one constraint transition
    per run -- the regime where the single per-step mask IS exact) and is
    re-masked from the now-current skill state on the next step. This scripts a
    decoder that WOULD commit the whole x/y/size run in one step and asserts the
    cap holds it to one token per step, with the mask the scheduler hands each
    step advancing in lockstep with the single committed token -- so every
    committed position is constrained by ITS OWN stage, never the whole run under
    the first-position mask.
    """
    COORD, SIZE, EOS = 7, 8, 9

    class _DetectLikeState(_RecordingState):
        # Mirror DetectSkillState: allowed set cycles x -> y -> size per token.
        def __init__(self, request: GenerationRequest) -> None:
            super().__init__(request)
            self._stage = "x"

        def consume_step(self, runtime: object, step: DecodeStep) -> None:
            self.append_token(step.token)
            if self._stage == "x":
                self._stage = "y"
            elif self._stage == "y":
                self._stage = "size"
            else:
                self._stage = "x"

        def allowed_token_ids(self, runtime: object):
            if self._stage == "x":
                return [COORD, EOS]
            if self._stage == "y":
                return [COORD]
            return [SIZE]

    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 19},
        # The decoder OFFERS a 3-token run (x, y, size) in a single macro-step.
        # Without the cap this whole run would commit under the first-position
        # mask; the cap must split it to one token per step.
        plans={0: [[20, 21, 22]]},
    )
    rt = _spec_runtime(dec, eos_id=EOS)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=20)
    detect = _DetectLikeState(req)
    req.lifecycle.skill_state = detect
    req.skill_state = detect

    assert sched._spec_admit() is True
    # ``admit`` sampled token 19 and advanced the stage x -> y, so the first
    # ``step`` is launched under the y mask. The admit-time snapshot was the x
    # mask -- which the bug would have reused for the whole run.
    assert list(dec.admit_kwargs[0]["allowed_token_ids"]) == [COORD, EOS]

    # Four macro-steps drive the depth-1 pipeline so the 3-token run commits ONE
    # token per step (commit lands one tick after launch): step1 launches /
    # step2..4 each commit one of 20, 21, 22.
    for _ in range(4):
        sched._spec_decode_step()

    # The scheduler signalled a one-token cap for this stateful row every step.
    assert dec.step_commit_caps[:4] == [[1], [1], [1], [1]]

    # Exactly one token committed per macro-step -- the run was NOT swallowed
    # under a single mask: tokens 20, 21, 22 staged across the steps, in order,
    # after the admit token 19.
    assert [int(t.token_id) for t in req.lifecycle.skill_state.tokens] == [
        19,
        20,
        21,
        22,
    ]

    # ...and the mask the scheduler hands each launch advances with the single
    # committed token (y -> size -> x -> y), so the token committed by each
    # step's commit was drafted/verified under ITS OWN stage's mask -- not all
    # three positions under the first (x) mask.
    per_step = [call[0] for call in dec.step_allowed]
    assert per_step[:4] == [[COORD], [SIZE], [COORD, EOS], [COORD]]


def test_spec_step_does_not_cap_unconstrained_or_constant_mask_row() -> None:
    """A non-stateful row keeps the full multi-token speculative accept.

    The STATEFUL-mask cap is *only* for rows whose mask can change within a run.
    An unconstrained row (no allowed/suppressed set) and a row that declares its
    mask constant via ``mask_is_stateful = False`` must not be throttled to one
    token per step, so they still commit their whole accepted run in a single
    macro-step (the throughput win spec decode exists for). With a generous
    ``max_new_tokens`` the only ``commit_caps`` entry is the remaining-budget cap
    (well above the run length, so it does not truncate) -- never the stateful
    ``1``. Asserts the scheduler forwards the loose budget cap (NOT the stateful
    cap) for both rows and the full 3-token run commits in a single step.
    """

    class _ConstantMaskState(_RecordingState):
        # A constant suppression set (like caption's ``[answer_id]``): present
        # every step but never transitions, so the per-step mask is already exact
        # for a multi-token run. Declares itself non-stateful so it is uncapped.
        mask_is_stateful = False

        def suppressed_token_ids(self, runtime: object):
            return [42]

    # Row 0: unconstrained (no mask). Row 1: constant-mask, declared not stateful.
    dec = _FakeDecoder(
        n_rows=2,
        first_tokens={0: 11, 1: 21},
        plans={0: [[12, 13, 14]], 1: [[22, 23, 24]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    # max_new far above the run length so the remaining-budget cap never bites
    # (isolating the stateful-cap behaviour this test is about).
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=20)
    r1 = _enqueue(sched, 1, prompt_len=3, max_new=20)
    constant = _ConstantMaskState(r1)
    r1.lifecycle.skill_state = constant
    r1.skill_state = constant

    assert sched._spec_admit() is True
    sched._spec_decode_step()  # launch macro-step over both rows
    sched._spec_decode_step()  # commit it

    # Neither row was capped by the STATEFUL cap (which would be ``1``): the only
    # cap is the loose remaining budget (max_new 20 - 1 staged at admit == 19),
    # far above the 3-token run, so the full run still commits this step.
    assert dec.step_commit_caps[0] == [19, 19]
    # Each row committed its WHOLE 3-token run in the single macro-step (the
    # admit token plus the full accepted run -- not throttled to one/step).
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [
        11,
        12,
        13,
        14,
    ]
    assert [int(t.token_id) for t in r1.lifecycle.skill_state.tokens] == [
        21,
        22,
        23,
        24,
    ]


def test_spec_real_caption_skill_not_capped_while_stateful_skills_are() -> None:
    """The REAL moondream skills carry the right ``mask_is_stateful`` verdict.

    Regression for codex P2 @ scheduler ``_mask_is_stateful``: caption's
    ``suppressed_token_ids`` returns a CONSTANT set (``[answer_id]`` on
    moondream2) at every position, so its per-step mask is already exact for a
    whole committed run -- it must NOT be capped to one token per macro-step.
    The behavioural fallback ("any active constraint is stateful") would wrongly
    cap it, so ``CaptionSkillState`` declares ``mask_is_stateful = False``; this
    pins that declaration on the real class (the generic ``_ConstantMaskState``
    test above would still pass even if the real skill forgot it).

    Conversely the genuinely stateful skills -- detect (cycles x->y->size) and
    point (toggles coord/eos) -- whose allowed set changes per committed token
    must STAY capped (``commit_caps = 1``), so this asserts the scheduler keeps
    throttling them. Drives the scheduler's ``_build_spec_step_masks`` directly
    over real skill states attached to enqueued rows.
    """
    from kestrel.models.moondream.skills.caption import (
        CaptionRequest,
        CaptionSkill,
        CaptionSkillState,
    )
    from kestrel.models.moondream.skills.detect import (
        DetectRequest,
        DetectSkill,
        DetectSkillState,
    )
    from kestrel.models.moondream.skills.point import (
        PointRequest,
        PointSkill,
        PointSkillState,
    )

    # Minimal runtime/template the real mask queries read: caption needs
    # ``model_name`` + ``answer_id``; detect needs coord/eos/size; point needs
    # coord/eos. The ids are arbitrary-but-distinct.
    COORD, EOS, SIZE, ANSWER = 3, 4, 5, 6
    runtime = SimpleNamespace(
        model_name="moondream2",
        prompt_template=SimpleNamespace(
            answer_id=ANSWER, coord_id=COORD, eos_id=EOS, size_id=SIZE
        ),
    )

    dec = _FakeDecoder(
        n_rows=3,
        first_tokens={0: 10, 1: 20, 2: 30},
        plans={0: [[]], 1: [[]], 2: [[]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    # The scheduler queries the REAL skills against the stub ``runtime`` above,
    # not the bare ``_spec_runtime`` fake -- point ``_build_spec_step_masks`` at it.
    sched.runtime = runtime

    # Generous budget so the remaining-budget cap never bites: the only cap that
    # can appear is the STATEFUL cap (``1``). caption => no stateful cap; detect
    # / point => stateful cap of ``1``.
    r_cap = _enqueue(sched, 0, prompt_len=3, max_new=1000)
    r_det = _enqueue(sched, 1, prompt_len=3, max_new=1000)
    r_pt = _enqueue(sched, 2, prompt_len=3, max_new=1000)

    cap_state = CaptionSkillState(
        CaptionSkill(),
        r_cap,
        CaptionRequest(length="normal", image=None, stream=False),
    )
    det_state = DetectSkillState(
        DetectSkill(),
        r_det,
        DetectRequest(object="cat", image=None, stream=False, max_objects=10),
    )
    pt_state = PointSkillState(
        PointSkill(),
        r_pt,
        PointRequest(object="cat", image=None, stream=False),
    )
    r_cap.lifecycle.skill_state = cap_state
    r_det.lifecycle.skill_state = det_state
    r_pt.lifecycle.skill_state = pt_state

    seqs = [r_cap.lifecycle, r_det.lifecycle, r_pt.lifecycle]

    # Caption: constant ``[answer_id]`` suppression, declared non-stateful.
    assert cap_state.suppressed_token_ids(runtime) == [ANSWER]
    assert sched._mask_is_stateful(r_cap.lifecycle, None, [ANSWER]) is False
    # Detect / point: their allowed set evolves per committed token -> stateful.
    assert det_state.allowed_token_ids(runtime) == [COORD, EOS]
    assert pt_state.allowed_token_ids(runtime) == [COORD, EOS]
    assert sched._mask_is_stateful(r_det.lifecycle, [COORD, EOS], None) is True
    assert sched._mask_is_stateful(r_pt.lifecycle, [COORD, EOS], None) is True

    _allowed, _suppressed, commit_caps = sched._build_spec_step_masks(seqs)

    # Caption row is NOT capped to one token: its only cap is the loose remaining
    # budget (1000 - 0 staged == 1000), never the stateful ``1``. The constant
    # suppression is still forwarded so masking stays correct.
    assert commit_caps[0] == 1000
    assert list(_suppressed[0]) == [ANSWER]
    # Detect and point rows stay capped at one committed token per macro-step.
    assert commit_caps[1] == 1
    assert commit_caps[2] == 1


def test_spec_zombie_lifecycle_marked_completed_before_retire() -> None:
    """Regression for codex P2 @ L1012: a finished spec row ends COMPLETED, retired once.

    A finishing spec request must end ``COMPLETED`` (not stuck ``FINALIZING``)
    with its decoder row retired exactly once -- the invariant this guards (the
    bug left a completed request in ``FINALIZING`` forever after its row was
    retired, because the zombie commit retired the row but never transitioned the
    lifecycle).

    L997 fix changes only the *timing*, not this invariant: a row finalized by the
    commit (it hit ``max_new_tokens``) is now dropped from the next launch and
    retired SYNCHRONOUSLY inside the finishing ``_spec_decode_step`` -- the
    ``transition(COMPLETED)`` + ``_retire_spec_row`` happen in the filter right
    after the commit, NOT a tick later via an optimistic follow-up (zombie)
    commit. So there is no longer a FINALIZING-with-pending window: the request
    goes ``FINALIZING -> COMPLETED -> retired`` within one step. This asserts the
    end state (COMPLETED, retired exactly once, no leak) and that the row was
    never left FINALIZING after its retire.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12, 13]]},  # admit(1) + 2 == max_new=3 -> finishes mid-pipeline
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    req = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()

    # Drive to quiescence, asserting the row is never left FINALIZING while still
    # holding the decoder row (the bug's stuck state). Under L997 the finishing
    # step transitions COMPLETED + retires in one go, so a retired row is never
    # observed in FINALIZING.
    for _ in range(10):
        if not sched._spec_decode_step():
            break
        if req.lifecycle.phase == RequestPhase.FINALIZING:
            # If ever FINALIZING, its row must still be live (not yet retired) --
            # i.e. the lifecycle is never stuck FINALIZING after a retire.
            assert req.lifecycle.state.batch_idx in rt.active_sequences

    # End state: lifecycle COMPLETED, row retired exactly once, nothing leaked.
    assert dec.retired == [0]
    assert dec.retired.count(0) == 1
    assert req.lifecycle.phase == RequestPhase.COMPLETED
    assert req.lifecycle.finished is True
    assert rt.active_sequences == {}
    assert sched._pending_spec is None
    assert [c.finish_reason for c in sched.pop_completed()] == ["length"]


def test_spec_step_caps_commit_to_remaining_max_new_tokens() -> None:
    """Regression for codex P1 @ L2079: cap the committed run to the budget.

    For an unconstrained row the stateful-mask cap is ``None``, so before the fix
    ``commit_caps[i]`` stayed ``None`` even when the request had fewer than the
    decoder's offered run left. ``_commit_spec`` stops *staging* at
    ``max_new_tokens``, but the decoder advances its KV / proposer state through
    the WHOLE accepted run, so the row (and the optimistic depth-1 zombie step
    launched off it) runs past the request's target length while the surplus
    tokens are silently dropped -- over-generation the non-spec path never does
    (it stops sampling exactly at the limit).

    The scheduler now folds the remaining output budget
    (``max_new_tokens - token_count``) into ``commit_caps``. This scripts a
    decoder that WOULD commit a 3-token run in the final macro-step of a request
    with only 2 new tokens left, and asserts the cap holds the committed run to
    exactly the remaining budget: the request finishes at exactly
    ``max_new_tokens`` (``length``), no surplus token is ever staged, and the
    decoder advance stays in lockstep with the (truncated) committed run.
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        # The decoder OFFERS a 3-token run, but admit already staged 1 of the 3
        # allowed tokens, so only 2 remain. Without the budget cap all 3 would
        # advance the decoder; the cap must hold the committed run to 2.
        plans={0: [[12, 13, 14]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()  # stages token 11 -> token_count == 1, budget == 2

    len_before = r0.lifecycle.state.length
    while sched._spec_decode_step():
        pass

    # The first launch saw budget == 2 (max_new 3 - 1 staged) and, with no skill
    # mask, forwarded that budget as the row's commit cap (NOT None).
    assert dec.step_commit_caps[0] == [2]
    # Exactly the remaining budget was staged: token 14 was NEVER committed, so
    # the request stops at exactly max_new_tokens with no over-generation.
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [11, 12, 13]
    # The decoder's KV length advanced by the committed (truncated) run only -- it
    # did not run past the target length through the dropped token.
    assert r0.lifecycle.state.length - len_before == 2
    # Finished at the limit, length-finished, row retired exactly once.
    assert r0.lifecycle.finished is True
    assert dec.retired == [0]
    assert r0.lifecycle.state.batch_idx not in rt.active_sequences
    assert sched._pending_spec is None
    assert [c.finish_reason for c in sched.pop_completed()] == ["length"]


def test_spec_step_rolls_back_refs_when_launch_raises() -> None:
    """Regression for codex P2 @ L941: roll back reserved refs on launch failure.

    ``_spec_decode_step`` reserves an ``inflight_refs`` for every active row
    BEFORE committing the previous macro-step and launching this one. If the
    commit or ``decoder.step`` raises before ``_pending_spec`` is installed, those
    reserved refs were owned by nothing -- no future ``_commit_spec`` decrements
    them -- so a later finish stays stuck in ``FINALIZING`` and the spec row /
    adapter slot leaks. The non-spec launch path (``launch_decode_step``) rolls
    back its reservation for exactly this reason; the spec path now does too. This
    scripts a decoder whose SECOND ``step`` raises and asserts the reservation is
    undone (no phantom ref), ``_pending_spec`` stays clear, and the row is left in
    a clean, still-running state rather than wedged.
    """

    class _RaisingDecoder(_FakeDecoder):
        def __init__(self, *, raise_on_step: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self._raise_on_step = raise_on_step
            self._step_calls = 0

        def step(self, states, **kwargs):
            self._step_calls += 1
            if self._step_calls == self._raise_on_step:
                raise RuntimeError("simulated decoder.step CUDA failure")
            return super().step(states, **kwargs)

    dec = _RaisingDecoder(
        raise_on_step=2,
        n_rows=1,
        first_tokens={0: 11},
        plans={0: [[12], [13]]},  # max_new large -> row keeps running across steps
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=20)
    sched._spec_admit()  # stages token 11

    # Step 1 launches cleanly: reserves a ref and installs the pending step.
    assert sched._spec_decode_step() is True
    assert sched._pending_spec is not None
    assert r0.lifecycle.inflight_refs == 1

    # Step 2 reserves a SECOND ref for this launch, commits step 1 (which
    # consumes step 1's ref and stages token 12), and then raises inside
    # ``decoder.step`` -- after step 2's ref was reserved but before
    # ``_pending_spec`` is reinstalled.
    try:
        sched._spec_decode_step()
        raise AssertionError("expected decoder.step to raise")
    except RuntimeError as exc:
        assert "simulated decoder.step CUDA failure" in str(exc)

    # Step 2's reservation was rolled back and step 1's was consumed by its
    # commit, so the row is left with ZERO in-flight refs -- no phantom ref. (The
    # bug would leave it at 1: step 2's reservation orphaned, since no future
    # ``_commit_spec`` would ever decrement it, wedging a later finish in
    # ``FINALIZING`` and leaking the pool/adapter slot.) No pending step dangles.
    assert r0.lifecycle.inflight_refs == 0
    assert sched._pending_spec is None
    # The commit that ran before the raise still took effect (token 12 staged),
    # and the row is left running -- not finished/finalized -- so a clean retry
    # could proceed without a wedged FINALIZING row or leaked pool slot.
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [11, 12]
    assert r0.lifecycle.finished is False
    assert r0.lifecycle.finalized is False
    assert r0.lifecycle in sched.running
    assert r0.lifecycle.state.batch_idx in rt.active_sequences


def test_spec_step_does_not_launch_row_finalized_by_commit() -> None:
    """Regression for codex P1 @ L997: a row finalized by the commit is not relaunched.

    ``_spec_decode_step`` snapshots ``active`` (the launch membership) BEFORE
    committing the previous macro-step. That commit can FINALIZE a row -- the row
    hit ``max_new_tokens`` so its last allowed token was ``length``-committed --
    which sets ``finalized`` and drops it from ``running`` but cannot retire it
    yet because this tick reserved an ``inflight_refs`` for the (about-to-launch)
    step. Because the snapshot predates the commit, that now-finalized row was
    still in ``active``; ``_build_spec_step_masks`` then returns a ``None`` commit
    cap for it (finalized rows come back unconstrained), so the scheduler asked
    ``decoder.step`` to run ANOTHER macro-step on a row whose
    ``SequenceState.length`` already equals ``max_length`` -- a real spec decoder
    that reserves only the request budget can advance/write past the row or raise.
    The non-spec launch set never includes a finalized row (``_can_dispatch``
    returns False for it); the spec path must match that by filtering the rows the
    commit just finalized out of the launch (and immediately retiring their
    reserved ref).

    This scripts a single row that finishes by ``max_new_tokens`` and asserts the
    decoder is launched EXACTLY ONCE (the legitimate macro-step that lands the
    final token) -- never a second time on the finalized row -- that the row's
    ``length`` never exceeds ``max_length``, and that the row retires cleanly with
    the ref count balanced (no phantom ref / pool leak).
    """
    dec = _FakeDecoder(
        n_rows=1,
        first_tokens={0: 11},
        # admit stages 1; this single macro-step commits 2 more == max_new=3, so
        # the row FINALIZES at the commit that runs on the NEXT tick (depth-1).
        plans={0: [[12, 13]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    r0 = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()  # stages token 11 -> token_count == 1

    max_len = r0.lifecycle.state.max_length
    # Drive the depth-1 pipeline to quiescence, asserting on every tick that the
    # row's KV length never runs past its absolute limit (the bug would launch a
    # macro-step off a row already at ``max_length``).
    while sched._spec_decode_step():
        assert r0.lifecycle.state.length <= max_len

    # The decoder.step was invoked exactly ONCE -- the one launch that commits the
    # final token. The pre-fix code launched a SECOND macro-step on the now-
    # finalized row (its tick-2 ``active`` snapshot still held the row the commit
    # had just finished), which this asserts never happens: row 0 appears in
    # exactly one launch and there is no extra launch after it finalized.
    assert dec.step_rows == [[0]]
    assert sum(1 for rows in dec.step_rows if 0 in rows) == 1

    # The request finished at exactly max_new_tokens (length), the row retired
    # exactly once, the ref count is balanced (no phantom ref), and nothing leaks.
    assert [int(t.token_id) for t in r0.lifecycle.skill_state.tokens] == [11, 12, 13]
    assert r0.lifecycle.finished is True
    assert r0.lifecycle.phase == RequestPhase.COMPLETED
    assert r0.lifecycle.inflight_refs == 0
    assert dec.retired == [0]
    assert dec.retired.count(0) == 1
    assert r0.lifecycle.state.batch_idx not in rt.active_sequences
    assert sched._pending_spec is None
    assert [c.finish_reason for c in sched.pop_completed()] == ["length"]


def test_spec_step_excludes_finalized_row_but_launches_continuing_row() -> None:
    """L997, multi-row: drop the finalized row from the launch, keep the continuer.

    The interesting case the single-row test cannot show: in one macro-step row A
    is committed to its ``max_new_tokens`` (finalizes) while row B keeps running.
    ``active`` is snapshotted before the commit, so BOTH rows are in it; after the
    commit finalizes A, the next ``decoder.step`` must launch ONLY B -- never A
    (A's ``state.length`` already equals ``max_length``; relaunching it is the
    unsafe over-advance L997 fixes) -- and A must retire synchronously while B's
    refs/launch continue undisturbed. Asserts (a) A appears in exactly the launch
    that lands its final token and never after it finalizes, (b) B is launched on
    every step it runs, (c) both rows retire exactly once with no leak and end
    COMPLETED.
    """
    dec = _FakeDecoder(
        n_rows=2,
        first_tokens={0: 11, 1: 22},
        plans={
            0: [[12, 13]],          # admit(1) + 2 == max_new=3 -> A finalizes
            1: [[23], [24], [999]],  # B keeps running, then EOS
        },
    )
    rt = _spec_runtime(dec, eos_id=999)
    sched = _make_scheduler(rt)
    rA = _enqueue(sched, 0, prompt_len=3, max_new=3)
    rB = _enqueue(sched, 1, prompt_len=2, max_new=20)
    sched._spec_admit()

    rowA = rA.lifecycle.state.batch_idx
    rowB = rB.lifecycle.state.batch_idx
    maxlenA = rA.lifecycle.state.max_length
    # Drive to quiescence, asserting on every tick that A's KV length never runs
    # past its limit (the bug would relaunch A off a row already at max_length).
    finalized_tick = None
    for tick in range(20):
        if not sched._spec_decode_step():
            break
        assert rA.lifecycle.state.length <= maxlenA
        if rA.lifecycle.finished and finalized_tick is None:
            finalized_tick = len(dec.step_rows)

    # A finalized; from the launch AT/AFTER its finalizing commit it is excluded.
    assert finalized_tick is not None
    # A was launched exactly once (the macro-step that lands its final token) and
    # never appears in a launch again -- the L997 fix dropped it from ``active``.
    a_launches = [rows for rows in dec.step_rows if 0 in rows]
    assert len(a_launches) == 1
    assert a_launches[0] == [0, 1]  # the single shared launch, with B alongside
    # Every launch after A finalized contains ONLY B (never A).
    assert all(rows == [1] for rows in dec.step_rows if rows != [0, 1])

    # B was launched on each of its running steps (1 shared + its solo steps).
    assert sum(1 for rows in dec.step_rows if 1 in rows) >= 3

    # Both rows finished cleanly: retired exactly once, COMPLETED, no leak.
    assert sorted(dec.retired) == [0, 1]
    assert dec.retired.count(0) == 1
    assert dec.retired.count(1) == 1
    assert rA.lifecycle.phase == RequestPhase.COMPLETED
    assert rB.lifecycle.phase == RequestPhase.COMPLETED
    assert rA.lifecycle.inflight_refs == 0
    assert rB.lifecycle.inflight_refs == 0
    assert rowA not in rt.active_sequences
    assert rowB not in rt.active_sequences
    assert sched._pending_spec is None
    assert sched.has_pending_work() is False
    reasons = sorted(c.finish_reason for c in sched.pop_completed())
    assert reasons == ["length", "stop"]  # A by length (max_new), B by eos


def test_spec_admit_zero_token_request_bypasses_saturated_batch() -> None:
    """Regression for the codex finding @ scheduler.py:608.

    A ``max_new_tokens <= 0`` request finalizes as ``"length"`` WITHOUT calling
    ``decoder.admit`` and WITHOUT consuming a spec pool row -- mirroring the
    non-spec ``_launch_prefill_step`` zero-length branch. That finalize therefore
    needs neither a free pool row nor a free running-batch slot.

    Before the fix, the zero-token branch lived INSIDE the admission loop whose
    guard is ``while decoder.free_slots > 0 and len(self.running) < max_running``.
    So a zero-token request queued behind a saturated spec batch -- either the
    pool is full (``decoder.free_slots == 0``) or the running set is already at
    ``max_batch_size`` -- never reached the branch and stalled until an unrelated
    row retired, diverging from its documented no-row contract. The fix drains
    launchable zero-token requests in a pre-pass that is NOT gated by that
    capacity guard. This covers BOTH saturating conditions.
    """
    # --- Condition A: running already at max_batch_size (pool still has rows) ---
    # ``_spec_runtime`` is max_batch_size=4 / max_batch_slots=8 over an 8-row
    # pool, so the batch cap stops real admissions at 4 while 4 pool rows stay
    # free. A zero-token request must still finalize despite ``running`` being
    # saturated -- this is exactly the ``len(self.running) < max_running`` clause
    # added in 6d27414 that the finding flags.
    n = 8
    dec = _FakeDecoder(
        n_rows=n,
        first_tokens={r: 10 + r for r in range(n)},
        plans={r: [[100 + r]] for r in range(n)},
    )
    rt = _spec_runtime(dec)
    assert rt.max_batch_size == 4 and dec.free_slots == 8
    sched = _make_scheduler(rt)
    # 4 real requests saturate the running batch at max_batch_size.
    for rid in range(4):
        _enqueue(sched, rid, prompt_len=3, max_new=8)
    # A zero-token request queued BEHIND the saturating batch.
    rz = _enqueue(sched, 99, prompt_len=5, max_new=0)

    assert sched._spec_admit() is True
    # Running saturated at the cap; pool NOT full (4 rows still free).
    assert len(dec.admitted) == rt.max_batch_size
    assert len(sched.running) == rt.max_batch_size
    assert dec.free_slots == n - rt.max_batch_size
    # The zero-token request finalized WITHOUT admit and WITHOUT a pool row:
    # it is no longer waiting, never entered ``running``, and ``decoder.admit``
    # was never called for it (row 99's id is not among the admitted pool rows;
    # admitted rows are the 4 real ones [0..3]).
    assert 99 not in dec.admitted
    # No spec POOL ROW is assigned (``decoder.admit`` never ran), but a
    # metrics-only ``SequenceState`` is attached so ``build_metrics`` reports the
    # KV prompt length; it is unbacked (``batch_idx == -1``).
    assert rz.lifecycle.sequence_state is not None
    assert rz.lifecycle.sequence_state.batch_idx == -1
    assert rz.lifecycle.sequence_state.prompt_length == rz.prompt_length
    assert all(r.request_id != 99 for r in sched.waiting)
    assert all(lc.request.request_id != 99 for lc in sched.running)
    completed = {c.request_id: c for c in sched.pop_completed()}
    assert 99 in completed
    assert completed[99].finish_reason == "length"
    assert completed[99].tokens == []

    # --- Condition B: pool fully drained (decoder.free_slots == 0) ---
    # Size the runtime so the batch cap does NOT block draining the pool: a
    # 2-row pool with max_batch_size=4 lets both rows admit, leaving
    # ``free_slots == 0``. A zero-token request behind the full pool must still
    # finalize immediately.
    dec2 = _FakeDecoder(
        n_rows=2,
        first_tokens={0: 11, 1: 22},
        plans={0: [[12]], 1: [[23]]},
    )
    rt2 = FakeRuntime(max_batch_size=4, max_batch_slots=6)
    rt2.spec = SpecDecodeCaps(
        proposer=SimpleNamespace(num_speculative_tokens=3, num_lookahead_tokens=4),
        decoder=dec2,
    )
    rt2.prompt_template = SimpleNamespace(eos_id=999)
    sched2 = _make_scheduler(rt2)
    _enqueue(sched2, 0, prompt_len=3, max_new=8)
    _enqueue(sched2, 1, prompt_len=3, max_new=8)
    rz2 = _enqueue(sched2, 77, prompt_len=4, max_new=0)

    assert sched2._spec_admit() is True
    # Pool fully consumed by the two real admissions.
    assert dec2.free_slots == 0
    assert dec2.admitted == [0, 1]
    assert len(sched2.running) == 2
    # The zero-token request finalized despite the empty pool: not waiting, not
    # running, never admitted.
    assert 77 not in dec2.admitted
    assert rz2.lifecycle.sequence_state is not None
    assert rz2.lifecycle.sequence_state.batch_idx == -1
    assert rz2.lifecycle.sequence_state.prompt_length == rz2.prompt_length
    assert all(r.request_id != 77 for r in sched2.waiting)
    completed2 = {c.request_id: c for c in sched2.pop_completed()}
    assert 77 in completed2
    assert completed2[77].finish_reason == "length"
    assert completed2[77].tokens == []


def test_spec_inactive_to_active_reasoning_mask_is_capped() -> None:
    """A reasoning mask that is None at step start but forces a prefix mid-run is capped.

    Regression for the codex P1 @ scheduler ``_mask_is_stateful``: the
    behavioural fallback inspects the constraint at the committed run's FIRST
    position only, so it cannot see a mask that is INACTIVE at step start but
    transitions to ACTIVE within the run. Reasoning skills do exactly that --
    while collecting reasoning, ``QuerySkillState`` (non-moondream2) and
    ``ChatSkillState`` expose NO constraint (both ``allowed_token_ids`` and
    ``suppressed_token_ids`` return ``None``); once the model emits ``answer_id``
    they force ``post_reasoning_prefix`` one id at a time. If a spec macro-step
    committed ``answer_id`` plus following tokens under that None-at-start mask,
    the post-boundary tokens would be accepted WITHOUT the required prefix
    constraint. So both states declare ``mask_is_stateful = True`` and the
    scheduler must cap them to one committed token per step EVEN THOUGH both
    masks read ``None`` this step -- the verdict the bare fallback would miss.

    The contrast row is caption: under the same non-moondream2 runtime its mask
    is also unconstrained (no allowed/suppressed), and it declares
    ``mask_is_stateful = False``, so it must STAY uncapped. Same None/None
    step-start signature, opposite cap verdict -- proving the declaration (not
    the fallback) decides.
    """
    from kestrel.models.moondream.skills.caption import (
        CaptionRequest,
        CaptionSkill,
        CaptionSkillState,
    )
    from kestrel.models.moondream.skills.query import (
        QueryRequest,
        QuerySkill,
        QuerySkillState,
    )
    from kestrel.skills.chat import (
        ChatMessage,
        ChatContentPart,
        ChatRequest,
        ChatSkill,
        ChatSkillState,
    )

    # A NON-moondream2 runtime: this is the case the bug lived in -- query's
    # ``suppressed_token_ids`` returns ``None`` off-model, so a reasoning row's
    # mask is fully None at step start. ``answer_id`` is only read once a step is
    # consumed (not on the bare mask query this test drives), but provide it for
    # completeness.
    ANSWER = 7
    runtime = SimpleNamespace(
        model_name="qwen3",
        prompt_template=SimpleNamespace(answer_id=ANSWER),
    )

    dec = _FakeDecoder(
        n_rows=3,
        first_tokens={0: 10, 1: 20, 2: 30},
        plans={0: [[]], 1: [[]], 2: [[]]},
    )
    rt = _spec_runtime(dec)
    sched = _make_scheduler(rt)
    # Query the REAL skills against the stub ``runtime`` above.
    sched.runtime = runtime

    # Generous budget so the remaining-budget cap never bites: the only cap that
    # can appear is the STATEFUL cap (``1``).
    r_query = _enqueue(sched, 0, prompt_len=3, max_new=1000)
    r_chat = _enqueue(sched, 1, prompt_len=3, max_new=1000)
    r_cap = _enqueue(sched, 2, prompt_len=3, max_new=1000)

    # Reasoning ON for query + chat: collecting reasoning, no constraint yet, but
    # the mask WILL force a prefix after ``answer_id`` -- the transition the
    # fallback can't see.
    query_state = QuerySkillState(
        QuerySkill(),
        r_query,
        QueryRequest(
            question="q", image=None, reasoning=True, stream=False, spatial_refs=None
        ),
    )
    chat_state = ChatSkillState(
        ChatSkill(),
        r_chat,
        ChatRequest(
            messages=(
                ChatMessage(role="user", parts=(ChatContentPart(text="hi"),)),
            ),
            images=(),
            reasoning=True,
            stream=False,
        ),
        post_reasoning_prefix=[ANSWER],
        turn_end_ids=[],
    )
    cap_state = CaptionSkillState(
        CaptionSkill(),
        r_cap,
        CaptionRequest(length="normal", image=None, stream=False),
    )
    r_query.lifecycle.skill_state = query_state
    r_chat.lifecycle.skill_state = chat_state
    r_cap.lifecycle.skill_state = cap_state

    seqs = [r_query.lifecycle, r_chat.lifecycle, r_cap.lifecycle]

    # The exact bug precondition: every row's mask reads None/None THIS step.
    assert query_state.allowed_token_ids(runtime) is None
    assert query_state.suppressed_token_ids(runtime) is None
    assert chat_state.allowed_token_ids(runtime) is None
    assert chat_state.suppressed_token_ids(runtime) is None
    assert cap_state.suppressed_token_ids(runtime) is None
    # The bare behavioural fallback would call all three non-stateful (both masks
    # empty); the declarations override that for the reasoning rows only.
    assert (bool(None) or bool(None)) is False  # what the fallback alone returns
    assert sched._mask_is_stateful(r_query.lifecycle, None, None) is True
    assert sched._mask_is_stateful(r_chat.lifecycle, None, None) is True
    assert sched._mask_is_stateful(r_cap.lifecycle, None, None) is False

    _allowed, _suppressed, commit_caps = sched._build_spec_step_masks(seqs)

    # Reasoning rows capped to one committed token per macro-step despite the
    # None-at-start mask; caption keeps the full multi-token accept (its only cap
    # is the loose remaining budget).
    assert commit_caps[0] == 1
    assert commit_caps[1] == 1
    assert commit_caps[2] == 1000
