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

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def admit(
        self,
        state,
        prompt_token_ids,
        *,
        image=None,
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

    def step(self, states):
        tokens, accepts, logprobs = [], [], []
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
    )
    lc = RequestLifecycle(
        request=req, skill_state=skill_state or _RecordingState(req)
    )
    lc.has_image = image is not None
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

    def materialize_spec_tokens(token_ids_cpu, sequences, batch_idx, side_values):
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

    def materialize_spec_tokens(token_ids_cpu, sequences, batch_idx, side_values):
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


def test_spec_side_values_guarded_when_no_spec_hook() -> None:
    """Round-2 codex finding @ L836: SpecSideValues never hits the non-spec hook.

    When a macro-step returns spatial ``SpecSideValues`` but the runtime exposes
    only the non-spec ``materialize_tokens`` hook (whose handle is the
    ``post_sample`` ``(slot, batch_size)`` aux value), the scheduler must NOT feed
    the side-values into it -- doing so raises ``cannot unpack SpecSideValues`` or
    reads the wrong layout. The guard materialises plain ``TextToken``s instead.
    Asserts (a) no crash, (b) the non-spec hook is never handed the side-values,
    (c) the committed ids still materialise (as text) and stage correctly.
    """
    real_side_values = SpecSideValues(
        hidden=__import__("torch").zeros(1, 3, 4),
        temperatures=__import__("torch").zeros(1),
        top_ps=__import__("torch").ones(1),
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
    req = _enqueue(sched, 0, prompt_len=3, max_new=3)
    sched._spec_admit()
    # Must not raise (the bug fed SpecSideValues into the non-spec hook -> unpack
    # TypeError, aborting the scheduler).
    while sched._spec_decode_step():
        pass

    # The non-spec hook was never handed the SpecSideValues (guard held).
    assert real_side_values not in nonspec_handles
    # Committed ids still staged (materialised as plain text via the guard).
    assert [int(t.token_id) for t in req.lifecycle.skill_state.tokens] == [11, 12, 13]
    assert req.lifecycle.finished is True
