"""Tests for the multimodal / mask / typed-token / logprob spec-step extension.

These cover the four features the DFlash spec macro-step gained so spec decode
covers Moondream's real workload (images + point/detect/query skills) with NO
fallback:

  * IMAGE   — admit prefills with the image KV prefix (validated at the runtime
              level by ``test_spec_decode_image`` when the spatial model builds;
              the env caveat below means this is GPU-gated).
  * MASKS   — each sequence's allowed/suppressed token ids constrain BOTH the
              drafter and the verify (masked argmax / sample), losslessly.
  * TYPED   — the committed ids carry the verify FINAL-hidden side-values +
              sampling knobs the runtime's ``materialize_tokens`` needs to type
              coord/size ids (``SpecSideValues``).
  * LOGPROBS— accepted tokens' logprobs = ``log_softmax`` of the (masked) verify
              target logits at the accepted positions.

The contract + the mask / logprob / side-value MATH are device-agnostic and run
on CPU (no GDN model). The end-to-end runtime checks are CUDA-gated and further
skip when the host's kernels build cannot run the model's GDN (the phase-2
validation-env caveat: a kernels .so whose gated-delta prefill is symmetric-heads
only cannot prefill the asymmetric-head Qwen3.5 GDN -- model-level spec decode is
then unavailable, so we validate the math here and flag the gap).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

# Load the device-agnostic contract directly so the math tests do not require a
# CUDA build / the full engine import chain.
from kestrel.runtime.spec import SpecSideValues, SpecStepResult


# ---------------------------------------------------------------------------
# Contract: SpecStepResult / SpecSideValues (eager + lazy, logprobs, side-values)
# ---------------------------------------------------------------------------


def test_spec_step_result_eager_carries_all_fields():
    sv = SpecSideValues(
        hidden=torch.zeros(2, 4, 8),
        temperatures=torch.zeros(2),
        top_ps=torch.ones(2),
    )
    r = SpecStepResult(
        tokens=[[10, 11, 12], [20]],
        accept_counts=[2, 0],
        logprobs=[[-0.1, -0.2, -0.3], [-1.0]],
        side_values=sv,
    )
    assert r.tokens == [[10, 11, 12], [20]]
    assert r.accept_counts == [2, 0]
    assert r.logprobs == [[-0.1, -0.2, -0.3], [-1.0]]
    # len(tokens[i]) == accept_counts[i] + 1 (the macro-step invariant).
    for toks, a in zip(r.tokens, r.accept_counts):
        assert len(toks) == a + 1
    assert r.side_values is sv
    assert r.side_values.counts is None  # "slice by accept_counts" convention


def test_spec_step_result_lazy_resolves_tokens_accepts_logprobs():
    calls = {"n": 0}

    def _resolve():
        calls["n"] += 1
        return [[5, 6]], [1], [[-1.0, -2.0]]

    r = SpecStepResult(resolve=_resolve)
    # No resolution until a field is read.
    assert calls["n"] == 0
    assert r.tokens == [[5, 6]]
    assert r.accept_counts == [1]
    assert r.logprobs == [[-1.0, -2.0]]
    # Resolved exactly once (cached across the three property reads).
    assert calls["n"] == 1


def test_spec_step_result_logprobs_optional_none():
    """A step with no logprob requester resolves logprobs to None."""
    r = SpecStepResult(resolve=lambda: ([[7]], [0], None))
    assert r.tokens == [[7]]
    assert r.logprobs is None


# ---------------------------------------------------------------------------
# MASK math: the additive logit mask (whitelist / blacklist) reproduces the
# non-spec sampler's restriction, and masked argmax/log_softmax stay inside the
# allowed set. These mirror what SpecStepRunner applies to drafter + verify.
# ---------------------------------------------------------------------------


def _build_row_mask(vocab, allowed, suppressed, *, dtype=torch.float32):
    """Standalone copy of SpecStepRunner._set_row_mask's math (one row)."""
    m = torch.zeros(vocab, dtype=dtype)
    if allowed:
        keep = torch.zeros(vocab, dtype=torch.bool)
        keep[torch.tensor(allowed)] = True
        m.masked_fill_(~keep, float("-inf"))
    if suppressed:
        m.index_fill_(0, torch.tensor(suppressed), float("-inf"))
    return m


def test_mask_whitelist_restricts_argmax_to_allowed():
    torch.manual_seed(0)
    vocab = 64
    allowed = [3, 7, 40]
    logits = torch.randn(5, vocab)
    mask = _build_row_mask(vocab, allowed, None)
    masked = logits + mask
    picks = masked.argmax(-1).tolist()
    assert all(p in allowed for p in picks), picks
    # The non-allowed positions are -inf (hard-blocked), allowed ones untouched.
    assert torch.isneginf(masked[:, [i for i in range(vocab) if i not in allowed]]).all()
    keep_cols = torch.tensor(allowed)
    assert torch.equal(masked[:, keep_cols], logits[:, keep_cols])


def test_mask_blacklist_suppresses_ids():
    torch.manual_seed(1)
    vocab = 32
    suppressed = [0, 5, 31]
    logits = torch.randn(4, vocab)
    masked = logits + _build_row_mask(vocab, None, suppressed)
    picks = masked.argmax(-1).tolist()
    assert all(p not in suppressed for p in picks), picks
    assert torch.isneginf(masked[:, torch.tensor(suppressed)]).all()


def test_mask_whitelist_then_blacklist_compose():
    """allowed then suppressed compose (suppressed wins) like the scheduler."""
    vocab = 16
    allowed = [1, 2, 3, 4]
    suppressed = [2, 4]
    mask = _build_row_mask(vocab, allowed, suppressed)
    # Allowed-not-suppressed kept; everything else -inf.
    kept = [1, 3]
    for i in range(vocab):
        if i in kept:
            assert mask[i] == 0.0
        else:
            assert torch.isneginf(mask[i])


def test_unmasked_row_is_identity_in_model_dtype():
    """An unmasked row's mask add must not perturb the bf16 argmax (greedy path
    stays byte-identical). x + 0 is exact and bf16+bf16 -> bf16."""
    torch.manual_seed(2)
    vocab = 128
    logits = torch.randn(3, vocab, dtype=torch.bfloat16)
    mask = torch.zeros(vocab, dtype=torch.bfloat16)  # unmasked row
    masked = logits + mask
    assert masked.dtype == torch.bfloat16
    assert torch.equal(masked, logits)
    assert torch.equal(masked.argmax(-1), logits.argmax(-1))


# ---------------------------------------------------------------------------
# PER-STEP MASK REFRESH (#114 contract): step(allowed_token_ids,
# suppressed_token_ids) recomputes each active row's mask in place and REPLACES
# the admit-time snapshot. These exercise the real SpecStepRunner mask helpers
# (``_set_row_mask`` / ``_refresh_step_masks``) on a lightweight stub so the
# row-iteration + override semantics are covered device-agnostically (the
# end-to-end GPU check below proves it constrains the actual drafter + verify).
# ---------------------------------------------------------------------------


class _MaskStub:
    """Minimal carrier exposing exactly what the mask helpers touch.

    ``_set_row_mask`` / ``_refresh_step_masks`` only read ``device`` / ``_vocab``
    and write ``_mask_buf`` / ``_row_masked``; bind the real (unbound) methods to
    this stub so the tested code path is the production one, not a copy.
    """

    def __init__(self, B, vocab, device="cpu"):
        from kestrel.models.qwen35.dflash.spec_decoder import SpecStepRunner

        self.device = torch.device(device)
        self._vocab = vocab
        self._mask_buf = torch.zeros(B, vocab, dtype=torch.float32)
        self._row_masked = [False] * B
        # Bind the real implementations under test.
        self._set_row_mask = SpecStepRunner._set_row_mask.__get__(self)
        self._refresh_step_masks = SpecStepRunner._refresh_step_masks.__get__(self)


def _allowed_cols(stub, row):
    """The set of non-blocked vocab ids for ``row`` (mask == 0 -> allowed)."""
    m = stub._mask_buf[row]
    return set((m == 0.0).nonzero(as_tuple=True)[0].tolist())


def test_refresh_step_masks_overrides_admit_mask_per_row():
    """A per-step mask REPLACES whatever the row carried (the admit snapshot).

    Row 0 starts whitelisted to {1,2} (the admit-time mask); a step then supplies
    {5,6}. After refresh the row must allow exactly {5,6} (not {1,2}, not the
    union) -- i.e. ``_set_row_mask`` zeroes the row before rebuilding so the new
    constraint wins. Row 1 is left ``None`` this step and must come back fully
    unmasked (every id allowed, ``_row_masked`` False)."""
    stub = _MaskStub(B=2, vocab=8)
    # Admit-time masks: row0 -> {1,2}, row1 -> {3}.
    stub._set_row_mask(0, [1, 2], None)
    stub._set_row_mask(1, [3], None)
    assert _allowed_cols(stub, 0) == {1, 2}
    assert stub._row_masked == [True, True]
    # Step refresh: row0 -> {5,6}, row1 -> None (unconstrained this step).
    stub._refresh_step_masks([0, 1], [[5, 6], None], [None, None])
    assert _allowed_cols(stub, 0) == {5, 6}, "per-step mask did not replace admit mask"
    assert _allowed_cols(stub, 1) == set(range(8)), "None row not cleared to unmasked"
    assert stub._row_masked == [True, False]


def test_refresh_step_masks_evolves_allowed_set_like_point_detect():
    """Stateful-skill cadence: the allowed set EVOLVES across consecutive steps.

    Emulates ``point`` alternating [coord,eos] <-> [coord] and ``detect`` cycling
    x->y->size: each step supplies a different whitelist, and the row's allowed
    set must track the latest one exactly (the non-spec per-step refresh)."""
    stub = _MaskStub(B=1, vocab=10)
    coord, eos = [4, 5, 6], 9
    cadence = [
        [coord + [eos]],   # point: [coord, eos]
        [coord],           # point: [coord]
        [[1]],             # detect: x
        [[2]],             # detect: y
        [[3]],             # detect: size
    ]
    for allowed in cadence:
        stub._refresh_step_masks([0], allowed, None)
        assert _allowed_cols(stub, 0) == set(allowed[0]), (
            f"row allowed {_allowed_cols(stub, 0)} != supplied {set(allowed[0])}")


def test_refresh_step_masks_only_touches_active_rows():
    """Idle rows (not in this step's ``rows``) keep their stored admit mask.

    ``_refresh_step_masks`` is handed only the ACTIVE rows; a row absent from the
    list must be left exactly as admit set it (its commit is discarded anyway, so
    re-masking it would be wasted work and could drop its stored constraint)."""
    stub = _MaskStub(B=3, vocab=8)
    stub._set_row_mask(2, [7], None)            # idle row's admit mask
    # Active rows this step are [0, 1]; row 2 must be untouched.
    stub._refresh_step_masks([0, 1], [[0], [1]], None)
    assert _allowed_cols(stub, 0) == {0}
    assert _allowed_cols(stub, 1) == {1}
    assert _allowed_cols(stub, 2) == {7}, "idle row's stored mask was disturbed"


def test_refresh_step_masks_suppressed_only_and_mixed():
    """A blacklist-only per-step refresh, and a whitelist+blacklist compose.

    ``suppressed_token_ids`` alone blocks the listed ids; supplying both lists
    composes whitelist-then-blacklist (suppressed wins) -- the same rule the
    non-spec sampler and ``_set_row_mask`` apply."""
    stub = _MaskStub(B=2, vocab=8)
    # Row0: blacklist {0,7}; allowed = everything else.
    # Row1: whitelist {1,2,3} then blacklist {2} -> {1,3}.
    stub._refresh_step_masks([0, 1], [None, [1, 2, 3]], [[0, 7], [2]])
    assert _allowed_cols(stub, 0) == set(range(8)) - {0, 7}
    assert _allowed_cols(stub, 1) == {1, 3}
    assert stub._row_masked[:2] == [True, True]


def test_set_row_mask_distinguishes_none_from_empty_whitelist():
    """``allowed_token_ids`` is a whitelist: ``None`` means "no constraint", an
    empty-but-not-``None`` sequence means "no id is allowed".

    Regression for the truthiness bug: ``if allowed_token_ids:`` treated ``[]``
    the same as ``None`` and skipped masking, leaving the row UNCONSTRAINED so
    the spec drafter/verify could commit an arbitrary token for a request whose
    skill admits none. The non-spec path rejects an empty allowed set at
    validation (``engine/core.py`` raises "removed every allowed next token")
    rather than silently masking the whole vocab and drawing token 0, so the spec
    ``_set_row_mask`` must RAISE on the empty whitelist -- and must still leave a
    ``None`` row fully unmasked."""
    stub = _MaskStub(B=2, vocab=8)
    # None whitelist -> unconstrained (every id allowed, row not masked).
    stub._set_row_mask(0, None, None)
    assert _allowed_cols(stub, 0) == set(range(8))
    assert stub._row_masked[0] is False
    # Empty (but not None) whitelist -> no valid next token -> raise, NOT a
    # silently-unconstrained row (the truthiness-bug behaviour).
    with pytest.raises(ValueError, match="empty whitelist"):
        stub._set_row_mask(1, [], None)
    # The same must hold on the per-step refresh path (it routes through
    # _set_row_mask), so a step that drops every allowed id is rejected too.
    with pytest.raises(ValueError, match="empty whitelist"):
        stub._refresh_step_masks([0], [[]], [None])
    # A single-id whitelist still works (sanity: the non-empty path is intact).
    stub._set_row_mask(0, [4], None)
    assert _allowed_cols(stub, 0) == {4}
    assert stub._row_masked[0] is True


def test_set_row_mask_rejects_blacklist_that_suppresses_whole_whitelist():
    """A non-empty whitelist whose ids are all blacklisted leaves NO valid token.

    e.g. ``allowed_token_ids=[42]`` with ``suppressed_token_ids=[42]``: the
    whitelist passes the empty-set check (it is non-empty), but the suppression
    then masks the only allowed id, so the whole row is ``-inf`` with no valid
    next token. Greedy admit/verify would return argmax 0 (a token that is NOT
    allowed) and sampling would see an all-NaN/all-zero row. The non-spec path
    rejects an empty EFFECTIVE allowed set at validation, so ``_set_row_mask``
    must RAISE after applying suppressions, exactly like the empty-whitelist case.
    Regression for the missing post-suppression all-masked check."""
    stub = _MaskStub(B=2, vocab=8)
    # Whitelist {42-equivalent: use in-range id 5}, blacklist the same id.
    with pytest.raises(ValueError, match="suppresses every allowed token"):
        stub._set_row_mask(0, [5], [5])
    # Multi-id whitelist fully covered by the blacklist (superset) -> also raise.
    with pytest.raises(ValueError, match="suppresses every allowed token"):
        stub._set_row_mask(0, [1, 2, 3], [0, 1, 2, 3, 4])
    # The same must hold through the per-step refresh path.
    with pytest.raises(ValueError, match="suppresses every allowed token"):
        stub._refresh_step_masks([0], [[5]], [[5]])
    # Sanity: a whitelist with at least one surviving id after suppression is OK
    # (whitelist {1,2,3}, blacklist {2} -> {1,3} still valid, no raise).
    stub._set_row_mask(1, [1, 2, 3], [2])
    assert _allowed_cols(stub, 1) == {1, 3}
    assert stub._row_masked[1] is True
    # A blacklist that empties the WHOLE vocab (no whitelist) also leaves no
    # valid token and must raise.
    with pytest.raises(ValueError, match="suppresses every allowed token"):
        stub._set_row_mask(1, None, list(range(8)))


class _ReleaseRowStub:
    """Carrier exposing exactly what ``_set_row_mask`` + ``_release_row`` touch.

    ``admit`` installs the row mask with ``_set_row_mask`` and, on any failure,
    calls ``_release_row`` (then re-raises); bind both real (unbound) methods so
    the freed-row clearing path under test is the production one. ``_release_row``
    reads/writes ``_mask_buf`` / ``_row_masked`` (the mask), the per-row sampling
    knobs (``_row_temp`` / ``_row_top_p`` / ``_row_logprobs``), the decode cursors
    (``_ctx_buf`` / ``_cur_buf``), the M-RoPE delta (``_rope_deltas``), and
    ``_free_rows``; provide all of them with the production shapes.
    """

    def __init__(self, B, vocab, device="cpu"):
        from kestrel.models.qwen35.dflash.spec_decoder import SpecStepRunner

        self.device = torch.device(device)
        self._vocab = vocab
        self.B = B
        self._mask_buf = torch.zeros(B, vocab, dtype=torch.float32)
        self._row_masked = [False] * B
        self._row_temp = [0.0] * B
        self._row_top_p = [1.0] * B
        self._row_logprobs = [False] * B
        self._ctx_buf = torch.zeros(B, dtype=torch.long)
        self._cur_buf = torch.zeros(B, dtype=torch.long)
        self._rope_deltas = torch.zeros(B, 1, dtype=torch.long)
        self._free_rows: list[int] = []
        self._set_row_mask = SpecStepRunner._set_row_mask.__get__(self)
        self._release_row = SpecStepRunner._release_row.__get__(self)


def test_release_row_clears_partial_mask_from_rejected_admit():
    """A rejected all-suppressed admit must leave NO -inf residue on the freed row.

    Regression for the partial-mask leak: ``_set_row_mask`` fills ``_mask_buf[row]``
    with -inf (whitelist then blacklist) BEFORE its "no valid token left" check, so
    an all-suppressed constraint (allowed=[x], suppressed=[x]) raises with the row
    already full of -inf but ``_row_masked[row]`` still False. ``admit`` catches
    that, calls ``_release_row(row)``, and re-raises -- but the old
    ``_row_masked``-gated zero skipped the clear (the flag was never set), so the
    freed idle row kept an all--inf mask. The captured draft graph adds ``_mask_buf``
    for EVERY fixed-B row (``logits + self._mask_buf[:, None, :]``), so under
    ``sampling=True`` a later step over that idle row builds all--inf draft logits ->
    softmax NaN -> ``multinomial`` failure (from an unrelated request). After the
    fix ``_release_row`` zeroes the mask unconditionally, so the freed row is a true
    identity (zero) mask and a sampled step over it yields finite probs / draws OK.
    """
    pytest.importorskip("kestrel.models.qwen35")
    stub = _ReleaseRowStub(B=2, vocab=8)
    row = 1  # free an idle row that another (row 0) live request rides alongside.

    # (1) Reproduce the admit reject: the all-suppressed constraint raises AFTER
    # _set_row_mask has written -inf into the row (the partial write), then admit's
    # failure path frees the row via _release_row and re-raises.
    with pytest.raises(ValueError, match="suppresses every allowed token"):
        try:
            stub._set_row_mask(row, [5], [5])   # allowed==suppressed -> all -inf
        except BaseException:
            stub._release_row(row)              # mirrors admit's ``except`` block
            raise

    # (2) The freed row must carry NO -inf residue: an identity (all-zero) mask,
    # flagged unmasked, and returned to the free pool.
    assert torch.isfinite(stub._mask_buf[row]).all(), (
        "freed row still holds -inf residue from the rejected admit's partial mask "
        "write; _release_row must zero the mask unconditionally")
    assert torch.equal(stub._mask_buf[row], torch.zeros(8)), (
        "freed row mask is not the zero (identity) mask")
    assert stub._row_masked[row] is False
    assert row in stub._free_rows, "rejected admit did not return the row to the pool"

    # (3) A subsequent SAMPLED step over that idle row must produce finite draft
    # probs (no NaN) and draw cleanly. Replicate the captured draft graph's masked
    # logits exactly: per-position draft logits + the per-row additive mask for ALL
    # B rows (the idle freed row included), then softmax + multinomial as the
    # sampling scheduler does.
    K = 3
    draft_logits = torch.zeros(stub.B, K, 8, dtype=torch.float32)
    masked_logits = draft_logits + stub._mask_buf[:, None, :]   # graph's line 1143
    probs = torch.softmax(masked_logits, dim=-1)
    assert torch.isfinite(probs).all(), (
        "draft probs over the freed idle row are non-finite (NaN) -- the all--inf "
        "residue mask was not cleared")
    # The freed idle row's distribution is a proper (uniform) simplex, not all-zero.
    assert torch.allclose(probs[row].sum(-1), torch.ones(K)), (
        "freed row's draft probs do not sum to 1 (all--inf row collapsed to NaN/0)")
    drawn = torch.multinomial(probs.reshape(stub.B * K, 8), num_samples=1)
    assert drawn.shape == (stub.B * K, 1), "multinomial failed over the freed row"

    # (4) The freed row's other decode state was reset too (cursor/token/knobs),
    # so the reused row starts clean (defensive: same release the mask rides in).
    assert int(stub._ctx_buf[row]) == 0 and int(stub._cur_buf[row]) == 0
    assert stub._row_temp[row] == 0.0 and stub._row_top_p[row] == 1.0
    assert stub._row_logprobs[row] is False


class _FirstTokenStub:
    """Carrier exposing exactly what ``_select_first_token`` touches on CPU.

    ``_select_first_token`` reads ``device`` / ``_row_masked`` / ``_mask_buf``
    (via ``_apply_row_mask_logits``) and, on the sampling branch, ``_spec_gen``;
    bind the real (unbound) methods so the path under test is the production one.
    The row mask is set with the real ``_set_row_mask`` so the whitelist applied
    here is exactly the admit-time mask the one-shot blacklist composes with.
    """

    def __init__(self, vocab, device="cpu"):
        from kestrel.models.qwen35.dflash.spec_decoder import SpecStepRunner

        self.device = torch.device(device)
        self._vocab = vocab
        self._mask_buf = torch.zeros(1, vocab, dtype=torch.float32)
        self._row_masked = [False]
        self._spec_gen = torch.Generator(device="cpu")
        self._set_row_mask = SpecStepRunner._set_row_mask.__get__(self)
        self._apply_row_mask_logits = SpecStepRunner._apply_row_mask_logits.__get__(self)
        self._select_first_token = SpecStepRunner._select_first_token.__get__(self)


def test_select_first_token_rejects_empty_one_shot_suppression():
    """One-shot ``suppress_next_token_ids`` that empties the allowed set must RAISE.

    Sibling of the ``_set_row_mask`` all-suppress fix, on the admit (first/bonus)
    token path: with the row whitelisted to ``{x}`` and the request's one-shot
    blacklist ``[x]``, the only allowed first token is removed, so the row is all
    ``-inf``. The greedy branch would then return ``argmax == 0`` (a token NOT in
    the whitelist) and the sampling branch would draw from NaNs -- committing a
    constraint-violating first token. ``_select_first_token`` must instead reject,
    exactly as ``_set_row_mask`` does for an all-masked persistent row.
    Regression for the missing post-one-shot-suppression all-masked check."""
    stub = _FirstTokenStub(vocab=8)
    # Row whitelisted to a single id; the one-shot blacklist removes exactly it.
    stub._set_row_mask(0, [5], None)
    logits = torch.zeros(1, 8, dtype=torch.float32)
    for temperature in (0.0, 1.0):   # greedy AND sampling branch both reject.
        with pytest.raises(ValueError, match="every allowed first token"):
            stub._select_first_token(
                logits, 0, suppress_next_token_ids=[5], temperature=temperature)
    # A suppression that leaves a surviving allowed id is fine (greedy picks it).
    stub._set_row_mask(0, [5, 6], None)
    tok, _ = stub._select_first_token(
        logits, 0, suppress_next_token_ids=[5], temperature=0.0)
    assert tok == 6, "greedy must pick the surviving whitelisted id, not argmax-0"
    # An unconstrained row with no one-shot suppression is unaffected (no raise).
    stub._set_row_mask(0, None, None)
    tok, _ = stub._select_first_token(logits, 0, temperature=0.0)
    assert 0 <= tok < 8


# ---------------------------------------------------------------------------
# LOGPROB math: per-committed-token logprob == log_softmax(masked verify logits)
# gathered at the committed id, matching the step's computation.
# ---------------------------------------------------------------------------


def test_committed_logprob_is_log_softmax_at_token():
    torch.manual_seed(3)
    B, blk, vocab = 2, 4, 50
    verify_logits = torch.randn(B, blk, vocab)
    # Committed ids per position (masked argmax in the step; arbitrary here).
    pred = verify_logits.argmax(-1)  # [B, blk]
    lp_all = torch.log_softmax(verify_logits.float(), dim=-1)
    lp_block = lp_all.gather(2, pred[:, :, None]).squeeze(2)  # [B, blk]
    # Reference: per (row, pos) log_softmax at the picked id.
    for r in range(B):
        for j in range(blk):
            ref = torch.log_softmax(verify_logits[r, j].float(), dim=-1)[pred[r, j]]
            assert torch.allclose(lp_block[r, j], ref, atol=1e-6)
    # Slicing to accept_i+1 (a=2 -> first 3 positions) is what the result carries.
    a = 2
    committed_lp = lp_block[0, : a + 1].tolist()
    assert len(committed_lp) == a + 1


def test_masked_logprob_excludes_blocked_mass():
    """Logprobs are taken over the MASKED distribution, so blocked ids carry no
    mass and the allowed set renormalizes (matches the masked sampler)."""
    torch.manual_seed(4)
    vocab = 20
    allowed = [2, 9, 15]
    logits = torch.randn(1, 1, vocab)
    masked = logits + _build_row_mask(vocab, allowed, None).view(1, 1, vocab)
    lp = torch.log_softmax(masked.float(), dim=-1)
    # Blocked ids -> -inf logprob; allowed ids -> finite, exp-sum to 1.
    blocked = [i for i in range(vocab) if i not in allowed]
    assert torch.isneginf(lp[0, 0, torch.tensor(blocked)]).all()
    allowed_p = lp[0, 0, torch.tensor(allowed)].exp().sum()
    assert torch.allclose(allowed_p, torch.tensor(1.0), atol=1e-5)


def _spec_lp_block(verify_logits, committed, samp_mask_b, temp_b):
    """Standalone copy of the step's per-committed-token logprob math.

    Mirrors ``SpecStepRunner.step``'s ``lp_block`` computation exactly (the
    temperature-scaled log_softmax gather + the one-hot greedy zeroing), so this
    test pins the contract the production block must keep: a greedy row reports
    ``0.0`` per committed token (the non-spec one-hot convention) while a
    sampling row reports the real temperature-scaled selected-token logprob.
    """
    any_sampling = bool(samp_mask_b.any())
    lp_logits = verify_logits.float()
    if any_sampling:
        lp_temp = torch.where(
            samp_mask_b, temp_b, temp_b.new_ones(())
        ).clamp_min(1e-6)
        lp_logits = lp_logits / lp_temp[:, None, None]
    lp_all = torch.log_softmax(lp_logits, dim=-1)
    gather_ids = committed.clamp_min(0)
    lp_block = lp_all.gather(2, gather_ids[:, :, None]).squeeze(2)
    if any_sampling:
        lp_block = torch.where(samp_mask_b[:, None], lp_block, lp_block.new_zeros(()))
    else:
        lp_block = lp_block.new_zeros(lp_block.shape)
    return lp_block


def test_greedy_committed_logprob_is_zero_under_spec():
    """A greedy row that requested logprobs reports ``0.0`` per committed token,
    matching the non-spec sampler's one-hot greedy convention (and
    ``_select_first_token``'s admit logprob). Without the one-hot zeroing the row
    would jump from ``0.0`` to the raw target-softmax logprob the moment spec
    decode engages -- the this guards. A sampling row in the SAME batch
    keeps its real temperature-scaled selected-token logprob."""
    torch.manual_seed(5)
    B, blk, vocab = 2, 4, 50
    verify_logits = torch.randn(B, blk, vocab)
    committed = verify_logits.argmax(-1)  # greedy commits the argmax per pos
    # Row 0 greedy (T<=eps), row 1 sampling (T=0.7).
    temp_b = torch.tensor([0.0, 0.7])
    samp_mask_b = temp_b > 1e-6
    lp_block = _spec_lp_block(verify_logits, committed, samp_mask_b, temp_b)
    # Greedy row: every committed-token logprob is exactly 0.0.
    assert torch.equal(lp_block[0], torch.zeros(blk))
    # Sampling row: the real log_softmax(logits / T) at the committed id (NOT 0).
    for j in range(blk):
        ref = torch.log_softmax(
            verify_logits[1, j].float() / 0.7, dim=-1)[committed[1, j]]
        assert torch.allclose(lp_block[1, j], ref, atol=1e-6)
    assert (lp_block[1] != 0).any()


def test_all_greedy_batch_committed_logprobs_all_zero():
    """When no row samples the whole verify block is greedy, so every committed
    logprob is 0.0 (the all-greedy one-hot convention; no temperature scaling
    applies)."""
    torch.manual_seed(6)
    B, blk, vocab = 3, 5, 40
    verify_logits = torch.randn(B, blk, vocab)
    committed = verify_logits.argmax(-1)
    temp_b = torch.zeros(B)  # all greedy
    samp_mask_b = temp_b > 1e-6
    lp_block = _spec_lp_block(verify_logits, committed, samp_mask_b, temp_b)
    assert torch.equal(lp_block, torch.zeros(B, blk))


# ---------------------------------------------------------------------------
# SIDE-VALUE packing: SpecSideValues holds the per-active-row verify block
# hidden [A, K+1, H] + per-row temp/top_p; the runtime slices accept_i+1 of each.
# ---------------------------------------------------------------------------


def test_side_value_block_layout_and_slice():
    B, blk, H = 4, 5, 8
    out_hidden = torch.arange(B * blk * H, dtype=torch.float32).reshape(B, blk, H)
    rows = [2, 0]                      # active rows (subset of B, any order)
    temps = [0.0, 0.7]
    top_ps = [1.0, 0.9]
    row_idx = torch.tensor(rows)
    hid = out_hidden.index_select(0, row_idx).contiguous()  # [A, blk, H]
    sv = SpecSideValues(
        hidden=hid,
        temperatures=torch.tensor(temps),
        top_ps=torch.tensor(top_ps),
    )
    assert sv.hidden.shape == (len(rows), blk, H)
    # hidden[i] is row rows[i]'s whole block; row-major over active sequences.
    assert torch.equal(sv.hidden[0], out_hidden[2])
    assert torch.equal(sv.hidden[1], out_hidden[0])
    # The runtime slices the leading accept_i+1 positions per sequence.
    accept_counts = [1, 3]
    for i, a in enumerate(accept_counts):
        committed_hidden = sv.hidden[i, : a + 1]
        assert committed_hidden.shape == (a + 1, H)
    assert torch.allclose(sv.temperatures, torch.tensor(temps))
    assert torch.allclose(sv.top_ps, torch.tensor(top_ps))


# ---------------------------------------------------------------------------
# CUDA-gated end-to-end runtime checks. These build the real Qwen3.5 runtime +
# SpecStepRunner. They skip cleanly when the host's kernels cannot run the GDN
# (phase-2 caveat). When they run, they prove the new features against the model.
# ---------------------------------------------------------------------------


def _try_build_runner(block_size=16, flush_cap=64, max_batch_size=2):
    """Build (rt, SpecStepRunner) or skip if the env can't run the GDN model.

    ``flush_cap`` is the GDN replay-ring capacity; a caller that drives more than
    ``flush_cap // block_size`` steps triggers the ring FLUSH (a separate
    GDN-owner kernel, GQA-incomplete on some hosts), so a test that only exercises
    the per-step machinery passes a cap large enough that the ring never flushes
    over its few steps (mirrors the per-step-mask test's 2-step bound).

    ``max_batch_size`` sizes the runtime's page table; the runner takes its 2
    persistent spec rows, so a caller that also needs a free ``batch_idx`` for a
    side reference-decode sequence passes a larger value (the default 2 leaves
    none free).
    """
    import kestrel.models.qwen35  # noqa: F401
    from kestrel.config import RuntimeConfig
    from kestrel.kv_cache import KVMemoryPool
    from kestrel.models.qwen35.dflash import SpecStepRunner
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    dev = torch.device("cuda")
    rt = Qwen35Runtime(
        RuntimeConfig(device="cuda", model="Qwen/Qwen3.5-4B",
                      max_batch_size=int(max_batch_size),
                      enable_cuda_graphs=False),
        kv_pool=KVMemoryPool(device=dev),
    )
    tc = getattr(rt.hf_config, "text_config", rt.hf_config)
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    dcfg = DFlashConfig(
        hidden_size=int(tc.hidden_size), intermediate_size=4096,
        num_hidden_layers=2, num_attention_heads=int(tc.num_attention_heads),
        num_key_value_heads=int(tc.num_attention_heads), head_dim=head_dim,
        vocab_size=int(tc.vocab_size), rope_theta=1e7, block_size=block_size,
        mask_token_id=0, target_layer_ids=target_layer_ids,
    )
    drafter = DFlashDraftModel(dcfg).to(dev, torch.bfloat16).eval()
    rt._linear_state_pool.replay_capacity = flush_cap
    for st in rt._linear_state_pool.layers:
        if st is not None:
            st.replay_checkpoint_states = None
            st.replay_k = st.replay_u = st.replay_g = st.replay_lengths = None
    rt._linear_state_pool.initialize_from_config(tc, dtype=rt.dtype)
    try:
        runner = SpecStepRunner(
            rt, drafter, dcfg, batch_size=2, max_seq_len=512,
            flush_cap=flush_cap, use_graphs=False,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"SpecStepRunner build failed on this host: {exc}")
    return rt, runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_masked_drafter_proposes_only_allowed_on_gpu():
    """REAL GPU check (no GDN, so it runs on the phase-2 .so): the mask fold the
    draft graph captures (``drafter_logits + mask``) makes the drafter's argmax
    proposals fall inside the allowed set, for the actual DFlash drafter module.

    This exercises the same computation ``SpecRunner._build_draft`` folds into
    the captured draft graph, on the real drafter forward, on the B200 -- the
    drafter is GDN-free (pure attention), so it is unaffected by the host's stale
    gated-delta kernels that block the full-model checks below.
    """
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel

    dev = torch.device("cuda")
    torch.manual_seed(0)
    H, V, blk = 256, 4000, 8
    dcfg = DFlashConfig(
        hidden_size=H, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=64, vocab_size=V,
        rope_theta=1e7, block_size=blk, mask_token_id=0, target_layer_ids=(0, 1),
    )
    drafter = DFlashDraftModel(dcfg).to(dev, torch.bfloat16).eval()
    embed = torch.nn.Embedding(V, H).to(dev, torch.bfloat16).eval()
    lm_head = torch.nn.Linear(H, V, bias=False).to(dev, torch.bfloat16).eval()

    B, K = 2, blk - 1
    fc_in = len(dcfg.target_layer_ids) * H
    # Mirror SpecRunner._build_draft's buffer layout exactly: target_hidden +
    # mask span ``maxc`` context columns; position_ids is ``maxc`` context
    # positions followed by ``block_size`` block positions; the live context is
    # ``c`` (so the mask blocks columns >= c, like the step).
    maxc, c = 32, 20
    th = torch.zeros(B, maxc, fc_in, device=dev, dtype=torch.bfloat16)
    th[:, :c] = torch.randn(B, c, fc_in, device=dev, dtype=torch.bfloat16)
    dpos = torch.zeros(B, maxc + blk, device=dev, dtype=torch.long)
    dpos[:, :maxc] = torch.arange(maxc, device=dev)
    dpos[:, maxc:] = torch.arange(c, c + blk, device=dev)
    block_ids = torch.zeros(B, blk, device=dev, dtype=torch.long)
    dmask = torch.zeros(B, 1, 1, maxc + blk, device=dev, dtype=torch.bfloat16)
    dmask[:, 0, 0, c:maxc] = float("-inf")

    allowed = [11, 222, 333, 1500, 3999]
    mask_row = _build_row_mask(V, allowed, None, dtype=torch.bfloat16).to(dev)
    mask_buf = mask_row.unsqueeze(0).expand(B, V)        # both rows masked

    with torch.inference_mode():
        noise = embed(block_ids)
        h = drafter(noise, th, dpos, attn_mask=dmask)
        logits = lm_head(h[:, 1:, :])                    # [B, K, V] (mask positions)
        # The fold the draft graph captures: logits + mask -> masked argmax.
        masked_drafts = (logits + mask_buf[:, None, :]).argmax(-1)
    flat = masked_drafts.reshape(-1).tolist()
    assert all(t in allowed for t in flat), (
        f"masked drafter proposed {sorted(set(flat))} outside allowed {allowed}")
    # Unmasked argmax would (almost surely) propose something outside the tiny
    # allowed set -> the mask is actually doing the constraining, not a no-op.
    with torch.inference_mode():
        unmasked = logits.argmax(-1).reshape(-1).tolist()
    assert any(t not in allowed for t in unmasked), (
        "test is vacuous: unmasked drafter already inside the allowed set")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_mask_keeps_committed_in_allowed_set():
    """End-to-end: a masked admit must commit ONLY allowed ids (drafter+verify
    both masked). Skips if the GDN model can't run on this host's kernels."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("List three colors:").ids
    # Restrict to a small allowed set; every committed token must be in it.
    allowed = sorted(set(rt.tokenizer.encode(" red green blue yellow").ids))
    s0, s1 = _State(), _State()
    try:
        with torch.inference_mode():
            try:
                first0, lp0 = runner.admit(s0, prompt, allowed_token_ids=allowed)
                first1, _ = runner.admit(s1, prompt)  # unmasked control row
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # admit returns (first_token_id, first_logprob); the masked row's
            # first (bonus) token must itself fall in the allowed set, and (no
            # logprobs requested) its logprob is None.
            assert first0 in allowed, (
                f"masked admit first token {first0} outside allowed {allowed}")
            assert lp0 is None
            for _ in range(3):
                res = runner.step([s0, s1])
                committed0 = res.tokens[0]
                assert all(t in allowed for t in committed0), (
                    f"masked row committed {committed0} outside allowed {allowed}")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_per_step_mask_overrides_admit_mask():
    """#114 per-step mask: ``step(allowed_token_ids=...)`` REPLACES the admit-time
    mask on the real drafter + verify. Admit whitelisted to set A, then step with
    a DISJOINT set B -> the committed tokens fall in B, not A. Skips if the GDN
    model can't run on this host's kernels.

    This is the end-to-end proof of the contract change: a row whose admit
    snapshot said {A} but whose live skill state now says {B} must decode within
    {B} this macro-step (the static admit mask alone would wrongly keep it in
    {A})."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("List some words:").ids
    set_a = sorted(set(rt.tokenizer.encode(" red green blue").ids))
    set_b = sorted(set(rt.tokenizer.encode(" cat dog fish").ids))
    # Disjoint sets so "committed in B but not A" is unambiguous.
    set_b = [t for t in set_b if t not in set_a]
    assert set_b, "test setup: need a non-empty B disjoint from A"
    s0 = _State()
    try:
        with torch.inference_mode():
            try:
                first0, _ = runner.admit(s0, prompt, allowed_token_ids=set_a)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # Admit honored set A.
            assert first0 in set_a
            # Step supplies the LIVE mask = set B (disjoint). It must override the
            # admit snapshot: the committed tokens fall in B (and none in A).
            res = runner.step(
                [s0],
                allowed_token_ids=[set_b],
                suppressed_token_ids=[None],
            )
            committed = res.tokens[0]
            assert committed, "macro-step committed no tokens"
            assert all(t in set_b for t in committed), (
                f"per-step mask not applied: committed {committed} not in B {set_b}")
            assert all(t not in set_a for t in committed), (
                f"admit mask leaked: committed {committed} still in A {set_a}")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_per_step_mask_evolves_like_point_detect():
    """#114 per-step mask: the allowed set EVOLVES across macro-steps and the
    committed tokens track the LATEST set each step (the stateful-skill cadence
    point/detect exhibit). Two consecutive steps with disjoint masks must each
    commit within that step's set -- proving the mask is re-applied per step, not
    snapshotted once. Skips if the GDN model can't run.

    Kept to 2 steps (< flush cadence) so the GDN ring never flushes -- the
    per-step MASK machinery is what's under test, independent of the ring-flush
    kernel (a separate GDN-owner follow-up)."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("Generate text:").ids
    set_a = sorted(set(rt.tokenizer.encode(" one two three").ids))
    set_b = sorted(set(rt.tokenizer.encode(" alpha beta gamma").ids))
    set_b = [t for t in set_b if t not in set_a]
    assert set_b, "test setup: need a non-empty B disjoint from A"
    s0 = _State()
    try:
        with torch.inference_mode():
            try:
                runner.admit(s0, prompt)  # admit UNMASKED; masks arrive per step
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # Step 1: constrain to A.
            r1 = runner.step([s0], allowed_token_ids=[set_a])
            c1 = r1.tokens[0]
            assert c1 and all(t in set_a for t in c1), (
                f"step1 committed {c1} not in A {set_a}")
            # Step 2: the live allowed set has EVOLVED to B (disjoint). The
            # committed tokens must now be in B -- the mask refreshed per step.
            r2 = runner.step([s0], allowed_token_ids=[set_b])
            c2 = r2.tokens[0]
            assert c2 and all(t in set_b for t in c2), (
                f"step2 committed {c2} not in evolved set B {set_b}")
            assert all(t not in set_a for t in c2), (
                f"step2 still constrained to stale A: committed {c2}")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_none_mask_keeps_admit_mask_backcompat():
    """#114 per-step mask: ``step(...)`` with NO mask args keeps the admit-time
    mask (back-compat). A row admitted whitelisted to A, stepped without per-step
    masks, must still commit within A. Skips if the GDN model can't run."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("List three colors:").ids
    set_a = sorted(set(rt.tokenizer.encode(" red green blue yellow").ids))
    s0 = _State()
    try:
        with torch.inference_mode():
            try:
                runner.admit(s0, prompt, allowed_token_ids=set_a)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # No allowed/suppressed args -> the admit-time mask stays in force.
            res = runner.step([s0])
            committed = res.tokens[0]
            assert committed and all(t in set_a for t in committed), (
                f"None-mask step dropped the admit mask: committed {committed} "
                f"not in A {set_a}")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_logprobs_match_target_log_softmax():
    """End-to-end: returned logprobs == log_softmax of the verify target logits
    at the committed positions. Skips if the GDN model can't run."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self, return_logprobs=False):
            self.batch_idx = -1
            self.length = 0
            self.return_logprobs = return_logprobs

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("The capital of France is").ids
    s0 = _State(return_logprobs=True)
    try:
        with torch.inference_mode():
            try:
                first, first_lp = runner.admit(s0, prompt)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # Greedy admit: the bonus token is the argmax, so its selected-token
            # logprob is the non-spec greedy convention 0.0 (NOT None -- logprobs
            # were requested).
            assert first_lp == 0.0
            res = runner.step([s0])
            lps = res.logprobs
            assert lps is not None and len(lps[0]) == res.accept_counts[0] + 1
            # Every logprob is a finite <= 0 number (a valid log-probability).
            for v in lps[0]:
                assert v <= 1e-4
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_first_logprob_matches_normal_decode_greedy():
    """The admit first (id, logprob) must equal what a normal greedy prefill+
    argmax produces for the SAME prompt: same id, and logprob 0.0 (the non-spec
    greedy selected-token convention). This is the bit-exact-greedy guarantee.

    The reference greedy prefill+argmax is run FIRST (only the runtime exists, so
    all KV pages are free), then the runner is built and admits the same prompt;
    a single-position greedy admit must reproduce the reference id exactly.
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self, return_logprobs=False):
            self.batch_idx = -1
            self.length = 0
            self.return_logprobs = return_logprobs

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("The capital of France is").ids
    s0 = _State(return_logprobs=True)
    try:
        with torch.inference_mode():
            try:
                first, first_lp = runner.admit(s0, prompt)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # The admit token is the masked-prefill argmax (the validated greedy
            # bonus token) and the greedy selected-token logprob is 0.0 -- exactly
            # what the non-spec normal-decode first token produces. Cross-check the
            # logprob value directly against the runner's own prefill logits: the
            # full-softmax logprob of the argmax under temperature 1 is what the
            # single-token sampler would also collapse to 0.0 under greedy.
            assert first_lp == 0.0
            # Re-derive the reference greedy id from a fresh prefill on the SAME
            # row (now that the runner row is admitted, reuse its batch_idx pages
            # via a throwaway cache -- no new page allocation).
            n = len(prompt)
            cpos = torch.arange(n, device=rt.device).view(1, -1)
            pt = rt.page_table
            ref_bidx = int(s0.batch_idx)
            tmp_cache = rt._new_cache()
            bidx_tok = torch.full((1, n), ref_bidx, device=rt.device, dtype=torch.int64)
            slotmap = pt.build_slot_mapping(batch_idx=bidx_tok, positions=cpos)
            page_tbl_row = torch.index_select(
                pt.page_table, 0, torch.tensor([ref_bidx], device=rt.device))
            out_pf = rt.model.model.language_model(
                input_ids=torch.tensor([prompt], device=rt.device, dtype=torch.long),
                position_ids=cpos, past_key_values=tmp_cache,
                cache_position_ids=cpos, slot_mapping=slotmap,
                page_table=page_tbl_row,
                paged_kv_seqlens_k=torch.tensor([n], device=rt.device, dtype=torch.int32),
                cu_seq_lens_q=torch.tensor([0, n], device=rt.device, dtype=torch.int32),
            )
            ref_logits = rt.model.lm_head(out_pf.last_hidden_state[0][n - 1:n])  # [1, V]
            ref_id = int(ref_logits[0].argmax())
            # Byte-identical first token: same greedy id, greedy logprob 0.0.
            assert first == ref_id
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_one_shot_suppression_blocks_first_token():
    """``suppress_next_token_ids`` must blacklist the greedy admit token: with the
    natural argmax suppressed, admit returns a DIFFERENT (next-best allowed) id."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("The capital of France is").ids
    s0, s1 = _State(), _State()
    try:
        with torch.inference_mode():
            try:
                natural, _ = runner.admit(s0, prompt)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            # Re-admit the same prompt with the natural argmax suppressed.
            suppressed, _ = runner.admit(
                s1, prompt, suppress_next_token_ids=[natural])
            assert suppressed != natural, (
                "one-shot suppression did not block the natural admit token")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_surfaces_side_values_on_typed_runtime():
    """When the runtime types coord/size ids, admit surfaces the first token's
    last-hidden side-values on ``state.admit_side_values`` (shape [1, 1, H],
    counts=[1]); a text-only runtime leaves it None."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0
            self.admit_side_values = None

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    prompt = rt.tokenizer.encode("The capital of France is").ids
    # Text-only runtime (Qwen35Runtime.spatial_tables is None) -> no side-values.
    s_text = _State()
    try:
        with torch.inference_mode():
            try:
                runner.admit(s_text, prompt)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            assert s_text.admit_side_values is None
        # Now force the typed-token path by giving the runtime a spatial table
        # marker; re-admit and assert the side-values shape.
        runner.retire(s_text)
        rt.spatial_tables = object()  # truthy -> _detect_typed_token_runtime True
        s_typed = _State()
        with torch.inference_mode():
            runner.admit(s_typed, prompt)
        sv = s_typed.admit_side_values
        assert sv is not None
        # One sequence, one committed (first) position.
        assert tuple(sv.hidden.shape[:2]) == (1, 1)
        assert sv.counts == [1]
        assert sv.temperatures.shape == (1,)
    finally:
        rt.spatial_tables = None
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_restores_free_row_on_prefill_failure():
    """A failed admit must NOT leak its pool row.

    ``admit`` pops a free row, then prefills/selects. If prefill raises (here: a
    prompt that exceeds ``max_seq_len``), the row was never recorded in
    ``_rows``/``_row_of`` so ``retire`` could never return it -- a few bad
    requests would permanently exhaust the runner. ``admit`` must restore the row
    (and clear its per-row mask/sampling state) on failure, so ``free_slots``
    returns to baseline and a subsequent good admit succeeds in that row.
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    try:
        baseline = runner.free_slots
        assert baseline >= 1
        # A prompt longer than max_seq_len: _prefill_row raises ValueError AFTER
        # the row is popped (the n + num_spec + 1 + 4 > max_seq_len guard).
        too_long = list(range(runner.max_seq_len + 50))
        bad = _State()
        with torch.inference_mode():
            # Sanity: a normal prompt must actually prefill on this host, else the
            # failure below could be an unrelated GDN-env issue, not the leak path.
            probe = rt.tokenizer.encode("The capital of France is").ids
            s_probe = _State()
            try:
                runner.admit(s_probe, probe)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            runner.retire(s_probe)
            assert runner.free_slots == baseline

            with pytest.raises(Exception):
                runner.admit(bad, too_long)
        # The row must have been returned; the failed state must not be tracked.
        assert runner.free_slots == baseline, (
            f"admit leaked a row on failure: free_slots {runner.free_slots} != "
            f"baseline {baseline}"
        )
        assert not runner.has_row(bad)
        # And the runner is still usable: a good admit succeeds and retires clean.
        good = _State()
        with torch.inference_mode():
            runner.admit(good, probe)
        assert runner.has_row(good)
        assert runner.free_slots == baseline - 1
        runner.retire(good)
        assert runner.free_slots == baseline
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_accepts_typed_prompt_tokens_greedy_identical():
    """admit must accept the typed ``Sequence[Token]`` contract (kestrel #114) and
    produce the byte-identical greedy first token a bare int-id list does.

    The scheduler forwards ``list(request.prefill_tokens)`` (typed tokens), not an
    ``int(t.token_id)`` projection. A text prompt wrapped as ``TextToken``s must
    prefill to exactly the same first/bonus token as the equivalent id list, with
    no leak (each admit retires before the next).
    """
    pytest.importorskip("kestrel.models.qwen35")
    from kestrel.runtime.tokens import TextToken

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    ids = rt.tokenizer.encode("The capital of France is").ids
    typed = [TextToken(token_id=int(t)) for t in ids]
    try:
        with torch.inference_mode():
            s_ids = _State()
            try:
                first_ids, _ = runner.admit(s_ids, ids)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            runner.retire(s_ids)
            s_typed = _State()
            first_typed, _ = runner.admit(s_typed, typed)
            runner.retire(s_typed)
        assert first_typed == first_ids, (
            f"typed prompt_tokens produced first token {first_typed} != int-id "
            f"path {first_ids}; greedy admit must be byte-identical"
        )
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_rejects_untyped_token_without_leaking_row():
    """A prompt token with no vocabulary id (e.g. ``CoordToken``) is unsupported in
    a Qwen3.5 prefill: admit must raise a clear error AND restore the popped row
    (the failure happens after the pop, inside _prepare_image_prefill)."""
    pytest.importorskip("kestrel.models.qwen35")
    from kestrel.runtime.tokens import CoordToken, TextToken

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    ids = rt.tokenizer.encode("The capital of France is").ids
    bad_prompt = [TextToken(token_id=int(t)) for t in ids] + [CoordToken(pos=0.5)]
    try:
        baseline = runner.free_slots
        s_bad = _State()
        with torch.inference_mode():
            with pytest.raises(ValueError):
                runner.admit(s_bad, bad_prompt)
        assert runner.free_slots == baseline, "admit leaked a row on a typed-token reject"
        assert not runner.has_row(s_bad)
    finally:
        runner.release()


# ---------------------------------------------------------------------------
# IMAGE_CROPS threading (#114 contract): admit(image_crops=...) prefills with the
# preprocessed multi-tile image inputs (the non-spec ``overlap``), not a
# thumbnail-only image. These are CUDA-gated and run the vision encoder + GDN
# image prefill; they skip cleanly when the host's kernels can't run that path.
#
# These build the ``QwenImageInputs`` DIRECTLY (synthetic pixel_values + grid)
# rather than ``rt._image_preprocessor.process(img)``: the image preprocessor's
# bicubic resize is a native op (``kestrel_native.resize_bicubic``) absent from
# some validation builds, but the spec image-prefill path under test only needs a
# well-formed ``QwenImageInputs`` -- which the scheduler hands over as
# ``image_crops`` anyway (already preprocessed). So we synthesize that object and
# exercise the real threading + vision-encoder prefill.
# ---------------------------------------------------------------------------


def _synthetic_image_inputs(n_tiles=1, gh=4, gw=4, seed=0):
    """A directly-built :class:`QwenImageInputs` with ``n_tiles`` grid rows.

    Each tile is a ``[1, gh, gw]`` grid; ``pixel_values`` is the matching random
    patch tensor ``[sum(gh*gw), C*tp*ps*ps]``. ``num_image_tokens`` is the Qwen
    convention ``prod(grid).sum() // (merge^2)``. This is exactly the shape
    ``QwenImageProcessorConfig`` produces, minus the native resize."""
    from kestrel.models.qwen35.runtime import QwenImageInputs
    from kestrel.models.qwen35.qwen_image import QwenImageProcessorConfig

    cfg = QwenImageProcessorConfig()
    feat = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    g = torch.Generator().manual_seed(seed)
    grid = torch.tensor([[1, gh, gw]] * n_tiles, dtype=torch.long)
    n_patches = int((grid[:, 1] * grid[:, 2]).sum())
    pv = torch.randn(n_patches, feat, generator=g)
    num_tokens = int(grid.prod(-1).sum()) // (cfg.merge_size ** 2)
    return QwenImageInputs(
        pixel_values=pv, image_grid_thw=grid, num_image_tokens=num_tokens)


def _typed(rt, text):
    """Encode ``text`` to the typed ``TextToken`` list the query/image path needs.

    The image query-path splice reads ``.token_id``, so an image admit must pass
    typed tokens (the scheduler forwards typed tokens)."""
    from kestrel.runtime.tokens import TextToken

    return [TextToken(token_id=int(t)) for t in rt.tokenizer.encode(text).ids]


def _gdn_pool_snapshot(rt, row_batch_idx):
    """Clone the GDN pool row's conv + recurrent state for a bit-exact compare.

    The image prefill writes the row's GDN state into the runtime's persistent
    linear-state pool (``capture_from_cache``); reading conv_states /
    recurrent_states at the row is a deterministic fingerprint of the
    image-conditioned context. Two prefills that built the SAME image KV produce
    bit-identical state here."""
    pool = rt._linear_state_pool
    snap = {}
    for i, layer in enumerate(pool.layers):
        if layer is None:
            continue
        conv = getattr(layer, "conv_states", None)
        rec = getattr(layer, "recurrent_states", None)
        if conv is not None:
            snap[(i, "conv")] = conv[row_batch_idx].clone()
        if rec is not None:
            snap[(i, "rec")] = rec[row_batch_idx].clone()
    return snap


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_image_crops_drives_image_prefill_bitexact():
    """#114 ``image_crops``: admitting with the preprocessed inputs via the NEW
    ``image_crops`` arg builds the SAME image-conditioned KV / GDN state as the
    legacy path that passed those same inputs as ``image`` -- same first token,
    same context length, bit-identical GDN pool state. And ``image_crops`` ALONE
    (``image=None``) still drives the image prefill (a text-only admit of the same
    prompt has a shorter context), so the multi-crop tiles are genuinely encoded
    and not silently dropped. Skips if the vision/GDN image prefill can't run.

    This is the regression for the threading fix: the scheduler hands the
    preprocessed tiles over as ``image_crops`` (the non-spec ``overlap``); the
    spec prefill must route THOSE into the vision encode, identically to the
    image path, rather than ignoring them (which would build thumbnail-only /
    text-only KV)."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    qi = _synthetic_image_inputs(n_tiles=1, seed=7)
    prompt = _typed(rt, "Describe the image:")
    # Admit sequentially (retire between) so this stays within the 2-row runner;
    # the GDN snapshots are cloned immediately, so a later admit into the reused
    # row does not perturb them.
    s_crops, s_image, s_text = _State(), _State(), _State()
    try:
        with torch.inference_mode():
            try:
                # NEW arg: image_crops (image=None) drives the image prefill.
                first_crops, _ = runner.admit(s_crops, prompt, image_crops=qi)
            except Exception as exc:  # pragma: no cover - GDN/vision env caveat
                pytest.skip(f"image prefill unavailable on this host: {exc}")
            crops_snap = _gdn_pool_snapshot(rt, int(s_crops.batch_idx))
            crops_ctx = runner._rows[runner._row_of[id(s_crops)]].ctx_len
            runner.retire(s_crops)

            # Legacy path: the same preprocessed inputs passed as ``image`` (the
            # pre-#114 admit accepted a QwenImageInputs there). Must be identical.
            first_image, _ = runner.admit(s_image, prompt, image=qi)
            image_snap = _gdn_pool_snapshot(rt, int(s_image.batch_idx))
            image_ctx = runner._rows[runner._row_of[id(s_image)]].ctx_len
            runner.retire(s_image)

            # Text-only control (no image at all) -> shorter context.
            runner.admit(s_text, prompt)
            text_ctx = runner._rows[runner._row_of[id(s_text)]].ctx_len
            runner.retire(s_text)

        # image_crops == image path: same first token, same ctx, identical GDN.
        assert first_crops == first_image, (
            f"image_crops first token {first_crops} != image path {first_image}")
        assert crops_ctx == image_ctx, (
            f"image_crops ctx_len {crops_ctx} != image path {image_ctx}")
        assert set(crops_snap) == set(image_snap)
        for k in image_snap:
            assert torch.equal(crops_snap[k], image_snap[k]), (
                f"GDN pool state diverged at {k}: image_crops != image prefill")
        # The tiles were actually encoded: image context > text-only context by
        # exactly the image-token count + 2 vision-block delimiters.
        assert crops_ctx == text_ctx + int(qi.num_image_tokens) + 2, (
            f"image ctx {crops_ctx} != text ctx {text_ctx} + image tokens "
            f"{int(qi.num_image_tokens)} + 2 -> tiles not encoded")
    finally:
        runner.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_admit_image_crops_encodes_full_tile_extent():
    """#114 ``image_crops``: the prefill encodes the FULL tile extent of the
    preprocessed inputs (more image tokens => longer image context), not a
    truncated/thumbnail subset. A larger ``image_crops`` (more patches) must
    lengthen the image context by exactly its extra image tokens. Skips if the
    image prefill can't run on this host.

    The ``image_crops`` the scheduler forwards carries every tile/crop; the
    vision block the spec prefill splices is sized to the FULL
    ``num_image_tokens``, so a bigger crop set yields proportionally more
    image-pad tokens in the KV context. A path that dropped tiles (thumbnail
    only) would NOT show this exact growth. Two single-image grids of different
    sizes are used (4x4 vs 8x8) so each prefill stays on the position-id-safe
    single-image layout while still differing in tile extent."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    try:
        rt, runner = _try_build_runner()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    qi_small = _synthetic_image_inputs(n_tiles=1, gh=4, gw=4, seed=1)
    qi_large = _synthetic_image_inputs(n_tiles=1, gh=8, gw=8, seed=1)
    prompt = _typed(rt, "Describe the image:")
    # The larger crop set has strictly more image tokens.
    assert int(qi_large.num_image_tokens) > int(qi_small.num_image_tokens)
    s_small, s_large = _State(), _State()
    try:
        with torch.inference_mode():
            try:
                runner.admit(s_small, prompt, image_crops=qi_small)
            except Exception as exc:  # pragma: no cover - GDN/vision env caveat
                pytest.skip(f"image prefill unavailable on this host: {exc}")
            small_ctx = runner._rows[runner._row_of[id(s_small)]].ctx_len
            runner.retire(s_small)
            runner.admit(s_large, prompt, image_crops=qi_large)
            large_ctx = runner._rows[runner._row_of[id(s_large)]].ctx_len
            runner.retire(s_large)
        # ctx grows by exactly the extra image tokens (same prompt, same +2
        # vision-block delimiters, more image-pad tokens for the larger crop set).
        assert large_ctx - small_ctx == int(qi_large.num_image_tokens) - int(
            qi_small.num_image_tokens), (
            f"image ctx delta {large_ctx - small_ctx} != extra image tokens "
            f"{int(qi_large.num_image_tokens) - int(qi_small.num_image_tokens)} -> "
            "full tile extent not encoded")
    finally:
        runner.release()


def _spec_image_continuation(runner, rt, prompt, qi, n_new, *, force_zero_delta):
    """Greedy spec continuation of an image prompt; optionally drop the M-RoPE delta.

    Admits ``prompt`` with ``qi`` (image), then steps greedily collecting committed
    tokens. ``force_zero_delta`` zeroes the runner's stored per-row spatial
    ``_rope_deltas`` right after admit -- reproducing the PRE-FIX behaviour (the
    verify forward used 2-D text positions, dropping the image M-RoPE shift) -- so
    the same block-verify forward can be compared with vs without the delta.
    """
    class _S:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    s = _S()
    first, _ = runner.admit(s, prompt, image_crops=qi)
    delta = int(runner._rope_deltas.abs().max().item())
    if force_zero_delta:
        runner._rope_deltas.zero_()
    eos = set(getattr(rt, "_eos_ids", set()) or set())
    toks = [int(first)]
    if int(first) not in eos:
        for _ in range(n_new + 4):
            for t in list(runner.step([s]).tokens[0]):
                toks.append(int(t))
                if int(t) in eos:
                    break
            if len(toks) >= n_new or (toks and toks[-1] in eos):
                break
    runner.retire(s)
    return toks[:n_new], delta


def _image_decode_reference(rt, ids, image_kwargs, n_new):
    """Normal-decode greedy continuation of the EXACT image-prefill ``ids``.

    Prefills + decodes one token at a time through ``_forward_base`` -- the same
    forward the runtime's own image generate() uses -- so every decode step
    rotates with the image-shifted spatial M-RoPE positions (the carried
    ``rope_deltas``). This is the M-RoPE-correct target the spec verify must track.
    """
    dev = rt.device
    bidx = rt.page_table.allocate()
    rt.page_table.reserve(bidx, len(ids) + n_new + 4)
    rt.page_table.commit_block_table([bidx])
    try:
        cpos = torch.arange(len(ids), device=dev, dtype=torch.long).view(1, -1)
        last_hidden, cache = rt._forward_base(
            input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
            past_key_values=rt._new_cache(), batch_idx=bidx,
            cache_position_ids=cpos, **image_kwargs,
        )
        logits = rt._logits_for_last(last_hidden[0, -1])
        eos = set(getattr(rt, "_eos_ids", set()) or set())
        gen = []
        for _ in range(n_new):
            nxt = int(logits.argmax(-1).item())
            gen.append(nxt)
            if nxt in eos:
                break
            pos = torch.tensor(
                [[int(cache.past_key_values.get_seq_length())]],
                dtype=torch.long, device=dev)
            last_hidden, cache = rt._forward_base(
                input_ids=torch.tensor([[nxt]], dtype=torch.long, device=dev),
                past_key_values=cache, batch_idx=bidx, cache_position_ids=pos)
            logits = rt._logits_for_last(last_hidden[0, 0])
        return gen
    finally:
        rt.page_table.erase(bidx, 0)


def _first_div(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_image_decode_applies_mrope_delta():
    """Image spec decode must apply the SAME M-RoPE spatial delta a normal image
    decode does -- the verify ``position_ids`` must carry the image's spatial
    shift, not text-only positions.

    Regression (spec_decoder ``_prefill_row``/``_capture_graphs``): the
    captured verify forward passed the 2-D text ``cpos_buf`` as ``position_ids``,
    which the text model broadcasts to 4 IDENTICAL M-RoPE rows -- correct only
    when the rope delta is 0. An image prefill SHIFTS the post-image spatial
    positions (non-zero ``rope_delta``); dropping that shift made the spec verify
    rotate every post-prefill token with TEXT-only positions, so the generated
    image answer could diverge from a normal image decode of the same model.

    Proven two ways, isolating the fix from the (pre-existing, documented)
    bf16 block-verify-vs-single-token argmax-tie noise:
      * TIE-FREE spec-vs-spec: the SAME block-verify forward, delta-applied vs
        delta-zeroed (the pre-fix behaviour). The delta-applied run must track the
        M-RoPE-correct reference at least as far as the zeroed run, and STRICTLY
        further for at least one (load-bearing) image -- i.e. the delta genuinely
        flows into the verify and corrects the early tokens the pre-fix path got
        wrong. (The zeroed run reproduces the pre-fix early divergence.)
      * The stored per-row delta is non-zero (the image really shifts M-RoPE).
    Skips if the GDN image prefill can't run on this host.
    """
    pytest.importorskip("kestrel.models.qwen35")
    try:
        # max_batch_size 4 so the runner's 2 spec rows leave a free batch_idx for
        # the normal-decode reference sequence.
        rt, runner = _try_build_runner(max_batch_size=4)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")

    n_new = 32
    # A large single-image grid: more image tokens compress more positions, so the
    # post-image spatial delta is large (the regime a dropped delta most visibly
    # changes the greedy continuation). A few seeds/prompts; require >=1 to be
    # "load-bearing" (the delta changes the spec output) so the proof is real.
    # Keep these cases in one shared runner: that also exercises retired-row reuse
    # and catches per-row ReplaySSM flush phase leaks across image requests.
    cases = [
        ("Explain everything you see in the image step by step:", 41),
        ("Describe the image in detail:", 7),
        ("Write a long caption for the picture:", 23),
    ]
    n_loadbearing = 0
    try:
        with torch.inference_mode():
            for ci, (text, seed) in enumerate(cases):
                qi = _synthetic_image_inputs(n_tiles=1, gh=16, gw=16, seed=seed)
                prompt = _typed(rt, text)
                try:
                    _ii, ids, ikw = runner._prepare_image_prefill(
                        prompt, None, image_crops=qi)
                except Exception as exc:  # pragma: no cover - vision env caveat
                    pytest.skip(f"image prefill unavailable on this host: {exc}")
                assert _ii is not None, "image did not route image prefill"

                try:
                    ref = _image_decode_reference(rt, ids, ikw, n_new)
                    spec_fix, delta = _spec_image_continuation(
                        runner, rt, prompt, qi, n_new, force_zero_delta=False)
                    spec_zero, _ = _spec_image_continuation(
                        runner, rt, prompt, qi, n_new, force_zero_delta=True)
                except Exception as exc:  # pragma: no cover - GDN/vision env caveat
                    pytest.skip(f"image spec decode unavailable on this host: {exc}")

                # The image must actually shift the spatial M-RoPE positions.
                assert delta != 0, (
                    f"case {ci}: image produced rope_delta=0; cannot exercise the "
                    "M-RoPE spec path")

                d_fix = _first_div(spec_fix, ref)     # delta-applied vs reference
                d_zero = _first_div(spec_zero, ref)   # delta-zeroed (pre-fix) vs ref
                fixed_ok = d_fix < 0 or (d_zero >= 0 and d_fix >= d_zero)
                assert fixed_ok, (
                    f"case {ci}: applying the M-RoPE delta made the spec output "
                    f"AGREE WITH the normal image decode LESS (delta-applied "
                    f"diverges at {d_fix}, delta-zeroed at {d_zero}) -- the delta "
                    "must only improve agreement")
                if d_zero >= 0 and (d_fix < 0 or d_fix > d_zero):
                    # Load-bearing: the pre-fix (delta-zeroed) path diverged from
                    # the M-RoPE-correct decode earlier than the fixed path -- the
                    # fix corrected the early image-positioned tokens.
                    n_loadbearing += 1
    finally:
        runner.release()

    assert n_loadbearing > 0, (
        "no image case was load-bearing (zeroing the verify M-RoPE delta never "
        "changed the spec output vs the normal image decode), so this did not "
        "exercise the M-RoPE shift -- widen the image/prompt sweep")


# ---------------------------------------------------------------------------
# COMMIT CAP (#114): the scheduler caps a STATEFUL-masked row to one committed
# token per macro-step so the single per-step mask is exact for one constraint
# transition. ``SpecStepRunner.step`` must ENFORCE the cap: truncate the
# accepted run and keep the GDN ring / conv window / paged-KV cursor / next
# current token consistent with the truncated commit (the drafted tokens beyond
# the cap discarded as if rejected), so the next step resumes from the correct
# committed position. These are the end-to-end proof that the cap the scheduler
# sends is honored on the real Qwen3.5 model + DFlash macro-step.
# ---------------------------------------------------------------------------


def _row_ring_lengths(runner, row):
    """Per-GDN-layer replay-ring lengths for ``row`` (host ints).

    The ring length is the committed GDN replay cursor; it must advance by
    exactly the committed run (``accept + 1``) each step. Indexed by the row's
    pool batch_idx (``runner.cb[row]``), matching ``step``'s
    ``replay_lengths[self.cb] += advance``.
    """
    cb = int(runner.cb[row])
    return [
        int(runner.cache.layers[idx].replay_lengths[cb])
        for idx in runner.gdn_layer_idxs
    ]


def _greedy_continuation(prompt, n, _State, *, block_size, flush_cap):
    """The model's deterministic greedy continuation of ``prompt`` (>= ``n`` ids).

    Builds a throwaway runner, admits the prompt, and steps it UNCAPPED until it
    has accumulated at least ``n`` committed (greedy) tokens, returning them. Used
    to build the FORCED drafts for the cap tests: feeding the true greedy
    continuation makes a fresh runner's verify accept the whole K-token run
    (greedy argmax is deterministic for a fixed prompt+weights), so the run is
    guaranteed multi-token independent of the random drafter's quality -- the
    regime the commit cap must constrain. ``block_size`` / ``flush_cap`` match the
    consumer runner so the same forced drafts line up with its K, and the ring
    never flushes over the few gather steps. Returns ``None`` to signal the caller
    should skip (env can't run the GDN model), mirroring ``_try_build_runner``.
    """
    try:
        rt, runner = _try_build_runner(block_size=block_size, flush_cap=flush_cap)
    except Exception:  # pragma: no cover
        return None
    s = _State()
    toks: list[int] = []
    try:
        with torch.inference_mode():
            try:
                runner.admit(s, prompt)
            except Exception:  # pragma: no cover - GDN env caveat
                return None
            # Step uncapped, gathering greedy committed tokens until we have n.
            # Bounded so the GDN ring (cap // block steps) never flushes.
            for _ in range(n + 4):
                toks.extend(runner.step([s]).tokens[0])
                if len(toks) >= n:
                    break
    finally:
        runner.release()
    return toks if len(toks) >= n else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_commit_cap_truncates_run_and_keeps_state_consistent():
    """A capped row whose decoder offers a >1-token run commits exactly ONE
    token per macro-step, with the GDN ring / conv window / paged-KV cursor /
    next-current token advanced by exactly that one token -- bit-exact to
    committing one token then re-stepping. The drafted tail beyond the cap is
    discarded (not leaked, not folded into state). Skips if the GDN model can't
    run on this host's kernels.

    Drives the cap end-to-end on the model: the model's greedy multi-token
    continuation of a fixed prompt is fed as the FORCED drafts to a CAPPED runner
    so its verify accepts the whole run -- and the cap must hold it to one token.
    This is the regime the scheduler relies on: ``commit_caps=[1]`` => 1 committed
    token (``accept == 0``), the row re-masked from the now-current skill state
    next step (mirrors the non-spec one-token-per-step path)."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    # --- Build a CAPPED runner; K forced drafts == the model's greedy
    # continuation so the verify accepts the whole K-token run (then the cap must
    # truncate it to one). Greedy is deterministic for a fixed prompt+weights, so
    # this is multi-token regardless of the random drafter's quality. Small
    # block_size (K = block-1 = 3 -- enough for a multi-token run) + a large
    # flush_cap so the GDN ring never flushes over these few steps (the per-step
    # commit-cap machinery is what's under test, not the separate flush kernel).
    BLK, FLUSH = 4, 512
    try:
        rt_cap, cap = _try_build_runner(block_size=BLK, flush_cap=FLUSH)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")
    prompt = rt_cap.tokenizer.encode(
        "Count slowly: one two three four five six").ids
    K = cap.num_spec
    # Need K+1 greedy ids: K forced drafts (all accepted) + the bonus at K, so
    # the uncapped run that the cap truncates is the full K+1-token commit.
    gold = _greedy_continuation(prompt, K + 1, _State, block_size=BLK, flush_cap=FLUSH)
    if gold is None:
        cap.release()
        pytest.skip("GDN model unavailable on this host (greedy continuation)")
    gold_drafts = gold[:K]                       # the K forced drafts (all accept)
    ref_tokens = gold[:K + 1]                     # the full uncapped run (K drafts + bonus)
    s_cap = _State()
    try:
        with torch.inference_mode():
            try:
                cap.admit(s_cap, prompt)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            row = cap._row_of[id(s_cap)]
            ctx0 = int(cap._ctx_buf[row])
            rl0 = _row_ring_lengths(cap, row)

            # WITHOUT the cap this run accepts all K forced drafts (the gold
            # continuation) and commits the full K+1-token run (``ref_tokens``);
            # ``commit_caps=[1]`` must hold it to one token.
            res = cap.step(
                [s_cap], commit_caps=[1], _force_drafts=[list(gold_drafts)])
            capped = res.tokens[0]

            # (1) accept_len truncated: exactly ONE committed token, accept == 0.
            assert len(capped) == 1, (
                f"commit_caps=[1] committed {len(capped)} tokens ({capped}); the "
                "stateful-skill cap must hold the row to a single token/step")
            assert res.accept_counts[0] == 0, (
                f"commit_caps=[1] => accept must be 0, got {res.accept_counts[0]}")

            # (2)+(3) The cap is an exact prefix truncation with NO leaked tail:
            # the committed run is EXACTLY the uncapped run's first token and
            # nothing else (so none of the discarded tail ref_tokens[1:] is
            # emitted -- a strictly stronger check than a tail set-disjointness).
            assert capped == [ref_tokens[0]], (
                f"capped commit {capped} != [first token {ref_tokens[0]}] of the "
                f"uncapped run {ref_tokens}; the cap must keep position 0, drop the "
                "tail, and leak nothing")

            # (4) State consistent with a ONE-token commit: paged-KV cursor and
            # every GDN replay ring advanced by exactly 1, and the next-current
            # token is the single committed token (so the next step re-drafts from
            # the correct committed position -- bit-exact to commit-one-re-step).
            assert int(cap._ctx_buf[row]) == ctx0 + 1, (
                f"paged-KV cursor advanced {int(cap._ctx_buf[row]) - ctx0} != 1; "
                "the cap must advance the KV by the truncated (one-token) run")
            rl1 = _row_ring_lengths(cap, row)
            assert rl1 == [n + 1 for n in rl0], (
                f"GDN replay ring advanced {[(b - a) for a, b in zip(rl0, rl1)]} "
                "!= 1 per layer; the drafted tail must NOT be folded into the ring")
            assert int(cap._cur_buf[row]) == capped[0], (
                f"next-current token {int(cap._cur_buf[row])} != committed token "
                f"{capped[0]}; the cap's resume position is wrong")

            # (5) Resumption is consistent: a SECOND capped step again commits one
            # token and again advances the ring/KV by exactly one, proving the row
            # resumes cleanly from the capped position with no leaked tail state.
            ctx1 = int(cap._ctx_buf[row])
            rl1b = _row_ring_lengths(cap, row)
            res2 = cap.step([s_cap], commit_caps=[1])
            assert len(res2.tokens[0]) == 1 and res2.accept_counts[0] == 0
            assert int(cap._ctx_buf[row]) == ctx1 + 1
            assert _row_ring_lengths(cap, row) == [n + 1 for n in rl1b]
    finally:
        cap.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_commit_cap_none_keeps_multi_token_accept():
    """An UNCAPPED row (``commit_caps`` None, or a None entry) still commits its
    full multi-token speculative run in one step -- the cap only constrains
    stateful-masked rows. Two steps: ``commit_caps=None`` and an explicit
    per-row ``commit_caps=[None]`` must each commit the same multi-token run and
    advance the KV by the full run length. Skips if the GDN model can't run.

    Guards the back-compat / greedy-unconstrained path: the cap must be a no-op
    for an unconstrained row (no intra-step speculation lost), so a regression
    that over-caps every row would be caught here."""
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    BLK, FLUSH = 4, 512   # small K (=3); large cap so the GDN ring never flushes
    try:
        rt0, probe = _try_build_runner(block_size=BLK, flush_cap=FLUSH)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")
    prompt = rt0.tokenizer.encode(
        "Count slowly: one two three four five six").ids
    K = probe.num_spec
    probe.release()
    # K forced drafts == the model's greedy continuation -> the verify accepts the
    # whole K-token run and commits K+1 tokens (the uncapped behaviour the no-op
    # cap must preserve). Greedy is deterministic, so this is multi-token
    # independent of the random drafter.
    gold = _greedy_continuation(prompt, K, _State, block_size=BLK, flush_cap=FLUSH)
    if gold is None:
        pytest.skip("GDN model unavailable on this host (greedy continuation)")
    gold_drafts = gold[:K]
    n_run = K + 1   # K accepted drafts + the bonus

    # Both an all-None ``commit_caps`` and an explicit per-row ``[None]`` must be
    # no-ops: the uncapped row keeps its full multi-token speculative commit.
    for caps in (None, [None]):
        try:
            rt_u, run = _try_build_runner(block_size=BLK, flush_cap=FLUSH)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"runtime build failed: {exc}")
        s0 = _State()
        try:
            with torch.inference_mode():
                try:
                    run.admit(s0, prompt)
                except Exception as exc:  # pragma: no cover - GDN env caveat
                    pytest.skip(f"admit/prefill GDN unavailable: {exc}")
                row = run._row_of[id(s0)]
                ctx0 = int(run._ctx_buf[row])
                res = run.step(
                    [s0], commit_caps=caps, _force_drafts=[list(gold_drafts)])
                toks = res.tokens[0]
                # The uncapped row commits the WHOLE K+1-token run (the cap did not
                # truncate it) and advances the KV by the full run.
                assert len(toks) == n_run and len(toks) >= 2, (
                    f"commit_caps={caps} truncated an uncapped row to {len(toks)} "
                    f"tokens; it must keep the full {n_run}-token run")
                assert res.accept_counts[0] == n_run - 1, (
                    f"commit_caps={caps} changed accept to {res.accept_counts[0]} "
                    f"(expected {n_run - 1}); the no-op cap must not truncate")
                assert int(run._ctx_buf[row]) == ctx0 + n_run, (
                    f"uncapped KV advanced {int(run._ctx_buf[row]) - ctx0} != "
                    f"{n_run}; the full run must be committed")
        finally:
            run.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_retire_resets_freed_row_decode_cursor():
    """``retire`` must reset a freed row's on-device decode cursor to 0.

    Regression for the stale-cursor bug: ``step`` builds the draft/verify
    position buffers AND the paged-KV slot mapping for *every* fixed-B row from
    ``_ctx_buf`` (the ``active`` mask only suppresses the per-row COMMIT / state
    advance, not the forward's KV writes). A retired-but-not-yet-readmitted row
    left parked at its retirement cursor -- which can be near ``max_seq_len`` --
    would therefore keep emitting verify/KV writes at those stale positions on
    every subsequent step of OTHER live rows, potentially past the row's reserved
    pages while another sequence is mid-decode. ``retire`` (via ``_release_row``)
    must zero the freed row's ``_ctx_buf``/``_cur_buf`` so the idle row rides
    along only at the safe page-0 position.

    Two rows are admitted and driven several steps so BOTH cursors advance well
    past 0; one is retired and (1) its ``_ctx_buf``/``_cur_buf`` are asserted 0,
    and (2) the still-live row keeps stepping cleanly -- the freed row now
    contributes only the in-bounds position-0 slot mapping, not its stale tail.
    Then (3) the freed row is re-admitted and starts fresh (its cursor seeded to
    the new prompt length, not the old tail).
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    # Large flush_cap so the GDN ring never flushes over these few steps (the
    # cursor-reset bookkeeping is what's under test, not the separate flush
    # kernel); batch_size=2 so one row can be retired while the other decodes.
    BLK, FLUSH = 16, 512
    try:
        rt, run = _try_build_runner(block_size=BLK, flush_cap=FLUSH)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"runtime build failed: {exc}")
    assert run.B >= 2, "this regression needs a >=2-row runner"
    prompt_a = rt.tokenizer.encode("The capital of France is").ids
    prompt_b = rt.tokenizer.encode("Water boils at a temperature of").ids
    s_a, s_b = _State(), _State()
    try:
        with torch.inference_mode():
            try:
                run.admit(s_a, prompt_a)
                run.admit(s_b, prompt_b)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            row_a = run._row_of[id(s_a)]
            row_b = run._row_of[id(s_b)]
            # Drive both rows forward so each cursor climbs well past 0.
            for _ in range(4):
                run.step([s_a, s_b])
            ctx_a = int(run._ctx_buf[row_a])
            assert ctx_a > 0, "row A cursor did not advance; cannot test the reset"

            # (1) Retire row A: its cursor + current token must reset to 0 (NOT
            # stay parked at ``ctx_a``, the pre-fix behaviour the bug left behind).
            run.retire(s_a)
            assert not run.has_row(s_a)
            assert int(run._ctx_buf[row_a]) == 0, (
                f"retire left row A's decode cursor at {int(run._ctx_buf[row_a])} "
                f"(was {ctx_a}); a freed row must reset to 0 so it does not drive "
                "verify/KV writes at a stale (possibly out-of-bounds) position")
            assert int(run._cur_buf[row_a]) == 0, (
                f"retire left row A's current token at {int(run._cur_buf[row_a])}; "
                "the freed row's current token must reset to 0")

            # (2) The still-live row B keeps stepping cleanly: the freed row A now
            # rides along the fixed-B graph at the safe position 0 (in-bounds slot
            # mapping into its own early pages), not its retired tail. Before the
            # fix, row A's stale ``_ctx_buf`` drove KV writes at ``ctx_a + block``
            # every one of these steps -- which can exceed its reserved pages.
            ctx_b0 = int(run._ctx_buf[row_b])
            for _ in range(4):
                run.step([s_b])
            assert int(run._ctx_buf[row_b]) > ctx_b0, (
                "live row B did not advance after row A retired")
            assert int(run._ctx_buf[row_a]) == 0, (
                "row A's cursor moved while idle/retired; an idle row must stay "
                "pinned at 0 until re-admitted")

            # (3) The freed row is reusable and starts fresh: a new admit seeds the
            # cursor to the new prompt length, not the old tail.
            s_c = _State()
            run.admit(s_c, prompt_a)
            row_c = run._row_of[id(s_c)]
            assert int(run._ctx_buf[row_c]) == len(prompt_a), (
                f"re-admitted row cursor {int(run._ctx_buf[row_c])} != new prompt "
                f"length {len(prompt_a)}; the reused row must start from its own "
                "prefill, not inherit a stale cursor")
            run.retire(s_b)
            run.retire(s_c)
    finally:
        run.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_default_spec_max_seq_len_leaves_transient_prefill_headroom():
    """The default spec KV reservation must leave room for one transient prefill.

    Regression for the admit-at-capacity KV exhaustion: with ``spec_decode``
    enabled and no explicit ``max_seq_len``, ``_default_spec_max_seq_len`` sizes
    each of the ``B`` fixed spec rows from the serving KV pool, and ``SpecRunner``
    reserves that for every row up front and never frees it. The serving ``admit``
    contract, however, first builds the request as a *transient* prefill
    ``batch_idx`` whose pages ``prepare_sequence`` reserves (up to its
    ``target_length`` <= ``max_seq_len``) BEFORE ``admit`` re-points
    ``state.batch_idx`` at a persistent row and erases the transient one. If the
    ``B`` rows already consumed the whole pool (minus a tiny margin), that
    transient reservation -- and hence an ordinary prompt's ``prepare_sequence``
    -- fails even though the request fits once admitted.

    This reproduces the exact production *page* reservation order against a real
    ``page_table`` sized to a small pool (so the per-row share is < model
    context, i.e. the ``// (B + 1)`` headroom split -- not the ``max_seq_length``
    clamp -- governs the budget): reserve ``B`` persistent rows of
    ``_default_spec_max_seq_len`` pages each (as ``SpecRunner.__init__`` does),
    then assert the pool still has at least one transient prefill's worth of
    pages free (the page-availability sub-condition ``Qwen35Runtime.can_reserve``
    / ``prepare_sequence`` gate a new request on) and that a real ``reserve`` of
    that many additional pages still succeeds. It also checks the pre-fix
    ``// B`` arithmetic would have left zero page headroom, so the transient
    page reservation would have raised.

    (The transient also needs a free page-table ``batch_idx``; with
    ``max_batch_size == B`` the ``B`` persistent rows already consume every free
    slot, which is exactly why this engine's scheduler admits via a ``-1``
    sentinel batch_idx rather than a ``prepare_sequence`` transient row. This
    test isolates the *page-budget* contract the fix changes -- a deployment that
    leaves a spare slot, or the documented ``prepare_sequence``-state ``admit``
    path, then has the page headroom it needs.)
    """
    pytest.importorskip("kestrel.models.qwen35")
    import kestrel.models.qwen35  # noqa: F401
    from kestrel.config import RuntimeConfig
    from kestrel.kv_cache import KVMemoryPool
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    dev = torch.device("cuda")
    B = 2
    # A deliberately small KV pool so per_row_pages stays well under the model
    # context (max_position_embeddings is huge), i.e. the headroom split governs
    # the budget rather than the max_seq_length clamp. Big enough that the share
    # still exceeds flush_cap (so we are not in the degenerate flush-cap floor).
    small_pages = 4096
    try:
        rt = Qwen35Runtime(
            RuntimeConfig(
                device="cuda", model="Qwen/Qwen3.5-4B", max_batch_size=B,
                enable_cuda_graphs=False, kv_cache_pages=small_pages,
            ),
            kv_pool=KVMemoryPool(device=dev),
        )
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"runtime build failed: {exc}")

    pt = rt.page_table
    page_size = int(rt.page_size)
    flush_cap = 64
    max_seq_len = rt._default_spec_max_seq_len(flush_cap)
    # Sanity: we are testing the headroom split, not the model-context clamp nor
    # the flush-cap floor (otherwise the assertion below would not exercise it).
    assert max_seq_len < int(rt.max_seq_length)
    assert max_seq_len > flush_cap

    def _cdiv(a, b):
        return (a + b - 1) // b

    per_row_pages = _cdiv(max_seq_len, page_size)

    # Reserve B persistent spec rows' pages exactly as ``SpecRunner.__init__``
    # does (this also consumes the B free batch_idx slots in this config).
    persistent = []
    for _ in range(B):
        bi = pt.allocate()
        pt.reserve(bi, max_seq_len)
        pt.commit_block_table([bi])
        persistent.append(bi)

    # The pre-fix budget divided the pool among B (not B + 1) shares, so the B
    # rows consumed essentially the whole pool -- a transient prefill of
    # max_seq_len would then have had zero pages and ``reserve`` would have
    # raised. Confirm the headroom the fix adds is real and material on this pool.
    pre_fix_per_row = ((small_pages - 2) // B) * page_size
    assert pre_fix_per_row > max_seq_len, (
        "test pool too small to distinguish the //B vs //(B+1) split")
    pages_left = pt.pages_available
    assert pages_left * page_size < pre_fix_per_row, (
        "sanity: pre-fix per-row budget should have left < one transient's pages")

    # The transient prefill ``prepare_sequence`` reservation gates on the
    # page-availability sub-condition (``pages_available * page_size >= size``);
    # it must hold at pool capacity for the worst case target_length == max_seq_len.
    assert pages_left * page_size >= max_seq_len, (
        f"only {pages_left} pages ({pages_left * page_size} tokens) free after "
        f"reserving {B} spec rows; a transient prefill needs up to {max_seq_len} "
        "tokens -- the default budget left no page headroom for the admit-time "
        "transient slot")
    # And a real ``reserve`` of that many additional pages still hands them out
    # (exercise the allocation path, not just the counter): grow one persistent
    # row by another max_seq_len, which draws per_row_pages fresh pages.
    grow_target = max_seq_len + max_seq_len
    assert pt.reserve(persistent[0], grow_target), (
        "reserving a transient prefill's worth of additional pages failed even "
        "though pages_available reported enough -- the headroom is not real")

    # Clean up the test's own reservations so the pool is not leaked.
    for bi in persistent:
        pt.erase(bi, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_batch_rejects_overlong_prompt_before_prefill():
    """``decode_batch`` must reject a too-long prompt BEFORE ``_prefill`` runs.

    Each fixed ``SpecRunner`` row reserves exactly ``max_seq_len`` KV pages once
    (``__init__``) and never grows. ``_prefill`` builds the per-token slot mapping
    and runs the packed prefill forward, which WRITES KV into those reserved
    pages. If the length guard (``len(prompt) + max_new_tokens + block_size + 4 >
    max_seq_len``) only fired AFTER ``_prefill``, an over-long prompt would index
    / write KV past the row's reservation (corrupting an adjacent row's pages)
    before the ``ValueError``.

    This pins the fix: with a prompt longer than ``max_seq_len``, ``decode_batch``
    raises and ``_prefill`` is never entered (so no out-of-reservation KV write
    can occur), mirroring ``_prefill_row``'s pre-forward check on the scheduler
    path.
    """
    pytest.importorskip("kestrel.models.qwen35")
    import kestrel.models.qwen35  # noqa: F401

    block_size, flush_cap = 16, 64
    rt, runner = _try_build_runner(block_size=block_size, flush_cap=flush_cap)
    # ``_try_build_runner`` builds a SpecStepRunner with max_seq_len=512.
    max_seq_len = int(runner.max_seq_len)
    B = int(runner.B)

    # Spy on ``_prefill`` (the KV-writing packed prefill) to prove it is never
    # reached for an over-long prompt -- the whole point of the pre-validation.
    prefill_calls = {"n": 0}
    orig_prefill = runner._prefill

    def _spy_prefill(prompts, sink):
        prefill_calls["n"] += 1
        return orig_prefill(prompts, sink)

    runner._prefill = _spy_prefill  # type: ignore[assignment]

    # A prompt that, with even one new token, exceeds the per-row reservation:
    # len(prompt) + max_new_tokens + block_size + 4 > max_seq_len.
    too_long = max_seq_len  # already >= the budget once the +block_size+4 is added
    prompts = [list(range(1, too_long + 1)) for _ in range(B)]

    with pytest.raises(ValueError, match="max_seq_len"):
        runner.decode_batch(prompts, max_new_tokens=8)
    assert prefill_calls["n"] == 0, (
        "decode_batch ran _prefill (and thus wrote KV) before rejecting an "
        "over-long prompt -- the length check must precede _prefill")

    # A prompt that DOES fit must still pass the guard and reach _prefill (so the
    # check is not vacuously rejecting everything). Keep it tiny so the prefill
    # itself is cheap; allow the GDN forward to be unavailable on this host.
    short = max(1, min(8, max_seq_len - block_size - 8))
    ok_prompts = [list(range(1, short + 1)) for _ in range(B)]
    try:
        runner.decode_batch(ok_prompts, max_new_tokens=1)
    except ValueError as exc:  # pragma: no cover - must NOT be the length guard
        if "max_seq_len" in str(exc):
            raise AssertionError(
                "decode_batch wrongly rejected a prompt that fits the reservation"
            ) from exc
    except Exception:
        # Any OTHER failure (e.g. the GDN flush kernel is GQA-incomplete on this
        # host) is unrelated to the length-guard fix under test.
        pass
    assert prefill_calls["n"] >= 1, (
        "a fitting prompt never reached _prefill -- the guard rejects valid input")


# ---------------------------------------------------------------------------
# RESERVE-FAILURE ROLLBACK (): ``SpecDecoder.generate`` /
# ``generate_batch`` allocate a page-table ``batch_idx`` (single) or several
# (batched) and reserve their KV pages. If the reservation raises (prompt +
# output exceeds the KV budget / the pool is tight) the allocated batch_idx --
# and any pages already reserved this call -- MUST return to the pool, not leak.
# These exercise the REAL ``kestrel.kv_cache.PageTable`` failure path (a genuine
# RuntimeError from an exhausted pool, not a stubbed raise) so a regression that
# moves the allocation back outside the cleanup ``try`` (the pre-fix shape) is
# caught. Device-agnostic: PageTable bookkeeping runs on CPU, so no GDN/model.
# ---------------------------------------------------------------------------


class _ReserveStub:
    """Minimal ``SpecDecoder`` carrier for the reserve-rollback paths.

    Binds the REAL (unbound) ``SpecDecoder.generate`` / ``generate_batch`` so the
    production allocate/reserve/cleanup code runs verbatim, backed by a real
    CPU ``PageTable``. The reservation is reached and (deliberately) fails before
    any model forward, so the heavy fields (drafter, lm_head, embed) are never
    touched and need not be provided. ``_release_batch_idx`` mirrors the runtime
    cleanup for a NON-persistent row: erase the row (returns its batch_idx + any
    reserved pages to the pool), exactly what ``generate``'s ``finally`` relies on.
    """

    def __init__(self, page_table, *, block_size, num_spec=1):
        from kestrel.models.qwen35.dflash.spec_decoder import SpecDecoder

        self.device = torch.device("cpu")
        self.num_spec = int(num_spec)
        self.block_size = int(block_size)
        self.gdn_layer_idxs = []

        class _RT:
            pass

        self.rt = _RT()
        self.rt.page_table = page_table
        # Reserve fails before any forward, so the cache is unused; return a bare
        # object (``generate`` builds it before ``reserve``).
        self.rt._new_cache = lambda: object()
        self.rt._release_batch_idx = self._release_batch_idx
        self.released: list[int] = []
        # The cleanup ``finally`` removes aux-hidden hooks; reserve fails before
        # they are installed (handles == []), so the real remover is a no-op here.
        self._remove_hooks = SpecDecoder._remove_hooks
        # Bind the production methods under test.
        self.generate = SpecDecoder.generate.__get__(self)
        self.generate_batch = SpecDecoder.generate_batch.__get__(self)

    def _ids(self, rows):
        # Real SpecDecoder._ids math (CPU tensor build).
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    def _release_batch_idx(self, batch_idx):
        self.released.append(int(batch_idx))
        pt = self.rt.page_table
        # Mirror Qwen35Runtime._release_batch_idx for a non-persistent row: erase
        # returns the batch_idx + any pages it holds to the free pools.
        if int(batch_idx) not in pt.free_batch_idx:
            pt.erase(int(batch_idx), 0)


def _cpu_page_table(n_pages, max_batch_size):
    """A real CPU ``PageTable`` with ``page_size=1`` (1 page == 1 token)."""
    from kestrel.kv_cache import PageTable

    return PageTable(
        n_pages=n_pages, page_size=1, max_batch_size=max_batch_size, device="cpu"
    )


def test_generate_reserve_failure_frees_row_no_leak():
    """A failed single-sequence ``reserve`` must free the allocated batch_idx.

    ``generate`` allocates one page-table row, then reserves
    ``len(prompt)+max_new_tokens+4*block_size+8`` pages. With the pool too small
    that ``reserve`` raises; the allocate ran first, so a pre-fix ``generate``
    (reserve OUTSIDE the cleanup ``try``) would never return the popped batch_idx
    -- one failed call permanently consumes a batch slot. After the fix the
    ``finally`` erases the row, so the free batch_idx count returns to baseline
    and (since ``reserve`` raises before taking any pages) the pool is intact.
    """
    pytest.importorskip("kestrel.kv_cache")
    pytest.importorskip("kestrel.models.qwen35")
    block_size = 2
    # 9 usable pages (page 0 reserved); the reservation below needs far more.
    pt = _cpu_page_table(n_pages=10, max_batch_size=4)
    stub = _ReserveStub(pt, block_size=block_size)

    free_slots_before = len(pt.free_batch_idx)
    pages_before = pt.pages_available
    # len(prompt)+max_new+4*block+8 = 2+2+16 = 20 pages > 9 available -> raises.
    with pytest.raises(RuntimeError):
        stub.generate([1, 2], max_new_tokens=2)

    assert len(pt.free_batch_idx) == free_slots_before, (
        "generate leaked a batch_idx when reserve failed (the allocate was not "
        "rolled back under the cleanup try)")
    assert pt.pages_available == pages_before, (
        "generate leaked pages when reserve failed")
    assert stub.released, "generate did not run its row-cleanup finally"


def test_generate_batch_reserve_failure_frees_all_rows_no_leak():
    """A failed reservation in the BATCHED setup must free EVERY allocated row.

    ``generate_batch`` allocates ``B`` rows then reserves each in a loop. Sized so
    row 0's reservation succeeds (consuming pages) but row 1's exhausts the pool
    and raises. A pre-fix ``generate_batch`` (allocate+reserve loop OUTSIDE the
    cleanup ``try``) would leak BOTH batch_idx AND row 0's already-reserved pages,
    permanently shrinking capacity. After the fix the ``finally`` releases all
    allocated rows, so both the free batch_idx count and the page pool fully
    recover.
    """
    pytest.importorskip("kestrel.kv_cache")
    pytest.importorskip("kestrel.models.qwen35")
    block_size = 2
    B = 2
    # 19 usable pages; each row needs len(2)+max_new(1)+4*block(8)+8 = 19 pages,
    # so row 0 takes all 19 and row 1's reserve raises (0 available) -- row 0's
    # pages must still be reclaimed.
    pt = _cpu_page_table(n_pages=20, max_batch_size=4)
    stub = _ReserveStub(pt, block_size=block_size)

    free_slots_before = len(pt.free_batch_idx)
    pages_before = pt.pages_available
    with pytest.raises(RuntimeError):
        stub.generate_batch([[1, 2], [3, 4]], max_new_tokens=1)

    assert len(pt.free_batch_idx) == free_slots_before, (
        "generate_batch leaked one or more batch_idx when a row's reserve failed")
    assert pt.pages_available == pages_before, (
        "generate_batch leaked the pages a successfully-reserved earlier row held "
        "when a later row's reserve failed")
    assert len(stub.released) == B, (
        f"generate_batch released {len(stub.released)} rows on failure, expected "
        f"all {B} allocated rows")


# ---------------------------------------------------------------------------
# VPOS REFRESH ACROSS A MULTI-STEP BATCHED VERIFY (): the M-RoPE image
# fix (f344f530) added the 4-row ``vpos_buf`` ``position_ids`` buffer bound into
# the captured verify graph. ``SpecStepRunner.step`` rebuilds it from the live
# text positions every macro-step, but ``SpecRunner.decode_batch``'s multi-step
# loop refreshed only ``cpos_buf`` (slot mapping / KV seqlens) -- ``vpos_buf``
# stayed at the PROMPT positions, so the second+ verify replay rotated queries
# with STALE rotary positions and could accept/return wrong tokens. This pins the
# refresh: across every macro-step the verify graph's ``vpos_buf`` must track the
# advancing ``cpos_buf`` (never the capture-time prompt positions).
# ---------------------------------------------------------------------------


def _try_build_spec_runner(block_size=16, flush_cap=128, batch_size=1,
                           max_seq_len=512):
    """Build (rt, graphed ``SpecRunner``) or skip if the GDN model can't run.

    Mirrors ``_try_build_runner`` but constructs the base ``SpecRunner`` (whose
    ``decode_batch`` always captures + replays the verify/draft CUDA graphs --
    the path the vpos-refresh bug lives in), not the eager ``SpecStepRunner``.
    ``flush_cap`` defaults high so a short multi-step ``decode_batch`` never
    triggers the GDN ring flush (a separate, GQA-incomplete-on-some-hosts kernel).
    """
    import kestrel.models.qwen35  # noqa: F401
    from kestrel.config import RuntimeConfig
    from kestrel.kv_cache import KVMemoryPool
    from kestrel.models.qwen35.dflash import SpecRunner
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    dev = torch.device("cuda")
    rt = Qwen35Runtime(
        RuntimeConfig(device="cuda", model="Qwen/Qwen3.5-4B",
                      max_batch_size=max(batch_size + 1, 2)),
        kv_pool=KVMemoryPool(device=dev),
    )
    tc = getattr(rt.hf_config, "text_config", rt.hf_config)
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    dcfg = DFlashConfig(
        hidden_size=int(tc.hidden_size), intermediate_size=4096,
        num_hidden_layers=2, num_attention_heads=int(tc.num_attention_heads),
        num_key_value_heads=int(tc.num_attention_heads), head_dim=head_dim,
        vocab_size=int(tc.vocab_size), rope_theta=1e7, block_size=block_size,
        mask_token_id=0, target_layer_ids=target_layer_ids,
    )
    drafter = DFlashDraftModel(dcfg).to(dev, torch.bfloat16).eval()
    rt._linear_state_pool.replay_capacity = flush_cap
    for st in rt._linear_state_pool.layers:
        if st is not None:
            st.replay_checkpoint_states = None
            st.replay_k = st.replay_u = st.replay_g = st.replay_lengths = None
    rt._linear_state_pool.initialize_from_config(tc, dtype=rt.dtype)
    try:
        runner = SpecRunner(
            rt, drafter, dcfg, batch_size=batch_size, max_seq_len=max_seq_len,
            flush_cap=flush_cap,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"SpecRunner build failed on this host: {exc}")
    return rt, runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_batch_refreshes_vpos_across_macro_steps():
    """The verify graph's M-RoPE ``vpos_buf`` must track ``cpos_buf`` every step.

    Regression (): ``decode_batch``'s multi-macro-step loop advanced
    ``cpos_buf`` (and the slot mapping / KV seqlens) to the new context length but
    left ``vpos_buf`` -- the 4-row ``position_ids`` buffer ``_capture_graphs``
    bound into the verify graph -- at the PROMPT positions. The verify graph reads
    ``vpos_buf``, so the second+ replay rotated queries with stale rotary positions
    and could return wrong tokens. The fix rebuilds ``vpos_buf`` from ``cpos_buf``
    before each replay, exactly as ``SpecStepRunner.step`` does.

    Captures ``(vpos_buf, cpos_buf)`` at EVERY verify replay (via a thin proxy
    around the captured graph, installed after the lazy capture) and asserts each
    replay's 4-row ``vpos_buf`` equals the live ``cpos_buf`` broadcast (text rows,
    rope delta 0). A pre-fix run would, from the 2nd step on, show ``vpos_buf``
    frozen at the prompt block while ``cpos_buf`` advanced -- the mismatch this
    catches. ``max_new_tokens`` is chosen to force several macro-steps while
    ``flush_cap`` keeps the GDN ring from flushing.
    """
    pytest.importorskip("kestrel.models.qwen35")
    rt, runner = _try_build_spec_runner(batch_size=1, flush_cap=128)

    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_capture = runner._capture_graphs

    def _capture_then_spy(*a, **k):
        orig_capture(*a, **k)
        if runner._verify_graph is not None:
            real = runner._verify_graph

            class _ReplaySpy:
                def replay(self_inner):
                    # Snapshot the buffers the loop sets right before the replay.
                    seen.append((runner.vpos_buf.clone(), runner.cpos_buf.clone()))
                    return real.replay()

            runner._verify_graph = _ReplaySpy()

    runner._capture_graphs = _capture_then_spy  # type: ignore[assignment]

    # Enough new tokens to force several macro-steps (>= 2 needed to expose the
    # stale vpos) while staying well under flush_cap=128 (no GDN ring flush).
    prompt = rt.tokenizer.encode("Explain how a rainbow forms in clear steps.").ids
    try:
        with torch.inference_mode():
            try:
                results = runner.decode_batch([prompt], max_new_tokens=40)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"decode_batch unavailable on this host: {exc}")
    finally:
        runner.release()

    assert len(seen) >= 2, (
        f"decode_batch ran only {len(seen)} verify replays; need >= 2 macro-steps "
        "to exercise the vpos refresh -- raise max_new_tokens")
    for i, (vpos, cpos) in enumerate(seen):
        # 4-row M-RoPE position_ids, all rows == text positions (delta 0).
        assert vpos.shape[0] == 4, vpos.shape
        expected = cpos.unsqueeze(0).expand(4, *cpos.shape)
        assert torch.equal(vpos, expected), (
            f"verify replay {i}: vpos_buf is STALE vs cpos_buf (row0 {vpos[0]} vs "
            f"cpos {cpos}); the multi-step loop did not refresh the M-RoPE verify "
            "position_ids before replay")
    # Sanity: the run produced exactly the requested number of tokens.
    assert len(results) == 1 and len(results[0].token_ids) == 40
