"""Math + distribution tests for the rejection-sampling spec primitives.

These are device-agnostic (run on CPU): they validate the accept/reject rule in
``kestrel.models.qwen35.dflash.sampling`` against the reference algorithm and against
the production single-token sampler's distribution. The on-GPU end-to-end
distribution-equivalence (spec loop vs non-spec sampler over the real model)
lives in the GPU validation harness.

Reference: Leviathan et al. 2022 / Chen et al. 2023 modified rejection sampling,
the rule vLLM implements in ``vllm/v1/sample/rejection_sampler.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import torch

# Load the sampling module directly by path so this pure-math test does not pull
# in the ``kestrel.models.qwen35`` package ``__init__`` (which imports the full engine /
# model registry, irrelevant to the device-agnostic sampler math).
_SAMPLING_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "kestrel" / "models" / "qwen35" / "dflash" / "sampling.py"
)
_spec = importlib.util.spec_from_file_location("_dflash_sampling", _SAMPLING_PATH)
_sampling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sampling)
logits_to_probs = _sampling.logits_to_probs
rejection_sample_block = _sampling.rejection_sample_block


def _ref_sample_next_distribution(logits, temperature, top_p):
    """Exact distribution of ``Qwen35Runtime._sample_next`` for one row.

    Reproduces softmax(logits/T) + sorted-cumsum top-p (keep smallest prefix
    whose cumulative mass first exceeds top_p; >=1 token) + renormalize.
    Returns a normalized prob vector (the sampling distribution).
    """
    logits = logits.to(torch.float32) / max(float(temperature), 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = int((cum <= top_p).sum().item()) + 1
        sorted_probs[keep:] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        out = torch.zeros_like(probs)
        out[sorted_idx] = sorted_probs
        return out
    return probs


def test_logits_to_probs_matches_sample_next():
    """``logits_to_probs`` must reproduce ``_sample_next``'s distribution."""
    torch.manual_seed(0)
    vocab = 512
    for temperature in (0.7, 1.0, 1.3):
        for top_p in (1.0, 0.9, 0.5):
            logits = torch.randn(7, vocab)
            got = logits_to_probs(logits, temperature, top_p)
            for r in range(logits.shape[0]):
                ref = _ref_sample_next_distribution(logits[r], temperature, top_p)
                assert torch.allclose(got[r], ref, atol=1e-6), (
                    f"T={temperature} top_p={top_p} row={r} max_err="
                    f"{(got[r] - ref).abs().max().item()}"
                )


def test_logits_to_probs_top_p_zero_is_disabled():
    """``top_p <= 0`` must DISABLE top-p, matching ``_sample_next``.

    Guards regression (sampling.py): ``_sample_next`` only truncates when
    ``0 < top_p < 1``. If the spec sampler instead treated ``top_p == 0`` as a
    real threshold, the ``prev < top_p`` keep mask would be empty and the row
    would renormalize to all zeros -> NaN / token-0 / ``multinomial`` failure.
    ``top_p <= 0`` must instead pass the full softmax through unchanged, both for
    a scalar and per-row (including a row mixed with normal ``top_p``), and the
    rows must stay finite and normalized (so ``multinomial`` is safe).
    """
    torch.manual_seed(1)
    vocab = 256
    logits = torch.randn(5, vocab)
    plain = torch.softmax(logits, dim=-1)

    # Scalar top_p == 0 and < 0 both disable -> identical to plain softmax.
    for tp in (0.0, -0.3):
        got = logits_to_probs(logits, 1.0, tp)
        assert torch.isfinite(got).all(), f"top_p={tp} produced non-finite probs"
        assert torch.allclose(got.sum(-1), torch.ones(5), atol=1e-6)
        assert torch.allclose(got, plain, atol=1e-6), (
            f"scalar top_p={tp} should be disabled (== plain softmax)"
        )

    # Per-row tensor: a 0 row, a <0 row, a normal-truncating row, a >=1 row.
    tp_vec = torch.tensor([0.0, -0.5, 0.8, 1.0])
    lg = torch.randn(4, vocab)
    plain4 = torch.softmax(lg, dim=-1)
    got = logits_to_probs(lg, 1.0, tp_vec)
    assert torch.isfinite(got).all()
    assert torch.allclose(got.sum(-1), torch.ones(4), atol=1e-6)
    # Disabled rows (0, -0.5, 1.0) pass softmax through unchanged.
    for r in (0, 1, 3):
        assert torch.allclose(got[r], plain4[r], atol=1e-6), (
            f"row {r} (top_p={tp_vec[r].item()}) should be disabled"
        )
    # The 0.8 row truncates (differs from plain) yet stays a valid distribution.
    assert not torch.allclose(got[2], plain4[2], atol=1e-6)
    assert (got[2] > 0).sum().item() < vocab

    # top_p == 0 combined with top_k: top-p disabled, top-k still applies.
    gk = logits_to_probs(logits, 1.0, 0.0, top_k=4)
    assert torch.isfinite(gk).all()
    assert torch.allclose(gk.sum(-1), torch.ones(5), atol=1e-6)
    for r in range(5):
        assert (gk[r] > 0).sum().item() == 4, "top_k=4 must keep exactly 4 tokens"


def _ref_top_k_then_top_p_distribution(logits, temperature, top_p, top_k):
    """Exact distribution of vLLM's ``apply_top_k_top_p`` ordering for one row.

    The reference for the combined top-k + top-p case: softmax(logits/T), keep
    the ``top_k`` highest-prob tokens, **renormalize that surviving mass**, then
    apply the sorted-cumsum top-p threshold (smallest prefix whose cumulative
    post-top-k mass first exceeds ``top_p``; >=1 token), then renormalize. This
    is the order vLLM uses (top-k cut precedes the top-p mass threshold), so the
    top-p cutoff is taken over the *post-top-k* distribution -- not the raw
    softmax (which would let a top-k set holding <top_p mass keep its whole self).
    """
    logits = logits.to(torch.float32) / max(float(temperature), 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    vocab = probs.shape[-1]
    if 0 < int(top_k) < vocab:
        sorted_probs[int(top_k):] = 0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
    if 0.0 < top_p < 1.0:
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = int((cum <= top_p).sum().item()) + 1
        sorted_probs[keep:] = 0
    sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
    out = torch.zeros_like(probs)
    out[sorted_idx] = sorted_probs
    return out


def test_logits_to_probs_top_k_then_top_p_renormalizes():
    """``logits_to_probs`` applies top-p over the *renormalized* post-top-k mass.

    Guards regression (sampling.py): when both ``top_k`` and ``top_p`` are
    set, top-p's cumulative threshold must be taken over the renormalized top-k
    distribution (vLLM order), not the raw softmax. Small ``vocab`` + small
    ``top_k`` (so the kept set holds well under ``top_p`` of the *raw* mass) makes
    the mis-ordering observable: without the renorm the cumsum never reaches
    ``top_p`` and top-p keeps the entire top-k set instead of its smallest prefix.
    """
    torch.manual_seed(0)
    vocab = 64
    # top_k small relative to vocab so the kept mass is a fraction of the whole;
    # top_p strictly inside that kept mass so the post-top-k threshold actually
    # truncates (and differs from "keep all of top_k").
    for temperature in (1.0, 0.8):
        for top_k in (3, 5, 8):
            for top_p in (0.5, 0.8, 0.9):
                logits = torch.randn(6, vocab)
                got = logits_to_probs(logits, temperature, top_p, top_k)
                for r in range(logits.shape[0]):
                    ref = _ref_top_k_then_top_p_distribution(
                        logits[r], temperature, top_p, top_k)
                    assert torch.allclose(got[r], ref, atol=1e-6), (
                        f"T={temperature} top_k={top_k} top_p={top_p} row={r} "
                        f"max_err={(got[r] - ref).abs().max().item()}; top-p must "
                        f"be taken over the renormalized post-top-k mass."
                    )
                    # The fix is observable: more than `top_k` survivors would mean
                    # top-p ran over raw mass; here the kept count must be <= top_k
                    # and (because top_p truncates inside it) typically < top_k.
                    assert int((got[r] > 0).sum()) <= top_k, (
                        f"T={temperature} top_k={top_k} top_p={top_p} row={r}: "
                        f"{int((got[r] > 0).sum())} survivors > top_k={top_k}."
                    )


def test_logits_to_probs_top_k_zero_is_disabled():
    """``top_k <= 0`` must DISABLE top-k (no truncation), for scalar AND per-row.

    Guards regression (sampling.py:117): the scalar ``top_k`` path already
    treats ``<= 0`` as disabled, but the PER-ROW tensor path only clamped the
    upper bound. A row with ``top_k == 0`` then made ``rank < 0`` False
    everywhere, zeroing the whole row; the final renormalize left it all zeros so
    ``multinomial`` / Gumbel-argmax returned token 0 instead of sampling from the
    full distribution. ``top_k <= 0`` must instead pass the full softmax through
    (a disabled row equals plain softmax), including a row mixed with a real
    ``top_k``, and a disabled-top_k row combined with ``top_p``.
    """
    torch.manual_seed(0)
    n, vocab = 4, 32
    logits = torch.randn(n, vocab)
    full = torch.softmax(logits, dim=-1)

    # Scalar top_k == 0 and < 0 both disable -> identical to plain softmax.
    for tk in (0, -1):
        got = logits_to_probs(logits, 1.0, None, top_k=tk)
        assert torch.isfinite(got).all(), f"top_k={tk} produced non-finite probs"
        assert torch.allclose(got, full, atol=1e-6), (
            f"scalar top_k={tk} should be disabled (== plain softmax)")

    # Per-row tensor: row 0 disabled (top_k=0), row 1 disabled (negative),
    # rows 2/3 keep a real top-k. The disabled rows must equal plain softmax;
    # the active rows must keep exactly their k highest-prob tokens.
    tk_vec = torch.tensor([0, -3, 4, 8], dtype=torch.long)
    got = logits_to_probs(logits, 1.0, None, top_k=tk_vec)
    assert torch.isfinite(got).all(), "per-row top_k produced non-finite probs"
    for r in (0, 1):
        assert torch.allclose(got[r], full[r], atol=1e-6), (
            f"row {r} (top_k={tk_vec[r].item()}) should be disabled (full softmax)")
        # A disabled row must NOT collapse to a one-hot at token 0.
        assert int((got[r] > 0).sum()) == int((full[r] > 0).sum()), (
            f"row {r} disabled top_k must keep the full support, not token 0")
    assert int((got[2] > 0).sum()) == 4, "row 2 top_k=4 must keep exactly 4 tokens"
    assert int((got[3] > 0).sum()) == 8, "row 3 top_k=8 must keep exactly 8 tokens"

    # Disabled per-row top_k combined with a real top_p: top-k off, top-p applies.
    tk_off = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    gp = logits_to_probs(logits, 1.0, 0.9, top_k=tk_off)
    assert torch.isfinite(gp).all()
    for r in range(n):
        ref = _ref_sample_next_distribution(logits[r], 1.0, 0.9)
        assert torch.allclose(gp[r], ref, atol=1e-6), (
            f"row {r}: top_k disabled + top_p=0.9 must equal pure top_p sampling")


def _ref_rejection_emit(p_row, q_row, x, u, residual_pick):
    """One reference accept/reject decision (scalar reference of the rule).

    p_row/q_row: target/draft dists at this position. x: drafted token. u:
    uniform. residual_pick: a token pre-sampled from norm(max(p-q,0)). Returns
    (accepted: bool, emitted_token).
    """
    q_x = float(q_row[x])
    if q_x > 0 and (float(p_row[x]) / q_x) >= u:
        return True, int(x)
    return False, int(residual_pick)


def test_rejection_block_matches_reference_decisions():
    """``rejection_sample_block`` accept count + emitted token == scalar reference.

    Drive both with the *same* uniforms and the same residual draw so the
    decision is deterministic, and check the batched op reproduces the per-row
    walk (accept longest prefix; replacement at first reject; bonus if all
    accepted) exactly.
    """
    torch.manual_seed(1)
    n, k, vocab = 16, 5, 200
    # Random target logits; random (independent) draft probs -> realistic q!=p.
    target_logits = torch.randn(n, k + 1, vocab)
    draft_logits = torch.randn(n, k, vocab)
    temperature, top_p = 1.0, 1.0
    q = logits_to_probs(draft_logits.reshape(n * k, vocab), temperature, top_p).reshape(n, k, vocab)
    # Draft tokens sampled from q (as the drafter would).
    gen = torch.Generator().manual_seed(2)
    x = torch.multinomial(q.reshape(n * k, vocab), 1, generator=gen).reshape(n, k)

    # Run the batched op with a fixed generator, then reconstruct the SAME
    # uniforms / residual draws it used is not possible (internal RNG), so
    # instead we recompute p/residual here and verify the *decision structure*:
    # accept count is consistent with the emitted prefix, and the emitted token
    # is either the drafted token (accepted) or a valid residual/bonus token.
    out, accept = rejection_sample_block(x, q, target_logits, temperature, top_p,
                                         generator=torch.Generator().manual_seed(3))
    p = logits_to_probs(target_logits.reshape(n * (k + 1), vocab), temperature, top_p).reshape(n, k + 1, vocab)
    residual = torch.clamp(p[:, :k, :] - q, min=0.0)
    for r in range(n):
        a = int(accept[r])
        assert 0 <= a <= k
        # Accepted prefix must equal the drafted tokens.
        for j in range(a):
            assert int(out[r, j]) == int(x[r, j]), (r, j)
        # The emitted token at column a is the replacement/bonus.
        emitted = int(out[r, a])
        if a < k:
            # Replacement must come from the residual support at the reject pos
            # (max(p-q,0) > 0 there) OR p>0 (residual can be all-zero only if
            # q dominates p everywhere, impossible when a reject happened with
            # p[x] < q[x]); require residual[emitted] > 0 OR p == 0 nowhere.
            assert residual[r, a, emitted] > 0 or residual[r, a].sum() == 0, (
                f"row {r}: replacement {emitted} not in residual support")
        else:
            assert p[r, k, emitted] > 0, f"row {r}: bonus {emitted} has p==0"


def test_rejection_block_is_distribution_exact():
    """The first committed token's distribution must equal the target ``p``.

    This is the whole point of the algorithm: regardless of the draft ``q``, the
    first committed token (``out_tokens[:, 0]``: the accepted draft if position
    0 accepts, else the residual replacement) has marginal distribution exactly
    ``p`` at position 0. Sample many blocks sharing a fixed (p, q) and compare
    the histogram to ``p`` via total-variation distance.
    """
    torch.manual_seed(4)
    vocab = 64
    target_logits = torch.randn(1, 2, vocab)            # [N=1, K+1=2, vocab]
    draft_logits = torch.randn(1, 1, vocab)
    T, top_p = 1.0, 1.0
    p = logits_to_probs(target_logits[:, 0, :], T, top_p)[0]    # target at pos 0
    q = logits_to_probs(draft_logits.reshape(1, vocab), T, top_p)  # [1, vocab]
    q3 = q.reshape(1, 1, vocab)

    S = 400_000
    # Batch the S samples as N=S rows sharing the same (p, q).
    tl = target_logits.expand(S, 2, vocab).contiguous()
    qS = q3.expand(S, 1, vocab).contiguous()
    gen = torch.Generator().manual_seed(5)
    x = torch.multinomial(q.expand(S, vocab), 1, generator=gen)   # [S, 1] drafts ~ q
    out, _accept = rejection_sample_block(
        x, qS, tl, T, top_p, generator=torch.Generator().manual_seed(6))
    emitted = out[:, 0]                                           # [S] first committed tok
    hist = torch.bincount(emitted, minlength=vocab).float() / S
    tv = 0.5 * (hist - p).abs().sum().item()
    # Sampling noise for S=4e5 over 64 bins: TV ~ O(sqrt(V/S)) ~ 6e-3.
    assert tv < 0.01, f"first-token distribution TV from target p = {tv:.4f} (want <0.01)"


def test_rejection_block_all_committed_positions_match_p():
    """EVERY committed position j matches the target dist ``p`` at position j.

    Generalizes the single-token test to the full block: for each output column
    j, conditioning on ``accept >= j`` (so column j is a committed token), the
    token at column j is distributed as ``p`` at position j. This is the
    end-to-end distribution-exactness the spec loop relies on (each committed
    token follows the target's per-position sampling distribution).
    """
    torch.manual_seed(7)
    vocab, K = 48, 3
    target_logits = torch.randn(1, K + 1, vocab)
    draft_logits = torch.randn(1, K, vocab)
    T, top_p = 1.0, 1.0
    p = logits_to_probs(target_logits.reshape(K + 1, vocab), T, top_p)  # [K+1, vocab]
    q = logits_to_probs(draft_logits.reshape(K, vocab), T, top_p)       # [K, vocab]

    S = 600_000
    tl = target_logits.expand(S, K + 1, vocab).contiguous()
    qS = q.reshape(1, K, vocab).expand(S, K, vocab).contiguous()
    gen = torch.Generator().manual_seed(8)
    x = torch.multinomial(q, S, replacement=True, generator=gen).t().contiguous()  # [S, K] ~ q per col
    out, accept = rejection_sample_block(
        x, qS, tl, T, top_p, generator=torch.Generator().manual_seed(9))
    for j in range(K + 1):
        committed = accept >= j           # column j is a committed token for these rows
        toks = out[committed, j]
        if toks.numel() < 20_000:
            continue                       # too few samples at this depth for a tight test
        hist = torch.bincount(toks, minlength=vocab).float() / toks.numel()
        tv = 0.5 * (hist - p[j]).abs().sum().item()
        assert tv < 0.02, (
            f"committed position {j}: TV from p[{j}] = {tv:.4f} "
            f"(n={toks.numel()}, want <0.02)")


def test_greedy_temperature_not_routed_here():
    """Sanity: ``logits_to_probs`` with T>0 never collapses to argmax-only.

    Greedy (T==0) is handled by the caller's exact-argmax path, not this module;
    this just documents that a positive temperature yields a soft distribution.
    """
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
    soft = logits_to_probs(logits, 1.0, 1.0)
    assert soft[0, 1] > 1e-3  # the runner-up keeps real mass at T=1


# ---------------------------------------------------------------------------
# BATCHED rejection sampling: the math ``SpecStepRunner.step`` runs per-row over
# B sequences x K drafts (the continuous-batching spec path). These validate the
# *batched* accept/reject rule -- per-row decisions independent across the batch
# (== the single-sequence sampler == the reference), the ragged per-row
# ``accept_len``, masked (constrained-decode) rows, and mixed greedy+sampling
# rows the way ``step`` selects them -- WITHOUT needing the GDN model (which the
# validation env's kernels cannot prefill; the full-model batched check is GPU-
# gated and skips there). The single-sequence sampler is just B==1 of this, so
# proving the batch matches the reference per row proves both share one rule.
# ---------------------------------------------------------------------------


def _ref_block_walk(p_row, q_row, x_row, u_row, recovered_row, bonus_tok):
    """Scalar reference of one row's modified-rejection walk (Leviathan/Chen).

    Returns ``(accept, committed)`` where ``committed`` is the length-(K+1) list
    of emitted tokens (``committed[:accept]`` the accepted drafts, ``committed[
    accept]`` the replacement/bonus, the tail undefined). Driven by explicit
    ``u`` / pre-sampled residual / bonus draws so the decision is deterministic
    (the same inputs the batched op consumes internally), making the reference
    exact rather than distributional.
    """
    K = x_row.shape[0]
    committed = [-1] * (K + 1)
    accept = 0
    for j in range(K):
        xj = int(x_row[j])
        qx = float(q_row[j, xj])
        ratio = (float(p_row[j, xj]) / qx) if qx > 0 else 0.0
        if qx > 0 and ratio >= float(u_row[j]):
            committed[j] = xj
            accept += 1
        else:
            committed[j] = int(recovered_row[j])
            return accept, committed
    committed[K] = int(bonus_tok)
    return accept, committed


def test_batched_rejection_is_per_row_independent_vs_reference():
    """Batched ``rejection_sample_block`` over B rows == the scalar per-row walk.

    Reproduce the op's exact internal draws (u, residual Gumbel-max, bonus) with
    the SAME seeded generator stream over the SAME [N,K] shape, then drive the
    scalar reference per row with those draws. The batched accept count and the
    whole committed prefix must match the reference for EVERY row -- i.e. a row's
    result depends only on its own (p, q, x), never its batch neighbors. This is
    the exactness ``SpecStepRunner.step`` relies on for ragged per-row commit.
    """
    torch.manual_seed(11)
    N, K, vocab = 24, 5, 128
    T, top_p = 1.0, 1.0
    target_logits = torch.randn(N, K + 1, vocab)
    draft_logits = torch.randn(N, K, vocab)
    q = logits_to_probs(draft_logits.reshape(N * K, vocab), T, top_p).reshape(N, K, vocab)
    p = logits_to_probs(target_logits.reshape(N * (K + 1), vocab), T, top_p).reshape(N, K + 1, vocab)
    x = torch.multinomial(q.reshape(N * K, vocab), 1,
                          generator=torch.Generator().manual_seed(12)).reshape(N, K)

    # Reproduce the op's internal draws bit-for-bit: it draws, in order, u
    # ([N,K] uniform), then recovered ([N*K] Gumbel-max over the residual), then
    # bonus ([N] Gumbel-max over p_K) -- all off the SAME generator. Replaying
    # that exact sequence here lets the scalar reference see identical randomness.
    gen_seed = 99
    g = torch.Generator().manual_seed(gen_seed)
    u = torch.empty(N, K)
    u.uniform_(generator=g)
    residual = torch.clamp(p[:, :K, :] - q, min=0.0)             # [N, K, vocab]
    e1 = torch.empty(N * K, vocab); e1.exponential_(generator=g)
    recovered = torch.argmax(residual.reshape(N * K, vocab) / e1, dim=-1).reshape(N, K)
    e2 = torch.empty(N, vocab); e2.exponential_(generator=g)
    bonus = torch.argmax(p[:, K, :] / e2, dim=-1)               # [N]

    out, accept = rejection_sample_block(
        x, q, target_logits, T, top_p,
        generator=torch.Generator().manual_seed(gen_seed))

    for r in range(N):
        ref_a, ref_commit = _ref_block_walk(
            p[r], q[r], x[r], u[r], recovered[r], int(bonus[r]))
        assert int(accept[r]) == ref_a, (
            f"row {r}: batched accept {int(accept[r])} != reference {ref_a}")
        # Committed prefix (accept drafts + replacement/bonus) must match exactly.
        for j in range(ref_a + 1):
            assert int(out[r, j]) == ref_commit[j], (
                f"row {r} col {j}: batched {int(out[r, j])} != reference "
                f"{ref_commit[j]}")


def test_batched_rejection_ragged_accept_len():
    """The per-row ``accept`` is the ragged longest-accepted-prefix, independent.

    Engineer B rows so row r accepts exactly r drafts: make the drafted token at
    every position have p==q (always accepted: ratio 1 >= u) EXCEPT position r,
    which is drafted as a token the target gives ~zero mass (forces a reject at
    r). The batched op must return accept == [0, 1, 2, ..., K, K] and the
    accepted prefix must echo the drafts -- the ragged shape ``step`` commits.
    """
    K, vocab = 6, 64
    N = K + 2                       # rows accepting 0,1,...,K, and one all-accept
    T, top_p = 1.0, 1.0
    # Build q as a real distribution; set the drafted token per position.
    base = torch.randn(N, K, vocab)
    q = logits_to_probs(base.reshape(N * K, vocab), T, top_p).reshape(N, K, vocab)
    x = torch.multinomial(q.reshape(N * K, vocab), 1,
                          generator=torch.Generator().manual_seed(3)).reshape(N, K)
    # Target logits: make p match q at the drafted token (huge logit there) so it
    # is always accepted, EXCEPT row r at position r, where we make the target
    # assign the drafted token ~no mass (huge NEGATIVE logit -> p(x)~0 -> reject).
    target_logits = torch.full((N, K + 1, vocab), -30.0)
    for r in range(N):
        for j in range(K):
            xj = int(x[r, j])
            if r < K and j == r:
                target_logits[r, j, xj] = -60.0      # reject here
                # give the rest of the row real mass so the residual is well-defined
                target_logits[r, j] = torch.where(
                    torch.arange(vocab) == xj,
                    torch.tensor(-60.0), torch.zeros(vocab))
            else:
                target_logits[r, j, xj] = 60.0       # accept (p[x] ~ 1 >= u)
    # Bonus position: arbitrary valid distribution.
    target_logits[:, K, :] = 0.0

    out, accept = rejection_sample_block(
        x, q, target_logits, T, top_p,
        generator=torch.Generator().manual_seed(4))
    expected = [min(r, K) for r in range(N)]          # rows 0..K-1 -> r; tail -> K
    assert accept.tolist() == expected, (
        f"ragged accept {accept.tolist()} != expected {expected}")
    # Accepted prefix per row must be exactly the drafted tokens.
    for r in range(N):
        a = int(accept[r])
        for j in range(a):
            assert int(out[r, j]) == int(x[r, j]), (r, j)


def test_batched_rejection_masked_row_stays_in_allowed_set():
    """A constrained-decode (masked) row only ever emits allowed ids, batched.

    ``SpecStepRunner`` adds an additive -inf logit mask to BOTH the drafter (q)
    and the verify (p) for a constrained row. Emulate that here: row 0 is masked
    to a small allowed set (its q and target logits both -inf outside it), rows
    1..B-1 are unmasked. Every committed token of the masked row (accepted draft,
    residual replacement, or bonus) must be in the allowed set, and the unmasked
    rows must be unaffected by the masked neighbor.
    """
    torch.manual_seed(21)
    N, K, vocab = 4, 4, 80
    T, top_p = 1.0, 1.0
    allowed = [3, 17, 42, 60]
    mask_row = torch.zeros(vocab)
    keep = torch.zeros(vocab, dtype=torch.bool)
    keep[torch.tensor(allowed)] = True
    mask_row.masked_fill_(~keep, float("-inf"))

    draft_logits = torch.randn(N, K, vocab)
    target_logits = torch.randn(N, K + 1, vocab)
    # Apply the mask to row 0's drafter + verify logits (the step's mask fold).
    draft_logits[0] = draft_logits[0] + mask_row[None, :]
    target_logits[0] = target_logits[0] + mask_row[None, :]

    q = logits_to_probs(draft_logits.reshape(N * K, vocab), T, top_p).reshape(N, K, vocab)
    x = torch.multinomial(q.reshape(N * K, vocab), 1,
                          generator=torch.Generator().manual_seed(22)).reshape(N, K)
    out, accept = rejection_sample_block(
        x, q, target_logits, T, top_p,
        generator=torch.Generator().manual_seed(23))

    # Masked row: every committed token (cols <= accept) must be allowed.
    a0 = int(accept[0])
    committed0 = [int(out[0, j]) for j in range(a0 + 1)]
    assert all(t in allowed for t in committed0), (
        f"masked row committed {committed0} outside allowed {allowed}")
    # The masked drafts themselves were restricted (q had no mass outside allowed).
    assert all(int(x[0, j]) in allowed for j in range(K)), (
        f"masked drafter proposed outside allowed: {x[0].tolist()}")
    # Unmasked rows are not forced into the tiny allowed set (mask is row-local):
    # at least one unmasked committed token lands outside it (the rows are
    # independent, so row 0's mask cannot leak into rows 1..N-1).
    other = [int(out[r, j]) for r in range(1, N) for j in range(int(accept[r]) + 1)]
    assert any(t not in allowed for t in other), (
        "mask leaked into unmasked rows (they should be independent)")


def test_step_assembly_selects_per_row_greedy_vs_sampled():
    """Mirror ``SpecStepRunner.step``'s mixed greedy/sampling per-row select.

    ``step`` computes the greedy argmax commit (``pred``/greedy ``accept``) AND
    the rejection-sampled commit (``samp_tokens``/``accept_samp``) over the whole
    fixed-B batch, then selects per row by ``temperature>0`` and slices each row
    to ``accept_r+1`` committed tokens. This reproduces that selection + ragged
    slice standalone (no GDN) and checks greedy rows take the exact-argmax result
    while sampling rows take the rejection result -- the contract the runner's
    committed-token / accept tensors must satisfy.
    """
    eps = _sampling.GREEDY_TEMPERATURE_EPS
    B, K = 4, 5
    # Per-row knobs: rows 0,2 greedy; rows 1,3 sampling.
    temp_b = torch.tensor([0.0, 0.8, 0.0, 1.0])
    samp_mask_b = temp_b > eps
    # Distinct sentinel tokens so a wrong select is unmistakable.
    pred = torch.tensor([[10, 11, 12, 13, 14, 15],          # greedy commit (argmax)
                         [20, 21, 22, 23, 24, 25],
                         [30, 31, 32, 33, 34, 35],
                         [40, 41, 42, 43, 44, 45]], dtype=torch.long)
    samp_tokens = torch.tensor([[-1] * (K + 1),             # row 0 greedy: unused
                                [200, 201, -1, -1, -1, -1],  # row 1 sampled
                                [-1] * (K + 1),              # row 2 greedy: unused
                                [400, -1, -1, -1, -1, -1]],  # row 3 sampled
                               dtype=torch.long)
    accept_greedy = torch.tensor([2, 4, 0, 5])
    accept_samp = torch.tensor([3, 1, 2, 0])
    # The select step() performs:
    committed = torch.where(samp_mask_b[:, None], samp_tokens, pred)
    accept = torch.where(samp_mask_b, accept_samp, accept_greedy)
    # Greedy rows: exact-argmax pred + greedy accept; sampling rows: rejection.
    assert accept.tolist() == [2, 1, 0, 0]
    out_tokens = [committed[r, : int(accept[r]) + 1].tolist() for r in range(B)]
    assert out_tokens[0] == [10, 11, 12]       # greedy row: pred[:3]
    assert out_tokens[1] == [200, 201]         # sampling row: samp_tokens[:2]
    assert out_tokens[2] == [30]               # greedy row: pred[:1]
    assert out_tokens[3] == [400]              # sampling row: samp_tokens[:1]
    # len(tokens[r]) == accept[r] + 1 (the macro-step invariant the scheduler relies on).
    for r in range(B):
        assert len(out_tokens[r]) == int(accept[r]) + 1


# ---------------------------------------------------------------------------
# End-to-end GPU distribution-equivalence: the spec loop's emitted token must
# follow the target sampling distribution, NOT just match it byte-for-byte
# (RNG streams differ from a plain sampler, so we validate the *distribution*).
# These need CUDA + the real Qwen3.5 target; the random-weight drafter from the
# correctness harness suffices (rejection sampling is distribution-exact for ANY
# draft q -- a bad drafter only lowers acceptance, never biases the output).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def _import_spec_decoder_or_skip():
    """Import the spec-decode entry points, or skip on a kernels-env mismatch.

    The full ``kestrel.models.qwen35`` import chain pulls in ``kestrel_kernels``; a
    validation host whose precompiled kernels ``.so`` predates the model's
    current attention / gated-delta API raises at *import* (e.g. a missing
    ``block_bidirectional_mask`` attr, or the JIT-only ``scripts.precompile``).
    That is the phase-2 env caveat: the same stale-kernels gap that blocks
    prefilling Qwen3.5's asymmetric-head GDN, surfacing here as an import error
    rather than a GDN-runtime error. Skip cleanly so these GPU end-to-end checks
    do not hard-fail on such a host -- the batched / single-seq rejection MATH is
    validated device-agnostically above; full-model batched validation is pending
    the kernels-env fix. Returns the ``SpecDecoder`` class when import succeeds.
    """
    try:
        from kestrel.models.qwen35.dflash import SpecDecoder
    except (ImportError, AttributeError) as exc:  # pragma: no cover - env dependent
        pytest.skip(
            "kestrel.models.qwen35/kestrel_kernels import failed on this host (stale "
            f"kernels .so vs the current model API): {exc}. Full-model spec-decode "
            "is unavailable here; the rejection-sampling math is validated above."
        )
    return SpecDecoder


def _build_runtime_and_drafter(block_size: int = 16):
    """Real Qwen3.5-4B runtime + a dimension-matched random DFlash drafter.

    Mirrors ``tests/qwen35/test_spec_decode_correctness.py`` so the spec loop
    runs the same verify/flush kernels; only the sampling temperature changes.
    """
    _import_spec_decoder_or_skip()      # skip early if kernels env is stale
    import kestrel.models.qwen35  # noqa: F401  (registers model specs)
    from kestrel.config import RuntimeConfig
    from kestrel.kv_cache import KVMemoryPool
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    dev = torch.device("cuda")
    rt = Qwen35Runtime(
        RuntimeConfig(device="cuda", model="Qwen/Qwen3.5-4B", max_batch_size=1),
        kv_pool=KVMemoryPool(device=dev),
    )
    tc = getattr(rt.hf_config, "text_config", rt.hf_config)
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    dcfg = DFlashConfig(
        hidden_size=int(tc.hidden_size),
        intermediate_size=4096,
        num_hidden_layers=2,
        num_attention_heads=int(tc.num_attention_heads),
        num_key_value_heads=int(tc.num_attention_heads),
        head_dim=head_dim,
        vocab_size=int(tc.vocab_size),
        rope_theta=1e7,
        block_size=block_size,
        mask_token_id=0,
        target_layer_ids=target_layer_ids,
    )
    drafter = DFlashDraftModel(dcfg).to(dev, torch.bfloat16).eval()
    return rt, drafter, dcfg


def _target_first_token_probs(spec, prompt_ids, temperature, top_p):
    """Analytic target distribution ``p`` for the FIRST generated token.

    Reproduces exactly what ``SpecDecoder.generate`` verifies against: prefill
    the prompt through the target (``rt._forward_base``), take the last-position
    logits through the (tied) LM head, and apply the SAME temperature/top_p as
    the spec loop (``logits_to_probs``). The spec loop's first committed token is
    -- by the rejection-sampling construction -- distributed as this ``p``.
    """
    import torch as _t

    rt = spec.rt
    dev = spec.device
    batch_idx = rt.page_table.allocate()
    cache = rt._new_cache()
    rt.page_table.reserve(batch_idx, len(prompt_ids) + 64)
    rt.page_table.commit_block_table([batch_idx])
    # No aux-hidden hooks needed: we only read the last-position next-token
    # logits (the spec loop's own ``cur_t = lm_head(lh[0])[-1]`` at prefill).
    try:
        lh, _fcache = rt._forward_base(
            input_ids=spec._ids([prompt_ids]),
            past_key_values=cache,
            batch_idx=batch_idx,
            cache_position_ids=_t.arange(len(prompt_ids), device=dev).view(1, -1),
        )
        logits = spec.lm_head(lh[0])[-1:].float()       # [1, vocab] last position
    finally:
        rt._release_batch_idx(batch_idx)
    return logits_to_probs(logits, temperature, top_p)[0]  # [vocab]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_loop_first_token_matches_target_distribution():
    """The spec loop's FIRST committed token is distributed as the target ``p``.

    Distribution-equivalence (not byte-equality): the spec RNG stream differs
    from a plain sampler, so we sample the FIRST committed token of the
    rejection-sampling spec loop many times (varying the seed) and check the
    histogram matches the analytic target distribution ``p`` (= what the
    non-spec sampler draws from) via total-variation distance, at a couple of
    temperatures. This is the on-GPU analog of
    ``test_rejection_block_is_distribution_exact`` over the real model.

    The first committed token is the prefill ``cur_t`` (``max_new_tokens==1``
    returns *only* token0), so this directly guards that token0 is sampled from
    ``p`` -- NOT chosen greedily before the rejection sampler runs (the bug where
    the prefill ``cur_t = lm_head(...).argmax()`` ignores temperature/top_p/
    top_k/seed). To make that bug detectable regardless of how peaked the target
    is, one config (``T=1.5``) is chosen to spread ``p`` and we assert the
    argmax-token's empirical frequency tracks ``p[argmax]`` instead of collapsing
    to ~1.0 (a greedy token0 would pin it at 1.0).
    """
    SpecDecoder = _import_spec_decoder_or_skip()

    rt, drafter, dcfg = _build_runtime_and_drafter(block_size=16)
    # A high-entropy continuation so the next-token distribution is spread (genuinely
    # exercises accept/reject + residual recovery, not a near-degenerate argmax).
    prompt = rt.tokenizer.encode(
        "Here is a list of ten random English words: apple, river, the"
    ).ids

    # Eager: a distribution test (not perf); eager skips 3000 graph captures and
    # is distribution-identical to graphed (same sampler ops). Greedy eager==graph
    # is separately proven byte-exact by test_spec_decode_correctness.
    S = 2000
    with torch.inference_mode():
        spec = SpecDecoder(rt, drafter, dcfg, flush_cap=64)
        # T=1.5 first: a deliberately *flatter* target so a greedy-collapsed
        # token0 (the bug) is impossible to hide behind a dominant argmax.
        for temperature, top_p in ((1.5, 1.0), (1.0, 1.0), (0.7, 1.0), (1.0, 0.9)):
            p = _target_first_token_probs(spec, prompt, temperature, top_p)
            # Effective support: only tokens p can emit (top_p truncates).
            support = int((p > 0).sum().item())
            argmax_tok = int(p.argmax())
            p_argmax = float(p[argmax_tok])
            counts = torch.zeros_like(p)
            for s in range(S):
                tok = spec.generate(
                    prompt, 1, eager=True,
                    temperature=temperature, top_p=top_p, seed=1000 + s
                ).token_ids[0]
                counts[tok] += 1
            hist = counts / S
            tv = 0.5 * (hist - p).abs().sum().item()
            # Expected TV from pure multinomial(S) sampling noise of the EXACT
            # distribution p: E|hist_i - p_i| ~ sqrt(p_i(1-p_i)/S) (half-normal
            # mean would add a 0.8 factor; keep the conservative sqrt). A
            # distribution-exact sampler lands at ~1.0-1.5x this; any real
            # q/p/accept bias (e.g. swapped ratio, missing residual recovery)
            # shifts the mass by O(0.1-0.5), i.e. many x the noise floor. The 4x
            # gate cleanly separates correct from incorrect without flaking.
            exp_tv = 0.5 * (p * (1.0 - p) / S).clamp_min(0.0).sqrt().sum().item()
            bound = max(0.02, 4.0 * exp_tv)
            assert tv < bound, (
                f"T={temperature} top_p={top_p}: spec-loop first-token TV from "
                f"target p = {tv:.4f} > {bound:.4f} (4x noise floor "
                f"{exp_tv:.4f}, support={support}, S={S}). The emitted "
                f"distribution must equal the target sampling distribution."
            )
            # Sanity: the loop did NOT collapse to greedy (it actually sampled).
            if support > 1 and float(p.max()) < 0.97:
                assert int((counts > 0).sum().item()) > 1, (
                    f"T={temperature} top_p={top_p}: spec loop emitted a single "
                    f"token over {S} samples -- not sampling from p."
                )
            # Direct anti-greedy guard for token0: a greedy ``cur_t`` (the bug)
            # pins the argmax token's empirical frequency at 1.0. A sampled token0
            # tracks p[argmax]. Assert the argmax frequency is near p[argmax] (3x
            # noise margin) whenever the target is non-degenerate. This fires
            # exactly on the "prefill token chosen greedily under temperature" bug
            # even when p[argmax] is fairly high (e.g. ~0.5-0.9), which the TV
            # bound alone tolerates.
            if p_argmax < 0.9:
                noise = (p_argmax * (1.0 - p_argmax) / S) ** 0.5
                hi = p_argmax + max(0.03, 6.0 * noise)
                assert float(hist[argmax_tok]) <= hi, (
                    f"T={temperature} top_p={top_p}: argmax-token frequency "
                    f"{float(hist[argmax_tok]):.3f} >> p[argmax]={p_argmax:.3f} "
                    f"(<= {hi:.3f}); token0 looks greedy, not sampled from p."
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_loop_greedy_token_equals_target_argmax():
    """Greedy (T==0) must commit exactly the target argmax (no regression).

    Guards that the temperature==0 default still takes the exact-argmax path:
    the first committed token equals ``argmax`` of the target's prefill logits,
    independent of any sampling machinery.
    """
    SpecDecoder = _import_spec_decoder_or_skip()

    rt, drafter, dcfg = _build_runtime_and_drafter(block_size=16)
    prompt = rt.tokenizer.encode("Describe the water cycle step by step.").ids
    with torch.inference_mode():
        spec = SpecDecoder(rt, drafter, dcfg, flush_cap=64)
        p = _target_first_token_probs(spec, prompt, 1.0, 1.0)   # T=1 just to read logits order
        argmax_tok = int(p.argmax())
        greedy_tok = spec.generate(prompt, 1, temperature=0.0).token_ids[0]
    assert greedy_tok == argmax_tok, (
        f"greedy first token {greedy_tok} != target argmax {argmax_tok}; the "
        f"temperature==0 path must stay exact-argmax."
    )


# ---------------------------------------------------------------------------
# Full-model BATCHED (SpecStepRunner) rejection-sampling validation. This is the
# continuous-batching spec path the batched sampler wires; it builds the real
# Qwen3.5 runtime + a SpecStepRunner(sampling=True), admits a sampling row + a
# greedy control row, and checks per-row distribution-exactness over the real
# model. It is GPU-gated AND skips when the host's kernels .so cannot run the
# model (the phase-2 caveat: a stale gated-delta/attention build that cannot
# prefill Qwen3.5's asymmetric-head GDN) -- on such a host this skips and the
# batched rejection MATH validated device-agnostically above stands in. When the
# kernels env is fixed this runs and proves the batched step() sampler end-to-end.
# ---------------------------------------------------------------------------


def _try_build_spec_step_runner(batch_size=2, block_size=16, sampling=True):
    """Build (rt, SpecStepRunner(sampling)) or skip on a kernels-env mismatch.

    Mirrors ``tests/qwen35/test_spec_step_features._try_build_runner`` but
    constructs the runner with ``sampling=sampling`` so the draft graph emits the
    per-position logits the batched rejection sampler consumes.
    """
    _import_spec_decoder_or_skip()
    import kestrel.models.qwen35  # noqa: F401
    from kestrel.config import RuntimeConfig
    from kestrel.kv_cache import KVMemoryPool
    from kestrel.models.qwen35.dflash import SpecStepRunner
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    dev = torch.device("cuda")
    # The runner reserves ``batch_size`` page-table rows for its address-stable
    # pool; the batched-sampling test then builds a reference SpecDecoder whose
    # target-distribution prefill needs one more transient batch slot (it
    # allocates + frees it). Size the pool to batch_size + 1 so that reference
    # prefill does not exhaust free_batch_idx (IndexError: pop from empty list).
    rt = Qwen35Runtime(
        RuntimeConfig(device="cuda", model="Qwen/Qwen3.5-4B",
                      max_batch_size=batch_size + 1, enable_cuda_graphs=False),
        kv_pool=KVMemoryPool(device=dev),
    )
    tc = getattr(rt.hf_config, "text_config", rt.hf_config)
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    flush_cap = 64
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
            rt, drafter, dcfg, batch_size=batch_size, max_seq_len=512,
            flush_cap=flush_cap, use_graphs=False, sampling=sampling,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"SpecStepRunner(sampling) build failed on this host: {exc}")
    return rt, runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_step_runner_batched_sampling_matches_target_distribution():
    """SpecStepRunner.admit (batched, sampling=True) commits the target dist per row.

    The batched continuous-batching analog of
    ``test_spec_loop_first_token_matches_target_distribution``. The FIRST
    committed token of a request is the prefill bonus that ``admit`` samples and
    returns as ``(first_token_id, first_logprob)``; subsequent tokens come from
    ``step`` (which verifies ``[cur_buf, drafts...]`` and so commits the SECOND
    token onward). The single-seq distribution test checks ``generate(prompt, 1)``
    -- i.e. that first bonus token -- so the batched analog must histogram the
    ``admit`` return, not ``step().tokens[row][0]`` (which is the second token,
    drawn from a different, first-token-conditioned distribution and so never
    equals ``p.argmax()``).

    Admit a sampling row (temperature>0) alongside a greedy control row, re-seed
    the spec RNG and re-admit each trial, and check the sampling row's first
    committed (admit) token follows the target distribution ``p`` (TV within the
    noise floor) while the greedy row commits the exact target argmax. One
    ``step`` per trial is still issued so the mixed batched macro-step is
    exercised end to end. Skips cleanly when the host's kernels cannot prefill
    the GDN model (phase-2 caveat); the batched rejection math is validated
    device-agnostically above regardless.
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    rt, runner = _try_build_spec_step_runner(batch_size=2, block_size=16,
                                             sampling=True)
    from kestrel.models.qwen35.dflash import SpecDecoder

    prompt = rt.tokenizer.encode(
        "Here is a list of ten random English words: apple, river, the"
    ).ids
    temperature, top_p = 1.0, 1.0
    # Target distribution p for the first generated token (same construction the
    # single-seq test uses): prefill + last-position logits + temperature/top_p.
    # The runner stores the drafter on its proposer (DFlashProposer), not as a
    # bare ``runner.drafter`` attribute.
    spec = SpecDecoder(rt, runner.proposer.drafter, runner.dcfg, flush_cap=64)
    p = _target_first_token_probs(spec, prompt, temperature, top_p)
    support = int((p > 0).sum().item())

    # Each trial does two full GDN prefills (admit x2) + a macro-step, so keep the
    # sample count modest; the TV bound auto-widens with the multinomial noise
    # floor (~1/sqrt(S)), so a smaller S stays a valid distribution gate.
    S = 200
    counts = torch.zeros_like(p)
    greedy_argmax = int(p.argmax())
    greedy_ok = True
    with torch.inference_mode():
        for s in range(S):
            s0, s1 = _State(), _State()
            # Seed the spec RNG per trial BEFORE admit -- admit samples the first
            # (bonus) token via runner._spec_gen, so the seed must be set first
            # for the sampling row to vary across S.
            runner._spec_gen = torch.Generator(device=runner.device)
            runner._spec_gen.manual_seed(7000 + s)
            try:
                # Sampling row (s0) + greedy control row (s1). admit returns the
                # request's first committed token (id, logprob).
                tok0, _ = runner.admit(s0, prompt, temperature=temperature, top_p=top_p)
                tok1, _ = runner.admit(s1, prompt, temperature=0.0)
            except Exception as exc:  # pragma: no cover - GDN env caveat
                pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")
            counts[int(tok0)] += 1                         # sampling row first tok
            if int(tok1) != greedy_argmax:
                greedy_ok = False
            # Drive one mixed batched macro-step so the step path is exercised.
            runner.step([s0, s1])
            runner.retire(s0)
            runner.retire(s1)
    hist = counts / S
    tv = 0.5 * (hist - p).abs().sum().item()
    exp_tv = 0.5 * (p * (1.0 - p) / S).clamp_min(0.0).sqrt().sum().item()
    bound = max(0.04, 4.0 * exp_tv)
    assert tv < bound, (
        f"batched admit sampling row first-token TV from target p = {tv:.4f} > "
        f"{bound:.4f} (4x noise floor {exp_tv:.4f}, support={support}, S={S})")
    assert greedy_ok, (
        "greedy control row in a mixed batched admit did not commit the target "
        "argmax -- the per-row greedy path must stay exact-argmax alongside a "
        "sampling row.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_admit_releases_transient_prefill_batch_idx():
    """``admit`` must return a serving-supplied transient prefill row to the pool.

    Regression for the transient page-table leak: serving builds the ``state`` via
    ``Qwen35Runtime.prepare_sequence``, which ``page_table.allocate()``s a
    transient prefill ``batch_idx`` (with its own reserved pages) and stores it on
    ``state.batch_idx``. ``admit`` then re-points ``state.batch_idx`` at one of the
    runner's fixed persistent spec rows. Before the fix, that overwrote the only
    reference to the transient row without erasing it, so the slot + its pages were
    never returned to the pool -- and since the runner reserves ``max_batch_size``
    rows up front, every spec admission leaked one slot until ``free_batch_idx``
    was exhausted. After the fix ``admit`` erases the prior transient row first.

    Build the runner, simulate ``prepare_sequence`` by allocating + reserving a
    real transient ``batch_idx``, admit, then assert the transient slot is back in
    ``free_batch_idx`` (and its pages recovered), ``state.batch_idx`` now addresses
    a persistent spec row, and the persistent rows themselves were NOT erased.
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    rt, runner = _try_build_spec_step_runner(batch_size=2, block_size=16,
                                             sampling=False)
    pt = rt.page_table
    prompt = rt.tokenizer.encode("The capital of France is").ids
    persistent = set(runner._persistent_batch_idx)

    with torch.inference_mode():
        # Simulate prepare_sequence: a transient prefill row with its own pages.
        transient = pt.allocate()
        pt.reserve(transient, len(prompt) + 8)
        pt.commit_block_table([transient])
        assert transient not in pt.free_batch_idx
        assert transient not in persistent
        state = _State()
        state.batch_idx = transient

        free_before = set(pt.free_batch_idx)
        pages_before = pt.pages_available
        try:
            runner.admit(state, prompt)
        except Exception as exc:  # pragma: no cover - GDN env caveat
            pt.erase(transient, 0)  # don't leak the test's own transient row
            runner.retire(state)
            pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")

        # The transient prefill row + its pages were returned to the pool.
        assert transient in pt.free_batch_idx, (
            "admit leaked the transient prefill batch_idx (it was overwritten "
            "without being erased)")
        assert pt.pages_available >= pages_before, (
            "admit did not return the transient row's pages to the pool")
        # state now addresses a persistent spec row (not the freed transient one).
        assert int(state.batch_idx) in persistent
        assert int(state.batch_idx) != transient
        # The persistent rows are address-stable: none were erased/freed.
        assert persistent.isdisjoint(pt.free_batch_idx), (
            "a persistent spec row was erased -- admit must only free the "
            "transient prefill row")
        # No double-accounting: exactly the transient row joined the free set
        # (the persistent row admit took was already reserved, not in free).
        assert set(pt.free_batch_idx) - free_before == {transient}

        runner.retire(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_release_sequence_preserves_admitted_spec_row():
    """End-to-end: ``release_sequence`` keeps a spec-admitted row address-stable.

    The serving cleanup path for a finished sequence is
    ``Qwen35Runtime.release_sequence(state)`` -> ``_release_batch_idx``. After
    ``admit`` re-points ``state.batch_idx`` at one of the runner's fixed persistent
    spec rows, that row's page-table mapping + GDN pool slot are captured by the
    spec graphs and MUST stay allocated + address-stable for the runner's lifetime.
    Before the P1 fix ``release_sequence`` erased the row (freeing it for
    reallocation under the live graphs -> page-table corruption). This drives the
    real runtime path and asserts the persistent row is NOT freed, its page-table
    entry is byte-identical, the pool's free count is unchanged, and a subsequent
    re-admit + macro-step over that SAME physical row still commits tokens.
    """
    pytest.importorskip("kestrel.models.qwen35")

    class _State:
        def __init__(self):
            self.batch_idx = -1
            self.length = 0

    rt, runner = _try_build_spec_step_runner(batch_size=2, block_size=16,
                                             sampling=False)
    # Wire the runner onto the runtime exactly as ``_maybe_init_spec_decode``
    # does in production (the manual harness builds the runner but does not set
    # this). ``_release_batch_idx`` reads ``self._spec_runner._persistent_batch_idx``
    # to recognise (and skip) a spec row, so the guard under test depends on it.
    rt._spec_runner = runner
    pt = rt.page_table
    prompt = rt.tokenizer.encode("The capital of France is").ids
    persistent = set(runner._persistent_batch_idx)

    with torch.inference_mode():
        state = _State()
        try:
            runner.admit(state, prompt)
        except Exception as exc:  # pragma: no cover - GDN env caveat
            runner.retire(state)
            pytest.skip(f"admit/prefill GDN unavailable on this host: {exc}")

        spec_row = int(state.batch_idx)
        assert spec_row in persistent, (
            "admit did not re-point state.batch_idx at a persistent spec row")
        # Snapshot the persistent row's page-table mapping + pool free count.
        page_tbl_before = pt.page_table[spec_row].clone()
        free_before = set(pt.free_batch_idx)
        pages_before = pt.pages_available

        # The spec runner's own cleanup for the finished sequence.
        runner.retire(state)
        # ...and then the runtime's normal release path runs for the SAME state
        # (this is the call that, pre-fix, erased the fixed spec row).
        rt.release_sequence(state)

        # The persistent row stayed allocated + address-stable.
        assert spec_row not in pt.free_batch_idx, (
            "release_sequence freed the persistent spec row back to the pool")
        assert persistent.isdisjoint(pt.free_batch_idx), (
            "a persistent spec row leaked into the free list after release")
        assert set(pt.free_batch_idx) == free_before, (
            "release_sequence mutated the page-table free list on a spec row")
        assert pt.pages_available == pages_before, (
            "release_sequence returned the persistent spec row's pages to the pool")
        assert torch.equal(pt.page_table[spec_row], page_tbl_before), (
            "release_sequence corrupted the persistent spec row's page-table entry "
            "(the spec graphs captured this mapping; it must be intact)")

        # Re-admit a fresh sequence: with the row preserved it lands back on the
        # same physical spec rows and a macro-step still commits, proving the
        # page-table entry the graphs reference is live and usable.
        s_a, s_b = _State(), _State()
        runner.admit(s_a, prompt)
        runner.admit(s_b, prompt)
        assert int(s_a.batch_idx) in persistent and int(s_b.batch_idx) in persistent
        result = runner.step([s_a, s_b])
        assert len(result.tokens) == 2 and all(len(t) >= 1 for t in result.tokens), (
            "a macro-step over the preserved spec rows committed no tokens -- the "
            "persistent page-table entry was not intact after release")
        runner.retire(s_a)
        runner.retire(s_b)


def _admit_erase_guard_harness():
    """Build a model-free ``SpecStepRunner`` exercising only ``admit``'s tail.

    The transient-prefill erase reconciliation (``state.batch_idx`` -> persistent
    spec row) is pure Python over ``_persistent_batch_idx`` /
    ``page_table.free_batch_idx`` / ``page_table.erase`` and does NOT depend on the
    GDN model or a GPU. Construct the runner via ``object.__new__`` (skipping
    ``__init__``, which would build the real runtime), populate only the
    attributes ``admit``'s tail reads, and stub the per-row prefill/sampling
    methods that run *before* that tail. A fake page table records every
    ``erase`` so the test can assert whether the guard fired.
    """
    pytest.importorskip("kestrel.models.qwen35")
    from kestrel.models.qwen35.dflash.spec_decoder import SpecStepRunner, _SpecRow

    class _FakePageTable:
        def __init__(self, persistent, free):
            # Fixed (reserved) spec rows live here and must NEVER be erased; the
            # free list is the pool's available transient rows.
            self.free_batch_idx = list(free)
            self._persistent = set(persistent)
            self.erased: list[int] = []

        def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
            self.erased.append(int(batch_idx))
            # Mirror a real page table: erasing a fixed reservation corrupts it.
            assert int(batch_idx) not in self._persistent, (
                f"erase() corrupted a fixed reservation: {batch_idx}")
            if int(batch_idx) not in self.free_batch_idx:
                self.free_batch_idx.append(int(batch_idx))

    class _FakeRT:
        def __init__(self, pt):
            self.page_table = pt

    persistent = [10, 11]
    pt = _FakePageTable(persistent=persistent, free=[20, 21, 22])
    runner = object.__new__(SpecStepRunner)
    runner.B = len(persistent)
    runner.device = torch.device("cpu")
    runner.rt = _FakeRT(pt)
    runner.batch_idx = list(persistent)            # row -> persistent batch_idx
    runner._persistent_batch_idx = set(persistent)
    runner._free_rows = list(range(runner.B))      # both pool rows available
    runner._rows = {}
    runner._row_of = {}
    runner._cur_buf = [0] * runner.B
    runner._ctx_buf = [0] * runner.B
    runner._row_temp = [0.0] * runner.B
    runner._row_top_p = [1.0] * runner.B
    runner._row_logprobs = [False] * runner.B

    # Stub the per-row prefill/sampling that runs before the erase reconciliation
    # (it is the model-dependent part; the guard under test runs after it).
    runner._set_row_mask = lambda row, allowed, suppressed: None
    runner._prefill_row = lambda row, prompt, image=None, image_crops=None: (
        len(prompt), object(), object())
    runner._select_first_token = lambda *a, **k: (7, None)
    runner._detect_typed_token_runtime = lambda: False
    return runner, pt, _SpecRow


def test_admit_sentinel_batch_idx_skips_erase():
    """A ``-1`` sentinel ``state.batch_idx`` must NOT trigger ``page_table.erase``.

    Regression for the sentinel erase bug. A scheduler-created ``state`` that
    never owned a transient prefill row carries the documented ``-1`` sentinel
    ``batch_idx`` ("no prior row to erase"). ``-1`` is neither in
    ``_persistent_batch_idx`` nor in ``free_batch_idx``, so without the
    non-negative guard ``admit`` would misread it as a live transient row and call
    ``erase(-1, 0)`` -- which either raises mid-admission or, under a page table
    that accepts negative indexing, frees the padding/last row and corrupts the
    fixed reservations *before* the real spec row is assigned. Assert no ``erase``
    is issued, the fixed reservations are untouched, and ``state.batch_idx`` is
    still re-pointed at a persistent spec row.
    """
    runner, pt, _SpecRow = _admit_erase_guard_harness()

    class _State:
        def __init__(self):
            self.batch_idx = -1   # the documented "no prior row" sentinel
            self.length = 0

    state = _State()
    free_before = list(pt.free_batch_idx)
    cur, logprob = runner.admit(state, [1, 2, 3])

    assert cur == 7 and logprob is None
    # The sentinel must short-circuit the erase entirely.
    assert pt.erased == [], (
        f"admit erased on a -1 sentinel batch_idx (erased={pt.erased})")
    # The fixed reservations + free pool are exactly as before (nothing freed).
    assert pt.free_batch_idx == free_before, (
        "admit mutated the page-table free list on a sentinel admit")
    assert runner._persistent_batch_idx.isdisjoint(pt.free_batch_idx), (
        "a fixed spec reservation leaked into the free list")
    # state still gets re-pointed at one of the runner's persistent spec rows.
    assert int(state.batch_idx) in runner._persistent_batch_idx


def test_admit_real_transient_batch_idx_still_erased():
    """The non-negative guard must NOT suppress erasing a genuine transient row.

    Companion to the sentinel test: a real serving-supplied transient prefill
    ``batch_idx`` (>= 0, not persistent, not already free) must still be erased so
    its row/pages return to the pool (the original transient-leak fix). Confirms
    the ``>= 0`` guard narrows the skip to the sentinel only.
    """
    runner, pt, _SpecRow = _admit_erase_guard_harness()
    transient = 20
    assert transient in pt.free_batch_idx
    pt.free_batch_idx.remove(transient)  # simulate prepare_sequence allocating it

    class _State:
        def __init__(self):
            self.batch_idx = transient
            self.length = 0

    state = _State()
    runner.admit(state, [1, 2, 3])

    assert pt.erased == [transient], (
        f"admit failed to erase the live transient row (erased={pt.erased})")
    assert int(state.batch_idx) in runner._persistent_batch_idx


def _release_batch_idx_guard_harness():
    """Build a model-free object exercising ``Qwen35Runtime._release_batch_idx``.

    The persistent-spec-row guard in ``_release_batch_idx`` is pure Python over
    ``self._spec_runner._persistent_batch_idx`` / ``page_table.free_batch_idx`` /
    ``page_table.erase`` and needs no GPU or model. Build a fake runtime carrying
    only the attributes the method touches, bind the *real* unbound method to it,
    and record every ``erase`` / ``clear`` so the test can assert what fired.
    """
    pytest.importorskip("kestrel.models.qwen35")
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    class _FakePageTable:
        def __init__(self, persistent, free):
            self.free_batch_idx = list(free)
            self._persistent = set(persistent)
            self.erased: list[int] = []

        def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
            self.erased.append(int(batch_idx))
            # Mirror a real page table: erasing a fixed reservation corrupts it.
            assert int(batch_idx) not in self._persistent, (
                f"erase() corrupted a fixed spec reservation: {batch_idx}")
            if int(batch_idx) not in self.free_batch_idx:
                self.free_batch_idx.append(int(batch_idx))

    class _FakeRunner:
        def __init__(self, persistent):
            self._persistent_batch_idx = set(persistent)

    class _FakeRT:
        def __init__(self, persistent, free):
            self.page_table = _FakePageTable(persistent, free)
            self._spec_runner = _FakeRunner(persistent)
            self.active_sequences = {}
            self._caches = {}
            self.cleared: list[int] = []
            # Bind the real method under test to this fake instance.
            self._release_batch_idx = (
                Qwen35Runtime._release_batch_idx.__get__(self, _FakeRT))
            self.release_sequence = (
                Qwen35Runtime.release_sequence.__get__(self, _FakeRT))

        def _clear_decode_state(self, batch_idx: int) -> None:
            # On a persistent spec row this would clobber the runner's GDN linear
            # state + RoPE deltas in the shared pool; the guard must prevent it.
            self.cleared.append(int(batch_idx))

    return _FakeRT(persistent=[10, 11], free=[20, 21, 22])


def test_release_batch_idx_skips_persistent_spec_row():
    """``release_sequence`` must NOT free/erase a persistent spec row (P1).

    Regression for the spec-row double-free: ``SpecRunner.admit`` re-points
    ``state.batch_idx`` at one of the runner's FIXED persistent spec rows (reserved
    once, address captured by the spec graphs). When that same ``state`` later runs
    the normal ``Qwen35Runtime.release_sequence`` cleanup, ``_release_batch_idx``
    would ``page_table.erase`` the fixed row -- freeing it for reallocation while
    the live spec graphs still assume it is allocated + address-stable (page-table
    corruption) -- and ``_clear_decode_state`` would wipe that row's GDN/RoPE state
    in the shared pool. The persistent-row guard must skip cleanup entirely; the
    spec runner's own ``retire()`` owns those rows.
    """
    rt = _release_batch_idx_guard_harness()
    pt = rt.page_table
    persistent = set(rt._spec_runner._persistent_batch_idx)
    free_before = list(pt.free_batch_idx)
    spec_row = next(iter(persistent))

    # Drive the *public* release path a spec-admitted sequence takes at finish.
    class _State:
        batch_idx = spec_row

    rt.release_sequence(_State())

    assert pt.erased == [], (
        f"release_sequence erased a persistent spec row (erased={pt.erased})")
    assert rt.cleared == [], (
        f"release_sequence cleared a persistent spec row's decode state "
        f"(cleared={rt.cleared})")
    # The fixed reservation stays allocated + address-stable (out of the pool).
    assert spec_row not in pt.free_batch_idx, (
        "a persistent spec row was returned to the free pool -- the spec graphs "
        "captured its address and it must persist for the runner's lifetime")
    assert persistent.isdisjoint(pt.free_batch_idx)
    assert pt.free_batch_idx == free_before, (
        "release_sequence mutated the free list on a persistent spec row")
    # A subsequent release of the SAME row is still a no-op (idempotent skip):
    # the page-table entry stays intact so the next spec step over that row works.
    rt.release_sequence(_State())
    assert pt.erased == [] and spec_row not in pt.free_batch_idx


def test_release_batch_idx_still_frees_non_persistent_row():
    """The guard must NOT suppress freeing a genuine non-spec transient row.

    Companion to the persistent-row skip: a normal (non-spec) sequence's
    ``batch_idx`` -- not in ``_persistent_batch_idx``, not already free -- must
    still be erased + its decode state cleared so its row/pages return to the pool.
    Confirms the guard narrows the skip to the fixed spec rows only.
    """
    rt = _release_batch_idx_guard_harness()
    pt = rt.page_table
    transient = 20
    assert transient in pt.free_batch_idx
    pt.free_batch_idx.remove(transient)  # simulate it being live/allocated

    class _State:
        batch_idx = transient

    rt.release_sequence(_State())

    assert pt.erased == [transient], (
        f"release_sequence failed to erase a live non-spec row (erased={pt.erased})")
    assert rt.cleared == [transient]
    assert transient in pt.free_batch_idx


def test_release_batch_idx_no_spec_runner_is_unguarded():
    """With no spec runner configured the release path is unchanged.

    ``self._spec_runner is None`` on a non-spec runtime; the guard must fall
    through to the normal erase/clear so non-spec behaviour is untouched.
    """
    rt = _release_batch_idx_guard_harness()
    rt._spec_runner = None
    pt = rt.page_table
    bi = 21
    assert bi in pt.free_batch_idx
    pt.free_batch_idx.remove(bi)

    class _State:
        batch_idx = bi

    rt.release_sequence(_State())
    assert pt.erased == [bi]
    assert rt.cleared == [bi]


def _runner_release_harness():
    """Build a model-free ``SpecRunner`` + fake runtime exercising ``release()``.

    ``SpecRunner.release()`` is pure Python over ``self.batch_idx`` /
    ``self._persistent_batch_idx`` and the runtime's ``_release_batch_idx`` (whose
    persistent-row guard reads ``self._spec_runner._persistent_batch_idx``); none
    of it needs a GPU or the model. Build a fake page table that models both the
    free-row list AND the reserved pages per row (so the test can assert
    ``release()`` actually reclaims them), a fake runtime carrying the real
    ``_release_batch_idx`` / ``release_sequence`` bound methods, and a
    ``SpecRunner`` built via ``object.__new__`` with only the attributes
    ``release()`` touches. The runner IS installed as ``rt._spec_runner`` -- the
    exact production wiring under which the guard otherwise blocks ``release()``.
    """
    pytest.importorskip("kestrel.models.qwen35")
    from kestrel.models.qwen35.dflash.spec_decoder import SpecRunner
    from kestrel.models.qwen35.runtime import Qwen35Runtime

    class _FakePageTable:
        def __init__(self, persistent, free, pages_per_row):
            self.free_batch_idx = list(free)
            # Reserved pages for every currently-allocated row (persistent rows
            # are reserved once at runner construction and never freed until
            # release()). erase() returns a row's pages to this counter.
            self._pages = {int(bi): int(pages_per_row) for bi in persistent}
            self.free_pages_count = 0
            self.erased: list[int] = []

        @property
        def pages_available(self) -> int:
            return self.free_pages_count

        def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
            bi = int(batch_idx)
            self.erased.append(bi)
            # Mirror the real page table: row -> free list, pages -> free pool.
            if bi not in self.free_batch_idx:
                self.free_batch_idx.append(bi)
            self.free_pages_count += self._pages.pop(bi, 0)

    class _FakeRunner:
        pass

    class _FakeRT:
        def __init__(self, persistent, free, pages_per_row):
            self.page_table = _FakePageTable(persistent, free, pages_per_row)
            self.active_sequences = {}
            self._caches = {}
            self.cleared: list[int] = []
            self._spec_runner = None  # set to the real runner below
            self._release_batch_idx = (
                Qwen35Runtime._release_batch_idx.__get__(self, _FakeRT))
            self.release_sequence = (
                Qwen35Runtime.release_sequence.__get__(self, _FakeRT))

        def _clear_decode_state(self, batch_idx: int) -> None:
            self.cleared.append(int(batch_idx))

    persistent = [10, 11]
    pages_per_row = 7
    rt = _FakeRT(persistent=persistent, free=[20, 21],
                 pages_per_row=pages_per_row)
    runner = object.__new__(SpecRunner)
    runner.rt = rt
    runner.batch_idx = list(persistent)
    runner._persistent_batch_idx = set(persistent)
    rt._spec_runner = runner
    return runner, rt, persistent, pages_per_row


def test_runner_release_frees_persistent_rows():
    """``SpecRunner.release()`` must ACTUALLY free its own persistent rows (P2).

    Follow-on to the ``e2985298`` guard (``_release_batch_idx`` skips persistent
    spec rows so a finished sequence's ``release_sequence`` never frees a fixed,
    graph-captured row). When the runner is installed as ``rt._spec_runner``, that
    same guard fires for EVERY ``bi`` the runner's own ``release()`` tears down --
    so a guarded ``_release_batch_idx`` returns without erasing and ``release()``
    leaks the fixed page-table + KV/GDN reservations (recreating a runner on the
    same runtime would permanently consume batch slots + pages). ``release()`` must
    bypass the guard for its own deliberate teardown: every persistent row erased,
    returned to ``free_batch_idx``, its pages reclaimed, and decode state cleared.
    """
    runner, rt, persistent, pages_per_row = _runner_release_harness()
    pt = rt.page_table
    assert pt.pages_available == 0
    for bi in persistent:
        assert bi not in pt.free_batch_idx  # reserved, out of the pool

    runner.release()

    # Every persistent row was actually erased (guard bypassed for teardown).
    assert sorted(pt.erased) == sorted(persistent), (
        f"release() did not erase all runner rows (erased={pt.erased}); the "
        "persistent-row guard blocked the runner's own teardown -> leak")
    # Rows returned to the pool and their pages reclaimed.
    for bi in persistent:
        assert bi in pt.free_batch_idx, (
            f"release() left persistent row {bi} reserved -> batch-slot leak")
    assert pt.pages_available == len(persistent) * pages_per_row, (
        "release() did not reclaim the persistent rows' reserved pages -> page leak")
    # Per-row decode state (GDN/RoPE) was cleared for each reclaimed row.
    assert sorted(rt.cleared) == sorted(persistent)


def test_runner_release_preserves_release_sequence_skip():
    """``release()``'s bypass must NOT weaken the ``e2985298`` per-sequence skip.

    The guard exists so the NORMAL ``release_sequence(state)`` cleanup of a
    finished spec-admitted sequence never frees a still-live persistent row out
    from under the captured graphs. Before the runner tears anything down, a
    ``release_sequence`` over one of its spec rows must STILL be skipped (no erase,
    no page churn, no decode-state clobber) -- i.e. ``release()``'s teardown bypass
    is scoped to the rows it is actively reclaiming and does not globally disarm
    the guard. Then ``release()`` reclaims the rows, proving both contracts hold on
    the same runner.
    """
    runner, rt, persistent, pages_per_row = _runner_release_harness()
    pt = rt.page_table
    spec_row = persistent[0]
    free_before = list(pt.free_batch_idx)
    pages_before = pt.pages_available

    # Normal per-sequence release over a LIVE spec row -- still a guarded no-op.
    class _State:
        batch_idx = spec_row

    rt.release_sequence(_State())
    assert pt.erased == [], (
        f"release_sequence freed a live persistent spec row (erased={pt.erased}) "
        "-- the e2985298 guard regressed")
    assert rt.cleared == []
    assert spec_row not in pt.free_batch_idx
    assert pt.free_batch_idx == free_before
    assert pt.pages_available == pages_before

    # The deliberate runner teardown still reclaims every row afterwards.
    runner.release()
    assert sorted(pt.erased) == sorted(persistent)
    for bi in persistent:
        assert bi in pt.free_batch_idx
    assert pt.pages_available == len(persistent) * pages_per_row
