"""Rejection-sampling (non-greedy) speculative decoding primitives.

Implements the *modified rejection sampling* of Leviathan et al. 2022 ("Fast
Inference from Transformers via Speculative Decoding") and Chen et al. 2023
("Accelerating Large Language Model Decoding with Speculative Sampling"), the
same accept/reject rule vLLM uses in ``vllm/v1/sample/rejection_sampler.py``:

    Draft token ``x_i`` is sampled from the *draft* distribution ``q``. Accept it
    with probability ``min(1, p(x_i) / q(x_i))`` where ``p`` is the *target*
    distribution. At the first reject (position ``i``), emit a replacement drawn
    from the normalized residual ``norm(max(p - q, 0))`` and stop. If all ``K``
    drafts are accepted, emit a *bonus* token drawn from ``p`` at the last
    position. This makes the emitted next-token distribution **exactly** the
    target sampling distribution ``p`` (distribution-exact by construction).

Temperature / top-p / top-k are applied to **both** ``q`` (the drafter logits)
and ``p`` (the target logits) with :func:`logits_to_probs`, which mirrors the
production single-token sampler (``Qwen35Runtime._sample_next``) bit-for-bit:
softmax of ``logits / temperature`` followed by a sorted-cumsum top-p truncation
(keep the smallest prefix whose cumulative mass first exceeds ``top_p``; always
keep at least one token) and an optional top-k truncation. Matching ``p`` to the
non-spec sampler is what makes the spec-decoded distribution equal the normal
sampled distribution.

Everything here is GPU-resident and CUDA-graph friendly: no ``.item()`` / host
syncs, no data-dependent shapes. The accept comparison and the residual
(Gumbel-max) draw are plain dense tensor ops over the ``[N, vocab]`` block, so a
graphed spec loop can call these between graph replays. ``temperature == 0`` is
*not* routed here at all -- the caller keeps its exact-argmax greedy path -- so
the validated lossless-greedy loop never changes.

The block layout matches the proposer / vLLM: for a sequence's verify block of
``K + 1`` positions, position ``j in [0, K)`` verifies draft token ``x_j``
against the target distribution at that position, and position ``K`` is the
bonus distribution sampled only when every draft is accepted.
"""

from __future__ import annotations

import torch

# Below this temperature a request is greedy; the caller must take the
# exact-argmax path instead of calling into this module. Matches
# ``Qwen35Runtime._sample_next`` (``temperature <= 0`` -> argmax).
GREEDY_TEMPERATURE_EPS = 1e-6


def logits_to_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor | float,
    top_p: torch.Tensor | float | None = None,
    top_k: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Temperature / top-p / top-k softmax, matching ``_sample_next``.

    ``logits`` is ``[N, vocab]``. ``temperature`` / ``top_p`` are scalars or
    ``[N]`` (broadcast per row). ``top_k`` is a scalar or ``[N]`` int (0 / <=0
    / >= vocab disables it). Returns ``[N, vocab]`` float32 probabilities that
    sum to 1 per row.

    The top-p rule keeps the smallest set of highest-probability tokens whose
    cumulative probability first **exceeds** ``top_p`` (``cumsum <= top_p`` then
    ``+1``), i.e. always at least one token, then renormalizes -- identical to
    the production single-token sampler so ``p`` here is the exact distribution
    the non-spec path samples from. top-k is applied before top-p (on the same
    sorted order), as in vLLM's ``apply_top_k_top_p``.
    """
    assert logits.ndim == 2, logits.shape
    n, vocab = logits.shape
    device = logits.device
    logits = logits.to(torch.float32)

    if not torch.is_tensor(temperature):
        temperature = torch.full((n,), float(temperature), device=device, dtype=torch.float32)
    else:
        temperature = temperature.to(device=device, dtype=torch.float32).reshape(-1)
    temperature = temperature.clamp_min(GREEDY_TEMPERATURE_EPS)
    logits = logits / temperature.unsqueeze(-1)

    has_top_p = top_p is not None
    has_top_k = top_k is not None
    if has_top_p and not torch.is_tensor(top_p):
        # Only ``0 < top_p < 1`` truncates, matching ``_sample_next``. A scalar
        # >= 1 is the usual "no truncation"; <= 0 must ALSO disable (an empty
        # ``prev < 0`` keep mask would otherwise renormalize the row to all
        # zeros and break ``multinomial`` / return token 0). Skip the sort.
        if float(top_p) >= 1.0 or float(top_p) <= 0.0:
            has_top_p = False
        else:
            top_p = torch.full((n,), float(top_p), device=device, dtype=torch.float32)
    elif has_top_p:
        top_p = top_p.to(device=device, dtype=torch.float32).reshape(-1)
        # Per-row disable for ``top_p <= 0`` (and >= 1): map to 1.0 so the
        # ``prev < top_p`` mask keeps every token for that row (no truncation),
        # instead of an empty mask that renormalizes to zeros. Rows in
        # ``(0, 1)`` truncate as before.
        top_p = torch.where(top_p <= 0.0, top_p.new_ones(()), top_p)
    if has_top_k and not torch.is_tensor(top_k):
        if int(top_k) <= 0 or int(top_k) >= vocab:
            has_top_k = False
        else:
            top_k = torch.full((n,), int(top_k), device=device, dtype=torch.long)
    elif has_top_k:
        top_k = top_k.to(device=device, dtype=torch.long).reshape(-1)
        # Per-row disable for ``top_k <= 0`` (mirrors the scalar ``top_k`` and the
        # per-row ``top_p`` disable above). A row with ``top_k <= 0`` means "no
        # top-k" -- map it to ``vocab`` so ``rank < k`` keeps every token for that
        # row. Without this, ``k == 0`` makes ``rank < 0`` False everywhere, the
        # row's surviving mass is all zeros, and the final renormalize / Gumbel
        # argmax returns token 0 instead of sampling from the full distribution.
        top_k = torch.where(top_k <= 0, top_k.new_full((), vocab), top_k)

    if not has_top_p and not has_top_k:
        return torch.softmax(logits, dim=-1)

    # Sorted-prob truncation (descending). Matches _sample_next's sort+cumsum.
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)

    if has_top_k:
        # Keep the top-k highest-prob tokens (rank < k). top_k clamped to vocab.
        k = top_k.clamp(max=vocab).unsqueeze(-1)                     # [N, 1]
        rank = torch.arange(vocab, device=device).unsqueeze(0)        # [1, V]
        sorted_probs = torch.where(rank < k, sorted_probs, sorted_probs.new_zeros(()))
        if has_top_p:
            # Renormalize the surviving top-k mass to sum to 1 BEFORE taking the
            # top-p cumulative threshold, so top-p is applied over the post-top-k
            # distribution (vLLM's apply_top_k_top_p order). Without this, the
            # cumsum runs over the raw softmax mass: if the top-k set holds only,
            # say, 50% of the mass then ``cum`` never reaches ``top_p`` and top-p
            # keeps the whole top-k set instead of its smallest >=top_p prefix.
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1, keepdim=True).clamp_min(1e-12)

    if has_top_p:
        cum = torch.cumsum(sorted_probs, dim=-1)
        # keep = (cum <= top_p).sum() + 1 -> mask positions with rank < keep.
        # Equivalent boolean mask: (cum - sorted_probs) < top_p keeps the first
        # `keep` entries (the cumulative mass *before* this token is < top_p),
        # which is exactly cumsum<=top_p plus the one token that crosses it, and
        # always keeps rank 0. The masses were renormalized after the top-k cut
        # above, so the threshold is taken over the post-top-k distribution.
        prev = cum - sorted_probs
        keep_mask = prev < top_p.unsqueeze(-1)
        sorted_probs = torch.where(keep_mask, sorted_probs, sorted_probs.new_zeros(()))

    # Renormalize and scatter back to vocab order.
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.empty_like(sorted_probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


def _gumbel_argmax(
    probs: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample one index per row from ``probs`` via the exponential (Gumbel-max)
    trick: ``argmax(probs / Exp(1))``.

    Identical in distribution to ``torch.multinomial(probs, 1)`` but kept as a
    dense, graph-safe op (no host sync, fixed shape) -- the same construction
    vLLM's ``sample_recovered_tokens_kernel`` uses (``argmax(prob / q)`` with
    ``q ~ Exp(1)``). ``probs`` need not be normalized (argmax is scale-free), so
    this also samples directly from the unnormalized residual ``max(p - q, 0)``.
    """
    g = torch.empty_like(probs)
    g.exponential_(generator=generator)
    return torch.argmax(probs / g, dim=-1)


def rejection_sample_block(
    draft_token_ids: torch.Tensor,   # [N, K] int  -- the sampled draft tokens
    draft_probs: torch.Tensor,       # [N, K, vocab] float -- q at each draft pos
    target_logits: torch.Tensor,     # [N, K+1, vocab] float -- raw target logits
    temperature: torch.Tensor | float,
    top_p: torch.Tensor | float | None = None,
    top_k: torch.Tensor | int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modified-rejection-sample one verify block per row (batched, GPU-resident).

    For each of the ``N`` rows: walk the ``K`` drafts left-to-right, accept
    ``x_j`` while ``u_j <= p_j(x_j) / q_j(x_j)`` (reject if ``q_j(x_j) == 0``),
    and at the first reject emit the residual sample. If all ``K`` accept, emit
    the bonus from ``p_K``.

    Returns ``(out_tokens, accept)`` where:
      * ``out_tokens`` is ``[N, K+1]`` int64: ``out_tokens[r, :accept_r]`` are
        the accepted drafts, ``out_tokens[r, accept_r]`` is the replacement /
        bonus token, and the tail (``> accept_r``) is undefined (-1).
      * ``accept`` is ``[N]`` int64: the number of accepted drafts ``a_r`` (so
        ``a_r + 1`` tokens are committed for row ``r``).

    This is the same emitted distribution as drawing the next token directly
    from ``p`` for every committed position -- i.e. distribution-exact vs the
    non-spec sampler that uses the same ``temperature`` / ``top_p`` / ``top_k``.
    """
    assert draft_token_ids.ndim == 2, draft_token_ids.shape
    n, k = draft_token_ids.shape
    assert draft_probs.shape[:2] == (n, k), draft_probs.shape
    assert target_logits.shape[0] == n and target_logits.shape[1] == k + 1, target_logits.shape
    vocab = target_logits.shape[-1]
    device = target_logits.device

    # Broadcast sampling params per draft position so p and q share the exact
    # same transform (temperature/top_p/top_k apply to both, per the reference).
    def _expand(x, fill_dtype):
        if x is None:
            return None
        if not torch.is_tensor(x):
            return x  # scalar: logits_to_probs handles broadcast + fast paths
        return x.to(device=device).reshape(n, 1).expand(n, k + 1).reshape(-1)

    flat_logits = target_logits.reshape(n * (k + 1), vocab)
    p_all = logits_to_probs(
        flat_logits,
        _expand(temperature, torch.float32) if torch.is_tensor(temperature)
        else temperature,
        _expand(top_p, torch.float32) if torch.is_tensor(top_p) else top_p,
        _expand(top_k, torch.long) if torch.is_tensor(top_k) else top_k,
    ).reshape(n, k + 1, vocab)

    p_draft = p_all[:, :k, :]                       # [N, K, vocab] target at draft pos
    p_bonus = p_all[:, k, :]                         # [N, vocab]    target at bonus pos

    # p(x_j) and q(x_j) at the *drafted* token for the accept ratio.
    idx = draft_token_ids.long().unsqueeze(-1)       # [N, K, 1]
    p_x = p_draft.gather(-1, idx).squeeze(-1)        # [N, K]
    q_x = draft_probs.gather(-1, idx).squeeze(-1)    # [N, K]

    # Accept iff q>0 and p/q >= u, u~Uniform(0,1). (Reject on q==0 to avoid NaN,
    # matching vLLM's guard.) Equivalent to u <= min(1, p/q) since u in [0,1).
    u = torch.empty_like(p_x)
    u.uniform_(generator=generator)
    ratio = torch.where(q_x > 0, p_x / q_x.clamp_min(1e-30), q_x.new_zeros(()))
    accept_step = (q_x > 0) & (ratio >= u)           # [N, K] per-position accept

    # accept_r = length of the leading all-True run (longest accepted prefix).
    accept = accept_step.int().cumprod(dim=1).sum(dim=1)              # [N] in [0, K]

    # Residual sample at EVERY draft position (norm(max(p - q, 0))), then pick
    # the one at the first reject. Sampling all K positions keeps the op dense /
    # graph-safe; only the position == accept_r is actually used per row.
    residual = torch.clamp(p_draft - draft_probs, min=0.0)            # [N, K, vocab]
    recovered = _gumbel_argmax(
        residual.reshape(n * k, vocab), generator
    ).reshape(n, k)                                                  # [N, K]

    # Bonus sample from p at the last position (used when accept_r == K).
    bonus = _gumbel_argmax(p_bonus, generator)                       # [N]

    # Assemble output: positions < accept are the accepted drafts; position ==
    # accept is recovered[accept] if accept < K else bonus.
    out = draft_token_ids.long().clone()                            # [N, K] accepted drafts
    # Replacement token per row at column == accept (when accept < K).
    safe_rej = accept.clamp(max=k - 1)                               # avoid OOB gather
    rep_at_reject = recovered.gather(1, safe_rej.unsqueeze(1)).squeeze(1)  # [N]
    replacement = torch.where(accept < k, rep_at_reject, bonus)      # [N]

    out_tokens = torch.full((n, k + 1), -1, dtype=torch.long, device=device)
    col = torch.arange(k + 1, device=device).unsqueeze(0)            # [1, K+1]
    # Accepted drafts at cols < accept.
    accepted_mask = col[:, :k] < accept.unsqueeze(1)                 # [N, K]
    out_tokens[:, :k] = torch.where(accepted_mask, out, out_tokens[:, :k])
    # Replacement / bonus at col == accept.
    out_tokens.scatter_(1, accept.unsqueeze(1), replacement.unsqueeze(1))
    return out_tokens, accept
