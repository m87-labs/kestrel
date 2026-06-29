"""Speculative-decoding contract: the proposer interface + capability hook.

This module defines the *seam* the scheduler and runtimes use to support
speculative decoding (DFlash and, later, other methods). It is intentionally
inert on its own: a runtime that leaves
:attr:`~kestrel.runtime.protocol.AutoregressiveRuntime.spec` as ``None`` behaves
exactly as a non-speculative runtime, so adding this module changes no behavior.

The full design lives in ``docs/speculative-decoding-design.md`` in
``kestrel-proprietary``. This scaffolding deliberately covers only the pieces a
proposer and a runtime advertise; the verify/accept and hybrid-state hooks land
with scheduler integration, where they are actually consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import torch


@dataclass
class DraftResult:
    """Draft tokens proposed for one decode step.

    ``token_ids`` is ``[batch, num_speculative_tokens]`` (int32, on device).
    ``draft_probs`` is ``[batch, num_speculative_tokens, vocab]`` when the
    proposer supports lossless rejection sampling; ``None`` selects the greedy
    acceptance path.
    """

    token_ids: torch.Tensor
    draft_probs: torch.Tensor | None = None


@runtime_checkable
class SpecProposer(Protocol):
    """Produces draft tokens for the target model to verify.

    Implementations live alongside the model they draft for (e.g. the DFlash
    drafter in ``kestrel-proprietary``). The context argument types are
    intentionally left open here; they are pinned down with scheduler
    integration, the PR that first consumes them.
    """

    # Drafts proposed per step (K).
    num_speculative_tokens: int

    # KV slots to reserve ahead per request before appending drafts. DFlash
    # uses ``K + 1`` (the bonus/last-sampled-token query plus K mask tokens);
    # a chain EAGLE-style proposer would use ``K``.
    num_lookahead_tokens: int

    def propose(self, ctx: Any) -> DraftResult:
        """Run the drafter and return ``[batch, K]`` draft tokens (+ probs)."""
        ...

    def commit_accept(self, ctx: Any) -> None:
        """Advance proposer-internal state to the accepted prefix."""
        ...


class SpecSideValues:
    """Per-macro-step typed-token side-values the runtime needs to *type* the
    committed ids (turn raw vocab ids into ``CoordToken`` / ``SizeToken`` / …).

    A multimodal runtime (Moondream) decodes some committed ids into spatial
    values from the target's last hidden state: a committed ``coord_id`` becomes
    a :class:`CoordToken` whose ``pos`` is sampled from the spatial head applied
    to that position's final hidden state, and likewise ``size_id`` →
    :class:`SizeToken`. The non-spec decode path does this in
    ``SamplingHooks.post_sample`` (``compute_spatial_values(token_ids,
    hidden_last, …)``) — one position per step. A spec macro-step commits a
    *variable run* of positions per sequence, so it carries the final hidden
    state **at every committed position**, packed, plus the per-sequence sampling
    knobs ``compute_spatial_values`` needs.

    Fields (device tensors, indexed by ``i`` parallel to the active
    ``sequences`` / :attr:`SpecStepResult.tokens`):

    * ``hidden`` ``[num_sequences, K + 1, hidden_dim]`` — the target's **final**
      ``last_hidden_state`` over sequence ``i``'s whole verify block. Only the
      leading ``a_i + 1`` positions (``j <= accept_counts[i]``) are committed;
      ``hidden[i, j]`` is the ``hidden_last`` for committed position ``j`` (the
      same tensor the non-spec ``post_sample`` reads). Keeping the full block on
      device (rather than pre-packing the variable run) avoids any per-step host
      sync in ``step``; the runtime slices by ``accept_counts``.
    * ``temperatures`` / ``top_ps`` ``[num_sequences]`` — the per-sequence
      sampling temperature / top-p (broadcast across that sequence's committed
      positions), so the spatial decode samples under the request's settings.
    * ``counts`` — ``list[int]`` of ``a_i + 1`` per sequence when the producer
      pre-split the rows, or ``None`` (the common path) meaning "slice the
      leading ``accept_counts[i] + 1`` positions of ``hidden[i]``".

    The runtime consumes this in its ``materialize_tokens`` hook (e.g. Moondream
    runs ``compute_spatial_values(committed_ids, hidden[i, j], …)`` to turn
    coord/size ids into ``CoordToken`` / ``SizeToken``); a text-only runtime
    ignores it entirely (the committed ids materialize as ``TextToken``).
    """

    __slots__ = ("hidden", "temperatures", "top_ps", "counts")

    def __init__(
        self,
        hidden: "torch.Tensor",
        temperatures: "torch.Tensor",
        top_ps: "torch.Tensor",
        counts: "list[int] | None" = None,
    ) -> None:
        self.hidden = hidden
        self.temperatures = temperatures
        self.top_ps = top_ps
        self.counts = counts


class SpecStepResult:
    """Per-sequence outcome of one speculative *macro-step*.

    A macro-step drafts ``K`` tokens per active sequence, verifies them in one
    target forward, accepts the longest matching prefix (greedy) or runs modified
    rejection sampling (``temperature > 0``), and commits ``a_i + 1`` tokens for
    sequence ``i`` (the ``a_i`` accepted drafts plus the bonus / replacement
    token at the first rejection). Sequences therefore advance a *variable*
    number of tokens in a single step.

    Everything below is **parallel to the ``sequences``** the scheduler passed to
    :meth:`SpecDecoder.step` (``i`` indexes that list):

    * ``tokens[i]`` — the list of newly committed token ids for ``sequences[i]``
      (length ``a_i + 1``, always ``>= 1``).
    * ``accept_counts[i]`` — ``a_i`` (drafts accepted, excludes the bonus);
      diagnostic — ``len(tokens[i]) == accept_counts[i] + 1``.
    * ``logprobs`` — ``None`` when no active sequence requested logprobs;
      otherwise ``logprobs[i]`` is the per-committed-token logprob list (same
      length as ``tokens[i]``), each the ``log_softmax`` of the target verify
      logits — under the request's temperature/top-p — at that position,
      gathered at the committed id. This is the spec analog of the non-spec
      per-token logprob; the scheduler stages it through ``stage_token`` exactly
      like the single-token path.
    * ``side_values`` — ``None`` for a text-only runtime; otherwise a
      :class:`SpecSideValues` carrying the target final-hidden + sampling knobs
      at every committed position so the runtime's ``materialize_tokens`` hook
      can type coord/size ids. It is *not* lazy (it holds device tensors read on
      the compute side at ``step`` time); ``tokens`` / ``accept_counts`` /
      ``logprobs`` resolve lazily off an in-flight D2H.

    Lazy resolution: a proposer builds this from an in-flight D2H of its
    committed-token / logprob buffers so ``step`` never stalls the GPU on a
    readback; the first access to :attr:`tokens` / :attr:`accept_counts` /
    :attr:`logprobs` blocks on that transfer. The async scheduler defers that
    access to a later step, so the wait overlaps GPU work (see
    ``docs/speculative-decoding-design.md`` §12). Eager construction
    (``SpecStepResult(tokens=..., accept_counts=..., logprobs=...)``) stays valid
    for synchronous callers and tests.
    """

    __slots__ = (
        "_tokens",
        "_accept_counts",
        "_logprobs",
        "_resolve",
        "side_values",
    )

    def __init__(
        self,
        tokens: "list[list[int]] | None" = None,
        accept_counts: "list[int] | None" = None,
        logprobs: "list[list[float]] | None" = None,
        *,
        resolve: "Any | None" = None,
        side_values: "SpecSideValues | None" = None,
    ) -> None:
        self._resolve = resolve
        self._tokens = tokens
        self._accept_counts = accept_counts
        self._logprobs = logprobs
        # Side-values are produced eagerly on the compute side (device tensors);
        # they do not participate in the lazy token/logprob D2H.
        self.side_values = side_values

    def _ensure(self) -> None:
        if self._resolve is not None:
            self._tokens, self._accept_counts, self._logprobs = self._resolve()
            self._resolve = None

    @property
    def tokens(self) -> "list[list[int]]":
        self._ensure()
        return self._tokens

    @property
    def accept_counts(self) -> "list[int]":
        self._ensure()
        return self._accept_counts

    @property
    def logprobs(self) -> "list[list[float]] | None":
        """Per-committed-token logprobs parallel to :attr:`tokens`, or ``None``.

        ``None`` means no active sequence this step requested logprobs (the
        proposer skipped the gather). When present, ``logprobs[i]`` has the same
        length as ``tokens[i]``.
        """
        self._ensure()
        return self._logprobs


@runtime_checkable
class SpecDecoder(Protocol):
    """The per-macro-step entry point a runtime exposes to the scheduler.

    The scheduler drives this once per decode macro-step in place of the
    one-token ``decode_with_slot`` path. Each call drafts, verify-replays,
    greedily accepts, and commits the accepted prefix into the runtime's
    persistent KV / hybrid-state pools for the supplied active sequences,
    returning the per-sequence accepted tokens.

    The runtime owns all device-side state (the persistent decode pool, the
    reused verify/draft CUDA graphs, the GDN ring buffers). The scheduler only
    supplies which sequences are active this step and consumes the returned
    tokens — it stays model-agnostic.

    Lifecycle: a sequence is introduced with :meth:`admit` (which prefills its
    prompt into a pool row and returns the first sampled token), advanced with
    repeated :meth:`step` calls, and removed with :meth:`retire` when it
    finishes. ``admit`` returns the bonus/first token the way prefill does, so
    the scheduler stages it exactly like the non-spec first decode token.
    """

    # K draft tokens proposed per sequence per macro-step (proposer's K).
    num_speculative_tokens: int

    # Number of free pool rows available to admit new sequences right now.
    @property
    def free_slots(self) -> int: ...

    def admit(
        self,
        state: Any,
        prompt_token_ids: Sequence[int],
        *,
        image: Any | None = None,
        allowed_token_ids: Sequence[int] | None = None,
        suppressed_token_ids: Sequence[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Prefill ``prompt_token_ids`` into a free pool row for ``state``.

        Assigns the row's device batch index onto ``state.batch_idx`` (so the
        scheduler's KV/finish bookkeeping addresses the same storage the spec
        loop commits into) and returns the first sampled (greedy/bonus) token
        id, mirroring prefill.

        Multimodal + constrained-decode admission (so the spec path covers the
        same requests the non-spec path does, with no fallback):

        * ``image`` — the request's image(s) (the runtime's preprocessed image
          inputs, or whatever object the runtime's image preprocessing accepts).
          When set, the prompt is prefilled *with the image KV prefix* exactly
          like a normal image prefill (vision block spliced + vision encoder
          run), not text-only. ``None`` keeps the text-only prefill.
        * ``allowed_token_ids`` / ``suppressed_token_ids`` — the sequence's skill
          mask (point/detect/query constrain the vocabulary). Stored per row and
          applied to **both** the drafter and the verify on every :meth:`step`,
          so masked decoding stays lossless and high-acceptance. ``None`` leaves
          the row unmasked.
        * ``temperature`` / ``top_p`` — the request's sampling knobs, recorded so
          :meth:`step` runs the matching greedy/rejection-sampling path and the
          per-token logprobs / spatial decode use the request's settings.
        """
        ...

    def step(self, states: Sequence[Any]) -> SpecStepResult:
        """Run one macro-step over the active ``states`` and commit accepts.

        Applies each row's admit-time token mask to the drafter + verify,
        returns a :class:`SpecStepResult` carrying the committed ids, accept
        counts, per-token logprobs (when any active sequence wants them), and the
        typed-token :class:`SpecSideValues` (when the runtime types coord/size
        ids). See :class:`SpecStepResult`.
        """
        ...

    def retire(self, state: Any) -> None:
        """Release the pool row held by a finished sequence's ``state``."""
        ...


@dataclass
class SpecDecodeCaps:
    """A runtime's speculative-decoding capability.

    A runtime advertises speculative decoding by setting
    :attr:`AutoregressiveRuntime.spec` to an instance of this; leaving it
    ``None`` means the runtime decodes one token per step as before.

    ``capture_hidden_layers`` are the target layer indices whose hidden states
    the proposer consumes (e.g. DFlash's ``target_layer_ids``); an empty tuple
    means the proposer needs no target hidden states.

    ``decoder`` is the runtime's :class:`SpecDecoder` — the per-macro-step
    entry point the scheduler drives (draft + verify + accept + commit). It is
    optional only so the inert scaffolding (and its CPU-only tests) can still
    build a ``SpecDecodeCaps`` that merely advertises a proposer; a runtime that
    actually wants the scheduler to speculate must set it.
    """

    proposer: SpecProposer
    capture_hidden_layers: tuple[int, ...] = ()
    decoder: SpecDecoder | None = None
