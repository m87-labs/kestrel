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


@dataclass
class SpecStepResult:
    """Per-sequence outcome of one speculative *macro-step*.

    A macro-step drafts ``K`` tokens per active sequence, verifies them in one
    target forward, greedily accepts the longest matching prefix, and commits
    ``a_i + 1`` tokens for sequence ``i`` (the ``a_i`` accepted drafts plus the
    bonus token at the first rejection). Sequences therefore advance a
    *variable* number of tokens in a single step.

    ``tokens`` is parallel to the ``sequences`` the scheduler passed to
    :meth:`SpecDecoder.step`: ``tokens[i]`` is the list of newly committed token
    ids for ``sequences[i]`` (length ``a_i + 1``, always ``>= 1``).
    ``accept_counts[i]`` is ``a_i`` (drafts accepted, excludes the bonus); it is
    diagnostic — ``len(tokens[i]) == accept_counts[i] + 1``.
    """

    tokens: list[list[int]]
    accept_counts: list[int]


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

    def admit(self, state: Any, prompt_token_ids: Sequence[int]) -> int:
        """Prefill ``prompt_token_ids`` into a free pool row for ``state``.

        Assigns the row's device batch index onto ``state.batch_idx`` (so the
        scheduler's KV/finish bookkeeping addresses the same storage the spec
        loop commits into) and returns the first sampled (greedy/bonus) token
        id, mirroring prefill.
        """
        ...

    def step(self, states: Sequence[Any]) -> SpecStepResult:
        """Run one macro-step over the active ``states`` and commit accepts."""
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
