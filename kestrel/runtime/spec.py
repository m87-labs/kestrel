"""Speculative-decoding contract: the proposer interface + capability hook.

This module defines the *seam* the scheduler and runtimes use to support
speculative decoding (DFlash and, later, other methods). It is intentionally
inert on its own: a runtime that leaves
:attr:`~kestrel.runtime.protocol.AutoregressiveRuntime.spec` as ``None`` behaves
exactly as a non-speculative runtime, so adding this module changes no behavior.

The full design lives in the speculative-decoding design doc. This scaffolding
deliberately covers only the pieces a proposer and a runtime advertise; the
verify/accept and hybrid-state hooks land with scheduler integration, where they
are actually consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    drafter). The context argument types are intentionally left open here; they
    are pinned down with scheduler integration, the PR that first consumes them.
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
class SpecDecodeCaps:
    """A runtime's speculative-decoding capability.

    A runtime advertises speculative decoding by setting
    :attr:`AutoregressiveRuntime.spec` to an instance of this; leaving it
    ``None`` means the runtime decodes one token per step as before.

    ``capture_hidden_layers`` are the target layer indices whose hidden states
    the proposer consumes (e.g. DFlash's ``target_layer_ids``); an empty tuple
    means the proposer needs no target hidden states.

    Verify/accept and hybrid-state (GDN) hooks are added with scheduler
    integration; they are omitted here to avoid specifying an interface ahead of
    its consumer.
    """

    proposer: SpecProposer
    capture_hidden_layers: tuple[int, ...] = ()
