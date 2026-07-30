"""DFlash speculative proposer: drafter -> draft tokens.

Wires :class:`DFlashDraftModel` to the target's (tied) token embedding and LM
head and implements the ``SpecProposer`` contract
(:mod:`kestrel.runtime.spec`). One ``propose`` builds the query block
``[last_token, MASK x K]``, embeds it with the target embedding, runs the draft
model conditioned on the target's hidden states, projects the K mask positions
through the target LM head, and samples ``K = block_size - 1`` draft tokens.

Block layout (matches the reference / vLLM): position 0 is the just-sampled
("bonus") token; positions ``1..K`` are mask tokens whose outputs are the draft
predictions. ``num_lookahead_tokens = block_size`` (the bonus query plus K masks).

The draft side is greedy by default (``draft_probs=None``); losslessness is
enforced later at verify/accept by the rejection sampler. Because the drafter
recomputes context K/V from the supplied target hidden each step (no persistent
draft KV yet), ``commit_accept`` is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from kestrel.runtime.spec import DraftResult

from .model import DFlashConfig, DFlashDraftModel


@dataclass
class ProposeContext:
    """Inputs for one draft step (populated by the runtime/scheduler later).

    - ``last_token_ids``: ``[batch]`` int, the just-sampled token per sequence
      (block position 0).
    - ``target_hidden``: ``[batch, ctx_len, len(target_layer_ids) * hidden]`` —
      the concatenated target hidden states at the context positions.
    - ``position_ids``: ``[batch, ctx_len + block_size]`` covering context + block.
    - ``temperature`` / ``generator``: optional sampling controls (greedy if
      ``temperature`` is ``None``).
    """

    last_token_ids: torch.Tensor
    target_hidden: torch.Tensor
    position_ids: torch.Tensor
    temperature: torch.Tensor | None = None
    generator: torch.Generator | None = None


class DFlashProposer:
    """``SpecProposer`` implementation backed by a DFlash draft model."""

    def __init__(
        self,
        drafter: DFlashDraftModel,
        embed_tokens: nn.Module,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        config: DFlashConfig,
    ) -> None:
        self.drafter = drafter
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.config = config
        # K draft tokens are the mask positions; block = [bonus] + K masks.
        self.num_speculative_tokens = config.block_size - 1
        self.num_lookahead_tokens = config.block_size

    @torch.no_grad()
    def propose(self, ctx: ProposeContext) -> DraftResult:
        bsz = ctx.last_token_ids.shape[0]
        block = self.config.block_size
        device = ctx.last_token_ids.device

        block_ids = torch.full(
            (bsz, block), self.config.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = ctx.last_token_ids
        noise = self.embed_tokens(block_ids)

        hidden = self.drafter(noise, ctx.target_hidden, ctx.position_ids)
        mask_hidden = hidden[:, 1:, :]  # [bsz, K, hidden] — the draft positions
        logits = self.lm_head(mask_hidden)  # [bsz, K, vocab]

        if ctx.temperature is None:
            token_ids = logits.argmax(dim=-1)
            return DraftResult(token_ids=token_ids.to(torch.int32), draft_probs=None)

        temp = ctx.temperature.view(bsz, 1, 1).clamp_min(1e-5)
        probs = torch.softmax(logits.float() / temp, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=ctx.generator).view(bsz, -1)
        return DraftResult(token_ids=sampled.to(torch.int32), draft_probs=probs)

    def commit_accept(self, ctx: ProposeContext) -> None:
        # Recompute drafter holds no persistent draft state between steps.
        return None
