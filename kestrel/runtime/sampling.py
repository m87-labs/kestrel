"""Runtime-side sampling hooks consumed by the generation scheduler.

The scheduler runs the generic parts of token sampling — sampling
itself, batch/lifecycle management, the staging-buffer + D2H transfer
pipeline. Anything model-specific that has to happen *around* that
sampling step (e.g. Moondream's per-step coord/size value decode from
hidden states) plugs in here.

A runtime exposes its hooks by assigning a :class:`SamplingHooks`
value to ``runtime.sampling_hooks``. Runtimes that don't need any
custom behaviour can leave the attribute unset — the scheduler uses
``SamplingHooks()`` as the default and, because every field has a
neutral default, the per-step path collapses to "sample tokens, ship
them home as ``TextToken``s." No buffers are allocated, no per-step
work is done, no per-step transfers happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from kestrel.runtime.tokens import TextToken, Token


def _default_post_sample(**_: object) -> None:
    """No-op. Default for runtimes without custom post-sample work."""


def _default_materialize_tokens(
    token_ids_cpu: Tensor,
    aux_values_cpu: list[Tensor],
) -> list[Token]:
    """Materialise sampled token ids as plain ``TextToken``s."""
    return [TextToken(token_id=int(tid)) for tid in token_ids_cpu.view(-1).tolist()]


@dataclass(frozen=True)
class AuxBufferSpec:
    """Declaration of one auxiliary per-step buffer the scheduler manages.

    ``dtype`` and ``width`` (the per-slot last-dim size) are enough for
    the scheduler to allocate the buffer + pin its companion on the
    host side; the runtime decides what the bytes mean and how to
    populate them in :attr:`SamplingHooks.post_sample`.
    """

    dtype: torch.dtype
    width: int


@dataclass(frozen=True)
class SamplingHooks:
    """Runtime contract for per-step sampling work.

    All fields default to neutral values: ``aux_buffers`` is empty so
    nothing extra is allocated or transferred, ``post_sample`` is a
    no-op, ``materialize_tokens`` produces ``TextToken``s only. A
    runtime that needs per-step decoded values (e.g. Moondream's
    coord/size decode) sets all three together so the buffers it
    declares are populated by ``post_sample`` and consumed by
    ``materialize_tokens``.
    """

    aux_buffers: tuple[AuxBufferSpec, ...] = ()
    post_sample: Callable[..., None] = field(default=_default_post_sample)
    materialize_tokens: Callable[..., list[Token]] = field(default=_default_materialize_tokens)


__all__ = ["AuxBufferSpec", "SamplingHooks"]
