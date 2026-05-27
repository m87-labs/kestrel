"""Runtime-side sampling hooks consumed by the generation scheduler.

The scheduler runs the generic parts of token sampling — sampling
itself, batch/lifecycle management, the D2H transfer of sampled token
ids + logprobs. Anything model-specific that has to happen *around*
that sampling step (e.g. Moondream's per-step coord/size decode from
hidden states) plugs in here.

A runtime exposes its hooks by assigning a :class:`SamplingHooks`
value to ``runtime.sampling_hooks``. Runtimes that don't need any
custom behaviour can leave the attribute unset — the scheduler uses
``SamplingHooks()`` as the default and, because every field is
``None``, the per-step path collapses to "sample tokens, ship them
home as ``TextToken``s." The runtime owns all storage and D2H for any
extra per-step values; the scheduler treats the handle the post-sample
hook returns as opaque.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SamplingHooks:
    """Per-step runtime hooks. All optional.

    Wiring:

    1. Scheduler samples token ids on the compute stream and records
       ``ready_event`` once they're written to the staging buffer.
    2. ``post_sample(...)`` fires next (compute stream) — runtime runs
       any GPU work it needs (e.g. decode side-values from
       ``hidden_last``) and initiates its own D2H against
       ``ready_event``. It returns an opaque handle the runtime
       understands later.
    3. Scheduler initiates its own D2H for token ids + logprobs.
    4. ``prepare_decode_inputs(...)`` fires before the next decode
       launch — runtime gathers any model-specific decode inputs from
       its own per-batch-idx state (the scheduler gathers token ids
       generically).
    5. On commit, scheduler reads CPU-side token ids + logprobs and
       calls ``materialize_tokens(token_ids_cpu, sequences, batch_idx,
       step_handle)`` to build the typed Token list it hands to skills.
    """

    # post_sample(slot, *, sampled_ids, hidden_last, sequences,
    #             batch_idx, temperatures, top_ps, token_logprobs,
    #             ready_event) -> Any
    # Decode-only. Receives the runtime's DecodeSlot so it can write
    # per-step side-values into slot-local staging and run its own D2H
    # against ``ready_event``. Returns an opaque handle threaded back
    # into ``materialize_tokens``. Prefill skips this hook entirely
    # (prefill's first token is always plain text).
    post_sample: Callable[..., Any] | None = None

    # materialize_tokens(token_ids_cpu, sequences, batch_idx, step_handle) -> list[Token]
    # Default: TextToken-only materialisation, step_handle ignored.
    materialize_tokens: Callable[..., list] | None = None

    # prepare_decode_inputs(slot, batch_idx, batch_size) -> None
    # Default: no-op. Runtime gathers any aux decode inputs into slot.
    prepare_decode_inputs: Callable[..., None] | None = None


__all__ = ["SamplingHooks"]
