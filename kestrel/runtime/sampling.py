"""Runtime-side sampling hooks consumed by the generation scheduler.

The scheduler runs the generic parts of token sampling — sampling
itself, batch/lifecycle management, the D2H transfer of sampled token
ids + logprobs. Anything model-specific that has to happen *around*
that sampling step (e.g. Moondream's per-step coord/size decode from
hidden states) plugs in here.

A runtime exposes its hooks through ``runtime.sampling_hooks``. Runtimes that
don't need custom behaviour expose ``SamplingHooks()`` and, because every field
is ``None``, the per-step path collapses to "sample tokens, ship them home as
``TextToken``s." The runtime owns all storage and D2H for any
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

    1. ``sample_greedy(...)`` optionally replaces ordinary argmax after all
       skill and request-level masks have been applied.
    2. Scheduler samples token ids on the compute stream and records
       ``ready_event`` once they're written to the staging buffer.
    3. ``post_sample(...)`` fires next (compute stream) — runtime runs
       any GPU work it needs (e.g. decode side-values from
       ``hidden_last``) and initiates its own D2H against
       ``ready_event``. It returns an opaque handle the runtime
       understands later.
    4. Scheduler initiates its own D2H for token ids + logprobs.
    5. ``prepare_decode_inputs(...)`` fires before the next decode
       launch — runtime gathers any model-specific decode inputs from
       its own per-batch-idx state (the scheduler gathers token ids
       generically).
    6. On commit, scheduler reads CPU-side token ids + logprobs and
       calls ``materialize_tokens(token_ids_cpu, sequences, batch_idx,
       step_handle)`` to build the typed Token list it hands to skills.
    """

    # sample_greedy(logits, out, *, sequences, batch_idx) -> Tensor
    # Runs for the first prefill sample and every non-speculative decode sample
    # after all static and one-shot masks. The hook must write the selected ids
    # into ``out`` with one batched device path and must not synchronize or read
    # device values on the host. It is restricted to greedy sampling without
    # token logprobs. Runtimes exposing it must leave ``spec`` disabled until
    # their speculative decoder implements equivalent semantics.
    sample_greedy: Callable[..., Any] | None = None

    # post_sample(slot, *, sampled_ids, hidden_last, sequences,
    #             batch_idx, temperatures, top_ps, token_logprobs,
    #             ready_event) -> Any
    # Receives the runtime's prefill/decode slot so it can write
    # per-step side-values into slot-local staging and run its own D2H
    # against ``ready_event``. Returns an opaque handle threaded back
    # into ``materialize_tokens``. It runs for prefill too because a skill may
    # constrain the first sampled token to a model-specific token type.
    post_sample: Callable[..., Any] | None = None

    # materialize_tokens(token_ids_cpu, sequences, batch_idx, step_handle) -> list[Token]
    # Default: TextToken-only materialisation, step_handle ignored.
    # ``step_handle`` is the *non-spec* ``post_sample`` return value (e.g.
    # Moondream's ``(slot, batch_size)`` aux handle). The speculative path's
    # per-step side-values have a different shape (``SpecSideValues``) and a
    # different decode (values come from a packed per-position hidden, not from
    # slot-local staging), so they go through ``materialize_spec_tokens`` below
    # rather than being forced into this hook.
    materialize_tokens: Callable[..., list] | None = None

    # materialize_spec_tokens(token_ids_cpu, sequences, batch_idx, side_values,
    #                         token_logprobs=None) -> list[Token]
    # Speculative-decode analog of ``materialize_tokens``. ``side_values`` is the
    # macro-step's :class:`~kestrel.runtime.spec.SpecSideValues` (the target's
    # per-committed-position final hidden + sampling knobs), from which a spatial
    # runtime decodes coord/size ids into ``CoordToken`` / ``SizeToken``. A
    # runtime that does not type spatial tokens on the spec path leaves this
    # ``None``; the scheduler then materialises plain ``TextToken``s (and never
    # passes ``SpecSideValues`` into ``materialize_tokens``, whose handle shape it
    # does not match).
    #
    # ``token_logprobs`` (optional) is the flat per-committed-position vocab-token
    # logprob list (parallel to ``token_ids_cpu``) the scheduler staged from the
    # spec decoder. A spatial runtime mutates it **in place** to add each spatial
    # position's coord/size head logprob, mirroring the non-spec ``post_sample``
    # path (where ``compute_spatial_values`` folds the spatial head logprob into
    # the vocab logprob in-place); the spec decoder only gathers the vocab logprob.
    # ``None`` means no request wanted logprobs (decode is value-producing only).
    materialize_spec_tokens: Callable[..., list] | None = None

    # prepare_decode_inputs(slot, batch_idx, batch_size) -> None
    # Default: no-op. Runtime gathers any aux decode inputs into slot.
    prepare_decode_inputs: Callable[..., None] | None = None


__all__ = ["SamplingHooks"]
