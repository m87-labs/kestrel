"""Speculative decoding driver for Qwen3.5 + a DFlash drafter.

Wraps a :class:`Qwen35Runtime` and a :class:`DFlashProposer` into a single-sequence
generate loop:

    prefill -> [ draft K -> verify block (one kernel forward) -> greedy accept ->
                 O(1) cursor + conv-window-roll commit -> ring-buffer flush ] -> ...

The verify step runs the GQA ReplaySSM verify kernel over ``[cur, d1..dK]`` in one
forward (``spec_verify=True``); acceptance is the longest prefix whose argmax matches
the draft, plus the bonus token at the first rejection. Commit is O(1): advance the
ring-buffer cursor by ``a+1`` and roll the conv window over the accepted prefix (using
the block pre-conv the GDN forward caches). When the ring buffer would overflow, the
committed deltas are folded into the checkpoint (exact ``torch.exp``) and the cursor is
reset, which keeps the kernel's reconstruct in its lossless regime indefinitely.

This is lossless versus greedy decoding (validated to 128/128 on p1) and is the
production entry point — the loop no longer lives in a bench.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Iterator

import torch

from kestrel.runtime.preprocessing import derive_image_insertion_offset

from .model import DFlashConfig, DFlashDraftModel
from .proposer import DFlashProposer
from .sampling import (
    GREEDY_TEMPERATURE_EPS,
    _gumbel_argmax,
    logits_to_probs,
    rejection_sample_block,
)


# The replay verify + materialize cubins are shipped in the precompiled
# ``gated_delta`` bundle (``scripts/precompile/families/gated_delta.py``) only for
# these ReplaySSM ring capacities. ``flush_cap`` selects that ring capacity, so a
# value outside this set has no shipped verify/materialize cubin: with
# ``KESTREL_CUTE_JIT`` unset the spec verify path falls through to the torch
# fallback (Python loops / ``.item()``), which cannot be captured into the spec
# runner's verify CUDA graph. We therefore reject unshipped caps at construction
# (unless dev JIT is enabled to compile them on demand) instead of failing
# opaquely during graph capture. Keep in sync with the ``replay_verify_indexed``
# / ``replay_materialize_*`` variant entries in the precompile registry.
_SHIPPED_FLUSH_CAPS = (32, 64)

# The replay VERIFY cubin variant key is parameterised by the block size as well
# as the ring capacity (``replay_verify_indexed_..._c{cap}_t{T}`` in
# ``dispatch.py``, where ``T`` is the per-verify block length == ``block_size``).
# The precompiled bundle ships verify cubins only at ``t16`` (the cap-32 and
# cap-64 entries are both ``_t16``), so a drafter whose ``block_size`` differs
# from this passes the ``flush_cap`` check above but still resolves NO verify
# cubin -- it then falls to the uncapturable torch verify path at graph capture,
# exactly like an unshipped cap. Keep in sync with the ``block_size`` field of
# the ``replay_verify_indexed`` variant entries in the precompile registry.
_SHIPPED_VERIFY_BLOCK_SIZES = (16,)


def _cute_jit_enabled() -> bool:
    """True when dev-time CUTE JIT is on (``KESTREL_CUTE_JIT=1``).

    With JIT on, an unshipped ``flush_cap`` can still resolve a verify cubin by
    compiling it on demand, so the shipped-cap restriction is relaxed.
    """
    import os

    return os.environ.get("KESTREL_CUTE_JIT") == "1"


def _validate_flush_cap(flush_cap: int, block_size: int) -> None:
    """Validate ``flush_cap``/``block_size`` before any ReplaySSM/verify-graph setup.

    Enforces the ``flush_cap >= block_size`` ring invariant and, unless dev JIT
    is enabled, that BOTH the ring capacity (``flush_cap``) and the verify block
    length (``block_size``) match a shipped precompiled verify/materialize cubin
    -- the verify variant key is ``..._c{flush_cap}_t{block_size}``, so either
    one being unshipped forces the spec runner onto the uncapturable torch verify
    path at graph capture. Validate them up front so callers fail with a clear
    message instead of opaquely during CUDA-graph capture."""
    if flush_cap < block_size:
        raise ValueError(
            f"flush_cap ({flush_cap}) must be >= block_size ({block_size})"
        )
    if _cute_jit_enabled():
        # Dev JIT can compile any (cap, block_size) verify cubin on demand, so
        # the shipped-variant restriction below is relaxed.
        return
    if flush_cap not in _SHIPPED_FLUSH_CAPS:
        raise ValueError(
            f"flush_cap ({flush_cap}) has no shipped replay verify/materialize "
            f"cubin; the precompiled gated_delta bundle ships only "
            f"{sorted(_SHIPPED_FLUSH_CAPS)}. An unshipped cap forces the torch "
            "verify fallback, which cannot be captured into the spec runner's "
            "CUDA graph. Choose a shipped flush_cap or set KESTREL_CUTE_JIT=1 to "
            "compile the variant on demand."
        )
    if block_size not in _SHIPPED_VERIFY_BLOCK_SIZES:
        raise ValueError(
            f"block_size ({block_size}) has no shipped replay verify cubin; the "
            f"precompiled gated_delta bundle ships verify cubins only at "
            f"block_size {sorted(_SHIPPED_VERIFY_BLOCK_SIZES)} (the verify "
            f"variant key is ..._c{{cap}}_t{{block_size}}). A non-shipped "
            "block_size forces the torch verify fallback, which cannot be "
            "captured into the spec runner's CUDA graph. Use a shipped "
            "block_size or set KESTREL_CUTE_JIT=1 to compile the variant on "
            "demand."
        )


@dataclass
class SpecDecodeResult:
    token_ids: list[int]
    mean_accept: float          # mean drafts accepted per block (excludes the bonus)
    steps: int

    @property
    def mean_advance(self) -> float:
        """Tokens committed per verify forward (accepted drafts + bonus)."""
        return self.mean_accept + 1.0


class SpecDecoder:
    """Single-sequence speculative decoding with a DFlash drafter.

    ``flush_cap`` bounds the ReplaySSM ring buffer; it is applied to the cache via
    ``linear_replay_capacity`` so the buffers are allocated at the right size. Must
    satisfy ``flush_cap >= block_size`` (one block must fit before a flush is forced);
    a comfortable margin (e.g. 4x block) minimizes flush frequency.
    """

    def __init__(
        self,
        runtime,
        drafter: DFlashDraftModel,
        dcfg: DFlashConfig,
        *,
        flush_cap: int = 32,
    ) -> None:
        self.rt = runtime
        self.dcfg = dcfg
        self.lm = runtime.model.model.language_model
        self.lm_head = runtime.model.lm_head
        self.embed = self.lm.embed_tokens
        self.device = next(self.lm.parameters()).device

        self.proposer = DFlashProposer(drafter, self.embed, self.lm_head, dcfg)
        self.num_spec = int(self.proposer.num_speculative_tokens)   # K drafts
        self.block_size = int(dcfg.block_size)                      # K + 1 (cur + drafts)
        self.target_layer_ids = tuple(dcfg.target_layer_ids)

        # Same shipped-cubin restriction as ``SpecRunner``: the single-sequence
        # reference path also builds the verify ring at ``flush_cap``.
        _validate_flush_cap(int(flush_cap), self.block_size)
        self.flush_cap = int(flush_cap)
        pool_capacity = int(runtime._linear_state_pool.replay_capacity)
        if pool_capacity != self.flush_cap:
            raise ValueError(
                f"runtime linear-state pool replay_capacity ({pool_capacity}) "
                f"must equal flush_cap ({self.flush_cap}); configure "
                "spec_decode.flush_cap before building the runtime"
            )

        self.gdn_layer_idxs = [
            i for i, layer in enumerate(self.lm.layers)
            if getattr(layer, "linear_attn", None) is not None
        ]

    # -- aux-hidden capture ----------------------------------------------------
    # The model loop calls ``decoder_layer._forward_from_normalized(...)`` directly
    # (a fused path that bypasses ``nn.Module.__call__``), so torch forward hooks
    # never fire. We scope-wrap that method on the target layers for the duration of
    # generate() and restore it in the finally block — temporary instrumentation,
    # not a persistent monkeypatch. The aux hidden the drafter consumes is the
    # second return value (the next layer's normalized residual).

    def _install_hooks(self, sink: dict[int, torch.Tensor]) -> list:
        saved = []
        for i in self.target_layer_ids:
            layer = self.lm.layers[i]
            orig = layer._forward_from_normalized

            def make(idx, fn):
                def wrapped(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    sink[idx] = (out[1] if out[1] is not None else out[0]).detach()
                    return out
                return wrapped

            layer._forward_from_normalized = make(i, orig)
            saved.append((layer, orig))
        return saved

    @staticmethod
    def _remove_hooks(saved: list) -> None:
        for layer, orig in saved:
            layer._forward_from_normalized = orig

    def _target_hidden(self, sink: dict[int, torch.Tensor]) -> torch.Tensor:
        # [seq, len(target_layer_ids) * hidden] for batch 0
        return torch.cat([sink[i][0] for i in self.target_layer_ids], dim=-1)

    def _ids(self, rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    def _target_hidden_b(self, sink: dict[int, torch.Tensor]) -> torch.Tensor:
        """Batched aux hidden: [B, len, len(target_layer_ids) * hidden]."""
        return torch.cat([sink[i] for i in self.target_layer_ids], dim=-1)

    # -- generate -------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        eager: bool = False,
        profile: dict[str, Any] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
    ) -> SpecDecodeResult:
        """Speculative decode one sequence.

        ``temperature == 0`` (the default) keeps the validated lossless-greedy
        path: drafts are the drafter's argmax, accept is the exact argmax match,
        and the bonus is the target argmax. ``temperature > 0`` switches to
        *rejection sampling* (Leviathan/Chen): the drafter samples from its
        ``temperature``/``top_p``/``top_k`` distribution ``q`` (recording
        ``q(x_i)``), and verify runs the accept/reject rule against the target
        distribution ``p`` built with the SAME knobs, so the committed tokens are
        distributed exactly as the non-spec sampler's ``p``. ``seed`` seeds the
        per-sequence RNG used for both the draft sampling and the accept/residual
        draws.
        """
        rt, dev = self.rt, self.device
        K, block_size = self.num_spec, self.block_size
        gdn = self.gdn_layer_idxs
        # Spec sampling mode: greedy keeps the exact-argmax path untouched.
        greedy = float(temperature) <= GREEDY_TEMPERATURE_EPS
        spec_gen: torch.Generator | None = None
        if not greedy and seed is not None:
            spec_gen = torch.Generator(device=dev)
            spec_gen.manual_seed(int(seed))
        pt = rt.page_table

        pending_events: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | None, float]] = []
        profile_events = profile.setdefault("events", []) if profile is not None else None
        profile_counters = profile.setdefault("counters", {}) if profile is not None else None

        @contextmanager
        def record(name: str) -> Iterator[None]:
            if profile_events is None:
                yield
                return
            start_event = end_event = None
            if dev.type == "cuda":
                stream = torch.cuda.current_stream(dev)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                torch.cuda.nvtx.range_push(name)
            cpu_start = time.perf_counter()
            try:
                yield
            finally:
                cpu_ms = (time.perf_counter() - cpu_start) * 1000.0
                if dev.type == "cuda":
                    torch.cuda.nvtx.range_pop()
                if end_event is not None:
                    end_event.record(torch.cuda.current_stream(dev))
                pending_events.append((name, start_event, end_event, cpu_ms))

        def count(name: str, value: int | float = 1) -> None:
            if profile_counters is not None:
                profile_counters[name] = profile_counters.get(name, 0) + value

        def finalize_profile() -> None:
            if profile_events is None:
                return
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            for name, start_event, end_event, cpu_ms in pending_events:
                gpu_ms = 0.0
                if start_event is not None and end_event is not None:
                    gpu_ms = float(start_event.elapsed_time(end_event))
                profile_events.append(
                    {"name": name, "gpu_ms": gpu_ms, "cpu_ms": cpu_ms}
                )

        # Allocate the page-table row first, then bring the reservation +
        # hook install UNDER the cleanup ``try`` below. ``reserve`` /
        # ``commit_block_table`` can raise (e.g. prompt + output exceeds the KV
        # budget); if they ran before the ``try`` a failure would skip the
        # ``finally`` and leak this ``batch_idx`` (allocate() popped it off the
        # free list) plus any partially-reserved pages -- a single failed
        # ``generate()`` would permanently consume a batch slot / pages.
        # ``_release_batch_idx`` (the finally) erases the row, returning the
        # batch_idx + its pages to the pool.
        batch_idx = rt.page_table.allocate()
        slot = 0  # per-request cache: GDN state lives at local slot 0
        sink: dict[int, torch.Tensor] = {}
        handles: list = []
        try:
            cache = rt._new_cache()
            rt.page_table.reserve(batch_idx, len(prompt_ids) + max_new_tokens + 4 * block_size + 8)
            rt.page_table.commit_block_table([batch_idx])
            slot_idx = self._ids([[slot]])[0]

            handles = self._install_hooks(sink)
            # Prefill (eager — one-shot, variable length).
            with record("prefill_forward"):
                lh, fcache = rt._forward_base(
                    input_ids=self._ids([prompt_ids]),
                    past_key_values=cache,
                    batch_idx=batch_idx,
                    cache_position_ids=torch.arange(len(prompt_ids), device=dev).view(1, -1),
            )
            cache = fcache.past_key_values  # the inner Qwen35InferenceCache
            with record("prefill_lm_head_argmax"):
                # The first committed token (``cur_t``) is the request's token0.
                # Greedy: the target argmax (byte-identical to the validated
                # lossless path). Non-greedy: SAMPLE it from the target
                # distribution ``p`` built with the SAME temperature/top_p/top_k
                # as the rejection-sampling spec loop below -- otherwise token0
                # (the only token at ``max_new_tokens==1``, and the first of every
                # sampled run) would ignore temperature/top_p/top_k/seed and
                # collapse to greedy before the sampler ever runs. Mirrors
                # ``SpecStepRunner._select_first_token`` (``_gumbel_argmax`` is the
                # graph-safe multinomial the rest of this module uses).
                last_logits = self.lm_head(lh[0])[-1:]              # [1, vocab]
                if greedy:
                    cur_t = last_logits[0].argmax()  # GPU-resident current token
                else:
                    first_probs = logits_to_probs(
                        last_logits, temperature, top_p, top_k)     # [1, vocab]
                    cur_t = _gumbel_argmax(first_probs, spec_gen)[0]
            hid = self._target_hidden(sink).clone()
            conv_k = cache.layers[gdn[0]].conv_states.shape[-1]
            replay_len = int(cache.layers[gdn[0]].replay_lengths[slot])

            # Capture the verify forward as a CUDA graph. We capture the inner
            # language model directly (not _forward_base, which allocates metadata)
            # with static buffers updated in place each step.
            cb = self._ids([batch_idx])
            n_prompt = len(prompt_ids)
            ids_buf = torch.zeros(1, block_size, device=dev, dtype=torch.long)
            cpos_buf = torch.arange(n_prompt, n_prompt + block_size, device=dev).view(1, -1).clone()
            slot_buf = pt.build_slot_mapping(batch_idx=cb, positions=cpos_buf).clone()
            page_tbl = torch.index_select(pt.page_table, 0, cb).clone()
            seqk_buf = (cpos_buf.max(dim=1).values.to(torch.int32) + 1).clone()
            gdn_idx = rt._gdn_state_indices_for_cache(fcache, cache_batch_idx=cb, batch_count=1)
            kw = dict(
                input_ids=ids_buf, position_ids=cpos_buf, past_key_values=cache,
                cache_position_ids=cpos_buf, slot_mapping=slot_buf, page_table=page_tbl,
                paged_kv_seqlens_k=seqk_buf, gdn_state_indices=gdn_idx, spec_verify=True,
                cu_seq_lens_q=torch.tensor([0, block_size], device=dev, dtype=torch.int32),
            )
            # Warmup + capture pollute the GDN cache (the conv over-advances), so
            # snapshot the GDN state before capture and restore it after.
            _gst = ("conv_states", "recurrent_states", "replay_checkpoint_states",
                    "replay_k", "replay_u", "replay_g", "replay_lengths")
            gdn_snap = {
                idx: {n: getattr(cache.layers[idx], n).clone()
                      for n in _gst if getattr(cache.layers[idx], n, None) is not None}
                for idx in gdn
            }
            if not eager:
                with record("verify_capture_warmup"):
                    warm = torch.cuda.Stream(); warm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warm):
                        for _ in range(3):
                            self.lm(**kw)
                    torch.cuda.current_stream().wait_stream(warm)
                with record("verify_capture"):
                    verify_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(verify_graph):
                        graph_out = self.lm(**kw)
                out_hidden = graph_out.last_hidden_state
                for idx in gdn:
                    for n, v in gdn_snap[idx].items():
                        getattr(cache.layers[idx], n).copy_(v)

            # Capture the draft (DFlash proposer) as a CUDA graph too. The drafter
            # re-reads target_hidden each step; keep it left-aligned in a fixed-size
            # buffer and mask the padded tail. Rotary is relative, so only the
            # block's position_ids need to track the live context length.
            maxc = n_prompt + max_new_tokens + self.flush_cap + 2 * block_size
            fc_in = hid.shape[-1]
            th_buf = torch.zeros(1, maxc, fc_in, dtype=hid.dtype, device=dev)
            th_buf[0, :n_prompt] = hid
            block_ids_buf = torch.full((1, block_size), self.dcfg.mask_token_id, dtype=torch.long, device=dev)
            dpos_buf = torch.cat([
                torch.arange(maxc, device=dev),
                torch.arange(n_prompt, n_prompt + block_size, device=dev),
            ]).view(1, -1).clone()
            dmask_buf = torch.zeros(1, 1, 1, maxc + block_size, dtype=hid.dtype, device=dev)
            dmask_buf[0, 0, 0, n_prompt:maxc] = float("-inf")

            # Greedy captures argmax drafts directly (unchanged). Non-greedy
            # captures the draft LOGITS [K, vocab]; sampling (q + draft tokens)
            # happens outside the graph (RNG can't be captured).
            def _draft_logits():
                noise = self.embed(block_ids_buf)
                h = self.proposer.drafter(noise, th_buf, dpos_buf, attn_mask=dmask_buf)
                return self.lm_head(h[:, 1:, :])[0]   # [K, vocab]

            def _draft():
                return _draft_logits().argmax(-1)     # greedy: [K]

            if not eager:
                with record("draft_capture_warmup"):
                    dwarm = torch.cuda.Stream(); dwarm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(dwarm):
                        for _ in range(3):
                            (_draft() if greedy else _draft_logits())
                    torch.cuda.current_stream().wait_stream(dwarm)
                with record("draft_capture"):
                    draft_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(draft_graph):
                        if greedy:
                            drafts_out = _draft()
                        else:
                            draft_logits_out = _draft_logits()

            # --- GPU-resident chunked decode loop ---
            # accept / commit / current-token / context all stay on-device, so the
            # host issues whole steps without blocking on the verify (no per-step
            # pred/draft D2H). The host reads the committed count only once per
            # chunk of ``chunk`` steps to bound the loop; the ring is flushed
            # (state-preserving) at each chunk boundary, with ``chunk`` sized so the
            # ring cannot overflow within a chunk. One D2H at the end returns the
            # committed tokens.
            committed_buf = torch.empty(
                max_new_tokens + 2 * self.flush_cap + block_size, dtype=torch.long, device=dev
            )
            n_committed_t = torch.zeros((), dtype=torch.long, device=dev)
            accepts_sum_t = torch.zeros((), dtype=torch.long, device=dev)
            ar_block = torch.arange(block_size, device=dev)
            ar_conv = torch.arange(conv_k, device=dev)
            ar_maxc = torch.arange(maxc, device=dev)
            chunk = max(1, self.flush_cap // block_size)
            n_steps = 0
            while True:
                committed = int(n_committed_t.item())
                if committed >= max_new_tokens:
                    break
                # Cap this chunk's verify blocks by the remaining output. Each step
                # commits at least one token (``accept + 1 >= 1``), so at most
                # ``max_new_tokens - committed`` more blocks are needed; running the
                # full ``flush_cap // block_size`` blocks once ``n_committed_t`` has
                # already reached ``max_new_tokens`` would speculate past the end and
                # (with ``page_size=1`` and only ``4*block_size+8`` reserved headroom,
                # while a chunk can advance up to ``flush_cap`` positions) write KV
                # beyond the reserved row. Reading the already-synced ``committed``
                # bounds the loop with no extra per-step D2H (the ``while`` synced it).
                steps_this_chunk = min(chunk, max_new_tokens - committed)
                for _ in range(steps_this_chunk):
                    ctx_t = n_prompt + n_committed_t  # GPU scalar
                    # Draft (graphed): buffers updated from GPU scalars only.
                    block_ids_buf[0, 0] = cur_t
                    dpos_buf[0, maxc:] = ar_block + ctx_t
                    dmask_buf[0, 0, 0, :maxc] = torch.where(
                        ar_maxc < ctx_t, 0.0, float("-inf")
                    ).to(dmask_buf.dtype)
                    # Draft: greedy -> argmax tokens; non-greedy -> sample from
                    # q (record draft_probs for the accept ratio).
                    draft_probs_t = None
                    if greedy:
                        if eager:
                            drafts_tensor = _draft()
                        else:
                            draft_graph.replay()
                            drafts_tensor = drafts_out
                    else:
                        if eager:
                            d_logits = _draft_logits()
                        else:
                            draft_graph.replay()
                            d_logits = draft_logits_out
                        draft_probs_t = logits_to_probs(
                            d_logits, temperature, top_p, top_k)        # [K, vocab]
                        drafts_tensor = torch.multinomial(
                            draft_probs_t, 1, generator=spec_gen).squeeze(-1)  # [K]
                    # Snapshot conv windows (verify over-advances; commit rolls).
                    conv_snap = {idx: cache.layers[idx].conv_states[slot].clone() for idx in gdn}
                    # Verify (graphed).
                    ids_buf[0, 0] = cur_t
                    ids_buf[0, 1:].copy_(drafts_tensor)
                    cpos_buf.copy_((ar_block + ctx_t).view(1, -1))
                    slot_buf.copy_(pt.build_slot_mapping(batch_idx=cb, positions=cpos_buf))
                    seqk_buf.copy_(cpos_buf.max(dim=1).values.to(torch.int32) + 1)
                    if eager:
                        verify_hidden = self.lm(**kw).last_hidden_state[0]
                    else:
                        verify_graph.replay()
                        verify_hidden = out_hidden[0]
                    block_hid = self._target_hidden(sink)
                    if greedy:
                        # GPU-resident accept = longest matching draft prefix; the
                        # next current token is the target argmax at the boundary.
                        pred_tensor = self.lm_head(verify_hidden).argmax(-1)  # [block_size]
                        accept_t = (pred_tensor[:K] == drafts_tensor).int().cumprod(0).sum()
                        commit_block = torch.cat([cur_t.view(1), drafts_tensor])
                        next_cur = pred_tensor[accept_t]
                    else:
                        # Rejection sampling: accept/reject against the target
                        # distribution p (same temperature/top_p/top_k as q), emit
                        # the recovered/bonus replacement at the boundary.
                        tgt_logits = self.lm_head(verify_hidden)             # [block, vocab]
                        out_tokens, accept_t = rejection_sample_block(
                            drafts_tensor.unsqueeze(0),
                            draft_probs_t.unsqueeze(0),
                            tgt_logits.unsqueeze(0),
                            temperature, top_p, top_k, generator=spec_gen,
                        )
                        out_tokens = out_tokens[0]                          # [block]
                        accept_t = accept_t[0]
                        # Committed block = [cur, accepted drafts]; out_tokens[:K]
                        # holds the accepted drafts at cols < accept (== drafts),
                        # the tail is unused (only accept+1 entries are committed).
                        commit_block = torch.cat([cur_t.view(1), out_tokens[:K]])
                        next_cur = out_tokens[accept_t]                     # recovered/bonus
                    # Commit: O(1) cursor + conv-window roll over the accepted prefix
                    # (gather the last conv_k cols of [conv_snap | preconv] starting
                    # at accept+1 -- the dynamic offset stays on-device).
                    for idx in gdn:
                        lc = cache.layers[idx]
                        lc.replay_lengths[slot] += (accept_t + 1)
                        cat_conv = torch.cat([conv_snap[idx], lc.spec_block_preconv[0]], dim=-1)
                        lc.conv_states[slot] = cat_conv.index_select(-1, ar_conv + (accept_t + 1))
                    # Write the whole block at the GPU cursor; only accept+1 are
                    # valid, the tail is overwritten next step (or trimmed at end).
                    # The drafter reads only th_buf[:ctx], so the stale tail is unread.
                    th_buf[0].index_copy_(0, ctx_t + ar_block, block_hid[:block_size])
                    committed_buf.index_copy_(
                        0, n_committed_t + ar_block, commit_block
                    )
                    cur_t = next_cur
                    accepts_sum_t = accepts_sum_t + accept_t
                    n_committed_t = n_committed_t + accept_t + 1
                    n_steps += 1
                # Forced flush at the chunk boundary (state-preserving; resets ring).
                for idx in gdn:
                    cache.layers[idx].materialize_recurrent_from_replay(
                        slot_idx.view(1), write_recurrent=False
                    )

            out = committed_buf[:max_new_tokens].tolist()
            mean_accept = (float(accepts_sum_t.item()) / n_steps) if n_steps else 0.0
            finalize_profile()
            return SpecDecodeResult(token_ids=out, mean_accept=mean_accept, steps=n_steps)
        finally:
            self._remove_hooks(handles)
            rt._release_batch_idx(batch_idx)

    # -- batched generate -----------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        *,
        eos_id: int | None = None,
    ) -> list[SpecDecodeResult]:
        """Batched speculative decoding over ``B`` sequences sharing the spec loop.

        Each of the ``B`` sequences is decoded with its own page-table slot, its
        own GDN ReplaySSM ring buffer (cache local slot ``i``), and its own
        ragged commit: sequences accept and advance *different* amounts per step.
        The verify forward (attn + MLP) and the draft forward are each run once
        per step as a single ``B``-batched CUDA-graph replay, amortizing the
        weight reads over the batch.

        Finish policy (documented): a finished sequence (hit ``eos_id`` or its
        ``max_new_tokens``) stays in the batch in a *frozen* state until all
        sequences finish. Its block still runs through both graphs (harmless),
        but its commit is suppressed (cursor / conv / KV position do not advance)
        and its outputs are discarded. No mid-run compaction -- the graphs are
        captured once at a fixed ``B`` and reused for every step.
        """
        rt, dev = self.rt, self.device
        K, block_size = self.num_spec, self.block_size
        gdn = self.gdn_layer_idxs
        pt = rt.page_table
        B = len(prompts)
        if B == 0:
            return []

        # B page-table slots; GDN state lives at cache local slot i for seq i.
        # The per-row reservation runs UNDER the cleanup ``try`` below: it
        # allocates pages for every row before the loop body, so if any row's
        # ``reserve`` raises (oversized prompt / tight pool) the rows already
        # reserved this call would otherwise never be erased -- their pages +
        # batch_idx would leak, permanently shrinking pool capacity for later
        # requests. The ``finally`` releases ALL allocated batch_idx (each erase
        # returns the row's batch_idx + any pages it did get), so a failed setup
        # frees every row with no slot/page leak.
        batch_idx = [pt.allocate() for _ in range(B)]
        cb = self._ids([batch_idx])[0].to(torch.int64)          # [B]
        # GDN slot i == cache row i (fresh per-request cache -> arange(B)).
        gdn_idx = torch.arange(B, dtype=torch.long, device=dev)
        slot_idx = gdn_idx                                       # alias

        sink: dict[int, torch.Tensor] = {}
        handles: list = []
        try:
            cache = rt._new_cache()
            for bi, p in zip(batch_idx, prompts):
                pt.reserve(bi, len(p) + max_new_tokens + 4 * block_size + 8)
            pt.commit_block_table(batch_idx)

            handles = self._install_hooks(sink)
            # ---- Prefill: rectangular equal-length batches use the runtime path
            # validated by target greedy; ragged batches keep the packed prefill. ----
            prompt_lens = [len(p) for p in prompts]
            page_tbl_full = torch.index_select(pt.page_table, 0, cb)   # [B, pages]
            if len(set(prompt_lens)) == 1:
                cpos = torch.arange(prompt_lens[0], device=dev).view(1, -1).expand(B, -1).clone()
                lh_pf, fcache = rt._forward_base(
                    input_ids=self._ids(prompts),
                    past_key_values=cache,
                    batch_idx=batch_idx,
                    cache_position_ids=cpos,
                )
                cache = fcache.past_key_values
                cur = [int(self.lm_head(lh_pf[r, -1:])[0].argmax()) for r in range(B)]
                prefill_hid = self._target_hidden_b(sink).clone()
                fc_in = prefill_hid.shape[-1]
            else:
                total = sum(prompt_lens)
                flat_ids = [t for p in prompts for t in p]
                seq_idx_rows = [r for r, n in enumerate(prompt_lens) for _ in range(n)]
                pos_rows = [j for n in prompt_lens for j in range(n)]
                cu = [0]
                for n in prompt_lens:
                    cu.append(cu[-1] + n)
                in_ids = self._ids([flat_ids])                       # [1, total]
                cpos = self._ids([pos_rows])                         # [1, total]
                sidx = torch.tensor([seq_idx_rows], device=dev, dtype=torch.int32)
                cu_t = torch.tensor(cu, device=dev, dtype=torch.int32)
                # Per-token batch_idx for slot mapping (packed layout, batch dim 1).
                bidx_tok = torch.tensor([[batch_idx[r] for r in seq_idx_rows]],
                                        device=dev, dtype=torch.int64)
                slotmap = pt.build_slot_mapping(batch_idx=bidx_tok, positions=cpos)
                seqk_pf = torch.tensor([n for n in prompt_lens], device=dev, dtype=torch.int32)
                rt.model.model.rope_deltas = None
                out_pf = rt.model.model.language_model(
                    input_ids=in_ids, position_ids=cpos, past_key_values=cache,
                    cache_position_ids=cpos, slot_mapping=slotmap,
                    page_table=page_tbl_full, paged_kv_seqlens_k=seqk_pf,
                    seq_idx=sidx, cu_seq_lens_q=cu_t,
                )
                cache.advance_to(max(prompt_lens))
                lh_pf = out_pf.last_hidden_state[0]                   # [total, H]
                # First sampled token + aux hidden per sequence.
                last_off = [cu[r + 1] - 1 for r in range(B)]
                cur = [int(self.lm_head(lh_pf[last_off[r]:last_off[r] + 1])[0].argmax())
                       for r in range(B)]
                # Aux hidden over the full prefill is packed [1, total, fc]; split it.
                hid_packed = self._target_hidden_b(sink)[0]          # [total, fc]
                fc_in = hid_packed.shape[-1]
                prefill_hid = torch.zeros(B, max(prompt_lens), fc_in, dtype=hid_packed.dtype, device=dev)
                for r in range(B):
                    prefill_hid[r, :prompt_lens[r]] = hid_packed[cu[r]:cu[r + 1]]
            conv_k = cache.layers[gdn[0]].conv_states.shape[-1]

            # ---- Build static buffers for the batched verify graph. ----
            # GDN takes [B, K+1] (verify-replay kernel, grid over B*v-heads).
            # Attention takes the same [B, K+1] queries via per-seq seqused_q/k
            # (no cu_seq_lens_q: each block attends causally to its own paged KV).
            ids_buf = torch.zeros(B, block_size, device=dev, dtype=torch.long)
            cpos_buf = torch.zeros(B, block_size, device=dev, dtype=torch.long)
            for r in range(B):
                cpos_buf[r] = torch.arange(prompt_lens[r], prompt_lens[r] + block_size,
                                           device=dev)
            slot_buf = pt.build_slot_mapping(batch_idx=cb, positions=cpos_buf).clone()
            seqk_buf = (cpos_buf.max(dim=1).values.to(torch.int32) + 1).clone()
            # Attention uses the shipped *packed* paged cubin: equal-length blocks
            # described by cu_seq_lens_q = [0, T, 2T, ..., B*T] over the flattened
            # [B, T] query block. GDN sees batch_size=B (its verify-replay kernel
            # is batched over B*v-heads), and ignores cu_seq_lens_q (batch_size>1).
            cu_q_buf = torch.arange(0, (B + 1) * block_size, block_size,
                                    device=dev, dtype=torch.int32)
            kw = dict(
                input_ids=ids_buf, position_ids=cpos_buf, past_key_values=cache,
                cache_position_ids=cpos_buf, slot_mapping=slot_buf,
                page_table=page_tbl_full, paged_kv_seqlens_k=seqk_buf,
                gdn_state_indices=gdn_idx, spec_verify=True,
                cu_seq_lens_q=cu_q_buf,
            )
            _gst = ("conv_states", "recurrent_states", "replay_checkpoint_states",
                    "replay_k", "replay_u", "replay_g", "replay_lengths")
            gdn_snap = {
                idx: {n: getattr(cache.layers[idx], n).clone()
                      for n in _gst if getattr(cache.layers[idx], n, None) is not None}
                for idx in gdn
            }
            warm = torch.cuda.Stream(); warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    self.lm(**kw)
            torch.cuda.current_stream().wait_stream(warm)
            verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(verify_graph):
                graph_out = self.lm(**kw)
            out_hidden = graph_out.last_hidden_state              # [B, K+1, H]
            for idx in gdn:
                for n, v in gdn_snap[idx].items():
                    getattr(cache.layers[idx], n).copy_(v)

            # ---- Build static buffers for the batched draft graph. ----
            maxc = max(prompt_lens) + max_new_tokens + block_size + 4
            th_buf = torch.zeros(B, maxc, fc_in, dtype=prefill_hid.dtype, device=dev)
            for r in range(B):
                th_buf[r, :prompt_lens[r]] = prefill_hid[r, :prompt_lens[r]]
            block_ids_buf = torch.full((B, block_size), self.dcfg.mask_token_id,
                                       dtype=torch.long, device=dev)
            # dpos: [B, maxc + block_size]; context positions then block positions.
            dpos_buf = torch.zeros(B, maxc + block_size, device=dev, dtype=torch.long)
            dpos_buf[:, :maxc] = torch.arange(maxc, device=dev)
            # dmask: per-seq additive mask over the maxc context positions.
            dmask_buf = torch.zeros(B, 1, 1, maxc + block_size,
                                    dtype=prefill_hid.dtype, device=dev)

            def _draft():
                noise = self.embed(block_ids_buf)
                h = self.proposer.drafter(noise, th_buf, dpos_buf, attn_mask=dmask_buf)
                return self.lm_head(h[:, 1:, :]).argmax(-1).to(torch.int32)  # [B, K]

            dwarm = torch.cuda.Stream(); dwarm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(dwarm):
                for _ in range(3):
                    _draft()
            torch.cuda.current_stream().wait_stream(dwarm)
            draft_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(draft_graph):
                drafts_out = _draft()                              # [B, K]

            # ---- Spec loop. ----
            ctx_len = list(prompt_lens)                            # live cache len / seq
            committed = [list(p) for p in prompts]
            done = [False] * B
            accepts: list[int] = []
            steps = 0
            arange_block = torch.arange(block_size, device=dev)

            def _all_done() -> bool:
                return all(done)

            while not _all_done():
                steps += 1
                # --- Draft (batched graph). Update buffers per seq, then replay. ---
                for r in range(B):
                    block_ids_buf[r, 0] = cur[r]
                    c = ctx_len[r]
                    dpos_buf[r, maxc:] = torch.arange(c, c + block_size, device=dev)
                    dmask_buf[r, 0, 0, :c] = 0
                    dmask_buf[r, 0, 0, c:maxc] = float("-inf")
                draft_graph.replay()
                drafts_b = drafts_out.tolist()                    # [[..K..]] * B

                # --- Flush ring buffers that would overflow this block. ---
                flush_rows = []
                for r in range(B):
                    if done[r]:
                        continue
                    lc0 = cache.layers[gdn[0]]
                    if int(lc0.replay_lengths[r]) + block_size > self.flush_cap:
                        flush_rows.append(r)
                if flush_rows:
                    fr = torch.tensor(flush_rows, device=dev, dtype=torch.long)
                    for idx in gdn:
                        lc = cache.layers[idx]
                        lc.materialize_recurrent_from_replay(fr, write_recurrent=False)

                # --- Snapshot conv windows (verify over-advances; commit rolls). ---
                conv_snap = {idx: cache.layers[idx].conv_states.clone() for idx in gdn}

                # --- Verify (batched graph). Build [B, K+1] block + metadata. ---
                for r in range(B):
                    ids_buf[r, 0] = cur[r]
                    for j in range(K):
                        ids_buf[r, j + 1] = drafts_b[r][j]
                    cpos_buf[r] = arange_block + ctx_len[r]
                slot_buf.copy_(pt.build_slot_mapping(batch_idx=cb, positions=cpos_buf))
                seqk_buf.copy_(cpos_buf.max(dim=1).values.to(torch.int32) + 1)
                verify_graph.replay()
                pred = self.lm_head(out_hidden).argmax(-1).tolist()   # [B, K+1]
                block_hid = self._target_hidden_b(sink)               # [B, K+1, fc]

                # --- Per-sequence accept + ragged commit. ---
                for r in range(B):
                    if done[r]:
                        continue
                    drafts = drafts_b[r]
                    accept = 0
                    for j in range(K):
                        if pred[r][j] == drafts[j]:
                            accept += 1
                        else:
                            break
                    bonus = pred[r][accept]
                    accepts.append(accept)

                    new_toks = [cur[r]] + drafts[:accept]
                    # Truncate at eos / max_new_tokens for the *output*, but the
                    # cache still committed exactly accept+1 tokens this step.
                    c = ctx_len[r]
                    for idx in gdn:
                        lc = cache.layers[idx]
                        lc.replay_lengths[r] += (accept + 1)
                        rolled = torch.cat(
                            [conv_snap[idx][r], lc.spec_block_preconv[r, :, : accept + 1]],
                            dim=-1,
                        )
                        lc.conv_states[r] = rolled[:, -conv_k:]
                    th_buf[r, c:c + accept + 1] = block_hid[r, : accept + 1]
                    committed[r] = committed[r] + new_toks
                    ctx_len[r] = c + accept + 1
                    cur[r] = bonus

                    # Finish check.
                    gen = committed[r][prompt_lens[r]:]
                    if eos_id is not None and eos_id in gen:
                        done[r] = True
                    elif len(gen) >= max_new_tokens:
                        done[r] = True

            results: list[SpecDecodeResult] = []
            n_acc = len(accepts)
            mean_accept = (sum(accepts) / n_acc) if n_acc else 0.0
            for r in range(B):
                gen = committed[r][prompt_lens[r]:]
                if eos_id is not None and eos_id in gen:
                    gen = gen[: gen.index(eos_id) + 1]
                gen = gen[:max_new_tokens]
                results.append(SpecDecodeResult(
                    token_ids=gen, mean_accept=mean_accept, steps=steps))
            return results
        finally:
            self._remove_hooks(handles)
            for bi in batch_idx:
                rt._release_batch_idx(bi)


# ---------------------------------------------------------------------------
# Graph-reusing runtime integration
# ---------------------------------------------------------------------------


class SpecRunner:
    """Graph-reusing batched speculative decoder for the serving runtime.

    ``SpecDecoder.generate_batch`` is correct but allocates a fresh cache and
    re-captures the verify + draft CUDA graphs on *every* call (~70 tok/s of
    per-request graph-capture tax). ``SpecRunner`` pays that cost once and reuses
    it across requests by mirroring how :class:`Qwen35Runtime`'s persistent
    decode path works:

    * a **fixed** set of ``B`` page-table ``batch_idx`` reserved once at
      construction (pages never freed -> page-table rows are address-stable);
    * a single **persistent** :class:`Qwen35InferenceCache` whose GDN linear
      layers are bound (``_linear_state_pool.bind_to_cache``) to the runtime's
      persistent ``_linear_state_pool`` tensors (fixed addresses);
    * the verify + draft graphs captured **lazily once** against those persistent
      buffers and replayed on every ``decode_batch`` call.

    Per request, ``decode_batch`` re-prefills the prompts into the *runtime's*
    GDN state pool and KV pages in place (no realloc), then runs the graphed spec
    loop. Because the bound buffers keep the same addresses, the captured graphs
    stay valid across requests.

    Why prefill cannot write the persistent pool directly: the native packed
    prefill path in ``qwen_model`` *reallocates* ``conv_states`` /
    ``recurrent_states`` to ``[B, ...]`` (it sizes them to the number of packed
    sequences), which would detach the cache from the pool and change addresses.
    So we prefill into a throwaway cache exactly like the engine's
    ``launch_prepared_batch``, then ``capture_batch_from_cache`` the resulting GDN
    state (conv / recurrent / replay ring buffer) into the persistent pool rows --
    an in-place ``index_copy_``, no allocation. KV is written straight into the
    runtime-shared paged KV pool during prefill, so it is already in place.
    """

    def __init__(
        self,
        runtime,
        drafter: DFlashDraftModel,
        dcfg: DFlashConfig,
        *,
        batch_size: int,
        max_seq_len: int,
        flush_cap: int = 32,
        sampling: bool = False,
    ) -> None:
        self.rt = runtime
        self.dcfg = dcfg
        self.lm = runtime.model.model.language_model
        self.lm_head = runtime.model.lm_head
        self.embed = self.lm.embed_tokens
        self.device = next(self.lm.parameters()).device
        # When True, the draft CUDA graph emits per-position LOGITS [B, K, vocab]
        # (so the host can sample q + run rejection sampling) instead of greedy
        # argmax draft tokens. Greedy (False, default) is the byte-identical
        # lossless path; this only changes the captured draft-graph output.
        self._sampling = bool(sampling)

        self.proposer = DFlashProposer(drafter, self.embed, self.lm_head, dcfg)
        self.num_spec = int(self.proposer.num_speculative_tokens)
        self.block_size = int(dcfg.block_size)
        self.target_layer_ids = tuple(dcfg.target_layer_ids)
        self.B = int(batch_size)
        self.max_seq_len = int(max_seq_len)

        # Reject ring capacities with no shipped verify cubin BEFORE reserving
        # pages / capturing the verify graph: an unshipped flush_cap would force
        # the uncapturable torch verify fallback (see ``_validate_flush_cap``).
        _validate_flush_cap(int(flush_cap), self.block_size)
        self.flush_cap = int(flush_cap)
        # The persistent pool must already be sized at flush_cap so the verify
        # kernel and the flush/reset shapes agree.
        pool_cap = int(getattr(runtime._linear_state_pool, "replay_capacity", 0))
        if pool_cap != self.flush_cap:
            raise ValueError(
                f"runtime linear-state pool replay_capacity ({pool_cap}) must "
                f"equal flush_cap ({self.flush_cap}); configure "
                "spec_decode.flush_cap before building the runtime."
            )

        self.gdn_layer_idxs = [
            i for i, layer in enumerate(self.lm.layers)
            if getattr(layer, "linear_attn", None) is not None
        ]

        # --- Persistent state owned once. ---
        pt = runtime.page_table
        self.batch_idx = [pt.allocate() for _ in range(self.B)]
        # The fixed persistent spec rows, as an int set, so ``admit`` can tell a
        # serving-supplied transient prefill ``batch_idx`` (which it must erase
        # before overwriting ``state.batch_idx``) apart from these reserved rows
        # (which it must never erase / double-free).
        self._persistent_batch_idx = {int(bi) for bi in self.batch_idx}
        # Reserve once, for the maximum sequence length, and never free. Pages
        # stay mapped so the page-table rows captured into the graph never move.
        for bi in self.batch_idx:
            pt.reserve(bi, self.max_seq_len)
        pt.commit_block_table(self.batch_idx)
        self.cb = self._ids([self.batch_idx])[0].to(torch.int64)        # [B] pool rows

        # Persistent cache bound to the runtime's persistent GDN pool. The pool
        # rows are addressed by the actual batch_idx (gdn_state_indices=self.cb).
        text_cfg = getattr(runtime.hf_config, "text_config", runtime.hf_config)
        runtime._linear_state_pool.initialize_from_config(text_cfg, dtype=runtime.dtype)
        self.cache = runtime._new_cache()
        runtime._linear_state_pool.bind_to_cache(self.cache)

        # Page-table rows for the fixed batch_idx set (address-stable copy).
        self.page_tbl = torch.index_select(pt.page_table, 0, self.cb).clone()

        self._graphs_ready = False
        # Filled lazily on the first decode_batch (capture once).
        self._verify_graph = None
        self._draft_graph = None

        # Per-row additive token mask folded into the captured draft graph (and
        # added to the verify) for constrained decoding. The base ``SpecRunner``
        # (decode_batch) never masks, so this stays all-zeros (identity) and the
        # captured draft logits are unchanged; ``SpecStepRunner`` populates rows
        # at admit. Defined here (not only on SpecStepRunner) so ``_build_draft``
        # -- shared by both -- can always reference ``self._mask_buf``.
        self._mask_dtype = getattr(runtime, "dtype", torch.float32)
        self._vocab = int(self.lm_head.weight.shape[0]) if hasattr(
            self.lm_head, "weight") else int(dcfg.vocab_size)
        self._mask_buf = torch.zeros(
            self.B, self._vocab, device=self.device, dtype=self._mask_dtype)

    # -- helpers reused from SpecDecoder (identical aux-hidden capture) ------
    _install_hooks = SpecDecoder._install_hooks
    _remove_hooks = staticmethod(SpecDecoder.__dict__["_remove_hooks"].__func__)
    _target_hidden_b = SpecDecoder._target_hidden_b

    def _ids(self, rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    # -- per-request prefill into the persistent pool -----------------------

    def _prefill(self, prompts: list[list[int]], sink: dict) -> tuple:
        """Prefill ``B`` prompts into the persistent KV pages + GDN pool.

        Returns (cur, hid_packed, cu, prompt_lens). Mirrors
        ``generate_batch``'s native packed prefill, but the resulting GDN state
        is *captured into the runtime pool* (in place) instead of left on a
        per-request cache.
        """
        rt, dev = self.rt, self.device
        pt = rt.page_table
        B = self.B
        prompt_lens = [len(p) for p in prompts]
        total = sum(prompt_lens)
        flat_ids = [t for p in prompts for t in p]
        seq_idx_rows = [r for r, n in enumerate(prompt_lens) for _ in range(n)]
        pos_rows = [j for n in prompt_lens for j in range(n)]
        cu = [0]
        for n in prompt_lens:
            cu.append(cu[-1] + n)

        in_ids = self._ids([flat_ids])
        cpos = self._ids([pos_rows])
        sidx = torch.tensor([seq_idx_rows], device=dev, dtype=torch.int32)
        cu_t = torch.tensor(cu, device=dev, dtype=torch.int32)
        bidx_tok = torch.tensor([[self.batch_idx[r] for r in seq_idx_rows]],
                                device=dev, dtype=torch.int64)
        slotmap = pt.build_slot_mapping(batch_idx=bidx_tok, positions=cpos)
        seqk_pf = torch.tensor(prompt_lens, device=dev, dtype=torch.int32)

        # Prefill into a throwaway cache: the native prefill path reallocates the
        # GDN conv/recurrent tensors to [B, ...], which must NOT clobber the bound
        # persistent pool. KV is written into the runtime-shared paged pool (in
        # place at the reserved pages) regardless.
        tmp_cache = rt._new_cache()
        rt.model.model.rope_deltas = None
        out_pf = rt.model.model.language_model(
            input_ids=in_ids, position_ids=cpos, past_key_values=tmp_cache,
            cache_position_ids=cpos, slot_mapping=slotmap,
            page_table=self.page_tbl, paged_kv_seqlens_k=seqk_pf,
            seq_idx=sidx, cu_seq_lens_q=cu_t,
        )
        self.cache.advance_to(max(prompt_lens))
        # Capture the freshly-prefilled GDN state into the persistent pool rows
        # addressed by the actual batch_idx (in-place index_copy_, no realloc).
        rt._linear_state_pool.capture_batch_from_cache(self.cb, tmp_cache, batch_size=B)

        lh_pf = out_pf.last_hidden_state[0]
        last_off = [cu[r + 1] - 1 for r in range(B)]
        cur = [int(self.lm_head(lh_pf[last_off[r]:last_off[r] + 1])[0].argmax())
               for r in range(B)]
        hid_packed = self._target_hidden_b(sink)[0]
        return cur, hid_packed, cu, prompt_lens

    # -- graph capture (once) ------------------------------------------------

    def _capture_graphs(self, prompt_lens: list[int], hid_packed, cu, sink: dict) -> None:
        rt, dev = self.rt, self.device
        B, block_size, K = self.B, self.block_size, self.num_spec
        gdn = self.gdn_layer_idxs
        pt = rt.page_table

        fc_in = hid_packed.shape[-1]
        self.conv_k = self.cache.layers[self.gdn_layer_idxs[0]].conv_states.shape[-1]

        # ---- Verify graph static buffers. ----
        self.ids_buf = torch.zeros(B, block_size, device=dev, dtype=torch.long)
        self.cpos_buf = torch.zeros(B, block_size, device=dev, dtype=torch.long)
        for r in range(B):
            self.cpos_buf[r] = torch.arange(
                prompt_lens[r], prompt_lens[r] + block_size, device=dev)
        # 4-row M-RoPE ``position_ids`` for the verify forward: row 0 is the text
        # position (== ``cpos_buf``), rows 1..3 are the spatial positions (text +
        # per-row rope delta). The runtime's normal decode passes this same 4-row
        # layout (``_prepare_decode_position_ids``); the verify forward previously
        # passed the 2-D ``cpos_buf`` as ``position_ids``, which the text model
        # broadcasts to 4 IDENTICAL rows -- correct only when the rope delta is 0
        # (text), but for an IMAGE row it dropped the spatial M-RoPE shift, so the
        # target verified post-prefill tokens with text-only rotary positions and
        # could diverge from a normal image decode. ``cache_position_ids`` stays
        # the 2-D text positions (it drives the paged-KV slot, not the rotary), so
        # KV writes are unchanged. ``vpos_buf`` is rebuilt from ``_rope_deltas``
        # each step (delta 0 -> all 4 rows == text -> byte-identical to pre-fix).
        self.vpos_buf = self.cpos_buf.unsqueeze(0).expand(4, B, block_size).contiguous()
        self.slot_buf = pt.build_slot_mapping(
            batch_idx=self.cb, positions=self.cpos_buf).clone()
        self.seqk_buf = (self.cpos_buf.max(dim=1).values.to(torch.int32) + 1).clone()
        cu_q_buf = torch.arange(0, (B + 1) * block_size, block_size,
                                device=dev, dtype=torch.int32)
        kw = dict(
            input_ids=self.ids_buf, position_ids=self.vpos_buf,
            past_key_values=self.cache, cache_position_ids=self.cpos_buf,
            slot_mapping=self.slot_buf, page_table=self.page_tbl,
            paged_kv_seqlens_k=self.seqk_buf, gdn_state_indices=self.cb,
            spec_verify=True, cu_seq_lens_q=cu_q_buf,
        )
        _gst = ("conv_states", "recurrent_states", "replay_checkpoint_states",
                "replay_k", "replay_u", "replay_g", "replay_lengths")
        gdn_snap = {
            idx: {n: getattr(self.cache.layers[idx], n).clone()
                  for n in _gst if getattr(self.cache.layers[idx], n, None) is not None}
            for idx in gdn
        }
        # Eager mode (validation / non-determinism-sensitive comparison): skip
        # graph capture; ``step`` runs ``self.lm(**self._verify_kw)`` directly.
        if not getattr(self, "_use_graphs", True):
            self._verify_kw = kw
            self._verify_sink = sink
            for idx in gdn:
                for n, v in gdn_snap[idx].items():
                    getattr(self.cache.layers[idx], n).copy_(v)
            self._build_draft(prompt_lens, hid_packed, cu, fc_in, capture=False)
            self._graphs_ready = True
            return
        warm = torch.cuda.Stream(); warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                self.lm(**kw)
        torch.cuda.current_stream().wait_stream(warm)
        self._verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._verify_graph):
            graph_out = self.lm(**kw)
        self.out_hidden = graph_out.last_hidden_state
        # The aux-hidden hooks fired during capture, so ``sink[i]`` now holds
        # views into the verify graph's buffers. Snapshot those views: every
        # ``verify_graph.replay()`` rewrites the same buffers, so reading these
        # views after a replay yields the fresh block aux-hidden -- regardless of
        # the per-request prefill (which overwrites the *live* sink with its own
        # batch-1 packed tensors on later calls).
        self._verify_sink = dict(sink)
        self._verify_kw = kw   # kwargs for the eager verify reference forward
        for idx in gdn:
            for n, v in gdn_snap[idx].items():
                getattr(self.cache.layers[idx], n).copy_(v)

        self._build_draft(prompt_lens, hid_packed, cu, fc_in, capture=True)
        self._graphs_ready = True

    def _build_draft(self, prompt_lens, hid_packed, cu, fc_in, *, capture: bool) -> None:
        # ``maxc`` is the drafter's fixed left-aligned context capacity. It must
        # cover the longest live context (max prompt + max_new + a block of
        # headroom). It is decoupled from the KV reservation so a caller can make
        # it match a per-call reference's context width for bit-exact comparison.
        B, block_size, dev = self.B, self.block_size, self.device
        # +block_size headroom: the GPU-resident commit scatters a whole block at
        # the row cursor (only accept+1 valid), so the cursor+block can run one
        # block past max_seq_len before the scheduler retires the row.
        self.maxc = self.max_seq_len + block_size
        self.th_buf = torch.zeros(B, self.maxc, fc_in,
                                  dtype=hid_packed.dtype, device=dev)
        for r in range(B):
            self.th_buf[r, :prompt_lens[r]] = hid_packed[cu[r]:cu[r + 1]]
        self.block_ids_buf = torch.full((B, block_size), self.dcfg.mask_token_id,
                                        dtype=torch.long, device=dev)
        self.dpos_buf = torch.zeros(B, self.maxc + block_size, device=dev, dtype=torch.long)
        self.dpos_buf[:, :self.maxc] = torch.arange(self.maxc, device=dev)
        self.dmask_buf = torch.zeros(B, 1, 1, self.maxc + block_size,
                                     dtype=hid_packed.dtype, device=dev)

        def _draft_logits():
            noise = self.embed(self.block_ids_buf)
            h = self.proposer.drafter(noise, self.th_buf, self.dpos_buf,
                                      attn_mask=self.dmask_buf)
            logits = self.lm_head(h[:, 1:, :])          # [B, K, vocab]
            # Constrained decoding: add each row's additive token mask so the
            # drafter proposes only within the allowed set (drafting outside it
            # would be rejected at verify and crater acceptance). ``_mask_buf`` is
            # all-zeros for unmasked rows (identity), so the unmasked path is
            # byte-identical. ``_mask_buf`` is address-stable, so this stays valid
            # inside the captured graph (the row contents are updated in place by
            # admit/retire between replays).
            return logits + self._mask_buf[:, None, :]

        def _draft():
            return _draft_logits().argmax(-1).to(torch.int32)  # masked greedy [B, K]

        # Eager fns the no-graph path calls directly.
        self._draft_fn = _draft
        self._draft_logits_fn = _draft_logits
        if not capture:
            return
        _warm = _draft_logits if self._sampling else _draft
        dwarm = torch.cuda.Stream(); dwarm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(dwarm):
            for _ in range(3):
                _warm()
        torch.cuda.current_stream().wait_stream(dwarm)
        self._draft_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._draft_graph):
            if self._sampling:
                self.draft_logits_out = _draft_logits()   # [B, K, vocab]
            else:
                self.drafts_out = _draft()                 # [B, K]

    def _seed_draft_buffers(self, prompt_lens: list[int], hid_packed, cu) -> None:
        """Reset the draft graph's persistent context buffer for a new request."""
        B = self.B
        self.th_buf.zero_()
        for r in range(B):
            self.th_buf[r, :prompt_lens[r]] = hid_packed[cu[r]:cu[r + 1]]

    def _assert_batched_greedy(self) -> None:
        """The batched decode loops are greedy-only for now.

        ``SpecRunner(sampling=True)`` captures the draft graph as per-position
        LOGITS (so the scheduler can sample ``q`` + run rejection sampling), but
        the batched accept/commit loops below still consume the greedy
        ``self.drafts_out`` argmax tokens. Rejection sampling is implemented and
        validated for the single-sequence path (``SpecDecoder.generate`` with
        ``temperature>0``); wiring it through the ragged batched commit is the
        separate scheduler-side change. Fail loudly here rather than hit a cryptic
        ``self.drafts_out`` ``AttributeError`` once that wiring lands.
        """
        if self._sampling:
            raise NotImplementedError(
                "SpecRunner batched decode does not yet sample (rejection "
                "sampling is wired in the single-sequence SpecDecoder.generate "
                "path). Construct SpecRunner with sampling=False for the greedy "
                "batched loop."
            )

    # -- main entry: one batched request, reusing the graphs ----------------

    @torch.inference_mode()
    def decode_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        *,
        eos_id: int | None = None,
        _force_gold=None,
    ) -> list[SpecDecodeResult]:
        if len(prompts) != self.B:
            raise ValueError(
                f"SpecRunner is fixed at batch_size={self.B}, got {len(prompts)}"
            )
        self._assert_batched_greedy()
        rt, dev = self.rt, self.device
        B, block_size, K = self.B, self.block_size, self.num_spec
        gdn = self.gdn_layer_idxs
        pt = rt.page_table

        # Validate the per-row length BEFORE prefilling: each fixed row reserves
        # exactly ``max_seq_len`` pages once (``__init__``), so a too-long prompt
        # would otherwise have ``_prefill`` build slot mappings and run the model
        # forward into pages past that reservation before this check fired. Reject
        # up front from the prompt lengths alone (``_prefill`` recomputes them the
        # same way), mirroring ``_prefill_row`` for the scheduler path.
        prompt_lens = [len(p) for p in prompts]
        for r in range(B):
            if prompt_lens[r] + max_new_tokens + block_size + 4 > self.max_seq_len:
                raise ValueError(
                    "prompt + max_new_tokens exceeds SpecRunner max_seq_len"
                )

        sink: dict[int, torch.Tensor] = {}
        handles = self._install_hooks(sink)
        try:
            cur, hid_packed, cu, prompt_lens = self._prefill(prompts, sink)

            if not self._graphs_ready:
                self._capture_graphs(prompt_lens, hid_packed, cu, sink)
            else:
                # Reuse the captured graphs; just re-seed the per-request draft
                # context buffer in place (verify buffers are written per step).
                self._seed_draft_buffers(prompt_lens, hid_packed, cu)

            arange_block = torch.arange(block_size, device=dev)
            maxc = self.maxc
            conv_k = self.conv_k

            ctx_len = list(prompt_lens)
            committed = [list(p) for p in prompts]
            done = [False] * B
            accepts: list[int] = []
            steps = 0

            while not all(done):
                steps += 1
                # --- Draft (graph replay). ---
                for r in range(B):
                    self.block_ids_buf[r, 0] = cur[r]
                    c = ctx_len[r]
                    self.dpos_buf[r, maxc:] = torch.arange(c, c + block_size, device=dev)
                    self.dmask_buf[r, 0, 0, :c] = 0
                    self.dmask_buf[r, 0, 0, c:maxc] = float("-inf")
                self._draft_graph.replay()
                drafts_b = self.drafts_out.tolist()
                if _force_gold is not None:
                    for r in range(B):
                        # gold = [cur_0, t1, t2, ...]; cur[r] == gold[committed_gen],
                        # so the drafts are the next K gold tokens after cur.
                        gen_len = len(committed[r]) - prompt_lens[r]
                        g = (_force_gold[r] + [0] * block_size)[gen_len + 1:gen_len + 1 + K]
                        drafts_b[r] = g

                # --- Flush ring buffers that would overflow this block. ---
                flush_rows = []
                for r in range(B):
                    if done[r]:
                        continue
                    pool_row = int(self.cb[r])
                    lc0 = self.cache.layers[self.gdn_layer_idxs[0]]
                    if int(lc0.replay_lengths[pool_row]) + block_size > self.flush_cap:
                        flush_rows.append(pool_row)
                if flush_rows:
                    fr = torch.tensor(flush_rows, device=dev, dtype=torch.long)
                    for idx in gdn:
                        lc = self.cache.layers[idx]
                        lc.materialize_recurrent_from_replay(fr, write_recurrent=False)

                # --- Snapshot conv windows (verify over-advances; commit rolls). ---
                conv_snap = {idx: self.cache.layers[idx].conv_states.clone()
                             for idx in gdn}

                # --- Verify (graph replay). ---
                for r in range(B):
                    self.ids_buf[r, 0] = cur[r]
                    for j in range(K):
                        self.ids_buf[r, j + 1] = drafts_b[r][j]
                    self.cpos_buf[r] = arange_block + ctx_len[r]
                # Rebuild the 4-row M-RoPE verify ``position_ids`` from the now-current
                # text positions BEFORE replaying, exactly as ``SpecStepRunner.step``
                # does. ``_capture_graphs`` bound ``self.vpos_buf`` (a 4-row buffer,
                # the f344f530 image-M-RoPE fix) into the captured verify graph; it is
                # seeded from the PROMPT positions and is never touched by the per-step
                # ``cpos_buf`` update below. Without this refresh the second+ macro-step
                # would replay the verify graph with the prompt's stale rotary positions
                # (``cpos_buf`` / slot mapping / KV seqlens all advanced to the new
                # ``ctx_len``, but the rotary stayed put) and accept/return WRONG tokens.
                # Row 0 is the text position; rows 1..3 add the per-row spatial rope
                # delta (0 for a text/non-image row -> all 4 rows == text positions,
                # byte-identical to the pre-fix 2-D ``cpos_buf``).
                self.vpos_buf[:] = self.cpos_buf.unsqueeze(0)
                rope_deltas = getattr(self, "_rope_deltas", None)
                if rope_deltas is not None:
                    self.vpos_buf[1:].add_(rope_deltas)
                self.slot_buf.copy_(
                    pt.build_slot_mapping(batch_idx=self.cb, positions=self.cpos_buf))
                self.seqk_buf.copy_(self.cpos_buf.max(dim=1).values.to(torch.int32) + 1)
                self._verify_graph.replay()
                pred = self.lm_head(self.out_hidden).argmax(-1).tolist()  # [B, K+1]
                block_hid = self._target_hidden_b(self._verify_sink)      # [B, K+1, fc]

                # --- Per-sequence accept + ragged commit. ---
                for r in range(B):
                    if done[r]:
                        continue
                    pool_row = int(self.cb[r])
                    drafts = drafts_b[r]
                    accept = 0
                    for j in range(K):
                        if pred[r][j] == drafts[j]:
                            accept += 1
                        else:
                            break
                    bonus = pred[r][accept]
                    accepts.append(accept)

                    new_toks = [cur[r]] + drafts[:accept]
                    c = ctx_len[r]
                    for idx in gdn:
                        lc = self.cache.layers[idx]
                        lc.replay_lengths[pool_row] += (accept + 1)
                        rolled = torch.cat(
                            [conv_snap[idx][pool_row],
                             lc.spec_block_preconv[r, :, : accept + 1]],
                            dim=-1,
                        )
                        lc.conv_states[pool_row] = rolled[:, -conv_k:]
                    self.th_buf[r, c:c + accept + 1] = block_hid[r, : accept + 1]
                    committed[r] = committed[r] + new_toks
                    ctx_len[r] = c + accept + 1
                    cur[r] = bonus

                    gen = committed[r][prompt_lens[r]:]
                    if eos_id is not None and eos_id in gen:
                        done[r] = True
                    elif len(gen) >= max_new_tokens:
                        done[r] = True

            results: list[SpecDecodeResult] = []
            n_acc = len(accepts)
            mean_accept = (sum(accepts) / n_acc) if n_acc else 0.0
            for r in range(B):
                gen = committed[r][prompt_lens[r]:]
                if eos_id is not None and eos_id in gen:
                    gen = gen[: gen.index(eos_id) + 1]
                gen = gen[:max_new_tokens]
                results.append(SpecDecodeResult(
                    token_ids=gen, mean_accept=mean_accept, steps=steps))
            return results
        finally:
            self._remove_hooks(handles)

    def release(self) -> None:
        """Free the fixed page-table slots (invalidates the captured graphs).

        ``_release_batch_idx`` skips any row in ``self._persistent_batch_idx`` so
        the NORMAL ``release_sequence`` cleanup of a finished spec-admitted
        sequence never frees the runner's address-stable rows out from under the
        live graphs (the ``e2985298`` guard). That same guard would, however,
        also block THIS deliberate teardown -- ``release()`` is the authoritative
        reclamation of the runner-owned rows, and a guarded ``_release_batch_idx``
        would return without erasing, leaking the fixed page-table + KV/GDN
        reservations forever (tearing down or recreating a runner on the same
        runtime would permanently consume batch slots + pages). So drop each row
        from ``_persistent_batch_idx`` BEFORE releasing it: the guard no longer
        recognises it as protected, ``_release_batch_idx`` runs its full
        erase/clear, and the row + its pages return to the pool. Rows not yet
        torn down stay protected, so a concurrent ``release_sequence`` on them is
        still a no-op.
        """
        for bi in self.batch_idx:
            # No-op if absent (idempotent re-release); discard so a guarded
            # ``_release_batch_idx`` cannot re-skip this row.
            self._persistent_batch_idx.discard(int(bi))
            self.rt._release_batch_idx(bi)


# ---------------------------------------------------------------------------
# Scheduler-driven spec step
# ---------------------------------------------------------------------------


@dataclass
class _SpecRow:
    """Live per-pool-row state for a sequence the scheduler is decoding."""

    state: object               # the runtime SequenceState occupying this row
    cur: int                    # next token to verify at block position 0 (bonus)
    ctx_len: int                # live KV / GDN context length for this row


class SpecStepRunner(SpecRunner):
    """``SpecRunner`` refactored into a *scheduler-callable* macro-step.

    ``SpecRunner.decode_batch`` owns a full generate loop with a frozen-finish
    fixed batch. The serving scheduler instead wants to drive ONE macro-step at
    a time, admit/retire sequences continuously, and advance each sequence by a
    variable amount. ``SpecStepRunner`` exposes exactly that by lifting
    ``decode_batch``'s inner loop body into :meth:`step` and adding per-row
    occupancy bookkeeping (:meth:`admit` / :meth:`retire`).

    It reuses every piece of ``SpecRunner``: the fixed pool of ``B``
    page-table rows, the persistent GDN/KV pool, and the verify + draft CUDA
    graphs captured once and replayed every step. A pool row is either free or
    holds one sequence; an idle row still runs through both fixed-``B`` graphs
    (harmless), but its commit is suppressed and its output discarded — the same
    frozen-row policy ``decode_batch`` already validates, now applied to *idle*
    rows so the graph batch dimension stays fixed for continuous batching.

    The scheduler maps each active ``SequenceState`` to a pool row at
    :meth:`admit` (which prefills the prompt into that row and returns the first
    sampled token, exactly like prefill), then passes the active states to
    :meth:`step` each macro-step. ``step`` returns, per supplied state, the list
    of newly committed tokens (``a_i + 1`` of them).
    """

    def __init__(self, *args, use_graphs: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pool-row free list + occupancy. Row r corresponds to self.batch_idx[r]
        # / self.cb[r]; rows are address-stable for the life of the runner.
        self._free_rows: list[int] = list(range(self.B))
        self._rows: dict[int, _SpecRow] = {}   # pool_row -> live state
        # id(state) -> pool_row, so the scheduler can pass plain states.
        self._row_of: dict[int, int] = {}
        # When False, verify/draft run eagerly (no CUDA-graph capture/replay).
        # The graph path is the production fast path; the eager path is bit-exact
        # to the model's eager forward (useful where the captured graph's kernel
        # selection is non-deterministic vs an eager reference -- validation).
        self._use_graphs = bool(use_graphs)
        # GPU-resident per-row decode state (indexed by graph row r == pool slot).
        # ``step`` keeps accept/commit/current-token/context entirely on-device so
        # the host issues a macro-step without a per-step pred/draft D2H; idle rows
        # carry stale values (their commit is masked off, output discarded).
        self._cur_buf = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._ctx_buf = torch.zeros(self.B, dtype=torch.long, device=self.device)
        # Per-row M-RoPE spatial delta (mirrors the runtime's
        # ``_decode_rope_deltas``). An IMAGE prefill shifts the post-image text
        # positions in the 3 spatial M-RoPE rows by this delta; ``step`` must add
        # it to the verify/decode ``position_ids`` spatial rows so the target
        # verify uses the SAME rotary positions as a normal image decode (a
        # text-only / non-image row keeps delta 0 -> the 4 M-RoPE rows collapse to
        # the text positions, byte-identical to the pre-fix 2-D position_ids). It
        # is set per row in ``_prefill_row`` from the image-prefill forward cache
        # and reset to 0 in ``_release_row`` (idle/text rows contribute no shift).
        self._rope_deltas = torch.zeros(
            self.B, 1, dtype=torch.long, device=self.device)
        # Fully-async step(): no per-step D2H. The flush decision is driven by a
        # per-row host step counter (flush every flush_cap//block_size active
        # steps; conservative since replay_len <= steps*block_size) instead of
        # reading int(replay_lengths) each step. The phase must be per row, not
        # global: rows are reused across requests, and inheriting another
        # request's flush phase materializes ReplaySSM at a different relative
        # token offset, which can flip bf16 argmax ties in image continuations.
        self._flush_every = max(1, self.flush_cap // self.block_size)
        self._row_steps = [0] * self.B
        self._cb_host = [int(bi) for bi in self.batch_idx]
        # Deferred per-step output (fully-async step, no pred.tolist()): D2H
        # pred[B,K+1]+accept[B] to a DOUBLE-BUFFERED pinned host buffer on a
        # private copy stream gated by a compute-stream ready event. The lazy
        # SpecStepResult resolves off the pinned copy at commit -- one tick later
        # under the async scheduler, so the wait overlaps the GPU. Ping-pong (2
        # slots) so step N's D2H can't clobber step N-1's not-yet-committed read.
        _pin = self.device.type == "cuda"
        self._out_copy_stream = (
            torch.cuda.Stream(device=self.device) if _pin else None)
        self._out_pred_pinned = [
            torch.empty(self.B, self.block_size, dtype=torch.long,
                        device="cpu", pin_memory=_pin)
            for _ in range(2)]
        self._out_acc_pinned = [
            torch.empty(self.B, dtype=torch.long, device="cpu", pin_memory=_pin)
            for _ in range(2)]
        self._out_copy_events = [
            torch.cuda.Event() if _pin else None for _ in range(2)]
        # Per-step logprob D2H rides the SAME ping-pong / copy stream: when any
        # active row this step wants logprobs we async-copy the per-position
        # committed-token logprob block [B, K+1] alongside pred/accept, and the
        # lazy result gathers per-row from it. None-filled when unused.
        self._out_lp_pinned = [
            torch.empty(self.B, self.block_size, dtype=torch.float32,
                        device="cpu", pin_memory=_pin)
            for _ in range(2)]
        self._out_slot = 0

        # --- Per-row token masks (skill constrained decoding) -----------------
        # The additive logit mask ``_mask_buf`` [B, vocab] (0 = allowed, -inf =
        # blocked) lives on the base SpecRunner (address-stable, folded into the
        # captured draft graph + added to verify). ``_row_masked`` tracks which
        # rows actually carry a mask, so an all-unmasked step skips the mask add
        # entirely (keeps the validated unmasked path byte-identical). The mask
        # dtype matches the lm_head output so an unmasked add (``logits + 0``) is
        # exact and never promotes/perturbs the bf16 argmax.
        self._row_masked = [False] * self.B

        # --- Per-row sampling knobs (temperature / top_p) ---------------------
        # Greedy (temperature 0) keeps the validated exact-argmax accept/commit;
        # a row with temperature > 0 runs modified rejection sampling in step().
        # Stored host-side (small, read only to branch greedy-vs-sample and to
        # build the spatial/logprob sampling params).
        self._row_temp = [0.0] * self.B
        self._row_top_p = [1.0] * self.B
        self._row_logprobs = [False] * self.B   # row wants per-token logprobs
        # Spec RNG for the sampling path (draft sample + accept/residual draws).
        self._spec_gen = None

    def _detect_typed_token_runtime(self) -> bool:
        """True when the runtime types some committed ids (coord/size) from the
        verify final-hidden, so :meth:`step` should pack :class:`SpecSideValues`.

        Two signals (either suffices): the runtime exposes a non-default
        ``sampling_hooks.post_sample`` (the engine scheduler↔runtime spatial
        hook), or it carries populated ``spatial_tables`` (the Qwen35 runtime's
        spatial-decode capability). A text-only runtime has neither, so the spec
        step carries no side-values and committed ids materialize as
        ``TextToken``. Re-evaluated each step (cheap attribute checks): the
        runtime can wire its spatial capability AFTER the runner is constructed
        (``_maybe_init_spec_decode`` runs before ``spatial_tables`` /
        ``sampling_hooks`` are populated), so latching it at __init__ would
        permanently disable typed tokens.
        """
        hooks = getattr(self.rt, "sampling_hooks", None)
        if hooks is not None and getattr(hooks, "post_sample", None) is not None:
            return True
        return getattr(self.rt, "spatial_tables", None) is not None

    # -- capacity / contract -------------------------------------------------

    @property
    def num_speculative_tokens(self) -> int:
        """K draft tokens per macro-step (SpecDecoder contract attribute)."""
        return self.num_spec

    @property
    def free_slots(self) -> int:
        return len(self._free_rows)

    def has_row(self, state: object) -> bool:
        return id(state) in self._row_of

    # -- admit: prefill one sequence into a free pool row --------------------

    @torch.inference_mode()
    def admit(
        self,
        state: object,
        prompt_tokens,
        *,
        image=None,
        image_crops=None,
        allowed_token_ids=None,
        suppressed_token_ids=None,
        suppress_next_token_ids=None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> "tuple[int, float | None]":
        """Prefill ``prompt_tokens`` into a free row; return its first token.

        ``prompt_tokens`` is the request's *typed* prefill sequence -- the same
        ``Sequence[Token]`` the non-spec ``Qwen35Runtime.prepare_sequence``
        receives (``list(request.prefill_tokens)``), not a text-only
        ``int(t.token_id)`` projection. A launchable prefill can carry
        ``ImageMarker`` tokens (multi-image chat prompts) and, for a resumed
        request, typed tokens in the generated prefix; none of those expose
        ``token_id``, so projecting to ints up front would raise
        ``AttributeError`` and abort admission. This method expands
        ``ImageMarker``s with the image KV prefix and reads the typed ids exactly
        like the non-spec prepare/prefill path. Bare ``int`` ids are still
        accepted (the standalone single-sequence tests pass token-id lists).

        Assigns a free pool row to ``state`` and writes the row's page-table
        ``batch_idx`` onto ``state.batch_idx`` (so the scheduler's KV/finish
        bookkeeping addresses the same pages the spec loop commits into).
        Mirrors prefill: returns ``(first_token_id, first_logprob)`` for the
        greedy/bonus first token. The sequence is then advanced by subsequent
        :meth:`step` calls until :meth:`retire`.

        ``first_logprob`` is the sampler's selected-token logprob for that first
        token (``None`` when the request did not request logprobs). It matches
        the non-spec prefill sampler exactly: ``0.0`` under greedy (the bonus is
        the argmax), and ``log(softmax(last_logits / temperature)[first])`` under
        sampling (the full-softmax selected logprob the single-token path
        transfers for token0; *not* renormalized over the top-p nucleus).

        ``image`` prefills the prompt *with the image KV prefix* (vision block +
        encoder), so the spec path covers Moondream's image workload with no
        fallback. ``image_crops`` is the request's multi-crop tiles -- for
        Qwen3.5 the already-preprocessed :class:`QwenImageInputs` (the full
        multi-tile ``pixel_values`` / ``image_grid_thw``) the non-spec
        ``prepare_sequence`` forwards alongside ``image`` so the vision encoder
        reads the high-resolution crop tiles (its ``overlap``) rather than only
        the global/thumbnail image. When given it is used directly as the image
        inputs (no re-preprocessing); a multi-crop request that passed only
        ``image`` would build thumbnail-only image KV and diverge from the
        non-spec path. ``None`` (single-crop / text-only) preprocesses ``image``
        alone, exactly as before. ``allowed_token_ids`` / ``suppressed_token_ids``
        install the row's *initial* skill mask (applied to drafter + verify each
        step); :meth:`step` re-supplies the mask per macro-step and that per-step
        value overrides this stored one for stateful skills whose allowed set
        evolves.
        ``suppress_next_token_ids`` is the request-level *one-shot* blacklist the
        non-spec path applies to a request's first generated token only; since
        ``admit`` samples that token, it is applied to this first sample alone
        (never recorded on the row, so :meth:`step` never re-applies it).
        ``temperature`` / ``top_p`` select greedy vs sampling and feed the
        logprob / spatial decode.

        When the runtime types coord/size ids (a spatial runtime), the first
        token's ``last_hidden_state`` (the spatial decode's ``hidden_last``) is
        surfaced on ``state.admit_side_values`` as a single-position
        :class:`SpecSideValues`, so the scheduler types the admit id through the
        runtime ``materialize_tokens`` hook (the spec analog of the non-spec
        prefill ``post_sample`` path). A text-only runtime leaves it unset.
        """
        if not self._free_rows:
            raise RuntimeError("SpecStepRunner has no free rows; admit over capacity")
        # Keep the typed prompt sequence intact: image-marker expansion + typed
        # id extraction happen in _prepare_image_prefill (the spec analog of the
        # non-spec prepare_sequence), so ImageMarker/typed tokens survive to the
        # prefill instead of being stripped by an int(t.token_id) projection.
        prompt = list(prompt_tokens)
        row = self._free_rows.pop(0)
        # The row is popped from _free_rows but not yet recorded in _rows/_row_of,
        # so retire() (which looks up _row_of[id(state)]) cannot return it. If any
        # step below raises (e.g. _prefill_row rejecting a prompt/image that
        # exceeds max_seq_len, an image-encode error, or a model OOM), restore the
        # row to the free pool and clear the per-row mask/sampling state it may
        # have installed -- otherwise a few bad requests permanently exhaust the
        # runner's capacity, and a stale -inf mask would leak onto the next admit
        # into this (address-stable) row.
        try:
            # Install the row's constrained-decode mask + sampling knobs BEFORE
            # the first token is sampled below, so the prefill bonus token is
            # itself produced under the mask (matching the non-spec prefill
            # sampler, which applies the skill mask to the first sampled token).
            self._set_row_mask(row, allowed_token_ids, suppressed_token_ids)
            self._row_temp[row] = float(temperature)
            self._row_top_p[row] = float(top_p)
            want_logprobs = bool(getattr(state, "return_logprobs", False)) or \
                bool(getattr(getattr(state, "request", None), "return_logprobs", False))
            self._row_logprobs[row] = want_logprobs
            ctx_len, last_logits, last_hidden = self._prefill_row(
                row, prompt, image=image, image_crops=image_crops)
            # Select the first/bonus token + its logprob from the final-position
            # logits, applying the row's skill mask and the request's one-shot
            # suppression (the non-spec path does both on a request's first token).
            cur, first_logprob = self._select_first_token(
                last_logits,
                row,
                suppress_next_token_ids=suppress_next_token_ids,
                temperature=float(temperature),
                top_p=float(top_p),
                want_logprob=want_logprobs,
            )
            # Surface the first-token side-values for a typed-token runtime so the
            # scheduler types the admit id (coord/size) via materialize_tokens. A
            # text-only runtime gets nothing (the id stays a plain TextToken).
            if hasattr(state, "admit_side_values"):
                state.admit_side_values = None
            if self._detect_typed_token_runtime():
                state.admit_side_values = self._pack_admit_side_values(
                    last_hidden, row)
        except BaseException:
            self._release_row(row)
            raise
        self._rows[row] = _SpecRow(state=state, cur=cur, ctx_len=ctx_len)
        self._row_of[id(state)] = row
        self._cur_buf[row] = cur
        self._ctx_buf[row] = ctx_len
        if hasattr(self, "_row_steps"):
            self._row_steps[row] = 0
        if hasattr(state, "batch_idx"):
            # Serving builds ``state`` via ``Qwen35Runtime.prepare_sequence``, which
            # ``page_table.allocate()``s a transient prefill ``batch_idx`` (with its
            # own reserved pages) and stores it here. Re-pointing ``state.batch_idx``
            # at the persistent spec row below would drop that only reference, so the
            # transient row/pages would never be returned to the pool -- and because
            # ``SpecRunner`` already reserves ``max_batch_size`` rows up front, every
            # spec admission would leak one slot until the KV pool is exhausted. Erase
            # the prior transient row first (return its pages + batch_idx to the pool).
            # Guard against double-free: skip if it is already free or is one of our
            # persistent spec rows (it never should be, but a defensive identity check
            # keeps a mis-wired caller from corrupting the fixed reservation).
            # A scheduler-created ``state`` that never owned a transient prefill row
            # carries the documented ``-1`` sentinel ``batch_idx`` ("no prior row to
            # erase"). ``-1`` is neither in ``_persistent_batch_idx`` nor in
            # ``free_batch_idx``, so without the non-negative guard the condition
            # would misread it as a live transient row and call ``erase(-1, 0)`` --
            # which either raises mid-admission or, under a page table that accepts
            # negative indexing, frees the padding/last row and corrupts the fixed
            # reservation before the real spec row is assigned. Only a row that was
            # actually allocated (``>= 0``) can need erasing.
            prev_batch_idx = getattr(state, "batch_idx", None)
            if (
                prev_batch_idx is not None
                and int(prev_batch_idx) >= 0
                and int(prev_batch_idx) not in self._persistent_batch_idx
                and int(prev_batch_idx) not in self.rt.page_table.free_batch_idx
            ):
                self.rt.page_table.erase(int(prev_batch_idx), 0)
            state.batch_idx = int(self.batch_idx[row])
        return cur, first_logprob

    def _select_first_token(
        self,
        last_logits: "torch.Tensor",
        row: int,
        *,
        suppress_next_token_ids=None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        want_logprob: bool = False,
    ) -> "tuple[int, float | None]":
        """Select the admit (first/bonus) token id + its selected-token logprob.

        ``last_logits`` is the ``[1, vocab]`` final-prefill logits. Applies the
        row's skill mask (``_apply_row_mask_logits``) and the request's one-shot
        ``suppress_next_token_ids`` blacklist, then:

        * greedy (``temperature <= eps``): the masked argmax, matching the
          validated greedy prefill byte-for-byte (the unmasked/unsuppressed bf16
          argmax is unchanged); its logprob is ``0.0`` (the non-spec greedy
          convention: the selected id *is* the argmax), or ``None`` when the
          request did not ask for logprobs.
        * sampling (``temperature > eps``): draw from the request's
          temperature/top-p distribution (``logits_to_probs`` -- the same
          distribution the non-spec single-token sampler draws from); the logprob
          is the FULL-softmax selected-token logprob ``log(softmax(masked_logits
          / temperature)[id])`` (matching ``sample_step_from_logits`` /
          ``_selected_logprobs_from_logits``, *not* the top-p-renormalized mass).
        """
        masked = self._apply_row_mask_logits(last_logits, row)   # [1, vocab]
        # One-shot suppression: blacklist these ids for this first sample only
        # (not stored on the row, so step never re-applies it). Use a copy so the
        # row mask buffer is untouched.
        if suppress_next_token_ids:
            masked = masked.clone()
            sup = torch.as_tensor(
                list(suppress_next_token_ids), device=masked.device, dtype=torch.long)
            masked[0, sup] = float("-inf")
            if not torch.isfinite(masked).any():
                # The one-shot blacklist removed every currently-allowed first
                # token (e.g. row whitelist allowed=[x] and suppress=[x]), leaving
                # the whole row at -inf with no valid token. The greedy branch
                # below would then return argmax == 0 and the sampling branch
                # would draw from NaNs (``logits_to_probs`` over an all--inf row),
                # committing a constraint-violating token. Reject instead, exactly
                # as ``_set_row_mask`` does for the all-suppressed persistent mask.
                raise ValueError(
                    "suppress_next_token_ids removes every allowed first token "
                    "(the row mask and one-shot blacklist leave no valid next "
                    "token); the spec admit cannot select a valid token."
                )
        greedy = float(temperature) <= GREEDY_TEMPERATURE_EPS
        if greedy:
            cur = int(masked[0].argmax())
            # Non-spec greedy convention: the selected id is the argmax, so its
            # selected-token logprob is 0.0 (matches the single-token sampler's
            # all-greedy logprobs path).
            logprob = 0.0 if want_logprob else None
            return cur, logprob
        # Sampling: draw the first token from the request's distribution and take
        # the full-softmax selected-token logprob (the value the non-spec prefill
        # transfers for token0). top-p truncates the SAMPLE only; the logprob is
        # over the untruncated softmax so it equals the single-token path.
        probs = logits_to_probs(masked, temperature, top_p)          # [1, vocab]
        cur_t = _gumbel_argmax(probs, self._spec_gen)                # [1]
        cur = int(cur_t[0])
        logprob = None
        if want_logprob:
            lse = torch.logsumexp(masked.float() / float(temperature), dim=-1)  # [1]
            sel = masked[0, cur].float() / float(temperature)
            logprob = float(sel - lse[0])
        return cur, logprob

    def _pack_admit_side_values(self, last_hidden: "torch.Tensor", row: int):
        """Build a single-position :class:`SpecSideValues` for the admit token.

        ``last_hidden`` is the ``[1, hidden]`` target ``last_hidden_state`` at the
        final prefill position -- the ``hidden_last`` the runtime's spatial decode
        reads to type the first id (coord/size). Shaped ``[1, 1, hidden]`` (one
        sequence, one committed position) with ``counts=[1]``, matching the
        single-token committed run (``accept_count == 0`` -> one committed
        position) the scheduler types via ``materialize_tokens``. Mirrors
        :meth:`_pack_side_values`, which packs the verify final-hidden for a
        macro-step's committed run.
        """
        from kestrel.runtime.spec import SpecSideValues

        dev = self.device
        hid = last_hidden.reshape(1, 1, -1).contiguous()          # [1, 1, hidden]
        temps = torch.tensor([self._row_temp[row]], device=dev, dtype=torch.float32)
        top_ps = torch.tensor([self._row_top_p[row]], device=dev, dtype=torch.float32)
        return SpecSideValues(
            hidden=hid, temperatures=temps, top_ps=top_ps, counts=[1])

    def _set_row_mask(self, row: int, allowed_token_ids, suppressed_token_ids) -> None:
        """Build pool ``row``'s additive logit mask from the skill constraints.

        ``allowed_token_ids`` (whitelist) blocks everything *not* listed;
        ``suppressed_token_ids`` (blacklist) blocks the listed ids. The result is
        a ``[vocab]`` additive mask (0 / -inf) written into the address-stable
        ``_mask_buf`` row, so the captured draft graph (which reads ``_mask_buf``)
        and the verify add both pick it up. Mirrors the non-spec sampler's mask
        application (``scheduler`` whitelist-then-blacklist) exactly.

        ``allowed_token_ids`` is a *whitelist*, so its emptiness is meaningful and
        must be distinguished by identity, not truthiness: ``None`` means "no
        whitelist constraint" (leave every id allowed), while an empty but
        non-``None`` sequence means "no id is allowed". The non-spec path rejects
        the latter at validation (``engine/core.py`` raises "removed every allowed
        next token") rather than silently masking the whole vocab to ``-inf`` and
        then drawing token 0 / failing ``multinomial``; match that here by raising,
        so the spec path never drafts/commits an arbitrary token for a request
        whose constraint admits none.
        """
        dev = self.device
        m = self._mask_buf[row]
        m.zero_()
        masked = False
        if allowed_token_ids is not None:
            allowed_list = list(allowed_token_ids)
            if not allowed_list:
                raise ValueError(
                    "allowed_token_ids is an empty whitelist (no token is "
                    "allowed); the spec mask cannot leave a valid next token. "
                    "Pass None for an unconstrained row."
                )
            idx = torch.as_tensor(allowed_list, device=dev, dtype=torch.long)
            keep = torch.zeros(self._vocab, dtype=torch.bool, device=dev)
            keep[idx] = True
            m.masked_fill_(~keep, float("-inf"))
            masked = True
        if suppressed_token_ids:
            idx = torch.as_tensor(list(suppressed_token_ids), device=dev, dtype=torch.long)
            m.index_fill_(0, idx, float("-inf"))
            masked = True
        if masked and not torch.isfinite(m).any():
            # The whitelist was non-empty but the blacklist removed every allowed
            # id (e.g. allowed=[42], suppressed=[42]), leaving the whole row at
            # -inf with no valid next token. Like the empty-whitelist case above,
            # the non-spec path rejects this at validation rather than masking the
            # vocab to -inf and drawing argmax 0 / an all-NaN sampling row; match
            # that by raising so the spec path never commits a disallowed token.
            raise ValueError(
                "token mask suppresses every allowed token (the whitelist and "
                "blacklist leave no valid next token); the spec mask cannot leave "
                "a valid next token for this row."
            )
        self._row_masked[row] = masked

    def _refresh_step_masks(self, rows, allowed_token_ids, suppressed_token_ids) -> None:
        """Rebuild the active rows' masks from this step's per-row constraints.

        ``rows`` are the active graph rows (parallel to the ``states`` passed to
        :meth:`step`); ``allowed_token_ids`` / ``suppressed_token_ids`` are the
        per-row constraint lists the scheduler recomputed THIS step (entry ``i``
        for ``states[i]``, ``None`` for an unconstrained row), or ``None`` for the
        whole list when that side was not supplied. Each active row's
        ``_mask_buf`` row is rebuilt in place via :meth:`_set_row_mask` (which
        zeros the row first, so a row whose constraint cleared this step goes back
        to unmasked) -- this REPLACES the admit-time mask for the step. Only the
        active rows are touched; idle rows keep their stored mask. Called before
        the draft graph reads ``_mask_buf`` so the constraint applies to the
        drafter (and the verify add) the same step.
        """
        allowed_seq = allowed_token_ids if allowed_token_ids is not None else None
        suppressed_seq = (
            suppressed_token_ids if suppressed_token_ids is not None else None)
        for i, row in enumerate(rows):
            row_allowed = (
                allowed_seq[i] if allowed_seq is not None and i < len(allowed_seq)
                else None)
            row_suppressed = (
                suppressed_seq[i]
                if suppressed_seq is not None and i < len(suppressed_seq)
                else None)
            self._set_row_mask(row, row_allowed, row_suppressed)

    @staticmethod
    def _prompt_token_id(tok) -> int:
        """Extract the vocabulary id from a typed prompt token (or a bare int).

        Accepts the typed ``Sequence[Token]`` the scheduler forwards
        (``list(request.prefill_tokens)``) as well as a bare ``int`` id list
        (the standalone single-sequence tests). ``TextToken`` carries
        ``token_id``; ``ImageMarker`` is handled separately (expanded to a vision
        block before materialization) and never reaches here. Other typed kinds
        (``CoordToken`` / ``SizeToken``) have no vocabulary id and are not
        supported in a Qwen3.5 prefill -- mirror the non-spec
        ``_build_packed_prefill_batch`` and reject them with a clear error rather
        than crashing on a missing attribute.
        """
        if isinstance(tok, int):
            return int(tok)
        token_id = getattr(tok, "token_id", None)
        if token_id is None:
            raise ValueError(
                "Qwen3.5 spec prefill only supports text/image prompt tokens; "
                f"got {type(tok).__name__} with no token_id"
            )
        return int(token_id)

    def _prepare_image_prefill(self, prompt, image, *, image_crops=None):
        """Resolve the typed prompt + optional image into prefill ids and kwargs.

        ``prompt`` is the typed ``Sequence[Token]`` (or a bare int-id list); the
        spec analog of the non-spec ``prepare_sequence`` image handling. Mirrors
        that path: an ``ImageMarker`` in the prompt (chat) is replaced in place by
        the image's vision block, otherwise (query) a single vision block is
        spliced at the runtime's image-insert offset.

        ``image_crops`` is the request's preprocessed multi-tile image inputs
        (for Qwen3.5 a :class:`QwenImageInputs` carrying the full multi-tile
        ``pixel_values`` / ``image_grid_thw``). When present it is used DIRECTLY
        as the image inputs (no re-preprocessing) -- the spec analog of the
        non-spec ``prepare_sequence``, which reads ``image_crops`` as the vision
        encoder's ``overlap`` so the high-res crop tiles are encoded rather than
        only the global/thumbnail image. ``None`` (single-crop / text-only)
        preprocesses ``image`` alone, exactly as before.

        Returns ``(image_inputs, ids, image_kwargs)``:
          * ``image_inputs`` -- the runtime ``QwenImageInputs`` (or ``None`` for a
            text-only prompt), so the caller can branch the forward.
          * ``ids`` -- the prompt token ids *with* the vision block
            (``<|vision_start|>`` + ``<|image_pad|>``xN + ``<|vision_end|>``).
            This is the sequence that occupies the KV/GDN context.
          * ``image_kwargs`` -- ``pixel_values`` / ``image_grid_thw`` /
            ``mm_token_type_ids`` for ``_forward_base`` (empty for text-only).
        """
        from kestrel.runtime.tokens import ImageMarker
        from ..runtime import (
            QwenImageInputs,
            IMAGE_PAD_ID,
            VISION_END_ID,
            VISION_START_ID,
        )
        from ..prompt_template import (
            IM_START_ID,
            _NEWLINE_ID,
            _USER_ID,
        )

        tokens = list(prompt)
        markers = [i for i, t in enumerate(tokens) if isinstance(t, ImageMarker)]

        # An image-bearing request may carry the pixels via ``image`` (raw or
        # preprocessed) OR via ``image_crops`` (the preprocessed multi-tile
        # inputs the scheduler forwards). Treat the request as imageful if either
        # is set; only a request with neither (and no marker) is text-only.
        has_image = image is not None or image_crops is not None

        # Text-only fast path: no image and no markers -> just materialize ids.
        # (Keeps the validated greedy/text path byte-identical: for an int-id or
        # all-``TextToken`` prompt this is exactly ``[int(id) ...]``.)
        if not has_image and not markers:
            return None, [self._prompt_token_id(t) for t in tokens], {}
        if not has_image and markers:
            raise ValueError(
                "Qwen3.5 spec prefill: prompt has image marker(s) but no image "
                "was provided to admit()"
            )

        rt = self.rt
        # Prefer the preprocessed multi-crop tiles when supplied (the non-spec
        # path's ``overlap`` -- all tiles of a large image, not the thumbnail);
        # else accept an already-preprocessed QwenImageInputs ``image`` or run the
        # runtime preprocessor on a raw ``image``. Mirrors ``prepare_sequence``'s
        # ``if image_crops is None: image_crops = process(image)``.
        if image_crops is not None:
            image_inputs = (
                image_crops
                if isinstance(image_crops, QwenImageInputs)
                else rt._image_preprocessor.process(image_crops)
            )
        else:
            image_inputs = (
                image
                if isinstance(image, QwenImageInputs)
                else rt._image_preprocessor.process(image)
            )
        n_img = int(image_inputs.num_image_tokens)
        block = (
            [VISION_START_ID]
            + [IMAGE_PAD_ID] * n_img
            + [VISION_END_ID]
        )

        if markers:
            # Chat path: one ImageMarker per image at its content position. The
            # spec runner takes a single ``request.image``, so support exactly one
            # marker; replace it in place with the image's vision block (matching
            # ``prepare_sequence``'s marker expansion).
            if len(markers) != 1:
                raise ValueError(
                    "Qwen3.5 spec prefill supports a single image; prompt has "
                    f"{len(markers)} image markers"
                )
            ids = [
                self._prompt_token_id(t)
                for t in tokens
                if not isinstance(t, ImageMarker)
            ]
            offset = markers[0]
            ids = ids[:offset] + block + ids[offset:]
        else:
            # Query path: no marker; splice a single vision block after the first
            # user-turn opener on the typed prompt.
            query_template = rt.prompt_template.query()
            offset = derive_image_insertion_offset(
                tokens,
                user_turn_opener=(IM_START_ID, _USER_ID, _NEWLINE_ID),
                fallback_offset=1 + (
                    len(query_template.prefix) if query_template else 0
                ),
            )
            ids = [self._prompt_token_id(t) for t in tokens]
            ids = ids[:offset] + block + ids[offset:]

        input_ids = self._ids([ids])
        image_kwargs = rt._image_forward_kwargs(input_ids, image_inputs)
        return image_inputs, ids, image_kwargs

    def _apply_row_mask_logits(self, logits: torch.Tensor, row: int) -> torch.Tensor:
        """Add pool ``row``'s additive mask to ``logits`` (no-op if unmasked).

        ``logits`` is ``[*, vocab]``; broadcasts the ``[vocab]`` mask. Used for
        the prefill bonus token so it respects the skill mask the way the
        non-spec prefill sampler does.
        """
        if not self._row_masked[row]:
            return logits
        return logits + self._mask_buf[row]

    def _pack_side_values(self, out_hidden, rows, accept):
        """Build :class:`SpecSideValues` for the typed-token runtime.

        ``out_hidden`` is the verify ``[B, K+1, H]`` FINAL hidden; rows ``rows``
        are this step's active graph rows. Returns the per-active-row block
        hidden ``[num_active, K+1, H]`` plus the per-active-row temperature /
        top_p (``[num_active]``), with ``counts=None`` -- the scheduler slices
        the leading ``accept_i + 1`` positions of each row's block using the
        ``accept_counts`` it already receives in the result, so this stays fully
        on-device (no per-step host sync). The runtime's ``materialize_tokens``
        consumes ``hidden[i, j]`` as the ``hidden_last`` for committed position
        ``j`` of sequence ``i`` (only ``j <= accept_i`` are committed).
        """
        from kestrel.runtime.spec import SpecSideValues

        dev = self.device
        row_idx = torch.as_tensor(rows, device=dev, dtype=torch.long)
        hid = out_hidden.index_select(0, row_idx).contiguous()   # [A, K+1, H]
        temps = torch.tensor([self._row_temp[r] for r in rows],
                             device=dev, dtype=torch.float32)
        top_ps = torch.tensor([self._row_top_p[r] for r in rows],
                              device=dev, dtype=torch.float32)
        return SpecSideValues(
            hidden=hid, temperatures=temps, top_ps=top_ps, counts=None)

    def _sampling_params_b(self, rows) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row temperature / top_p over the FIXED-B graph batch.

        Returns ``(temp_b, top_p_b)`` each ``[B]`` (indexed by pool row r), built
        from the per-row sampling knobs ``_row_temp`` / ``_row_top_p``. Inactive
        rows this step (not in ``rows``) are forced greedy (T=0 / top_p=1) so the
        batched rejection sampler treats them as exact-argmax -- their committed
        token is discarded anyway (idle rows commit nothing). Used by ``step``'s
        sampling path; ``rejection_sample_block`` / ``logits_to_probs`` broadcast
        these per row, so greedy rows (T<=eps) collapse to argmax and only true
        sampling rows draw from the soft distribution.
        """
        dev = self.device
        temp_b = torch.zeros(self.B, device=dev, dtype=torch.float32)
        top_p_b = torch.ones(self.B, device=dev, dtype=torch.float32)
        for r in rows:
            temp_b[r] = self._row_temp[r]
            top_p_b[r] = self._row_top_p[r]
        return temp_b, top_p_b

    @torch.inference_mode()
    def admit_batch(self, states, prompts) -> list[int]:
        """Admit ``len(states)`` sequences via ONE packed prefill (test helper).

        Used to validate the :meth:`step` body in isolation: it reproduces
        ``SpecRunner.decode_batch``'s exact packed prefill (so the prefill GDN /
        KV state and the drafter context buffers are bit-identical to
        ``decode_batch``), then seeds the per-row bookkeeping. The production
        scheduler uses per-sequence :meth:`admit` instead (continuous batching);
        per-row prefill is numerically equivalent up to the documented bf16-tie
        difference between packed and single-sequence attention.
        """
        if len(states) != self.B:
            raise ValueError("admit_batch is fixed at batch_size B")
        if self._rows:
            raise RuntimeError("admit_batch requires an empty runner")
        rt = self.rt
        prompts = [list(p) for p in prompts]
        sink: dict[int, torch.Tensor] = {}
        handles = self._install_hooks(sink)
        try:
            cur, hid_packed, cu, prompt_lens = self._prefill(prompts, sink)
            if not self._graphs_ready:
                self._capture_graphs(prompt_lens, hid_packed, cu, sink)
            else:
                self._seed_draft_buffers(prompt_lens, hid_packed, cu)
        finally:
            self._remove_hooks(handles)
        self._free_rows = []
        for r in range(self.B):
            self._rows[r] = _SpecRow(state=states[r], cur=cur[r], ctx_len=prompt_lens[r])
            self._row_of[id(states[r])] = r
            self._cur_buf[r] = cur[r]
            self._ctx_buf[r] = prompt_lens[r]
            if hasattr(states[r], "batch_idx"):
                states[r].batch_idx = int(self.batch_idx[r])
        return cur

    def _set_row_rope_delta(self, row: int, rope_deltas) -> None:
        """Store ``row``'s M-RoPE spatial delta (or clear it to 0).

        ``step`` adds this to the 3 spatial rows of the verify/decode
        ``position_ids`` (rows 1..3), matching the runtime's normal-decode
        ``_prepare_decode_position_ids`` (which adds ``_decode_rope_deltas`` to
        the spatial rows). ``rope_deltas`` is the image-prefill forward cache's
        delta (``[1, 1]`` / ``[1]`` / scalar) or ``None`` for a text prefill
        (delta 0). Validated/coerced to a scalar like ``_store_sequence_cache``."""
        if rope_deltas is None:
            self._rope_deltas[row].zero_()
            return
        rd = rope_deltas.to(device=self.device, dtype=torch.long).reshape(-1)
        if rd.numel() != 1:
            raise RuntimeError(
                f"Qwen spec M-RoPE delta must be a single value, got "
                f"{tuple(rope_deltas.shape)}"
            )
        self._rope_deltas[row, 0] = rd[0]

    def _prefill_row(
        self, row: int, prompt: list[int], *, image=None, image_crops=None
    ) -> "tuple[int, torch.Tensor, torch.Tensor]":
        """Prefill one prompt into pool ``row``.

        Returns ``(ctx_len, last_logits, last_hidden)``:
          * ``ctx_len`` -- the prefilled context length ``n`` (with the image
            block, when an image is given).
          * ``last_logits`` -- the ``[1, vocab]`` lm-head logits at the final
            prefill position. The first/bonus token is selected from these *in*
            :meth:`admit` (mask + one-shot suppression + greedy/sample + the
            selected-token logprob), mirroring the non-spec prefill sampler.
          * ``last_hidden`` -- the ``[1, hidden]`` target ``last_hidden_state``
            at the final prefill position. This is the ``hidden_last`` the
            runtime's spatial decode reads to type the first id into a
            coord/size value (the non-spec ``post_sample`` input), surfaced by
            :meth:`admit` on ``state.admit_side_values``.

        Single-sequence packed prefill into a throwaway cache, then capture the
        GDN state into the persistent pool row (in place). KV is written into
        the runtime-shared paged pool at this row's reserved pages. Also seeds
        the drafter's left-aligned context buffer for the row and ensures the
        graphs are captured (lazily, once).

        When ``image`` is given the prompt is prefilled *with the image KV
        prefix*: the vision block is spliced into the ids and the forward runs
        through the full multimodal model (vision encoder + multimodal position
        ids) exactly like a normal image prefill, so the GDN/KV state the spec
        loop then decodes from carries the image context (no text-only fallback).
        ``image_crops`` (the request's preprocessed multi-tile image inputs) is
        threaded through so a multi-crop image is encoded as all its tiles, not
        a thumbnail-only image (matching the non-spec ``prepare_sequence``).
        """
        rt, dev = self.rt, self.device
        pt = rt.page_table
        # Image prefill: expand the prompt with the vision block and build the
        # image forward kwargs the runtime's own image path uses. The returned
        # ``ids`` (with image-pad tokens) is what actually occupies the KV/GDN
        # context, so ``n`` (and the drafter context width) is its length.
        image_inputs, ids, image_kwargs = self._prepare_image_prefill(
            prompt, image, image_crops=image_crops)
        n = len(ids)
        pool_row = int(self.cb[row])
        if n + self.num_spec + 1 + 4 > self.max_seq_len:
            raise ValueError("prompt too long for SpecStepRunner max_seq_len")

        sink: dict[int, torch.Tensor] = {}
        handles = self._install_hooks(sink)
        try:
            cpos = torch.arange(n, device=dev).view(1, -1)
            page_tbl_row = torch.index_select(pt.page_table, 0, self.cb[row:row + 1])
            tmp_cache = rt._new_cache()
            rt.model.model.rope_deltas = None
            if image_inputs is not None:
                # Multimodal prefill: route through _forward_base, which runs the
                # vision encoder + multimodal position ids and writes GDN/KV state
                # into ``tmp_cache`` at this row's reserved pages (batch_idx).
                lh_pf_b, pf_fc = rt._forward_base(
                    input_ids=self._ids([ids]),
                    past_key_values=tmp_cache,
                    batch_idx=int(self.batch_idx[row]),
                    cache_position_ids=cpos,
                    **image_kwargs,
                )
                lh_pf = lh_pf_b[0]
                # Store this row's M-RoPE spatial delta so ``step`` shifts the
                # verify/decode spatial position rows by it (mirrors the runtime's
                # ``_store_sequence_cache`` -> ``_decode_rope_deltas``). The image
                # prefill compresses the vision block's positions, so post-image
                # text tokens sit at a SHIFTED spatial M-RoPE position; without
                # this the verify forward would rotate every post-prefill token
                # with text-only positions and could diverge from a normal image
                # decode of the same model. A text row never reaches this branch
                # (delta stays 0, set below).
                self._set_row_rope_delta(row, getattr(pf_fc, "rope_deltas", None))
            else:
                bidx_tok = torch.full((1, n), self.batch_idx[row], device=dev, dtype=torch.int64)
                slotmap = pt.build_slot_mapping(batch_idx=bidx_tok, positions=cpos)
                seqk = torch.tensor([n], device=dev, dtype=torch.int32)
                cu_t = torch.tensor([0, n], device=dev, dtype=torch.int32)
                # Single-seq prefill: match _forward_base's validated path (cu_seq_lens_q
                # = [0, n], NO seq_idx -> the model derives the single-seq layout).
                out_pf = rt.model.model.language_model(
                    input_ids=self._ids([ids]), position_ids=cpos,
                    past_key_values=tmp_cache, cache_position_ids=cpos,
                    slot_mapping=slotmap, page_table=page_tbl_row,
                    paged_kv_seqlens_k=seqk, cu_seq_lens_q=cu_t,
                )
                lh_pf = out_pf.last_hidden_state[0]
                # Text-only prefill: no M-RoPE spatial shift. Clear any stale
                # delta a prior image admit left on this (reused) row so the
                # verify positions stay plain text positions.
                self._set_row_rope_delta(row, None)
            self.cache.advance_to(max(int(self.cache.get_seq_length()), n))
            # Capture this row's freshly-prefilled GDN state into the pool row.
            rt._linear_state_pool.capture_from_cache(pool_row, tmp_cache)

            # Final-position lm-head logits + target last-hidden. The first/bonus
            # token (greedy argmax or sampled draw), the one-shot suppression, and
            # the selected-token logprob are all derived from ``last_logits`` in
            # ``admit`` -- mirroring the non-spec prefill sampler, which selects a
            # request's first generated token (and computes its logprob) from the
            # prefill's last-position logits. ``last_hidden`` is that position's
            # ``last_hidden_state`` (the spatial decode's ``hidden_last``).
            last_logits = self.lm_head(lh_pf[n - 1:n])     # [1, vocab] bf16
            last_hidden = lh_pf[n - 1:n]                    # [1, hidden]
            hid = self._target_hidden_b(sink)[0]    # [n, fc]

            if not self._graphs_ready:
                # Capture the fixed-B graphs once. ``_capture_graphs`` only uses
                # prompt_lens/hid/cu to seed static buffers (overwritten every
                # step) and the drafter context (re-seeded per admit), so feed it
                # a packed view that repeats this row's prefill hidden across all
                # B rows: prompt_lens=[n]*B, cu=[0,n,2n,...], hid tiled.
                prompt_lens = [n] * self.B
                cu = [n * r for r in range(self.B + 1)]
                hid_packed = hid.repeat(self.B, 1)
                self._capture_graphs(prompt_lens, hid_packed, cu, sink)
            # Seed this row's drafter context buffer (others are seeded on admit).
            self.th_buf[row].zero_()
            self.th_buf[row, :n] = hid
        finally:
            self._remove_hooks(handles)
        return n, last_logits, last_hidden

    # -- retire: free a finished sequence's row ------------------------------

    def _release_row(self, row: int) -> None:
        """Return ``row`` to the free pool and clear its per-row decode state.

        Shared by :meth:`retire` (normal finish) and :meth:`admit`'s failure
        path (a row popped from ``_free_rows`` that never became live). Clears
        the row's constrained-decode mask + sampling knobs so the next admit into
        this (address-stable) row starts unmasked/greedy: the mask row is read
        every step via the captured graph, so a stale -inf mask left on a freed
        row would silently constrain the next sequence.

        It also resets the row's on-device decode cursor (``_ctx_buf``) and
        current token (``_cur_buf``) to 0. ``step`` builds the draft/verify
        position buffers + KV slot mapping for *every* fixed-B row from
        ``_ctx_buf`` (the ``active`` mask only suppresses the COMMIT / state
        advance, not the forward's KV writes), so a freed row left parked at its
        retirement position -- which can be near ``max_seq_len`` -- would keep
        driving verify/KV writes at those stale positions on subsequent steps of
        OTHER rows, potentially past the row's reserved pages (out-of-bounds slot
        mapping) while another sequence is mid-decode. ``admit`` re-seeds the
        cursor for the next sequence (line below), so a retired-but-not-yet-
        readmitted row must sit at the safe page-0 position (ctx 0 -> the slot
        mapping stays inside this row's own already-committed early pages, a true
        no-op idle row) instead of a stale tail.
        """
        # Zero the mask UNCONDITIONALLY, not just when ``_row_masked[row]`` is set.
        # ``admit`` installs the row mask via ``_set_row_mask`` BEFORE marking the
        # row live, and ``_set_row_mask`` fills ``_mask_buf[row]`` with -inf *then*
        # runs its "no valid token left" check -- so an all-suppressed whitelist/
        # blacklist (e.g. allowed=[x], suppressed=[x]) raises with the row already
        # full of -inf but ``_row_masked[row]`` still False. ``admit``'s failure
        # path then calls this method to free the row; a ``_row_masked``-gated zero
        # would leave that -inf residue on the freed idle row. The mask row is read
        # every step for ALL fixed-B rows by the captured draft graph
        # (``logits + self._mask_buf[:, None, :]``), so under ``sampling=True`` the
        # stale all--inf row turns that row's draft logits all -inf -> softmax NaN
        # -> ``multinomial`` failure on an unrelated live step. Always clearing here
        # makes the freed row a true identity (zero) mask, covering this reject path
        # and any other release that left a partially-written row.
        self._mask_buf[row].zero_()
        self._row_masked[row] = False
        self._row_temp[row] = 0.0
        self._row_top_p[row] = 1.0
        self._row_logprobs[row] = False
        # Reset the freed row's decode cursor / current token so an idle (retired,
        # not-yet-readmitted) row contributes only safe, in-bounds positions to the
        # next ``step`` it rides along in (see docstring).
        self._ctx_buf[row] = 0
        self._cur_buf[row] = 0
        if hasattr(self, "_row_steps"):
            self._row_steps[row] = 0
        # Clear the M-RoPE spatial delta so an idle row contributes no spatial
        # shift to the verify position rows it rides along in, and the next
        # (text-or-image) admit into this reused row starts from delta 0.
        self._rope_deltas[row].zero_()
        # A retired row still rides along in the fixed-B verify forward while
        # other rows decode. Drop its replay cursor so the idle row is a real
        # no-op and the next request admitted into this row starts from a clean
        # per-row flush phase/state.
        if (
            hasattr(self, "_cb_host")
            and hasattr(self, "gdn_layer_idxs")
            and hasattr(self, "cache")
        ):
            pool_row = self._cb_host[row]
            for idx in self.gdn_layer_idxs:
                lc = self.cache.layers[idx]
                if getattr(lc, "replay_lengths", None) is not None:
                    lc.replay_lengths[pool_row].zero_()
        self._free_rows.append(row)

    def retire(self, state: object) -> None:
        row = self._row_of.pop(id(state), None)
        if row is None:
            return
        self._rows.pop(row, None)
        self._release_row(row)

    # -- step: one macro-step over the active rows ---------------------------

    @torch.inference_mode()
    def step(
        self,
        states,
        *,
        allowed_token_ids=None,
        suppressed_token_ids=None,
        commit_caps=None,
        _force_drafts=None,
    ) -> "SpecStepResult":
        """Draft + verify + accept + commit one macro-step for ``states``.

        ``states`` are the sequences active this step (each previously admitted).
        Returns, parallel to ``states``, the newly committed tokens per sequence
        (``a_i + 1`` each) and the accept counts (``a_i``).

        ``allowed_token_ids`` / ``suppressed_token_ids`` are the per-row skill
        masks the scheduler recomputes from each sequence's *current* skill state
        THIS macro-step (parallel to ``states``: entry ``i`` is the mask for
        ``states[i]``, ``None`` for an unconstrained / finalized row). They
        REPLACE the admit-time mask for this step and are applied to BOTH the
        drafter and the verify. Stateful constrained skills (point alternates
        ``[coord,eos]`` <-> ``[coord]``; detect cycles x->y->size) evolve their
        allowed set per committed token, so the mask snapshotted once at
        :meth:`admit` goes stale after the first position; refreshing it per step
        mirrors the non-spec sampler re-querying the mask every step. ``None`` for
        the whole argument keeps the admit-time mask (back-compat for a caller
        that does not refresh, e.g. the standalone tests). When supplied, each
        active row's mask is rebuilt in place into the address-stable
        ``_mask_buf`` (the captured draft graph reads it), so the draft + verify
        constrain to the live allowed/suppressed set; idle rows keep their stored
        mask (they commit nothing).

        ``commit_caps`` is the per-row upper bound on how many tokens this
        macro-step may COMMIT for ``states[i]`` (parallel to ``states``; ``None``
        for a row -- or ``None`` for the whole arg -- means uncapped: commit the
        full accepted run, the validated multi-token behaviour). The scheduler
        sets ``commit_caps[i] = 1`` for a STATEFUL-masked row: the single per-step
        mask above is exact only for ONE constraint transition per committed run,
        but a macro-step otherwise commits a *variable* run of ``a_i + 1`` tokens
        under that one mask, so a stateful skill's (point ``[coord,eos]`` <->
        ``[coord]``; detect x->y->size; query prefix injection) 2nd..Nth committed
        positions would be verified under the stale 1st-position mask. Capping the
        row to one committed token forces exactly one transition per run -- the
        regime where the per-step mask IS exact, identical to the non-spec
        one-token-per-step path -- and the row is re-masked from the now-current
        skill state next step.

        Enforcement (the scheduler relies on it for correctness): a row commits
        ``accept_i + 1`` tokens, so a cap of ``commit_caps[i]`` tokens truncates
        the accept count to ``min(accept_i, commit_caps[i] - 1)`` BEFORE any state
        advance. The GDN replay ring (``replay_lengths``), the conv-window roll,
        the paged-KV context cursor (``_ctx_buf``) and the next current token
        (``_cur_buf``) then all advance by the capped run only. The
        drafted/verified tokens beyond the cap are DISCARDED -- never folded into
        the ring or KV, exactly as if they had been rejected -- so the next step
        re-drafts from the capped commit position. With ``commit_caps[i] == 1`` the
        accept is forced to 0 and the row commits exactly its single first-position
        token under this step's mask (bit-exact to committing one token then
        re-stepping). Greedy/unconstrained rows are unaffected.

        ``_force_drafts`` (validation only): a per-``states`` list of draft token
        lists that overrides the drafter's proposals. Feeding the gold greedy
        continuation makes acceptance deterministic (accept == matching prefix),
        which proves the macro-step + variable-advance machinery is lossless
        independent of drafter quality (the free-running random drafter otherwise
        cascades through bf16 argmax ties + kernel non-determinism).
        """
        from kestrel.runtime.spec import SpecSideValues, SpecStepResult

        rt, dev = self.rt, self.device
        B, block_size, K = self.B, self.block_size, self.num_spec
        gdn = self.gdn_layer_idxs
        pt = rt.page_table
        if not self._graphs_ready:
            raise RuntimeError("SpecStepRunner.step before any admit (no graphs)")

        rows = [self._row_of[id(s)] for s in states]          # active graph rows
        # Per-step constrained-decode refresh: when the scheduler supplies the
        # live per-row masks (recomputed from each sequence's current skill state
        # AFTER committing the prior macro-step), rebuild each active row's mask
        # into the address-stable ``_mask_buf`` BEFORE the draft graph reads it.
        # This REPLACES the admit-time snapshot for this step so stateful skills
        # (point/detect) constrain the drafter + verify to their evolved allowed
        # set -- matching the non-spec sampler's per-step mask re-query. ``None``
        # for the whole arg leaves the stored (admit) mask untouched (back-compat).
        if allowed_token_ids is not None or suppressed_token_ids is not None:
            self._refresh_step_masks(rows, allowed_token_ids, suppressed_token_ids)
        # Per-step feature flags (host-side, cheap): does ANY active row carry a
        # mask / want logprobs / sample? The all-False fast paths keep the
        # validated greedy/unmasked step byte-identical.
        any_masked = any(self._row_masked[r] for r in rows)
        want_logprobs = any(self._row_logprobs[r] for r in rows)
        # Sampling rows (temperature>0) run modified rejection sampling; greedy
        # rows (temperature==0) keep the exact-argmax accept/commit. A step is
        # "sampling" iff ANY active row samples; greedy-only steps take the
        # byte-identical fast path (no q/multinomial/residual ops at all). The
        # rejection sampler needs the per-position draft LOGITS, which the draft
        # graph only emits when the runner was built sampling=True; refuse a
        # sampling row otherwise (the greedy draft graph captured argmax tokens,
        # not logits) rather than silently decode it greedily.
        sample_rows = [r for r in rows if self._row_temp[r] > GREEDY_TEMPERATURE_EPS]
        any_sampling = bool(sample_rows)
        if any_sampling and not self._sampling:
            raise NotImplementedError(
                "SpecStepRunner.step got a sampling row (temperature>0) but the "
                "runner was built with sampling=False, so the draft graph emits "
                "greedy argmax tokens, not the per-position logits the rejection "
                "sampler needs. Construct SpecStepRunner(sampling=True) to enable "
                "per-row rejection sampling."
            )
        maxc, conv_k = self.maxc, self.conv_k
        cur_buf, ctx_buf = self._cur_buf, self._ctx_buf        # [B] on-device, by row
        ar_block = torch.arange(block_size, device=dev)
        ar_conv = torch.arange(conv_k, device=dev)
        ar_maxc = torch.arange(maxc, device=dev)
        # Active-row mask over the fixed-B graph batch; idle rows commit nothing.
        active = torch.zeros(B, dtype=torch.bool, device=dev)
        active[torch.tensor(rows, device=dev, dtype=torch.long)] = True

        sink: dict[int, torch.Tensor] = {}
        handles = self._install_hooks(sink)
        try:
            # --- Draft (graph replay). All buffers updated from on-device state. ---
            self.block_ids_buf[:, 0] = cur_buf
            self.dpos_buf[:, maxc:] = ar_block[None, :] + ctx_buf[:, None]
            self.dmask_buf[:, 0, 0, :maxc] = torch.where(
                ar_maxc[None, :] < ctx_buf[:, None], 0.0, float("-inf")
            ).to(self.dmask_buf.dtype)
            # Draft. A sampling runner's graph emits per-position LOGITS [B,K,vocab]
            # (so the host can sample q + rejection-sample); a greedy runner's
            # graph emits the argmax draft tokens [B,K] directly. ``q`` (the
            # drafter's per-position distribution) is built only when THIS step
            # samples -- the rejection rule needs q(x_j) for the accept ratio and
            # q for the residual. On a greedy-only step (every active row T=0) the
            # fast path below never touches q (it stays None), so a sampling runner
            # whose batch is momentarily all-greedy still pays nothing extra.
            q = None
            if self._sampling:
                if self._use_graphs:
                    self._draft_graph.replay()
                    draft_logits = self.draft_logits_out           # [B, K, vocab]
                else:
                    draft_logits = self._draft_logits_fn()
                # Per-row temperature / top_p over the fixed-B batch (greedy rows
                # keep T=0 / top_p=1 -> argmax drafts below; their q is unused).
                temp_b, top_p_b = self._sampling_params_b(rows)    # [B], [B]
                samp_mask_b = (temp_b > GREEDY_TEMPERATURE_EPS)    # [B]
                greedy_drafts = draft_logits.argmax(-1)            # [B, K]
                if any_sampling:
                    q = logits_to_probs(
                        draft_logits.reshape(B * K, self._vocab),
                        temp_b.repeat_interleave(K),
                        top_p_b.repeat_interleave(K),
                    ).reshape(B, K, self._vocab)
                    # Sampling rows draw x_j ~ q_j; greedy rows take the argmax
                    # (exact). Sample every row densely (graph-safe), then overwrite
                    # greedy rows so the greedy accept stays byte-exact.
                    sampled = torch.multinomial(
                        q.reshape(B * K, self._vocab), 1, generator=self._spec_gen
                    ).reshape(B, K)                                # [B, K]
                    drafts = torch.where(samp_mask_b[:, None], sampled, greedy_drafts)
                else:
                    drafts = greedy_drafts                         # all-greedy step
            elif self._use_graphs:
                self._draft_graph.replay()
                drafts = self.drafts_out                       # [B, K]
            else:
                drafts = self._draft_fn()
            if _force_drafts is not None:
                for idx, r in enumerate(rows):
                    drafts[r] = torch.tensor(_force_drafts[idx], device=dev, dtype=drafts.dtype)

            # --- Flush rings every _flush_every ACTIVE steps, per row. ---
            #
            # Replay length grows by at most block_size each macro-step, so this
            # cadence prevents overflow without reading replay_lengths back to
            # host. The per-row phase is correctness-critical for continuous
            # batching: a reused row must not inherit another request's flush
            # phase, because materializing at a different relative token offset
            # changes bf16 accumulation order enough to flip image argmax ties.
            flush_rows = []
            for r in rows:
                self._row_steps[r] += 1
                if self._row_steps[r] % self._flush_every == 0:
                    flush_rows.append(self._cb_host[r])
            if flush_rows:
                fr = torch.tensor(flush_rows, device=dev, dtype=torch.long)
                for idx in gdn:
                    self.cache.layers[idx].materialize_recurrent_from_replay(
                        fr, write_recurrent=False)

            # --- Snapshot conv windows (verify over-advances; commit rolls). ---
            conv_snap = {idx: self.cache.layers[idx].conv_states.clone() for idx in gdn}

            # --- Verify (graph replay). ---
            self.ids_buf[:, 0] = cur_buf
            self.ids_buf[:, 1:] = drafts.long()
            self.cpos_buf[:] = ar_block[None, :] + ctx_buf[:, None]
            # Rebuild the 4-row M-RoPE verify ``position_ids`` from the text
            # positions + per-row spatial delta, mirroring the runtime's normal
            # decode ``_prepare_decode_position_ids`` (row 0 = text, rows 1..3 =
            # text + rope delta). For a non-image row the delta is 0, so all 4
            # rows equal the text positions and this is byte-identical to passing
            # the 2-D ``cpos_buf``. ``cache_position_ids`` (the KV slot/seqlen
            # source) stays the 2-D text positions, so KV writes are unchanged.
            self.vpos_buf[:] = self.cpos_buf.unsqueeze(0)
            self.vpos_buf[1:].add_(self._rope_deltas)
            self.slot_buf.copy_(
                pt.build_slot_mapping(batch_idx=self.cb, positions=self.cpos_buf))
            self.seqk_buf.copy_(self.cpos_buf.max(dim=1).values.to(torch.int32) + 1)
            if self._use_graphs:
                self._verify_graph.replay()
                out_hidden = self.out_hidden
                # Graph replay refreshes the capture-time sink views in place.
                block_hid = self._target_hidden_b(self._verify_sink)
            else:
                out_hidden = self.lm(**self._verify_kw).last_hidden_state
                # Eager forward fired the step-local hooks -> read that sink.
                block_hid = self._target_hidden_b(sink)
            verify_logits = self.lm_head(out_hidden)            # [B, K+1, vocab]
            if any_masked:
                # Apply the SAME per-row token mask to the verify so accept +
                # bonus stay inside the allowed set (matching the masked drafter
                # and the non-spec masked sampler). Identity for unmasked rows.
                verify_logits = verify_logits + self._mask_buf[:, None, :]
            pred = verify_logits.argmax(-1)                     # [B, K+1] masked argmax

            # --- Per-row committed tokens + accept (greedy OR rejection-sampled).
            # ``committed`` [B, K+1] is the committed token at each block position
            # (committed[r, :accept_r+1] are the tokens emitted for row r); accept
            # [B] is the per-row accepted-draft count. Greedy rows use the
            # exact-argmax rule (committed==pred, accept==matching prefix); a
            # sampling row uses the modified-rejection rule against the SAME masked
            # verify logits, so both share one downstream commit. ---
            # Greedy accept: longest matching draft prefix (the validated rule).
            accept = (pred[:, :K] == drafts).int().cumprod(1).sum(1).long()    # [B]
            committed = pred                                                   # [B, K+1]
            if any_sampling:
                # Modified rejection sampling per row over the WHOLE fixed-B batch
                # (dense / graph-safe): walk drafts left-to-right accepting x_j
                # with prob min(1, p_j(x_j)/q_j(x_j)); at the first reject emit the
                # normalized-residual sample, or the bonus from p_K if all accept.
                # ``verify_logits`` (== p's logits, masked) and ``q`` carry the
                # per-row temperature/top_p, so the emitted distribution equals the
                # non-spec sampler's p per committed position (distribution-exact).
                samp_tokens, accept_samp = rejection_sample_block(
                    drafts, q, verify_logits, temp_b, top_p_b,
                    generator=self._spec_gen,
                )                                                  # [B,K+1], [B]
                # Select per row: sampling rows take the rejection result, greedy
                # rows keep the exact-argmax result. (samp_tokens.dtype is long.)
                samp_sel = samp_mask_b[:, None]
                committed = torch.where(samp_sel, samp_tokens, pred.long())
                accept = torch.where(samp_mask_b, accept_samp.long(), accept)

            # --- Per-row commit cap (#114 stateful-skill mask exactness). The
            # scheduler caps a STATEFUL-masked row to ``commit_caps[r]`` committed
            # TOKENS this step (1 == one constraint transition per run, the regime
            # where the single per-step mask is exact -- identical to the non-spec
            # one-token-per-step path). A row commits ``accept+1`` tokens (the
            # accepted drafts plus the bonus), so a cap of ``c`` tokens is an
            # accept ceiling of ``c-1``: truncate the accept count to
            # ``min(accept, c-1)`` HERE -- before the state advance below -- so the
            # cap propagates uniformly. ``advance = accept+1`` then drives the GDN
            # replay ring length, the conv-window roll, the paged-KV context
            # cursor, the next-current (bonus) token and the returned token slice,
            # every one of which advances by the capped run only. The
            # drafted/verified tokens beyond the cap are never folded into the ring
            # or KV (replay_lengths / conv / ctx advance by the capped run), i.e.
            # DISCARDED exactly as if rejected; the next step re-drafts from the
            # capped commit position. With ``c == 1`` the accept is forced to 0, so
            # the row commits exactly the single first-position token under THIS
            # step's mask -- bit-exact to the non-spec sampler advancing one token
            # then re-querying the mask. ``None`` (row or whole arg) leaves the row
            # uncapped (full multi-token accept). Built on-device so the step stays
            # async; capped rows are rare (only constrained stateful skills) so the
            # tiny host loop over ``rows`` is off the hot text-decode path.
            if commit_caps is not None and any(c is not None for c in commit_caps):
                # Per-row accept ceiling over the fixed-B graph batch: a capped row
                # gets ``commit_caps[r] - 1`` (clamped at 0 so a cap of 1 forces a
                # single committed token); every other row gets a no-op ceiling
                # (the max possible accept K, which ``minimum`` can never lower).
                cap_b = accept.new_full((B,), K)
                for idx, r in enumerate(rows):
                    cap = commit_caps[idx]
                    if cap is not None:
                        cap_b[r] = max(0, int(cap) - 1)
                accept = torch.minimum(accept, cap_b)

            # --- GPU-resident batched accept + commit (idle rows masked off). ---
            # advance_i = accept_i+1 for active rows, 0 for idle -> every commit op
            # below is an identity on idle rows (conv-roll with offset 0 is the
            # identity gather, etc.).
            advance = torch.where(active, accept + 1, accept.new_zeros(()))     # [B]
            # Conv-window roll: gather the last conv_k cols of [conv_snap | preconv]
            # starting at `advance` (advance=0 -> identity gather, so idle rows
            # unchanged). Equivalent to the per-row rolled[:, -conv_k:].
            roll_idx = ar_conv[None, :] + advance[:, None]                     # [B, conv_k]
            for idx in gdn:
                lc = self.cache.layers[idx]
                lc.replay_lengths[self.cb] += advance.to(lc.replay_lengths.dtype)
                cat_conv = torch.cat([conv_snap[idx][self.cb], lc.spec_block_preconv], dim=-1)
                C = cat_conv.shape[1]
                lc.conv_states[self.cb] = cat_conv.gather(
                    2, roll_idx[:, None, :].expand(B, C, conv_k))
            # Aux-hidden: scatter the whole verified block at each row's cursor
            # (only accept+1 valid; tail overwritten next step / unread beyond ctx).
            th_idx = ctx_buf[:, None] + ar_block[None, :]                      # [B, block]
            self.th_buf.scatter_(
                1, th_idx[:, :, None].expand(B, block_size, self.th_buf.shape[-1]), block_hid)
            # Advance per-row on-device state (idle rows keep their stale values).
            # The next cur is the committed token at column == accept (the bonus
            # for greedy / the residual/bonus replacement for a sampling row).
            bonus = committed.gather(1, accept[:, None]).squeeze(1)            # [B]
            self._cur_buf = torch.where(active, bonus, cur_buf)
            self._ctx_buf = ctx_buf + advance

            # --- Per-committed-token logprobs: log_softmax of the (masked) verify
            # target logits at each block position, gathered at the COMMITTED id
            # (committed[r][j] is the token emitted at position j for j <=
            # accept_r). This is the logprob of the actually-emitted token under
            # the target distribution p -- correct for both the greedy argmax and
            # the rejection-sampled committed tokens. Computed on the compute side
            # and D2H'd alongside committed/accept; the lazy result slices per row
            # to a_r+1. ---
            lp_block = None
            if want_logprobs:
                # Greedy rows report ``0.0`` for every committed token (the
                # non-spec one-hot convention: greedy forces a one-hot
                # distribution at the argmax, so the selected-token logprob is
                # 0.0 -- matching ``_select_first_token``'s admit logprob and the
                # single-token sampler's all-greedy path; the row is zeroed below
                # after the gather). Sampling rows must report the logprob under
                # the SAME temperature-scaled distribution the sampler drew from
                # (full-softmax of logits/temperature, NOT top-p-renormalized --
                # matching ``_select_first_token``'s sampling logprob and the
                # non-spec single-token sampler), so divide the verify logits by
                # the per-row temperature before log_softmax (idle/greedy rows
                # keep T=1 -> no change). Done per row so a mixed greedy+sampling
                # batch stays consistent with non-spec decoding.
                lp_logits = verify_logits.float()                              # [B,K+1,V]
                if any_sampling:
                    lp_temp = torch.where(
                        samp_mask_b, temp_b, temp_b.new_ones(())
                    ).clamp_min(GREEDY_TEMPERATURE_EPS)                        # [B]
                    lp_logits = lp_logits / lp_temp[:, None, None]
                lp_all = torch.log_softmax(lp_logits, dim=-1)                  # [B,K+1,V]
                # Rejected sampling rows leave committed tail positions at -1; gather
                # would index the vocab dim OOB (negative -> device-side assert), so
                # clamp to a valid id before gathering. The tail logprobs are unused
                # (the lazy result slices each row to accept+1 before returning).
                gather_ids = committed.clamp_min(0)                           # [B, K+1]
                lp_block = lp_all.gather(2, gather_ids[:, :, None]).squeeze(2)  # [B, K+1]
                # Force the one-hot greedy convention: a greedy row (NOT in
                # samp_mask_b) reports 0.0 for every committed token, matching the
                # non-spec sampler / ``_select_first_token`` (without this a
                # greedy request's per-token logprob would jump from 0.0 to the
                # raw target-softmax value the moment spec decode engages). Idle
                # and greedy rows are masked; sampled rows keep their real
                # temperature-scaled logprobs. When no row samples the whole
                # block is greedy -> zero it outright.
                if any_sampling:
                    lp_block = torch.where(
                        samp_mask_b[:, None], lp_block, lp_block.new_zeros(()))
                else:
                    lp_block = lp_block.new_zeros(lp_block.shape)

            # --- Typed-token side-values: the verify FINAL hidden at every
            # committed position is what the runtime's materialize_tokens hook
            # decodes coord/size ids from (the non-spec ``hidden_last``). Pack the
            # committed positions (a_r+1 per active row) row-major, with the
            # per-row temperature/top_p broadcast, so the scheduler can batch the
            # spatial decode over the whole macro-step. Text-only runtimes skip
            # this (ids stay TextToken). ---
            side_values = None
            if self._detect_typed_token_runtime():
                side_values = self._pack_side_values(out_hidden, rows, accept)

            # --- Scheduler contract. The newly committed tokens for row r are
            # committed[r][:accept_r+1]: committed[:accept] are the accepted drafts
            # and committed[accept] is the bonus/replacement, so this is exactly
            # the accepted drafts followed by the next token (greedy argmax or the
            # rejection-sampled token). Instead of a blocking committed.tolist() we
            # kick off an async D2H of committed[B,K+1]+accept[B] into a pinned
            # ping-pong slot on a private copy stream (gated by a compute-stream
            # event) and hand back a LAZY SpecStepResult; the scheduler resolves it
            # at commit, one tick later, so the wait overlaps GPU work (SGLang's
            # per-step relay; design doc §12). ---
            if self._out_copy_stream is not None:
                slot = self._out_slot
                self._out_slot ^= 1
                pred_pin = self._out_pred_pinned[slot]
                acc_pin = self._out_acc_pinned[slot]
                done = self._out_copy_events[slot]
                lp_pin = self._out_lp_pinned[slot] if want_logprobs else None
                ready = torch.cuda.Event()
                ready.record()                              # current (compute) stream
                with torch.cuda.stream(self._out_copy_stream):
                    self._out_copy_stream.wait_event(ready)
                    pred_pin.copy_(committed, non_blocking=True)
                    acc_pin.copy_(accept, non_blocking=True)
                    if lp_pin is not None:
                        lp_pin.copy_(lp_block, non_blocking=True)
                    done.record(self._out_copy_stream)
                rs = list(rows)

                def _resolve(ev=done, pp=pred_pin, ap=acc_pin, lp=lp_pin, rs=rs):
                    ev.synchronize()                        # wait async D2H (host parse only)
                    toks = [pp[r, : int(ap[r]) + 1].tolist() for r in rs]
                    accs = [int(ap[r]) for r in rs]
                    lps = (
                        [lp[r, : int(ap[r]) + 1].tolist() for r in rs]
                        if lp is not None else None
                    )
                    return toks, accs, lps

                return SpecStepResult(resolve=_resolve, side_values=side_values)
            # CPU / non-CUDA fallback: eager (tests, no copy stream).
            pred_h = committed.tolist()
            acc_h = accept.tolist()
            out_tokens = [pred_h[r][: acc_h[r] + 1] for r in rows]
            out_accepts = [acc_h[r] for r in rows]
            out_logprobs = None
            if want_logprobs:
                lp_h = lp_block.tolist()
                out_logprobs = [lp_h[r][: acc_h[r] + 1] for r in rows]
            return SpecStepResult(
                tokens=out_tokens,
                accept_counts=out_accepts,
                logprobs=out_logprobs,
                side_values=side_values,
            )
        finally:
            self._remove_hooks(handles)
