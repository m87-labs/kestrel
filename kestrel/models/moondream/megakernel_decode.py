"""Route decode steps through the whole-model megakernel (MD2 + MD3).

The megakernel runs an entire model's decode step (all layers) in one persistent
CUDA launch, replacing the native ``text_decoder`` forward for eligible buckets.
This module owns the engine-side lifecycle: pick the right variant for the GPU,
decide per-bucket whether the megakernel is enabled, build the session lazily, and
run each decode token.

Every entry point is non-fatal. If the model/GPU is not supported, no bucket is
enabled, or the megakernel backend is missing, the manager disables itself and the
caller runs the native decode. It is never an error and never a deadlock.

How it works:

  * Variant selection ``(model_kind, arch, num_sms, bucket)``. The megakernel is
    built for a specific persistent-grid size (``num_sms``); an oversubscribed grid
    deadlocks, so we run the largest shipped same-arch ``num_sms`` that is <= the
    device SM count (``_select_num_ctas``), or fall back to native. The deploy set
    (``_DEPLOY_SM_COUNTS``) is listed here directly so the engine depends only on
    ``kestrel_kernels``.

  * Enable table. Being buildable is not the same as being faster: on H100 SXM the
    megakernel is 1.38x at batch 1 but 0.86x at batch 8. ``_ENABLE_TABLE`` records,
    per ``(model, arch, num_sms, bucket)``, which buckets are actually enabled and
    the measurement that justifies it. It is code (no env flags, no JSON).

  * KV cache. The engine's KV cache is paged; the megakernel owns a contiguous KV
    buffer that grows by one token per step. When a sequence starts decoding, the
    manager copies its prefill KV from the paged cache into the contiguous buffer
    once (``prepare`` -> ``_maybe_seed``); after that the megakernel appends each
    step's K/V in place. The engine does not manage this -- it is internal.

  * Positions. One CUDA graph is captured per bucket and serves every step. The
    engine writes the token positions into a resident device buffer between graph
    replays; an in-graph gather uses them to refill RoPE cos/sin (and MD3 TAU) and
    the KV length, so the same graph works at any position.

  * LoRA. ``lora_slot_ids`` select the per-row adapter; slot 0 (no adapter) is
    identical to the base decode.

The engine touches three things: construct the manager, call ``prepare`` in the
per-step decode prep, and call ``decode`` in the captured forward. Writing
``input_pos`` between replays is already part of the native decode contract.

Enabled today: MD3 and MD2 at batch 1 on H100 SXM (sm90, 132 SMs). Batches above 1
need a per-request KV copy for mixed sequence starts and are left for a follow-up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# The megakernel serves at most 8 rows per launch; above this the engine
# always runs native. The enable table restricts further within this range.
_MAX_MEGAKERNEL_BUCKET = 8

# Persistent-grid sizes the megakernel is shipped for, per arch. A variant is valid
# only when its grid size is <= the device SM count (an oversubscribed persistent
# grid deadlocks), and we ship one variant per distinct deploy SM count. Listed here
# directly so the engine depends only on kestrel_kernels.
_DEPLOY_SM_COUNTS: dict[int, tuple[int, ...]] = {
    90: (132, 114),  # 132: H100 SXM5/NVL, H200, GH200 ; 114: H100 PCIe
    100: (148,),  # 148: B200
    # 120: (188,) -- RTX PRO 6000 Blackwell Server: not shipped yet (native fallback).
}


def _select_num_ctas(arch: int, device_sms: int) -> Optional[int]:
    """Return the grid size of the megakernel variant to run on this device, or
    ``None`` if none is supported here (the caller then runs native decode).

    Returns the largest shipped same-arch grid size <= ``device_sms`` (never
    oversubscribe the persistent grid). ``None`` when nothing fits -- an SM count we
    have not built yet, or an arch with no variants."""
    candidates = [n for n in _DEPLOY_SM_COUNTS.get(int(arch), ()) if n <= int(device_sms)]
    return max(candidates) if candidates else None


@dataclass(frozen=True)
class _EnableRow:
    """One ``(model, arch, num_sms)`` row of the enable table: which decode buckets
    are enabled, and the measurement that justifies them."""

    buckets: frozenset[int] = field(default_factory=frozenset)
    note: str = ""


# Which decode buckets route through the megakernel, per (model, arch, grid size,
# bucket). The megakernel is not always faster (H100 SXM measured batch 1 at 1.38x
# but batch 8 at 0.86x), so only buckets measured at >= 1.0x are enabled here.
#
#   model_kind -> arch (SM major * 10) -> grid size -> _EnableRow
#
# A missing entry or an empty bucket set means native everywhere (non-fatal).
_ENABLE_TABLE: dict[str, dict[int, dict[int, _EnableRow]]] = {
    "moondream3": {
        90: {
            132: _EnableRow(
                buckets=frozenset({1}),
                note=(
                    "H100 SXM: batch 1 is 1.38x (~1917us vs 2306us native). Batches "
                    "2/4 are ~1.0x and deferred (need a per-request KV copy). Batch 8 "
                    "is 0.86x -> native."
                ),
            ),
        },
    },
    "moondream2": {
        90: {
            132: _EnableRow(
                buckets=frozenset({1}),
                note=(
                    "H100 SXM: batch 1 is 1.48x at seqlen 16 (1200.85us vs 1777.25us "
                    "native) and 1.41x at seqlen 512 (1314.05us vs 1856.72us); cosine "
                    "0.9998, argmax matches native. Uses num_splits=2 (see "
                    "_config_for_model). Batches above 1 deferred (per-request KV copy)."
                ),
            ),
        },
    },
}


def _model_kind(model_name: str) -> Optional[str]:
    """Map an engine model name (e.g. 'moondream3-preview', 'moondream2') to the
    enable-table key, or ``None`` when the model has no megakernel."""
    name = (model_name or "").lower()
    if name.startswith("moondream3"):
        return "moondream3"
    if name.startswith("moondream2"):
        return "moondream2"
    return None


def _enabled_buckets(model_kind: str, arch: int, num_sms: int) -> frozenset[int]:
    """The decode buckets enabled for this exact ``(model, arch, grid size)``
    variant. Missing entry means an empty set (native)."""
    row = _ENABLE_TABLE.get(model_kind, {}).get(int(arch), {}).get(int(num_sms))
    if row is None:
        return frozenset()
    return frozenset(b for b in row.buckets if 1 <= int(b) <= _MAX_MEGAKERNEL_BUCKET)


class MegakernelDecodeManager:
    """Owns the whole-model decode megakernel sessions for one runtime.

    Built once at runtime init; holds at most one session per enabled bucket.
    Sessions are built lazily on the first eager forward for a bucket: the build
    compiles the kernels and allocates the persistent buffers, so it must run before
    ``torch.cuda.graph`` capture and never inside it. ``prepare`` (eager, once per
    step) drives that lazy build and the internal KV copy; ``decode`` (in the
    captured graph) only rebinds inputs and launches.

    All entry points are non-fatal: on any ineligibility, missing config, import
    failure, or ``MegakernelNotEligible`` the manager disables itself / returns
    ``None`` so the caller runs the native decode.
    """

    def __init__(
        self,
        *,
        model_name: str,
        text: Any,
        text_config: Any,
        device: torch.device,
        page_table: Any,
        max_seq_length: int,
    ) -> None:
        self._text = text
        self._tc = text_config
        self._device = device
        self._page_table = page_table
        self._max_seq_length = int(max_seq_length)
        self._sessions: dict[int, Any] = {}
        # Per-bucket state used to detect when a new sequence starts decoding: the
        # (batch_idx tuple, last input_pos tuple) the session last decoded. A changed
        # composition or a non-contiguous input_pos means the contiguous KV must be
        # re-copied before the next launch (see ``_needs_seed``).
        self._session_state: dict[int, tuple] = {}
        self._enabled_buckets: frozenset[int] = frozenset()
        self._num_ctas: Optional[int] = None
        self._bridge = None
        self._disabled_reason: Optional[str] = None

        self._model_kind = _model_kind(model_name)
        if self._model_kind is None:
            self._disabled_reason = f"model {model_name!r} has no megakernel"
            return
        if device.type != "cuda" or not torch.cuda.is_available():
            self._disabled_reason = "no CUDA device"
            return
        try:
            from kestrel_kernels import megakernel as bridge
        except Exception as exc:  # megakernel backend absent -> native decode
            self._disabled_reason = f"megakernel backend import failed: {exc}"
            return
        self._bridge = bridge

        props = torch.cuda.get_device_properties(device.index or 0)
        arch = props.major * 10
        num_ctas = _select_num_ctas(arch, props.multi_processor_count)
        if num_ctas is None:
            self._disabled_reason = (
                f"no shipped variant for sm{arch} / {props.multi_processor_count} SMs"
            )
            return
        self._num_ctas = int(num_ctas)
        self._enabled_buckets = _enabled_buckets(self._model_kind, arch, self._num_ctas)
        if not self._enabled_buckets:
            self._disabled_reason = (
                f"no buckets enabled for {self._model_kind}/sm{arch}/{self._num_ctas}"
            )
        else:
            logger.info(
                "megakernel decode enabled: %s sm%d ctas=%d buckets=%s",
                self._model_kind,
                arch,
                self._num_ctas,
                sorted(self._enabled_buckets),
            )

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_buckets) and self._bridge is not None

    def bucket_enabled(self, bucket: int) -> bool:
        """Whether decode bucket ``bucket`` routes through the megakernel."""
        return self.enabled and int(bucket) in self._enabled_buckets

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    # -- eager per-step hook (session build + KV copy) -------------------------------
    # ``prepare`` runs outside the captured graph, once per step, on the prep the
    # engine already runs between replays (and at the pre-capture warmup). It keeps
    # both the session lifecycle and the KV copy internal, so the engine's contract
    # is only: call ``prepare`` in the decode-step prep, call ``decode`` in the
    # captured forward, and write ``input_pos`` between replays.

    def prepare(
        self,
        bucket: int,
        *,
        batch_idx: torch.Tensor,
        input_pos_gpu: torch.Tensor,
    ) -> None:
        """Eager per-step hook: (1) lazily build the session for ``bucket`` on the
        first call (the pre-capture warmup -- compilation happens here, so this must
        never run under capture), and (2) copy this sequence's prefill KV into the
        contiguous buffer once when a new sequence takes over a row (see
        ``_needs_seed``). No-op for a non-enabled bucket or under CUDA-graph
        capture."""
        if not self.bucket_enabled(bucket):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        sess = self._ensure_session(bucket, batch_idx=batch_idx, input_pos_gpu=input_pos_gpu)
        if sess is None:
            return
        self._maybe_seed(bucket, sess, batch_idx=batch_idx, input_pos_gpu=input_pos_gpu)

    def _config_for_model(self) -> Optional[Any]:
        """The backend config for this model, or ``None`` for the default (MD3:
        qmma up-projection, streamed tensor-core down-projection, num_splits=1). MD2
        (dense, no MoE) uses num_splits=2 -- the KV split its batch-1 win was
        measured at (1.48x at seqlen 16); the MoE knobs are ignored."""
        if self._model_kind == "moondream2":
            return self._bridge.MegakernelDecodeConfig(num_splits=2)
        return None

    def _ensure_session(
        self,
        bucket: int,
        *,
        batch_idx: torch.Tensor,
        input_pos_gpu: torch.Tensor,
    ) -> Optional[Any]:
        """Build (once) and return the megakernel session for ``bucket``. Must run at
        the pre-capture warmup (compilation happens here). Any build failure or
        ineligibility disables the bucket and returns ``None`` (native fallback)."""
        sess = self._sessions.get(int(bucket))
        if sess is not None:
            return sess
        # History (prefill length) for the initial build. The captured graph is
        # position-agnostic, so this only sizes the single-position build; kv_capacity
        # covers the full sequence so one graph advances every step. The build-time
        # embedding is a zero placeholder -- decode() overwrites the resident embedding
        # buffer every step, so the placeholder never reaches an output.
        history = int(input_pos_gpu[0].item())
        seqlen = history + 1
        dec_embed0 = torch.zeros(
            (int(bucket), 1, self._tc.dim), device=self._device, dtype=torch.bfloat16)
        try:
            sess = self._bridge.build_decode_session(
                self._model_kind,
                self._text,
                self._tc,
                dec_embed0,
                self._page_table,
                batch_idx[: int(bucket)],
                batch_bucket=int(bucket),
                seqlen=int(seqlen),
                device=self._device,
                ctas_per_row=self._num_ctas,
                config=self._config_for_model(),
                kv_capacity=int(self._max_seq_length),
                enable_position_rebind=True,
            )
        except self._bridge.MegakernelNotEligible as exc:
            logger.info("megakernel bucket %d not eligible (%s); native", bucket, exc)
            self._enabled_buckets = self._enabled_buckets - {int(bucket)}
            return None
        except Exception as exc:  # non-fatal: any build failure -> native
            logger.warning("megakernel bucket %d build failed (%s); native", bucket, exc)
            self._enabled_buckets = self._enabled_buckets - {int(bucket)}
            return None
        self._sessions[int(bucket)] = sess
        return sess

    def _needs_seed(self, bucket: int, idx: tuple, pos: tuple) -> bool:
        """Whether the contiguous KV must be (re-)copied from the paged cache before
        the next launch. This happens when a new sequence starts decoding on a row,
        detected against the manager's own per-bucket state:

          * FIRST decode of a bucket (no recorded state) -- the just-prefilled history
            has not been copied into the contiguous buffer yet.
          * batch_idx composition CHANGED -- a different sequence now occupies a row,
            so its history (not the previous row's) must be copied.
          * input_pos DISCONTINUITY -- a row's position is not exactly last+1. During
            contiguous megakernel decode every row advances by exactly one per step
            (the megakernel appended the previous step's K/V), so anything else means
            this row just came off prefill (or restarted) and its history must be
            re-copied.

        A steady contiguous step (same rows, each +1) needs no copy: the buffer grows
        by one per step across graph replays."""
        prev = self._session_state.get(int(bucket))
        if prev is None:
            return True
        prev_idx, prev_pos = prev
        if prev_idx != idx:
            return True
        return any(p != pp + 1 for p, pp in zip(pos, prev_pos))

    def _maybe_seed(
        self,
        bucket: int,
        sess: Any,
        *,
        batch_idx: torch.Tensor,
        input_pos_gpu: torch.Tensor,
    ) -> None:
        """Copy the contiguous KV when a new sequence starts decoding, then record the
        state so the next step is recognized as a continuation."""
        b = int(bucket)
        idx = tuple(int(x) for x in batch_idx[:b].tolist())
        pos = tuple(int(x) for x in input_pos_gpu[:b].tolist())
        if self._needs_seed(b, idx, pos):
            # Batch-1 copy: a shared contiguous KV for a fixed batch composition,
            # history = the current position (uniform at batch 1). Batches above 1
            # with different per-row histories need a per-row copy (deferred); those
            # buckets are not enabled in the table yet.
            history = pos[0]
            sess.seed_kv_from_paged(self._page_table, batch_idx[:b], int(history))
        self._session_state[b] = (idx, pos)

    # -- the decode call (inside the captured graph) --------------------------------
    def decode(
        self,
        bucket: int,
        *,
        embeds: torch.Tensor,
        input_pos_gpu: torch.Tensor,
        lora_slot_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Run one decode step through the megakernel; return the hidden state
        ``[B, 1, H]`` or ``None`` (bucket not enabled / no session built => the caller
        runs native).

        Writes the token ``embeds`` and the engine's ``input_pos`` into the session's
        resident buffers (device copies, safe under CUDA-graph capture) and launches.
        The KV was already copied for the active sequence by ``prepare``."""
        sess = self._sessions.get(int(bucket))
        if sess is None:
            return None
        if lora_slot_ids is not None:
            sess.lora_route(lora_slot_ids)
        step = self._bridge.DecodeStep(embed=embeds, position=input_pos_gpu)
        return sess.decode(step)


__all__ = ["MegakernelDecodeManager"]
