"""Engine-side integration of the whole-model decode megakernel (MD2 + MD3).

THE ENGINE SWITCH. This is the production seam that routes eligible decode steps
through the fused whole-model megakernel VM (one persistent CUDA launch for the
entire decode layer stack) in place of the native ``text_decoder`` forward. It is
the runtime counterpart of the research bridge ``mkl.megakernel.decode_bridge`` and
is governed by ``mkl/docs/MEGAKERNEL_SHIP_INTEGRATION.md``.

Design contract (all NON-FATAL -- any miss falls back to the native decode):

  * Selection key ``(model_kind, arch, num_sms, bucket)``.
      - Eligibility (a shipped variant fits the device) is decided HERE by
        ``_select_num_ctas(arch, device_sms)`` -- the largest same-arch shipped
        ``num_sms`` <= the device SM count (closest-<=; never oversubscribe a
        persistent grid). ``None`` => native. The deploy set (``_DEPLOY_SM_COUNTS``)
        is re-encoded here, not imported from the bridge/build tree, so the runtime
        never depends on either.
      - The perf on/off decision (the megakernel is not a universal win: SXM B1
        1.38x but B8 0.86x) is the typed ``_ENABLE_TABLE`` below, keyed by the same
        tuple. Env-free (the no-flags rule) and code-reviewed like code.

  * KV seam (paged -> contiguous), INTERNAL. The engine's KV is PAGED; the megakernel
    VM owns a CONTIGUOUS KV buffer that self-extends across decode steps. The manager
    detects each sequence's prefill->decode transition ITSELF (see ``prepare`` and
    ``_needs_seed`` below) and does a ONE-TIME copy of that sequence's prefill KV
    pages into the VM's contiguous ``kv_k8``/``kv_v8``. The megakernel then appends
    each step's K/V in place; the paged cache is untouched during megakernel decode.
    The engine never calls ``seed_kv`` -- it is a private consequence of writing
    ``input_pos``.

  * Per-step position rebind. The captured megakernel graph is position-agnostic: the
    engine writes ``slot.meta.input_pos.gpu`` into the session's resident ``input_pos``
    between graph replays and an in-graph gather refills RoPE cos/sin (+ MD3 TAU pos)
    and the KV scan length. ONE captured graph per bucket serves all steps.

  * LoRA. ``lora_slot_ids`` -> the VM's ``(slot, rt_rank)`` selection; rank-0 (no
    adapter) is byte-identical to the base decode.

The engine surface is intentionally tiny: construct the manager, write
``slot.meta.input_pos.gpu`` between replays (already the native decode contract), and
call ``decode(bucket, ...)`` inside the captured forward. Session build and KV seeding
are internal, driven off the ``prepare`` hook the engine already runs each step.

Scope of the shipped enable table: MD3 B1 on SXM (the measured >=1.0x, fingerprint-
gated bucket). MD2 is wired end-to-end; whether any MD2 bucket ships is a per-SKU perf
call recorded in ``_ENABLE_TABLE``. B>1 needs a per-request KV re-seed for arbitrary
batch compositions (staggered request starts) -- specced for a follow-up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Buckets the megakernel batch VM can serve at all (BATCH_TILE=8); above this the
# engine always runs native. The enable table further restricts within this set.
_MAX_MEGAKERNEL_BUCKET = 8

# Per-arch shipped persistent-grid variants: the ``num_sms`` the tape/grid are baked
# for. A variant is valid only when its ``num_sms`` <= the device SM count (an
# oversubscribed persistent grid deadlocks), and we ship ONE variant per distinct
# deploy SM count. Mirrors the build-side ``whole_model_eligible_num_sms``; re-encoded
# here (not imported) so the runtime never depends on the build tree.
_DEPLOY_SM_COUNTS: dict[int, tuple[int, ...]] = {
    90: (132, 114),  # 132: H100 SXM5/NVL, H200, GH200 ; 114: H100 PCIe
    100: (148,),  # 148: B200
    # 120: (188,) -- RTX PRO 6000 Blackwell Server: DEFERRED (native fallback).
}


def _select_num_ctas(arch: int, device_sms: int) -> Optional[int]:
    """The ``num_sms`` of the megakernel variant to run on this device, or ``None`` if
    the model is not megakernel-eligible here (-> the caller runs the NATIVE decode).

    Returns the LARGEST shipped same-arch ``num_sms`` <= ``device_sms`` (closest-<=:
    best SOL, never oversubscribe). ``None`` when no shipped count fits -- an SM count
    we haven't built yet, or an arch with no variants (sm120 today). NON-FATAL native
    fallback, not a deadlock."""
    candidates = [n for n in _DEPLOY_SM_COUNTS.get(int(arch), ()) if n <= int(device_sms)]
    return max(candidates) if candidates else None


@dataclass(frozen=True)
class _EnableRow:
    """One ``(model, arch, num_sms)`` row of the perf enable table: the decode buckets
    routed through the megakernel, plus the measurement note that justifies them."""

    buckets: frozenset[int] = field(default_factory=frozenset)
    note: str = ""


# Whole-model decode megakernel PERF enable table (config-as-code, env-free -- the
# no-flags rule; see mkl/docs/MEGAKERNEL_SHIP_INTEGRATION.md). Eligibility (a shipped
# variant fits the device) is ``_select_num_ctas``; this table is the *perf* on/off
# decision layered on top -- per (model_kind, arch, num_sms, bucket). The megakernel is
# NOT a universal win (p2/H100-SXM measured B1 1.38x but B8 0.86x), so only buckets
# that are >=1.0x AND fingerprint-gated are enabled here.
#
#   model_kind -> arch (SM major*10) -> variant num_sms -> _EnableRow
#
# ``num_sms`` is the shipped-variant grid the device matched closest-<= (132 = H100
# SXM5/NVL/H200/GH200). A missing model/arch/num_sms entry, or an empty bucket set =>
# native everywhere (non-fatal fallback).
_ENABLE_TABLE: dict[str, dict[int, dict[int, _EnableRow]]] = {
    "moondream3": {
        90: {
            132: _EnableRow(
                buckets=frozenset({1}),
                note=(
                    "SXM: B1 1.38x (~1917us vs 2306us native). B2/B4 ~1.0x deferred "
                    "(need B>1 per-request KV re-seed). B8 0.86x -> native."
                ),
            ),
        },
    },
    "moondream2": {
        90: {
            132: _EnableRow(
                buckets=frozenset({1}),
                note=(
                    "SXM: B1 1.48x @seqlen16 (1200.85us vs 1777.25us native), 1.41x "
                    "@seqlen512 (1314.05us vs 1856.72us); cos 0.9998, argmax matches "
                    "native. num_splits=2 (see _config_for_model). B>1 deferred (needs "
                    "per-request KV re-seed)."
                ),
            ),
        },
    },
}


def _model_kind(model_name: str) -> Optional[str]:
    """Map an engine model name (e.g. 'moondream3-preview', 'moondream2') onto the
    enable-table key, or ``None`` when the model is not megakernel-wired."""
    name = (model_name or "").lower()
    if name.startswith("moondream3"):
        return "moondream3"
    if name.startswith("moondream2"):
        return "moondream2"
    return None


def _enabled_buckets(model_kind: str, arch: int, num_sms: int) -> frozenset[int]:
    """The decode buckets enabled for this exact shipped ``(model, arch, num_sms)``
    variant. Missing entry => empty set => native."""
    row = _ENABLE_TABLE.get(model_kind, {}).get(int(arch), {}).get(int(num_sms))
    if row is None:
        return frozenset()
    return frozenset(b for b in row.buckets if 1 <= int(b) <= _MAX_MEGAKERNEL_BUCKET)


class MegakernelDecodeManager:
    """Owns the whole-model decode megakernel sessions for one runtime.

    Built once at runtime init; holds at most one megakernel session per enabled decode
    bucket. Sessions are built lazily on the first EAGER forward for a bucket (the
    pre-capture warmup) -- the build JIT-compiles the kernels + allocates the
    persistent/capture buffers, so it must run BEFORE ``torch.cuda.graph`` capture and
    never inside it. ``prepare`` (eager, once per step) drives that lazy build and the
    internal KV seed; ``decode`` (in the captured graph) only rebinds + launches.

    All public entry points are non-fatal: on any ineligibility, missing config, import
    failure, or ``MegakernelNotEligible`` they leave the manager disabled / return
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
        # Per-bucket session state used to detect a sequence's prefill->decode
        # transition: the (batch_idx tuple, last input_pos tuple) the session last
        # decoded. A new composition or a non-contiguous input_pos => re-seed the
        # contiguous KV before the next launch (see ``_needs_seed``).
        self._session_state: dict[int, tuple] = {}
        self._enabled_buckets: frozenset[int] = frozenset()
        self._num_ctas: Optional[int] = None
        self._bridge = None
        self._disabled_reason: Optional[str] = None

        self._model_kind = _model_kind(model_name)
        if self._model_kind is None:
            self._disabled_reason = f"model {model_name!r} not megakernel-wired"
            return
        if device.type != "cuda" or not torch.cuda.is_available():
            self._disabled_reason = "no CUDA device"
            return
        try:
            from mkl.megakernel import decode_bridge as bridge
        except Exception as exc:  # mkl is a build-time dep; absent on some hosts
            self._disabled_reason = f"decode bridge import failed: {exc}"
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

    # -- the single eager seam (session build + internal KV seed) --------------------
    # ``prepare`` runs OUTSIDE the captured graph, once per step, on the per-step prep
    # the engine ALREADY runs between replays (and at the pre-capture warmup). It keeps
    # both the session lifecycle and the KV seam INTERNAL, so the engine's whole
    # megakernel contract is: call ``prepare`` in the decode-step prep, call ``decode``
    # in the captured forward, and write ``input_pos`` between replays -- nothing else.

    def prepare(
        self,
        bucket: int,
        *,
        batch_idx: torch.Tensor,
        input_pos_gpu: torch.Tensor,
    ) -> None:
        """Eager per-step seam: (1) lazily builds the session for ``bucket`` on the first
        call (the pre-capture warmup -- JIT compiles here, so this must never run under
        capture), and (2) detects this sequence's prefill->decode transition and re-seeds
        the contiguous KV once when a new sequence takes over a row (see ``_needs_seed``).
        No-op for a non-enabled bucket or under CUDA-graph capture."""
        if not self.bucket_enabled(bucket):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        sess = self._ensure_session(bucket, batch_idx=batch_idx, input_pos_gpu=input_pos_gpu)
        if sess is None:
            return
        self._maybe_seed(bucket, sess, batch_idx=batch_idx, input_pos_gpu=input_pos_gpu)

    def _config_for_model(self) -> Optional[Any]:
        """The bridge ``MegakernelDecodeConfig`` for this model, or ``None`` for the
        bridge default (the MD3 canonical: qmma-up + streamed tensor-core down,
        num_splits=1). MD2 (dense, no MoE) ships at num_splits=2 -- the flash-decode KV
        split its B1 win was measured at (1.48x @ seqlen16); the MoE knobs are ignored."""
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
        """Build (once) and return the megakernel session for ``bucket``. MUST run at the
        pre-capture warmup (JIT compiles here). Any build failure or ineligibility
        disables the bucket and returns ``None`` (native fallback)."""
        sess = self._sessions.get(int(bucket))
        if sess is not None:
            return sess
        # History (prefill length) for the initial build. The captured graph is
        # position-agnostic, so this only sizes the single-position seed; kv_capacity
        # covers the full sequence so one graph advances every step. The build-time
        # embedding is a zero placeholder -- the resident x0 is overwritten every step by
        # decode()'s embed rebind, so the build's x0 content never reaches an output.
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
        """Whether the contiguous KV must be (re-)seeded from the paged cache before the
        next launch. The trigger is a prefill->decode transition, detected against the
        manager's per-bucket session state WITHOUT assuming the engine tells us:

          * FIRST decode of a bucket (no recorded state) -- the just-prefilled history
            has never been copied into the contiguous KV.
          * batch_idx composition CHANGED -- a different sequence now occupies a row,
            so its history (not the previous row's) must be seeded.
          * input_pos DISCONTINUITY -- a row's position is not exactly last+1. A
            contiguous megakernel decode advances every row by exactly 1 per step (the
            VM appended the previous step's K/V in place); anything else means this row
            just came off prefill (or restarted), so its paged history must be re-seeded.

        A steady contiguous step (same rows, each +1) needs NO copy: the cache
        self-extends across graph replays."""
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
        """Seed the contiguous KV on a detected transition, then record the state so the
        next step is recognized as a contiguous continuation."""
        b = int(bucket)
        idx = tuple(int(x) for x in batch_idx[:b].tolist())
        pos = tuple(int(x) for x in input_pos_gpu[:b].tolist())
        if self._needs_seed(b, idx, pos):
            # B1-scoped seed: a shared contiguous KV for a fixed batch composition,
            # history = the (uniform, at B1) current position. B>1 with staggered
            # per-row histories needs a per-row seed (deferred; see the ship doc); those
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
        """Run one decode step through the megakernel; return hidden ``[B, 1, H]`` or
        ``None`` (bucket not enabled / no session built => caller runs native).

        Writes the token ``embeds`` + the engine's ``input_pos`` into the session's
        resident buffers (device copies, CUDA-graph capturable) and launches. The KV was
        already seeded for the active sequence by ``prepare`` (the internal seam)."""
        sess = self._sessions.get(int(bucket))
        if sess is None:
            return None
        if lora_slot_ids is not None:
            sess.lora_route(lora_slot_ids)
        step = self._bridge.DecodeStep(embed=embeds, position=input_pos_gpu)
        return sess.decode(step)


__all__ = ["MegakernelDecodeManager"]
