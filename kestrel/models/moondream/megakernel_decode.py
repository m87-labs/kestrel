"""Engine-side integration of the whole-model decode megakernel (MD3).

THE ENGINE SWITCH. This is the production seam that routes eligible decode steps
through the fused whole-model megakernel VM (one persistent CUDA launch for the
entire decode layer stack) in place of the native ``text_decoder`` forward. It is
the runtime counterpart of the research bridge ``mkl.megakernel.decode_bridge``
and is governed by ``mkl/docs/MEGAKERNEL_SHIP_INTEGRATION.md``.

Design contract (all NON-FATAL -- any miss falls back to the native decode):

  * Selection key ``(model_kind, arch, device_sms, bucket)``.
      - Eligibility (a shipped variant fits the device) comes from
        ``select_megakernel_num_ctas(arch, device_sms)`` in the bridge (mirrors the
        AOT bundle family's ``whole_model_eligible_num_sms``: sm90 -> 132/114,
        sm100 -> 148; sm120 deferred). ``None`` => native.
      - The perf on/off decision (the megakernel is not a universal win: SXM B1
        1.38x but B8 0.86x) is a config table, ``megakernel_enable.json``, keyed by
        the same tuple. Env-free (the no-flags rule).

  * KV seam (paged -> contiguous). The engine's KV is PAGED; the megakernel VM owns
    a CONTIGUOUS KV buffer that self-extends across decode steps. So at each
    sequence's prefill->decode transition we do a ONE-TIME copy of that sequence's
    prefill KV pages into the VM's contiguous ``kv_k8``/``kv_v8`` (``seed_kv``,
    below -- the bridge's ``seed_kv_from_paged``). The megakernel then appends each
    step's K/V in place; the paged cache is untouched during megakernel decode.

  * Per-step position rebind. The captured megakernel graph is position-agnostic
    (bridge increment 3): the engine writes ``slot.meta.input_pos.gpu`` into the
    session's resident ``input_pos`` between graph replays and an in-graph gather
    refills RoPE cos/sin, TAU pos, and the KV scan length. ONE captured graph per
    bucket serves all steps.

  * LoRA. ``lora_slot_ids`` -> the VM's ``(slot, rt_rank)`` MoeLoraState seam
    (``lora_route``); rank-0 (no adapter) is byte-identical to the base decode.

Scope of THIS increment (the shipped slice): the wiring + selection + KV seam +
per-step rebind + non-fatal fallback, with the enable table shipping **B1-only** on
SXM (the measured >=1.0x bucket). B>1 needs a per-request KV re-seed for arbitrary
batch compositions (staggered request starts) -- specced for the production PR.
"""
from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Buckets the megakernel batch VM can serve at all (BATCH_TILE=8); above this the
# engine always runs native. The enable table further restricts within this set.
_MAX_MEGAKERNEL_BUCKET = 8


def _model_kind(model_name: str) -> Optional[str]:
    """Map an engine model name (e.g. 'moondream3-preview', 'moondream3.1-9B-A2B')
    onto the enable-table key. Only MD3 (MoE) is wired today; MD2 returns None until
    its VM spec ships."""
    name = (model_name or "").lower()
    if name.startswith("moondream3"):
        return "moondream3"
    return None


def _load_enable_table() -> dict:
    """Load the committed per-(model, arch, device_sms, bucket) enable table. Any
    load failure => empty table => native everywhere (non-fatal)."""
    try:
        with resources.files(__package__).joinpath("megakernel_enable.json").open(
            "r", encoding="utf-8"
        ) as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - non-fatal config load
        logger.warning("megakernel enable table load failed (%s); native decode", exc)
        return {}


def _enabled_buckets(
    table: dict, model_kind: str, arch: int, device_sms: int
) -> frozenset[int]:
    """The set of decode buckets enabled for this (model, arch, device_sms). The
    ``device_sms`` sub-key is the shipped variant num_sms (matched closest-<= by the
    caller via ``select_megakernel_num_ctas``), so this reads the exact shipped
    variant's row. Missing entry => empty set => native."""
    try:
        entry = table[model_kind][str(int(arch))][str(int(device_sms))]
        buckets = {int(b) for b in entry.get("buckets", [])}
        return frozenset(b for b in buckets if 1 <= b <= _MAX_MEGAKERNEL_BUCKET)
    except Exception:
        return frozenset()


class MegakernelDecodeManager:
    """Owns the whole-model decode megakernel sessions for one runtime.

    Built once at runtime init; holds at most one ``Md3MegakernelDecode`` session
    per enabled decode bucket. Sessions are built lazily (``ensure_session``) at the
    pre-capture warmup for a bucket -- the build JIT-compiles the kernels + allocates
    the persistent/capture buffers, so it must run BEFORE ``torch.cuda.graph``
    capture and never inside it.

    All public entry points are non-fatal: on any ineligibility, missing config,
    import failure, or ``MegakernelNotEligible`` they leave the manager disabled /
    return ``None`` so the caller runs the native decode.
    """

    def __init__(
        self,
        *,
        model_name: str,
        text: Any,
        text_config: Any,
        device: torch.device,
        page_table: Any,
    ) -> None:
        self._text = text
        self._tc = text_config
        self._device = device
        self._page_table = page_table
        self._sessions: dict[int, Any] = {}
        # Per-bucket last-seeded (batch_idx tuple, history) so we re-seed the
        # contiguous KV only when the active sequence(s) change (B1 today).
        self._seed_key: dict[int, tuple] = {}
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
        num_ctas = bridge.select_megakernel_num_ctas(arch, props.multi_processor_count)
        if num_ctas is None:
            self._disabled_reason = (
                f"no shipped variant for sm{arch} / {props.multi_processor_count} SMs"
            )
            return
        self._num_ctas = int(num_ctas)
        table = _load_enable_table()
        self._enabled_buckets = _enabled_buckets(
            table, self._model_kind, arch, self._num_ctas
        )
        if not self._enabled_buckets:
            self._disabled_reason = (
                f"no buckets enabled for {self._model_kind}/sm{arch}/{self._num_ctas}"
            )
        else:
            logger.info(
                "megakernel decode enabled: %s sm%d ctas=%d buckets=%s",
                self._model_kind, arch, self._num_ctas, sorted(self._enabled_buckets),
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

    # -- session lifecycle -------------------------------------------------------
    def ensure_session(
        self,
        bucket: int,
        *,
        dec_embed0: torch.Tensor,
        batch_idx: torch.Tensor,
        seqlen: int,
        kv_capacity: int,
        config: Any = None,
    ) -> Optional[Any]:
        """Build (once) and return the megakernel session for ``bucket``, or ``None``
        if the bucket is not enabled / build failed. MUST be called at the
        pre-capture warmup (JIT compiles here); never inside graph capture."""
        if not self.bucket_enabled(bucket):
            return None
        sess = self._sessions.get(int(bucket))
        if sess is not None:
            return sess
        try:
            sess = self._bridge.Md3MegakernelDecode.build(
                self._text, self._tc, dec_embed0, self._page_table, batch_idx,
                batch_bucket=int(bucket), seqlen=int(seqlen), device=self._device,
                ctas_per_row=self._num_ctas, config=config,
                kv_capacity=int(kv_capacity), enable_position_rebind=True,
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

    def seed_kv(
        self,
        bucket: int,
        *,
        batch_idx: torch.Tensor,
        history: int,
    ) -> None:
        """Copy the active sequence(s)' prefill KV from the engine's PAGED cache into
        the session's CONTIGUOUS ``kv_k8``/``kv_v8`` and reset the position base. A
        ONE-TIME copy per sequence-decode-start (NOT per step); de-duplicated by
        ``_seed_key`` so a re-decode of the same (batch_idx, history) is a no-op.

        B1-scoped today: a shared contiguous KV for a fixed batch composition. B>1
        with staggered request starts needs per-row re-seed (see the module docstring
        / the ship doc); those buckets are not enabled in the table yet."""
        sess = self._sessions.get(int(bucket))
        if sess is None:
            return
        key = (tuple(int(x) for x in batch_idx.tolist()), int(history))
        if self._seed_key.get(int(bucket)) == key:
            return
        sess.seed_kv_from_paged(self._page_table, batch_idx, int(history))
        self._seed_key[int(bucket)] = key

    def lora_route(self, bucket: int, lora_slot_ids: Optional[torch.Tensor]) -> None:
        """Map the engine's per-row ``lora_slot_ids`` onto the VM's ``(slot,
        rt_rank)`` MoeLoraState selection. No-op when the session has no LoRA compiled
        (``moe_lora_rank == 0``), which is byte-identical to the base decode."""
        sess = self._sessions.get(int(bucket))
        if sess is None or lora_slot_ids is None:
            return
        route = getattr(sess, "lora_route", None)
        if route is not None:
            route(lora_slot_ids)

    # -- the decode call ---------------------------------------------------------
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
        resident buffers (device copies, CUDA-graph capturable) and launches. The KV
        must already be seeded for the active sequence (see ``seed_kv``)."""
        sess = self._sessions.get(int(bucket))
        if sess is None:
            return None
        self.lora_route(bucket, lora_slot_ids)
        step = self._bridge.DecodeStep(embed=embeds, position=input_pos_gpu)
        return sess.decode(step)


__all__ = ["MegakernelDecodeManager"]
