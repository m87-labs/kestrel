"""Per-slot decode resources for Gemma 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


class GemmaDecodeMetaBuffers:
    """Per-step decode inputs packed into one staging buffer.

    ``batch_idx`` / ``input_pos`` / ``lora_slot_ids`` share a single
    :class:`PackedBuffer`, so the scheduler stages them with one H2D copy
    (``copy_inputs_to_gpu``) instead of three. They are exposed as
    :class:`PackedField` views, so the runtime's graph-capture zeroing and
    forward paths read/write ``.np``/``.cpu``/``.gpu`` exactly as before.
    """

    def __init__(
        self, *, max_batch_slots: int, device: torch.device, pin_memory: bool
    ) -> None:
        from kestrel.utils import PackedBuffer

        self._inputs = PackedBuffer(
            [
                ("batch_idx", (max_batch_slots,), torch.int64),
                ("input_pos", (max_batch_slots,), torch.int32),
                ("lora_slot_ids", (max_batch_slots,), torch.int32),
            ],
            device=device,
            pin_memory=pin_memory,
        )

    @property
    def batch_idx(self) -> Any:
        return self._inputs.batch_idx

    @property
    def input_pos(self) -> Any:
        return self._inputs.input_pos

    @property
    def lora_slot_ids(self) -> Any:
        return self._inputs.lora_slot_ids

    def copy_inputs_to_gpu(self) -> None:
        """Stage batch_idx/input_pos/lora_slot_ids to the GPU in one H2D copy."""
        self._inputs.copy_to_gpu()


@dataclass
class GemmaDecodeSlot:
    slot_id: int
    meta: GemmaDecodeMetaBuffers
    compute_stream: Any
    paged_kv_page_table: Tensor
    paged_kv_seqlens_k: Tensor
    slot_mapping: Tensor
    cache_position_ids: Tensor
    position_ids: Tensor
    sampled_ids: Tensor
    sampled_logprobs: Tensor
    logits: Tensor
    hidden_last: Tensor
    decode_token_ids: Tensor
    render: Any
    step_done_event: Any
    commit_done_event: Any
    disallow_mask: Any
    mask_ready_event: Any


def create_gemma_decode_slot(
    slot_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_slots: int,
    kv_cache_pages: int,
    vocab_size: int,
    hidden_dim: int,
    compute_stream: Any,
    copy_stream: Any,
) -> GemmaDecodeSlot:
    from kestrel.device import make_event
    from kestrel.scheduler.transfer import RenderBuffer
    from kestrel.utils import CpuGpuBuffer

    pin = device.type == "cuda"
    meta = GemmaDecodeMetaBuffers(
        max_batch_slots=max_batch_slots, device=device, pin_memory=pin
    )

    return GemmaDecodeSlot(
        slot_id=slot_id,
        meta=meta,
        compute_stream=compute_stream,
        paged_kv_page_table=torch.empty(
            (max_batch_slots, kv_cache_pages),
            dtype=torch.int32,
            device=device,
        ),
        paged_kv_seqlens_k=torch.empty(
            (max_batch_slots,),
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping=torch.empty(
            (max_batch_slots, 1),
            dtype=torch.long,
            device=device,
        ),
        cache_position_ids=torch.empty(
            (max_batch_slots, 1),
            dtype=torch.long,
            device=device,
        ),
        position_ids=torch.empty(
            (max_batch_slots, 1),
            dtype=torch.long,
            device=device,
        ),
        sampled_ids=torch.empty((max_batch_slots,), dtype=torch.long, device=device),
        sampled_logprobs=torch.empty(
            (max_batch_slots,), dtype=torch.float32, device=device
        ),
        logits=torch.empty((max_batch_slots, vocab_size), dtype=dtype, device=device),
        hidden_last=torch.empty(
            (max_batch_slots, hidden_dim), dtype=dtype, device=device
        ),
        decode_token_ids=torch.empty(
            (max_batch_slots,), dtype=torch.long, device=device
        ),
        render=RenderBuffer(max_batch_slots, device, copy_stream=copy_stream),
        step_done_event=make_event(device, enable_timing=False, blocking=False),
        commit_done_event=make_event(device, enable_timing=False, blocking=False),
        disallow_mask=CpuGpuBuffer(
            max_batch_slots,
            vocab_size,
            dtype=torch.bool,
            device=device,
            pin_memory=pin,
        ),
        mask_ready_event=make_event(device, enable_timing=False, blocking=False),
    )


__all__ = [
    "GemmaDecodeMetaBuffers",
    "GemmaDecodeSlot",
    "create_gemma_decode_slot",
]
