"""Model-agnostic resources for autoregressive decode slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from kestrel.device import make_event
from kestrel.scheduler.transfer import RenderBuffer
from kestrel.utils import CpuGpuBuffer, PackedBuffer


class DecodeMetaBuffers:
    """Scheduler-owned per-step inputs staged by one packed H2D copy."""

    def __init__(
        self, *, max_batch_slots: int, device: torch.device, pin_memory: bool
    ) -> None:
        self.inputs = PackedBuffer(
            [
                ("batch_idx", (max_batch_slots,), torch.int64),
                ("input_pos", (max_batch_slots,), torch.int32),
                ("lora_slot_ids", (max_batch_slots,), torch.int32),
            ],
            device=device,
            pin_memory=pin_memory,
        )
        self.batch_idx = self.inputs.batch_idx
        self.input_pos = self.inputs.input_pos
        self.lora_slot_ids = self.inputs.lora_slot_ids


@dataclass
class DecodeSlot:
    slot_id: int
    meta: DecodeMetaBuffers
    compute_stream: Any
    paged_kv_page_table: Tensor
    paged_kv_seqlens_k: Tensor
    slot_mapping: Tensor
    cache_position_ids: Tensor
    position_ids: Tensor
    scratch: dict[str, Tensor]
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


def create_decode_slot(
    slot_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_slots: int,
    kv_cache_pages: int,
    vocab_size: int,
    hidden_dim: int,
    position_shape: tuple[int, ...],
    scratch_specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]] | None = None,
    compute_stream: Any,
    copy_stream: Any,
) -> DecodeSlot:
    """Allocate the scheduler/runtime ABI from model-derived dimensions."""

    pin = device.type == "cuda"
    return DecodeSlot(
        slot_id=slot_id,
        meta=DecodeMetaBuffers(
            max_batch_slots=max_batch_slots,
            device=device,
            pin_memory=pin,
        ),
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
        position_ids=torch.empty(position_shape, dtype=torch.long, device=device),
        scratch={
            name: torch.empty(shape, dtype=tensor_dtype, device=device)
            for name, (shape, tensor_dtype) in (scratch_specs or {}).items()
        },
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


__all__ = ["DecodeMetaBuffers", "DecodeSlot", "create_decode_slot"]
