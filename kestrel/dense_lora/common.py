from __future__ import annotations

from dataclasses import dataclass

import torch

from .torch_backend import (
    DenseLoRATorchBatch,
    DenseLoRATorchOpScratch,
    apply_batch,
    prepare_batch,
)


@dataclass(frozen=True)
class DenseLoRAPreparedBatch:
    state: DenseLoRATorchBatch
    num_tokens: int


def prepare_dense_lora_batch(
    lora_slot_ids: torch.Tensor,
    *,
    segment_len: int = 1,
) -> DenseLoRAPreparedBatch:
    if lora_slot_ids.ndim != 1:
        raise ValueError("lora_slot_ids must have shape [num_segments]")
    if segment_len < 1:
        raise ValueError("segment_len must be >= 1")

    state = prepare_batch(
        lora_slot_ids,
        segment_len=segment_len,
    )
    return DenseLoRAPreparedBatch(
        state=state,
        num_tokens=int(lora_slot_ids.shape[0]) * segment_len,
    )


def apply_dense_lora(
    x: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor | None = None,
    *,
    prepared_batch: DenseLoRAPreparedBatch | None = None,
    scratch: DenseLoRATorchOpScratch | None = None,
) -> None:
    if x.ndim != 2:
        raise ValueError("x must have shape [num_tokens, hidden_dim]")
    if output.ndim != 2:
        raise ValueError("output must have shape [num_tokens, out_dim]")
    if lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError("LoRA weights must have shape [max_slots, ..., rank]")
    if lora_a.shape[0] != lora_b.shape[0]:
        raise ValueError("LoRA A/B must agree on max_slots")
    if lora_a.shape[1] != lora_b.shape[2]:
        raise ValueError("LoRA A/B must agree on rank")
    if lora_a.shape[2] != x.shape[1]:
        raise ValueError("LoRA A hidden_dim must match x")
    if lora_b.shape[1] != output.shape[1]:
        raise ValueError("LoRA B out_dim must match output")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if x.numel() == 0:
        return

    if x.shape[0] != output.shape[0]:
        raise ValueError("x and output must agree on num_tokens")

    x = x.contiguous()
    lora_a = lora_a.contiguous()
    lora_b = lora_b.contiguous()

    if prepared_batch is None:
        if lora_slot_ids is None:
            raise ValueError("lora_slot_ids is required when prepared_batch is not provided")
        prepared_batch = prepare_dense_lora_batch(
            lora_slot_ids,
        )
    elif x.shape[0] != prepared_batch.num_tokens:
        raise ValueError("prepared_batch num_tokens must match x/output")

    apply_batch(
        x,
        output,
        lora_a,
        lora_b,
        prepared_batch.state,
        scratch=scratch,
    )
