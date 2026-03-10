from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseLoRATorchBatch:
    segment_slot_ids: torch.Tensor
    num_segments: int
    segment_len: int


@dataclass(frozen=True)
class DenseLoRATorchOpScratch:
    selected_a: torch.Tensor
    selected_b: torch.Tensor
    intermediate: torch.Tensor


@dataclass(frozen=True)
class DenseLoRATorchMLPScratch:
    up: DenseLoRATorchOpScratch
    down: DenseLoRATorchOpScratch


def create_mlp_scratch(
    *,
    max_segments: int,
    max_segment_len: int,
    max_rank: int,
    d_model: int,
    d_ffn: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DenseLoRATorchMLPScratch:
    if max_segments < 1:
        raise ValueError("max_segments must be >= 1")
    if max_segment_len < 1:
        raise ValueError("max_segment_len must be >= 1")
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")

    def make_op_scratch(input_dim: int, output_dim: int) -> DenseLoRATorchOpScratch:
        return DenseLoRATorchOpScratch(
            selected_a=torch.empty(
                (max_segments, max_rank, input_dim),
                dtype=dtype,
                device=device,
            ),
            selected_b=torch.empty(
                (max_segments, output_dim, max_rank),
                dtype=dtype,
                device=device,
            ),
            intermediate=torch.empty(
                (max_segments, max_segment_len, max_rank),
                dtype=dtype,
                device=device,
            ),
        )

    return DenseLoRATorchMLPScratch(
        up=make_op_scratch(d_model, d_ffn),
        down=make_op_scratch(d_ffn, d_model),
    )


def prepare_batch(
    lora_slot_ids: torch.Tensor,
    *,
    segment_len: int = 1,
) -> DenseLoRATorchBatch:
    slot_ids = lora_slot_ids.to(dtype=torch.long)

    return DenseLoRATorchBatch(
        segment_slot_ids=slot_ids,
        num_segments=int(slot_ids.shape[0]),
        segment_len=segment_len,
    )


def apply_batch(
    x: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    batch: DenseLoRATorchBatch,
    scratch: DenseLoRATorchOpScratch | None = None,
) -> None:
    if batch.num_segments == 0:
        return

    segment_len = batch.segment_len
    if x.shape[0] != batch.num_segments * segment_len:
        raise ValueError("x shape does not match prepared dense LoRA batch")

    hidden_dim = x.shape[1]
    out_dim = output.shape[1]

    x_segments = x.view(batch.num_segments, segment_len, hidden_dim)
    output_segments = output.view(batch.num_segments, segment_len, out_dim)

    if scratch is None:
        a_weights = torch.index_select(lora_a, 0, batch.segment_slot_ids)
        b_weights = torch.index_select(lora_b, 0, batch.segment_slot_ids)
        intermediate = torch.bmm(x_segments, a_weights.transpose(1, 2))
        output_segments.add_(torch.bmm(intermediate, b_weights.transpose(1, 2)))
        return

    if scratch.selected_a.shape[0] < batch.num_segments:
        raise ValueError("dense LoRA scratch does not have enough segment capacity")
    if scratch.selected_a.shape[1:] != lora_a.shape[1:]:
        raise ValueError("dense LoRA scratch selected_a shape does not match weights")
    if scratch.selected_b.shape[1:] != lora_b.shape[1:]:
        raise ValueError("dense LoRA scratch selected_b shape does not match weights")
    if scratch.intermediate.shape[0] < batch.num_segments:
        raise ValueError("dense LoRA scratch intermediate does not have enough segments")
    if scratch.intermediate.shape[1] < segment_len:
        raise ValueError("dense LoRA scratch intermediate does not have enough sequence capacity")
    if scratch.intermediate.shape[2] != lora_a.shape[1]:
        raise ValueError("dense LoRA scratch intermediate rank does not match weights")

    a_weights = scratch.selected_a[:batch.num_segments]
    b_weights = scratch.selected_b[:batch.num_segments]
    intermediate = scratch.intermediate[:batch.num_segments, :segment_len]

    torch.index_select(lora_a, 0, batch.segment_slot_ids, out=a_weights)
    torch.index_select(lora_b, 0, batch.segment_slot_ids, out=b_weights)
    torch.bmm(x_segments, a_weights.transpose(1, 2), out=intermediate)
    torch.baddbmm(
        output_segments,
        intermediate,
        b_weights.transpose(1, 2),
        beta=1.0,
        alpha=1.0,
        out=output_segments,
    )
