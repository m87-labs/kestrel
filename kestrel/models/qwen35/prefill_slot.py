"""Per-slot prefill resources for Qwen 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kestrel.utils import CpuGpuBuffer, PackedBuffer


@dataclass
class Qwen35PrefillScratch:
    token_capacity: int
    batch_capacity: int
    kv_cache_pages: int
    dtype: torch.dtype
    device: torch.device
    # All per-token and per-batch text metadata lives in one packed staging
    # buffer so it stages to the GPU in a single H2D copy. Access the individual
    # fields through ``text_meta`` (e.g. ``text_meta.input_ids.np``); each is a
    # named view exposing the CpuGpuBuffer-style ``.np``/``.cpu``/``.gpu``
    # surface the fill code uses. Field set:
    #   input_ids, cache_position_ids, seq_idx, mm_token_type_ids, position_ids,
    #   slot_mapping, batch_indices, last_positions, last_token_offsets,
    #   cu_seq_lens_q, rope_deltas
    text_meta: PackedBuffer
    paged_kv_page_table: Tensor
    paged_kv_seqlens_k: Tensor
    pixel_values: CpuGpuBuffer | None = None
    pixel_capacity: int = 0
    pixel_dim: int = 0
    image_grid_thw: CpuGpuBuffer | None = None
    image_grid_capacity: int = 0
    vision_bilinear_indices: CpuGpuBuffer | None = None
    vision_bilinear_weights: CpuGpuBuffer | None = None
    vision_position_ids: CpuGpuBuffer | None = None
    vision_cu_seqlens: CpuGpuBuffer | None = None
    vision_token_capacity: int = 0
    vision_sequence_capacity: int = 0

    @classmethod
    def create(
        cls,
        *,
        token_capacity: int,
        batch_capacity: int,
        kv_cache_pages: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Qwen35PrefillScratch":
        pin = device.type == "cuda"
        token_capacity = max(1, int(token_capacity))
        batch_capacity = max(1, int(batch_capacity))
        text_meta = PackedBuffer(
            [
                ("input_ids", (1, token_capacity), torch.long),
                ("cache_position_ids", (1, token_capacity), torch.long),
                ("seq_idx", (1, token_capacity), torch.int32),
                ("mm_token_type_ids", (1, token_capacity), torch.int32),
                ("position_ids", (3, 1, token_capacity), torch.long),
                ("slot_mapping", (1, token_capacity), torch.int64),
                ("batch_indices", (batch_capacity,), torch.int64),
                ("last_positions", (batch_capacity,), torch.int32),
                ("last_token_offsets", (batch_capacity,), torch.long),
                ("cu_seq_lens_q", (batch_capacity + 1,), torch.int32),
                ("rope_deltas", (batch_capacity, 1), torch.long),
            ],
            device=device,
            pin_memory=pin,
        )
        return cls(
            token_capacity=token_capacity,
            batch_capacity=batch_capacity,
            kv_cache_pages=kv_cache_pages,
            dtype=dtype,
            device=device,
            text_meta=text_meta,
            paged_kv_page_table=torch.empty(
                (batch_capacity, kv_cache_pages),
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_seqlens_k=torch.empty(
                (batch_capacity,),
                dtype=torch.int32,
                device=device,
            ),
        )

    def ensure_size(
        self,
        *,
        total_tokens: int,
        batch_size: int,
        pixel_rows: int,
        pixel_dim: int,
        image_grid_rows: int,
        vision_sequence_count: int = 0,
    ) -> "Qwen35PrefillScratch":
        token_capacity = self.token_capacity
        while token_capacity < total_tokens:
            token_capacity *= 2

        batch_capacity = self.batch_capacity
        while batch_capacity < batch_size:
            batch_capacity *= 2

        if token_capacity != self.token_capacity or batch_capacity != self.batch_capacity:
            return Qwen35PrefillScratch.create(
                token_capacity=token_capacity,
                batch_capacity=batch_capacity,
                kv_cache_pages=self.kv_cache_pages,
                dtype=self.dtype,
                device=self.device,
            ).ensure_size(
                total_tokens=total_tokens,
                batch_size=batch_size,
                pixel_rows=pixel_rows,
                pixel_dim=pixel_dim,
                image_grid_rows=image_grid_rows,
                vision_sequence_count=vision_sequence_count,
            )

        if pixel_rows > self.pixel_capacity or (
            pixel_rows > 0 and pixel_dim != self.pixel_dim
        ):
            self.pixel_capacity = max(1, pixel_rows)
            self.pixel_dim = pixel_dim
            self.pixel_values = CpuGpuBuffer(
                self.pixel_capacity,
                pixel_dim,
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.device.type == "cuda",
                with_numpy=False,
            )

        if image_grid_rows > self.image_grid_capacity:
            self.image_grid_capacity = max(1, image_grid_rows)
            self.image_grid_thw = CpuGpuBuffer(
                self.image_grid_capacity,
                3,
                dtype=torch.long,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )

        if pixel_rows > self.vision_token_capacity:
            self.vision_token_capacity = max(1, pixel_rows)
            self.vision_bilinear_indices = CpuGpuBuffer(
                4,
                self.vision_token_capacity,
                dtype=torch.long,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )
            self.vision_bilinear_weights = CpuGpuBuffer(
                4,
                self.vision_token_capacity,
                dtype=torch.float32,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )
            self.vision_position_ids = CpuGpuBuffer(
                self.vision_token_capacity,
                2,
                dtype=torch.long,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )

        if vision_sequence_count + 1 > self.vision_sequence_capacity:
            self.vision_sequence_capacity = max(1, vision_sequence_count + 1)
            self.vision_cu_seqlens = CpuGpuBuffer(
                self.vision_sequence_capacity,
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )

        return self


@dataclass
class Qwen35PrefillSlot:
    slot_id: int
    batch_idx: Tensor
    step_done_event: Any
    step_done_event_pending: bool
    commit_done_event: Any
    scratch: Qwen35PrefillScratch


def create_qwen35_prefill_slot(
    slot_id: int,
    *,
    device: torch.device,
    max_batch_size: int,
    kv_cache_pages: int,
    dtype: torch.dtype,
    token_capacity: int,
) -> Qwen35PrefillSlot:
    from kestrel.device import make_event

    return Qwen35PrefillSlot(
        slot_id=slot_id,
        batch_idx=torch.zeros((max_batch_size,), dtype=torch.int64, device=device),
        step_done_event=make_event(device, enable_timing=False, blocking=False),
        step_done_event_pending=False,
        commit_done_event=make_event(device, enable_timing=False, blocking=False),
        scratch=Qwen35PrefillScratch.create(
            token_capacity=token_capacity,
            batch_capacity=max_batch_size,
            kv_cache_pages=kv_cache_pages,
            dtype=dtype,
            device=device,
        ),
    )


__all__ = [
    "Qwen35PrefillScratch",
    "Qwen35PrefillSlot",
    "create_qwen35_prefill_slot",
]
