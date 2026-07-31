"""Shared resource graph for paged autoregressive runtimes."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

from kestrel.device import make_event, make_stream
from kestrel.kv_cache import PageTable
from kestrel.runtime.decode_slot import DecodeSlot, create_decode_slot


def decode_slot_rows(max_batch_size: int) -> int:
    max_batch_size = int(max_batch_size)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    return max(
        max_batch_size + 2,
        1 << (max_batch_size - 1).bit_length(),
    )


@dataclass
class PrefillSlot:
    slot_id: int
    batch_idx: torch.Tensor
    step_done_event: Any
    commit_done_event: Any
    scratch: Any = None


@dataclass
class PagedRuntimeResources:
    compute_stream: Any
    copy_stream: Any
    graph_capture_lock: threading.RLock
    page_table: PageTable
    prefill_slot: PrefillSlot
    decode_slots: tuple[DecodeSlot, ...]
    max_batch_slots: int
    decode_rows: int
    padding_batch_idx: int


def create_paged_runtime_resources(
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    page_size: int,
    kv_cache_pages: int,
    vocab_size: int,
    hidden_dim: int,
    position_shape_prefix: tuple[int, ...] = (),
    compute_stream: Any = None,
    num_decode_slots: int = 2,
) -> PagedRuntimeResources:
    max_batch_slots = max_batch_size + 2
    rows = decode_slot_rows(max_batch_size)
    padding_batch_idx = max_batch_slots - 1
    compute = compute_stream if compute_stream is not None else make_stream(device)
    copy = make_stream(device)
    page_table = PageTable(
        n_pages=kv_cache_pages,
        page_size=page_size,
        max_batch_size=max_batch_slots,
        device=str(device),
        prefix_cache=None,
        h2d_stream=compute,
    )
    page_table.free_batch_idx.remove(padding_batch_idx)
    page_table.reserve(padding_batch_idx, 1)
    page_table.commit_block_table([padding_batch_idx])
    prefill_slot = PrefillSlot(
        slot_id=0,
        batch_idx=torch.zeros(
            (max_batch_size,),
            dtype=torch.int64,
            device=device,
        ),
        step_done_event=make_event(device, enable_timing=False, blocking=False),
        commit_done_event=make_event(device, enable_timing=False, blocking=False),
    )
    position_shape = (*position_shape_prefix, rows, 1)
    slots = tuple(
        create_decode_slot(
            slot_id=index,
            device=device,
            dtype=dtype,
            max_batch_slots=rows,
            kv_cache_pages=kv_cache_pages,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            position_shape=position_shape,
            compute_stream=compute,
            copy_stream=copy,
        )
        for index in range(num_decode_slots)
    )
    return PagedRuntimeResources(
        compute_stream=compute,
        copy_stream=copy,
        graph_capture_lock=threading.RLock(),
        page_table=page_table,
        prefill_slot=prefill_slot,
        decode_slots=slots,
        max_batch_slots=max_batch_slots,
        decode_rows=rows,
        padding_batch_idx=padding_batch_idx,
    )


__all__ = [
    "PagedRuntimeResources",
    "PrefillSlot",
    "create_paged_runtime_resources",
    "decode_slot_rows",
]
