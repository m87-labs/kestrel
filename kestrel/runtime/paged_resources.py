"""Small resource types shared by paged autoregressive runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def bound_kv_cache_pages(
    requested_pages: int,
    *,
    page_size: int,
    max_batch_size: int,
    max_seq_length: int,
) -> int:
    """Cap physical KV pages at the runtime's maximum reachable capacity."""

    requested_pages = int(requested_pages)
    page_size = int(page_size)
    max_batch_size = int(max_batch_size)
    max_seq_length = int(max_seq_length)
    if requested_pages < 1:
        raise ValueError("requested_pages must be positive")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")

    pages_per_sequence = (max_seq_length + page_size - 1) // page_size
    # PageTable reserves physical page 0, and these runtimes reserve one
    # additional page for their padding batch row.
    reachable_pages = 2 + max_batch_size * pages_per_sequence
    return min(requested_pages, reachable_pages)


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


__all__ = [
    "PrefillSlot",
    "bound_kv_cache_pages",
    "decode_slot_rows",
]
