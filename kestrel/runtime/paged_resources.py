"""Small resource types shared by paged autoregressive runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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
    "decode_slot_rows",
]
