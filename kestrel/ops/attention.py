"""Model-independent attention tensor operations."""

from __future__ import annotations

import torch


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand grouped K/V heads without materializing the broadcast dimension."""
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        kv_heads,
        repeats,
        sequence_length,
        head_dim,
    )
    return expanded.reshape(
        batch,
        kv_heads * repeats,
        sequence_length,
        head_dim,
    )


__all__ = ["repeat_kv"]
