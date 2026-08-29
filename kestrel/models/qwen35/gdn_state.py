"""Recurrent-state storage for Qwen 3.5/3.6 GDN layers."""

from __future__ import annotations

import torch


class LinearAttentionState:
    """One convolution state and one authoritative BF16 recurrent state."""

    def __init__(self) -> None:
        self.conv_states: torch.Tensor | None = None
        self.recurrent_states: torch.Tensor | None = None
        self.has_previous_state = False

    def clear(self, row: int | None = None) -> None:
        for tensor in (self.conv_states, self.recurrent_states):
            if tensor is None:
                continue
            if row is None:
                tensor.zero_()
            else:
                tensor[int(row)].zero_()


__all__ = ["LinearAttentionState"]
