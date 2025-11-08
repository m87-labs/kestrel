from __future__ import annotations

import torch
import torch.nn as nn


class ExpertWeights(nn.Module):
    """Simple container for per-expert weight tensors."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size, dtype=dtype)
        )


__all__ = ["ExpertWeights"]
