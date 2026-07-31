"""Model-independent normalization modules."""

from __future__ import annotations

import torch
from kestrel_kernels import get_runtime
from torch import nn


_rmsnorm = get_runtime().dense.rmsnorm


class RMSNorm(nn.Module):
    """RMS normalization with native CUDA/MPS dispatch and an exact fallback."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        with_scale: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        weight = torch.ones(dim, dtype=torch.float32)
        if with_scale:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rmsnorm(hidden_states, self.weight, self.eps)


__all__ = ["RMSNorm"]
