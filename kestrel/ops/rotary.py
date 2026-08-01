"""Model-independent rotary embedding tensor operations."""

from __future__ import annotations

import torch


def default_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct standard RoPE inverse frequencies."""
    rotated_dim = int(head_dim * partial_rotary_factor)
    exponents = (
        torch.arange(
            0,
            rotated_dim,
            2,
            dtype=torch.int64,
            device=device,
        ).float()
        / rotated_dim
    )
    return (1.0 / base**exponents) / float(factor)


__all__ = ["default_inv_freq"]
