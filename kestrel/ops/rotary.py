"""Model-independent rotary embedding operations."""

from __future__ import annotations

import torch
from torch import nn


def _validate_schedule(head_dim: int, base: float, partial: float, factor: float) -> None:
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("RoPE head dimension must be positive and even")
    if not 0.0 < partial <= 1.0 or base <= 0.0 or factor <= 0.0:
        raise ValueError("RoPE partial factor, base, and scaling must be positive")


def default_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct standard RoPE inverse frequencies."""
    _validate_schedule(head_dim, base, partial_rotary_factor, factor)
    rotated_dim = int(head_dim * partial_rotary_factor)
    if rotated_dim <= 0 or rotated_dim % 2:
        raise ValueError(f"partial RoPE dimension must be positive and even: {rotated_dim}")
    exponents = torch.arange(
        0, rotated_dim, 2, dtype=torch.int64, device=device
    ).float() / rotated_dim
    return (1.0 / base**exponents) / float(factor)


def proportional_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Rotate a proportional prefix while leaving remaining pairs unchanged."""
    _validate_schedule(head_dim, base, partial_rotary_factor, factor)
    rotated_pairs = int(partial_rotary_factor * head_dim // 2)
    exponents = torch.arange(
        0, 2 * rotated_pairs, 2, dtype=torch.int64, device=device
    ).float() / head_dim
    rotated = 1.0 / base**exponents
    unchanged = head_dim // 2 - rotated_pairs
    if unchanged:
        rotated = torch.cat(
            (rotated, torch.zeros(unchanged, device=device)),
        )
    return rotated / float(factor)


def apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    if cos.shape != sin.shape or cos.shape[-1] != tensor.shape[-1]:
        raise ValueError("rotary tables must match the tensor's last dimension")
    midpoint = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., midpoint:], tensor[..., :midpoint]), dim=-1)
    return (
        tensor * cos.unsqueeze(unsqueeze_dim)
        + rotated * sin.unsqueeze(unsqueeze_dim)
    )


def apply_multidimensional_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    dimensions: int,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    if dimensions <= 0 or tensor.shape[-1] % (2 * dimensions):
        raise ValueError("rotary channels must divide into even position blocks")
    shape = (*tensor.shape[:-1], dimensions, tensor.shape[-1] // dimensions)
    table_shape = (*cos.shape[:-1], dimensions, cos.shape[-1] // dimensions)
    return apply_rotary(
        tensor.reshape(shape),
        cos.reshape(table_shape),
        sin.reshape(table_shape),
        unsqueeze_dim=unsqueeze_dim,
    ).flatten(-2)


class MultidimensionalRotaryEmbedding(nn.Module):
    """Generate independent RoPE tables for each position-id dimension."""

    def __init__(
        self,
        head_dim: int,
        base: float,
        *,
        dimensions: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if dimensions <= 0 or head_dim % (2 * dimensions):
            raise ValueError("head channels must divide into even rotary blocks")
        self.dimensions = dimensions
        self.register_buffer(
            "inv_freq",
            default_inv_freq(
                head_dim // dimensions,
                base,
                device=device,
            ),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        tensor: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.shape[-1] != self.dimensions:
            raise ValueError(f"expected {self.dimensions} position dimensions")
        device_type = tensor.device.type if tensor.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = position_ids.float()[..., None] * self.inv_freq.float()
            embedding = torch.cat((frequencies, frequencies), dim=-1).flatten(-2)
            cos, sin = embedding.cos(), embedding.sin()
        return cos.to(tensor.dtype), sin.to(tensor.dtype)


__all__ = [
    "MultidimensionalRotaryEmbedding",
    "apply_multidimensional_rotary",
    "apply_rotary",
    "default_inv_freq",
    "proportional_inv_freq",
]
