"""Model-independent rotary embedding operations."""

from __future__ import annotations

import torch
from torch import nn


def default_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct standard RoPE inverse frequencies."""
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}")
    if not 0.0 < partial_rotary_factor <= 1.0:
        raise ValueError(
            "partial_rotary_factor must lie in (0, 1], got "
            f"{partial_rotary_factor}"
        )
    if base <= 0.0 or factor <= 0.0:
        raise ValueError("RoPE base and factor must be positive")
    rotated_dim = int(head_dim * partial_rotary_factor)
    if rotated_dim <= 0 or rotated_dim % 2:
        raise ValueError(
            "partial RoPE dimension must be positive and even, got "
            f"{rotated_dim}"
        )
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


def apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply split-half rotary embeddings along the last dimension."""
    if tensor.shape[-1] <= 0 or tensor.shape[-1] % 2:
        raise ValueError(
            "rotary input channels must be positive and even, got "
            f"{tensor.shape[-1]}"
        )
    if cos.shape[-1] != tensor.shape[-1] or sin.shape != cos.shape:
        raise ValueError(
            "rotary cos/sin must match the input channel dimension"
        )
    midpoint = tensor.shape[-1] // 2
    rotated = torch.cat(
        (-tensor[..., midpoint:], tensor[..., :midpoint]),
        dim=-1,
    )
    return (
        tensor * cos.unsqueeze(unsqueeze_dim)
        + rotated * sin.unsqueeze(unsqueeze_dim)
    )


def apply_multidimensional_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply independent rotary blocks for each position-id dimension."""
    dimensions = position_ids.shape[-1]
    channels = tensor.shape[-1]
    if dimensions <= 0:
        raise ValueError("multidimensional RoPE needs at least one dimension")
    if channels <= 0 or channels % (2 * dimensions):
        raise ValueError(
            f"{channels} channels must be divisible by "
            f"2 * {dimensions} position dimensions"
        )
    rotated_channels = 2 * (channels // (2 * dimensions))
    split_sizes = [rotated_channels] * dimensions
    tensor_parts = torch.split(tensor, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    return torch.cat(
        [
            apply_rotary(
                tensor_parts[index],
                cos_parts[index],
                sin_parts[index],
                unsqueeze_dim=unsqueeze_dim,
            )
            for index in range(dimensions)
        ],
        dim=-1,
    )


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
        self.dimensions = int(dimensions)
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if head_dim <= 0 or head_dim % (2 * self.dimensions):
            raise ValueError(
                f"{head_dim} head channels must be divisible by "
                f"2 * {self.dimensions} dimensions"
            )
        self.inv_freq = default_inv_freq(
            head_dim // self.dimensions,
            base,
            device=device,
        )

    def _ensure_device(self, device: torch.device) -> None:
        if self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device)

    @torch.no_grad()
    def forward(
        self,
        tensor: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.shape[-1] != self.dimensions:
            raise ValueError(
                f"expected {self.dimensions} position dimensions, "
                f"got {position_ids.shape[-1]}"
            )
        self._ensure_device(tensor.device)
        expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0],
            -1,
            1,
        )
        device_type = tensor.device.type if tensor.device.type != "mps" else "cpu"
        all_cos = []
        all_sin = []
        for dimension in range(self.dimensions):
            positions = position_ids[:, :, dimension][:, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                frequencies = (expanded @ positions).transpose(1, 2)
                embedding = torch.cat((frequencies, frequencies), dim=-1)
                all_cos.append(embedding.cos())
                all_sin.append(embedding.sin())
        return (
            torch.cat(all_cos, dim=-1).to(dtype=tensor.dtype),
            torch.cat(all_sin, dim=-1).to(dtype=tensor.dtype),
        )


__all__ = [
    "MultidimensionalRotaryEmbedding",
    "apply_multidimensional_rotary",
    "apply_rotary",
    "default_inv_freq",
]
