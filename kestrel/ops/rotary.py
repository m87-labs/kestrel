"""Model-independent rotary embedding operations."""

from __future__ import annotations

import math
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
    # Meta construction needs only shape/dtype, not compiler-backed decompositions.
    if torch.empty(0, device=device).is_meta:
        return torch.empty(rotated_dim // 2, dtype=torch.float32, device=device)
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
    if torch.empty(0, device=device).is_meta:
        dtype = (torch.promote_types(torch.float32, torch.get_default_dtype())
                 if rotated_pairs < head_dim // 2 else torch.float32)
        return torch.empty(head_dim // 2, dtype=dtype, device=device)
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


def yarn_inv_freq(head_dim: int, base: float, parameters: dict, *, device=None):
    """Construct the checkpoint's static YaRN frequency and amplitude schedule."""
    factor = float(parameters["factor"])
    original = int(parameters["original_max_position_embeddings"])
    fast, slow = float(parameters.get("beta_fast", 32)), float(parameters.get("beta_slow", 1))
    _validate_schedule(head_dim, base, 1.0, factor)
    if base <= 1 or original <= 0 or not fast >= slow > 0:
        raise ValueError("invalid YaRN correction range")
    def scale(mscale=1.0):
        return 1.0 if factor <= 1 else 1.0 + 0.1 * mscale * math.log(factor)
    amplitude = parameters.get("attention_factor")
    if amplitude is None:
        mscale, all_dim = parameters.get("mscale"), parameters.get("mscale_all_dim")
        amplitude = scale(mscale) / scale(all_dim) if mscale and all_dim else scale()
    amplitude = float(amplitude)
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("YaRN attention factor must be positive and finite")
    if torch.empty(0, device=device).is_meta:
        return torch.empty(head_dim // 2, device=device, dtype=torch.float32), amplitude
    low = head_dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(base))
    high = head_dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(base))
    if parameters.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(head_dim // 2, device=device, dtype=torch.float32) - low)
            / (high - low)).clamp(0, 1)
    frequencies = base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    return ramp / (factor * frequencies) + (1 - ramp) / frequencies, amplitude


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
        self.attention_factor = 1.0
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
            if self.attention_factor != 1.0:
                cos, sin = cos * self.attention_factor, sin * self.attention_factor
        return cos.to(tensor.dtype), sin.to(tensor.dtype)


__all__ = [
    "MultidimensionalRotaryEmbedding",
    "apply_multidimensional_rotary",
    "apply_rotary",
    "default_inv_freq",
    "proportional_inv_freq",
    "yarn_inv_freq",
]
