"""RoPE helper utilities used by Moondream modules."""

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Precompute RoPE cos/sin cache in vLLM format."""
    if device is not None:
        device = torch.device(device)

    if dim % 2 != 0:
        raise ValueError("dim must be even")

    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            / float(dim)
        )
    )
    t = torch.arange(end, dtype=torch.float32, device=device).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, sin], dim=-1).to(dtype=dtype)


__all__ = ["precompute_freqs_cis"]
