import torch

from kestrel_kernels.rotary_embedding import rotary_embedding as rotary_embedding_cuda


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Precompute RoPE cos/sin cache in vLLM format.

    Args:
        dim: Rotary dimension (must be even). The cache will have shape
            ``[end, dim]`` where the first ``dim//2`` columns are ``cos`` and
            the last ``dim//2`` columns are ``sin``.
        end: Maximum sequence length (number of positions).
        theta: RoPE base.
        dtype: Output dtype.
        device: Output device.
    """
    if device is not None:
        device = torch.device(device)

    if dim % 2 != 0:
        raise ValueError("dim must be even")

    # Compute in fp32 for stability, then cast to the requested dtype.
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            / float(dim)
        )
    )
    t = torch.arange(end, dtype=torch.float32, device=device).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)  # [end, dim//2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, sin], dim=-1).to(dtype=dtype)


__all__ = ["precompute_freqs_cis", "rotary_embedding_cuda"]
