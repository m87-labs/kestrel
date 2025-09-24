"""Rotary positional embedding helpers for the Moondream text model.

Adapted from the Moondream project (Apache-2.0).
"""

from __future__ import annotations

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 1_500_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> torch.Tensor:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    if interleave:
        xq_r = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 0]
        xq_i = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 1]
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

    position_ids = position_ids.to(torch.int64)
    freqs_cos = freqs_cis[..., 0][position_ids, :]
    freqs_sin = freqs_cis[..., 1][position_ids, :]

    if position_ids.ndim == 1:
        batch = position_ids.shape[0]
        freqs_cos = freqs_cos.view(batch, 1, 1, -1)
        freqs_sin = freqs_sin.view(batch, 1, 1, -1)
    elif position_ids.ndim == 2:
        batch, seq_len = position_ids.shape
        freqs_cos = freqs_cos.view(batch, 1, seq_len, -1)
        freqs_sin = freqs_sin.view(batch, 1, seq_len, -1)
    else:
        raise ValueError(f"Unsupported position_ids rank: {position_ids.ndim}")

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1)
    shape = xq_out.shape[:-2] + (xq_out.shape[-2] * xq_out.shape[-1],)
    xq_out = xq_out.reshape(shape)

    return torch.cat([xq_out.to(x.dtype), x_pass], dim=-1)


__all__ = ["precompute_freqs_cis", "apply_rotary_emb"]
