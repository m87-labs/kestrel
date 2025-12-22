from __future__ import annotations

import torch

from kestrel_kernels.fused_linear_residual import fused_linear_bias_residual_cuda


def fused_linear_bias_residual_into(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute: out = residual + (x @ w.T + b).

    Notes:
      - Uses a fused CUDA op (cublasLt epilogues) when available.
      - `x`/`residual` may be 2D (M,C) or 3D (B,T,C); weights are 2D.
      - Intended for inference (no backward).
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must be 2D or 3D, got {tuple(x.shape)}")
    if residual.shape != out.shape:
        raise ValueError(f"residual must match out shape {tuple(out.shape)}, got {tuple(residual.shape)}")

    if x.ndim == 3:
        bsz, t, c = x.shape
        x2 = x.reshape(bsz * t, c)
        r2 = residual.reshape(bsz * t, residual.shape[-1])
        out2 = out.reshape(bsz * t, out.shape[-1])
    else:
        x2 = x
        r2 = residual
        out2 = out

    if w.ndim != 2:
        raise ValueError("w must be rank-2 tensor")
    if b.ndim != 1:
        raise ValueError("b must be rank-1 tensor")

    m, in_dim = x2.shape
    if w.shape[1] != in_dim:
        raise ValueError(f"w must have in_dim={in_dim}, got {tuple(w.shape)}")
    out_dim = w.shape[0]
    if b.shape[0] != out_dim:
        raise ValueError(f"b must have shape ({out_dim},), got {tuple(b.shape)}")
    if out2.shape != (m, out_dim):
        raise ValueError(f"out must have shape {(m, out_dim)}, got {tuple(out2.shape)}")
    if r2.shape != (m, out_dim):
        raise ValueError(f"residual must have shape {(m, out_dim)}, got {tuple(r2.shape)}")

    fused_linear_bias_residual_cuda(out2, x2, w, b, r2)


__all__ = [
    "fused_linear_bias_residual_cuda",
    "fused_linear_bias_residual_into",
]
