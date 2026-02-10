from __future__ import annotations

import torch

from kestrel.utils.buffers import FixedBuffer
from kestrel_kernels import get_runtime

fused_mlp_gelu_bias_residual_cuda = (
    get_runtime().dense.fused_mlp_gelu_bias_residual_cuda
)


class FusedMLPWorkspaces:
    def __init__(self) -> None:
        self.hidden = FixedBuffer("Fused MLP hidden workspace")


_WORKSPACES = FusedMLPWorkspaces()


def preallocate_fused_mlp_workspaces(
    max_num_tokens: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-allocate fused MLP workspace to ensure stable pointers for CUDA graphs.

    Args:
        max_num_tokens: Maximum tokens in any forward pass.
        hidden_dim: Hidden dimension of the MLP (intermediate size).
        device: Target device.
        dtype: Data type.
    """
    _WORKSPACES.hidden.get((max_num_tokens, hidden_dim), device=device, dtype=dtype)


def fused_mlp_gelu_bias_residual_into(
    *,
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    workspaces: FusedMLPWorkspaces | None = None,
) -> None:
    """Compute: out = residual + (gelu(x @ w1.T + b1) @ w2.T + b2).

    Notes:
      - Uses a fused CUDA op (cublasLt epilogues) when available.
      - `x`/`residual` may be 2D (M,C) or 3D (B,T,C); weights are 2D.
      - Intended for inference (no backward).
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must be 2D or 3D, got {tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(f"residual must match x shape {tuple(x.shape)}, got {tuple(residual.shape)}")

    if x.ndim == 3:
        b, t, c = x.shape
        x2 = x.reshape(b * t, c)
        r2 = residual.reshape(b * t, c)
        out2 = out.reshape(b * t, c)
    else:
        x2 = x
        r2 = residual
        out2 = out

    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("w1 and w2 must be rank-2 tensors")
    if b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("b1 and b2 must be rank-1 tensors")

    m, in_dim = x2.shape
    if w1.shape[1] != in_dim:
        raise ValueError(f"w1 must have in_dim={in_dim}, got {tuple(w1.shape)}")
    hidden_dim = w1.shape[0]
    if b1.shape[0] != hidden_dim:
        raise ValueError(f"b1 must have shape ({hidden_dim},), got {tuple(b1.shape)}")
    if w2.shape[1] != hidden_dim:
        raise ValueError(f"w2 must have in_dim={hidden_dim}, got {tuple(w2.shape)}")
    out_dim = w2.shape[0]
    if b2.shape[0] != out_dim:
        raise ValueError(f"b2 must have shape ({out_dim},), got {tuple(b2.shape)}")
    if out2.shape != (m, out_dim):
        raise ValueError(f"out must have shape {(m, out_dim)}, got {tuple(out2.shape)}")
    if r2.shape != (m, out_dim):
        raise ValueError(f"residual must have shape {(m, out_dim)}, got {tuple(r2.shape)}")

    ws = _WORKSPACES if workspaces is None else workspaces
    hidden = ws.hidden.get((m, hidden_dim), device=x2.device, dtype=x2.dtype)
    fused_mlp_gelu_bias_residual_cuda(out2, hidden, x2, w1, b1, w2, b2, r2)


__all__ = [
    "FusedMLPWorkspaces",
    "fused_mlp_gelu_bias_residual_cuda",
    "fused_mlp_gelu_bias_residual_into",
    "preallocate_fused_mlp_workspaces",
]
