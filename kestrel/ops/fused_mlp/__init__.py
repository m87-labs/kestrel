from __future__ import annotations

import torch

from ..jit import cpp_jit


@cpp_jit()
def fused_mlp_gelu_bias_residual_cuda(
    out: torch.Tensor,
    hidden: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    residual: torch.Tensor,
) -> None: ...


class _ResizableBuffer:
    def __init__(self) -> None:
        self._tensor: torch.Tensor | None = None

    def get(self, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        numel = 1
        for dim in shape:
            numel *= dim
        if numel == 0:
            return torch.empty(shape, device=device, dtype=dtype)

        if (
            self._tensor is None
            or self._tensor.numel() < numel
            or self._tensor.device != device
            or self._tensor.dtype != dtype
        ):
            self._tensor = torch.empty(numel, device=device, dtype=dtype)
        return self._tensor[:numel].view(*shape)


class FusedMLPWorkspaces:
    def __init__(self) -> None:
        self.hidden = _ResizableBuffer()


_WORKSPACES = FusedMLPWorkspaces()


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
]

