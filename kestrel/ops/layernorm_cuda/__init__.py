from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..jit import cpp_jit


@cpp_jit()
def layernorm_bias_cuda(
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> None: ...

@cpp_jit()
def layernorm_bias_reload_cuda(
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> None: ...


def layernorm_bias_into(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
    variant: str = "auto",
    fallback_to_torch: bool = False,
) -> None:
    """Compute LayerNorm forward: out = (x - mean) / sqrt(var + eps) * weight + bias.

    Notes:
      - bf16 CUDA forward-only kernel.
      - `x` may be 2D (M,N) or 3D (B,T,N); `weight`/`bias` are 1D (N,).
      - `variant` selects between epilogue strategies for the N=1152/2048 rows4 kernel:
        "default" caches x in registers; "reload_x" trades extra reads for lower reg pressure.
      - If `fallback_to_torch=True`, unsupported inputs fall back to `torch.nn.functional.layer_norm`.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must be 2D or 3D, got {tuple(x.shape)}")
    if out.shape != x.shape:
        raise ValueError(f"out must match x shape {tuple(x.shape)}, got {tuple(out.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D")
    if x.shape[-1] != weight.shape[0] or weight.shape != bias.shape:
        raise ValueError("weight/bias must match x last dimension")

    if x.ndim == 3:
        b, t, n = x.shape
        x2 = x.reshape(b * t, n)
        out2 = out.reshape(b * t, n)
    else:
        x2 = x
        out2 = out

    # Notes on variants:
    # - "default": caches x vectors in registers for the output pass (fewer reads, more regs).
    # - "reload_x": reloads x for the output pass (more reads, fewer regs / higher occupancy).
    # - "auto": use "reload_x" for the vision width (N=1152), otherwise "default".
    try:
        if variant == "auto":
            variant = "reload_x" if int(x2.shape[1]) == 1152 else "default"
        if variant == "default":
            layernorm_bias_cuda(out2, x2, weight, bias, float(eps))
        elif variant == "reload_x":
            layernorm_bias_reload_cuda(out2, x2, weight, bias, float(eps))
        else:
            raise ValueError(f"Unknown layernorm_cuda variant: {variant!r}")
    except Exception:
        if not fallback_to_torch:
            raise
        out2.copy_(F.layer_norm(x2, (x2.shape[1],), weight, bias, float(eps)))


def layernorm_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    fallback_to_torch: bool = False,
) -> torch.Tensor:
    out = torch.empty_like(x)
    layernorm_bias_into(
        x=x, weight=weight, bias=bias, out=out, eps=eps, fallback_to_torch=fallback_to_torch
    )
    return out


__all__ = [
    "layernorm_bias",
    "layernorm_bias_into",
    "layernorm_bias_cuda",
    "layernorm_bias_reload_cuda",
]
