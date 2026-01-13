"""Tests for fused linear + bias + residual add (cublasLt epilogues)."""

import pytest
import torch
import torch.nn.functional as F

from kestrel_kernels.fused_linear_residual import fused_linear_bias_residual_cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


DTYPE = torch.bfloat16
RTOL = 5e-2
ATOL = 1e-1


def _weight_scale(in_dim: int) -> float:
    return in_dim**-0.5


def fused_linear_bias_residual_into(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute: out = residual + (x @ w.T + b)."""
    if x.ndim == 3:
        bsz, t, c = x.shape
        x2 = x.reshape(bsz * t, c)
        r2 = residual.reshape(bsz * t, residual.shape[-1])
        out2 = out.reshape(bsz * t, out.shape[-1])
    else:
        x2 = x
        r2 = residual
        out2 = out

    fused_linear_bias_residual_cuda(out2, x2, w, b, r2)


@pytest.mark.parametrize("crops", list(range(1, 13)))
def test_fused_linear_residual_supports_crop_sizes(device, crops: int):
    """Covers vision crop-count M values (1..12 crops, 27*27 patches each)."""
    torch.manual_seed(0)

    m = crops * 729
    in_dim = out_dim = 128
    w_scale = _weight_scale(in_dim)

    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)
    w = torch.randn((out_dim, in_dim), device=device, dtype=DTYPE) * w_scale
    b = torch.randn((out_dim,), device=device, dtype=DTYPE) * w_scale

    out = torch.empty((m, out_dim), device=device, dtype=DTYPE)
    fused_linear_bias_residual_into(x=x, w=w, b=b, residual=residual, out=out)

    expected = F.linear(x, w, b) + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_linear_residual_matches_pytorch_vision_proj_shape(device):
    """Moondream vision attention proj shape (1 crop)."""
    torch.manual_seed(0)

    m = 729
    in_dim = out_dim = 1152
    w_scale = _weight_scale(in_dim)

    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)
    w = torch.randn((out_dim, in_dim), device=device, dtype=DTYPE) * w_scale
    b = torch.randn((out_dim,), device=device, dtype=DTYPE) * w_scale

    out = torch.empty((m, out_dim), device=device, dtype=DTYPE)
    fused_linear_bias_residual_into(x=x, w=w, b=b, residual=residual, out=out)

    expected = F.linear(x, w, b) + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_linear_residual_supports_3d_inputs(device):
    torch.manual_seed(0)

    bsz, t, in_dim = 2, 16, 128
    out_dim = in_dim
    w_scale = _weight_scale(in_dim)

    x = torch.randn((bsz, t, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((bsz, t, out_dim), device=device, dtype=DTYPE)
    w = torch.randn((out_dim, in_dim), device=device, dtype=DTYPE) * w_scale
    b = torch.randn((out_dim,), device=device, dtype=DTYPE) * w_scale

    out = torch.empty((bsz, t, out_dim), device=device, dtype=DTYPE)
    fused_linear_bias_residual_into(x=x, w=w, b=b, residual=residual, out=out)

    expected = F.linear(x, w, b) + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_linear_residual_allows_out_alias_residual(device):
    """Enables in-place residual updates (C and D alias)."""
    torch.manual_seed(0)

    m, in_dim = 256, 128
    out_dim = 128
    w_scale = _weight_scale(in_dim)

    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)
    residual_orig = residual.clone()

    w = torch.randn((out_dim, in_dim), device=device, dtype=DTYPE) * w_scale
    b = torch.randn((out_dim,), device=device, dtype=DTYPE) * w_scale

    fused_linear_bias_residual_into(x=x, w=w, b=b, residual=residual, out=residual)

    expected = F.linear(x, w, b) + residual_orig
    torch.testing.assert_close(residual, expected, rtol=RTOL, atol=ATOL)
