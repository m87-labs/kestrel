"""Tests for fused MLP (cublasLt epilogues)."""

import pytest
import torch
import torch.nn.functional as F

from kestrel_kernels.fused_mlp import fused_mlp_gelu_bias_residual_cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


DTYPE = torch.bfloat16
RTOL = 5e-2
ATOL = 1e-1


def _weights_scales(in_dim: int, hidden_dim: int) -> tuple[float, float]:
    # Match common initialization scaling to keep activations in a realistic range.
    return (in_dim**-0.5, hidden_dim**-0.5)


def fused_mlp_gelu_bias_residual_into(
    *,
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute: out = residual + (gelu(x @ w1.T + b1) @ w2.T + b2)."""
    if x.ndim == 3:
        b, t, c = x.shape
        x2 = x.reshape(b * t, c)
        r2 = residual.reshape(b * t, c)
        out2 = out.reshape(b * t, c)
    else:
        x2 = x
        r2 = residual
        out2 = out

    m = x2.shape[0]
    hidden_dim = w1.shape[0]
    hidden = torch.empty((m, hidden_dim), device=x2.device, dtype=x2.dtype)
    fused_mlp_gelu_bias_residual_cuda(out2, hidden, x2, w1, b1, w2, b2, r2)


@pytest.mark.parametrize("crops", list(range(1, 13)))
def test_fused_mlp_supports_crop_sizes(device, crops: int):
    """Covers vision crop-count M values (1..12 crops, 27*27 patches each)."""
    torch.manual_seed(0)

    m = crops * 729
    in_dim, hidden_dim, out_dim = 128, 256, 128
    w1_scale, w2_scale = _weights_scales(in_dim, hidden_dim)

    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)
    w1 = torch.randn((hidden_dim, in_dim), device=device, dtype=DTYPE) * w1_scale
    b1 = torch.zeros((hidden_dim,), device=device, dtype=DTYPE)
    w2 = torch.randn((out_dim, hidden_dim), device=device, dtype=DTYPE) * w2_scale
    b2 = torch.zeros((out_dim,), device=device, dtype=DTYPE)

    out = torch.empty((m, out_dim), device=device, dtype=DTYPE)
    fused_mlp_gelu_bias_residual_into(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2, residual=residual, out=out
    )

    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2) + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_mlp_matches_pytorch_vision_shape(device):
    """Moondream vision MLP shape (1 crop)."""
    torch.manual_seed(0)

    m, in_dim, hidden_dim, out_dim = 729, 1152, 4304, 1152
    w1_scale, w2_scale = _weights_scales(in_dim, hidden_dim)
    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)

    w1 = torch.randn((hidden_dim, in_dim), device=device, dtype=DTYPE) * w1_scale
    b1 = torch.zeros((hidden_dim,), device=device, dtype=DTYPE)
    w2 = torch.randn((out_dim, hidden_dim), device=device, dtype=DTYPE) * w2_scale
    b2 = torch.zeros((out_dim,), device=device, dtype=DTYPE)

    out = torch.empty((m, out_dim), device=device, dtype=DTYPE)
    fused_mlp_gelu_bias_residual_into(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2, residual=residual, out=out
    )

    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2) + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_mlp_supports_3d_inputs(device):
    torch.manual_seed(0)

    b, t, in_dim = 2, 16, 128
    hidden_dim = 256
    out_dim = in_dim
    w1_scale, w2_scale = _weights_scales(in_dim, hidden_dim)

    x = torch.randn((b, t, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((b, t, out_dim), device=device, dtype=DTYPE)

    w1 = torch.randn((hidden_dim, in_dim), device=device, dtype=DTYPE) * w1_scale
    b1 = torch.zeros((hidden_dim,), device=device, dtype=DTYPE)
    w2 = torch.randn((out_dim, hidden_dim), device=device, dtype=DTYPE) * w2_scale
    b2 = torch.zeros((out_dim,), device=device, dtype=DTYPE)

    out = torch.empty((b, t, out_dim), device=device, dtype=DTYPE)
    fused_mlp_gelu_bias_residual_into(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2, residual=residual, out=out
    )

    expected = F.linear(
        F.gelu(F.linear(x, w1, b1), approximate="tanh"),
        w2,
        b2,
    )
    expected = expected + residual
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_fused_mlp_allows_out_alias_residual(device):
    """Enables in-place residual updates (C and D alias)."""
    torch.manual_seed(0)

    m, in_dim, hidden_dim, out_dim = 256, 128, 256, 128
    w1_scale, w2_scale = _weights_scales(in_dim, hidden_dim)
    x = torch.randn((m, in_dim), device=device, dtype=DTYPE)
    residual = torch.randn((m, out_dim), device=device, dtype=DTYPE)
    residual_orig = residual.clone()

    w1 = torch.randn((hidden_dim, in_dim), device=device, dtype=DTYPE) * w1_scale
    b1 = torch.zeros((hidden_dim,), device=device, dtype=DTYPE)
    w2 = torch.randn((out_dim, hidden_dim), device=device, dtype=DTYPE) * w2_scale
    b2 = torch.zeros((out_dim,), device=device, dtype=DTYPE)

    # Alias out == residual (in-place update).
    fused_mlp_gelu_bias_residual_into(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2, residual=residual, out=residual
    )

    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2) + residual_orig
    torch.testing.assert_close(residual, expected, rtol=RTOL, atol=ATOL)
