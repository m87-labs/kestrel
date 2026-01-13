"""Tests for GELU residual CuTe DSL kernel."""

import math

import pytest
import torch


def _reference_gelu_residual(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation - stays in FP32 throughout to match kernel."""
    hidden = x.shape[1] // 2
    h = x[:, :hidden].float()
    g = x[:, hidden:].float()
    gelu_f = 0.5 * h * (1.0 + torch.erf(h * (1.0 / math.sqrt(2.0))))
    result_f = gelu_f * (g + 1.0)
    return result_f.to(x.dtype)


@pytest.mark.parametrize("rows,hidden", [
    (1, 1024),
    (8, 1024),
    (32, 1024),
    (128, 1024),
    (5920, 1024),  # prefill
])
def test_gelu_residual_cute_matches_pytorch(sm90_device, rows, hidden):
    """Test that CuTe kernel produces correct results."""
    from kestrel_kernels.gelu_residual import gelu_residual_cute

    dtype = torch.bfloat16
    x = torch.randn((rows, hidden * 2), dtype=dtype, device=sm90_device)
    out = torch.empty((rows, hidden), dtype=dtype, device=sm90_device)

    gelu_residual_cute(out, x)
    expected = _reference_gelu_residual(x)

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_gelu_residual_cute_3d_input(sm90_device):
    """Test that 3D inputs are handled correctly."""
    from kestrel_kernels.gelu_residual import gelu_residual_cute

    dtype = torch.bfloat16
    hidden = 1024
    batch, seq = 4, 32
    x = torch.randn((batch, seq, hidden * 2), dtype=dtype, device=sm90_device)
    out = torch.empty((batch, seq, hidden), dtype=dtype, device=sm90_device)

    gelu_residual_cute(out, x)

    # Check against reference
    x_2d = x.view(-1, hidden * 2)
    expected = _reference_gelu_residual(x_2d).view(batch, seq, hidden)

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_gelu_residual_cute_cuda_graph(sm90_device):
    """Test that kernel works with CUDA graphs."""
    from kestrel_kernels.gelu_residual import gelu_residual_cute

    dtype = torch.bfloat16
    hidden = 1024
    rows = 128
    x = torch.randn((rows, hidden * 2), dtype=dtype, device=sm90_device)
    out = torch.empty((rows, hidden), dtype=dtype, device=sm90_device)

    # Warmup
    gelu_residual_cute(out, x)
    torch.cuda.synchronize()

    # Capture in CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gelu_residual_cute(out, x)
    torch.cuda.synchronize()

    # Replay and check
    g.replay()
    torch.cuda.synchronize()

    expected = _reference_gelu_residual(x)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_gelu_residual_cute_vs_cuda(sm90_device):
    """Test that CuTe kernel matches the CUDA kernel."""
    from kestrel_kernels.gelu_residual import gelu_residual_cute
    from kestrel_kernels.activation import gelu_residual_cuda

    dtype = torch.bfloat16
    hidden = 1024
    rows = 128
    x = torch.randn((rows, hidden * 2), dtype=dtype, device=sm90_device)

    out_cute = torch.empty((rows, hidden), dtype=dtype, device=sm90_device)
    out_cuda = torch.empty((rows, hidden), dtype=dtype, device=sm90_device)

    gelu_residual_cute(out_cute, x)
    gelu_residual_cuda(out_cuda, x)

    torch.testing.assert_close(out_cute, out_cuda, rtol=1e-2, atol=1e-2)
