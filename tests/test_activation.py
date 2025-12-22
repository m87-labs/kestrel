"""Tests for GELU residual activation CUDA kernel."""

import math

import pytest
import torch

from kestrel_kernels.activation import gelu_residual_cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.bfloat16


def _reference_gelu_residual(x: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[1] // 2
    h = x[:, :hidden]
    g = x[:, hidden:]
    h_f = h.float()
    gelu_f = 0.5 * h_f * (1.0 + torch.erf(h_f * (1.0 / math.sqrt(2.0))))
    gelu_bf16 = gelu_f.to(x.dtype)
    g_plus = (g + 1.0).to(x.dtype)
    return (gelu_bf16 * g_plus).to(x.dtype)


@pytest.mark.parametrize("rows,hidden", [(1, 1024), (8, 1024), (64, 1024)])
def test_gelu_residual_cuda_matches_pytorch(device, dtype, rows, hidden):
    x = torch.randn((rows, hidden * 2), dtype=dtype, device=device)
    out = torch.empty((rows, hidden), dtype=dtype, device=device)

    gelu_residual_cuda(out, x)
    expected = _reference_gelu_residual(x)

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_gelu_residual_cuda_rejects_hidden_not_multiple_of_8(device, dtype):
    hidden = 10
    x = torch.randn((2, hidden * 2), dtype=dtype, device=device)
    out = torch.empty((2, hidden), dtype=dtype, device=device)

    with pytest.raises(RuntimeError, match="hidden dimension must be a multiple of 8"):
        gelu_residual_cuda(out, x)
