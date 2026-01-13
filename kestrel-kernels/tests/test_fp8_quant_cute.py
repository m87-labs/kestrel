"""Tests for FP8 row-wise quantization CuTe DSL kernel."""

import pytest
import torch

FP8_E4M3_MAX = 448.0


def _reference_fp8_rowwise_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of per-row FP8 quantization."""
    absmax = x.abs().amax(dim=1).float()
    scale = absmax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-6)
    x_scaled = x.float() / scale.unsqueeze(1)
    x_clamped = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    out_bits = x_clamped.to(torch.float8_e4m3fn).view(torch.uint8)
    return out_bits, scale


@pytest.mark.parametrize("rows,hidden", [
    # hidden=1024 (MoE down projection input)
    (1, 1024),
    (8, 1024),
    (32, 1024),
    (128, 1024),
    (256, 1024),  # boundary for warps_per_block switch
    (512, 1024),
    (5920, 1024),  # prefill
    # hidden=2048 (MoE up projection input)
    (1, 2048),
    (8, 2048),
    (128, 2048),
    (5920, 2048),  # prefill
])
def test_fp8_quant_cute_matches_pytorch(sm90_device, rows, hidden):
    """Test that CuTe kernel produces correct results."""
    from kestrel_kernels.fp8_quant_cute import fp8_quant_cute

    dtype = torch.bfloat16
    x = torch.randn((rows, hidden), dtype=dtype, device=sm90_device)
    out_bits = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
    out_scale = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

    fp8_quant_cute(out_bits, out_scale, x)

    expected_bits, expected_scale = _reference_fp8_rowwise_quant(x)

    # Check scales match
    torch.testing.assert_close(out_scale, expected_scale, rtol=1e-3, atol=1e-6)

    # Check quantized values match (allow off-by-1 for rounding at boundaries)
    torch.testing.assert_close(out_bits.int(), expected_bits.int(), atol=1, rtol=0)


@pytest.mark.parametrize("hidden", [1024, 2048])
def test_fp8_quant_cute_vs_cuda(sm90_device, hidden):
    """Test that CuTe kernel matches the CUDA kernel."""
    from kestrel_kernels.fp8_quant_cute import fp8_quant_cute
    from kestrel_kernels.fp8_quant import fp8_e4m3fn_rowwise_quant_cuda

    dtype = torch.bfloat16

    for rows in [8, 32, 128, 5920]:
        x = torch.randn((rows, hidden), dtype=dtype, device=sm90_device)

        out_bits_cute = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
        out_scale_cute = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

        out_bits_cuda = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
        out_scale_cuda = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

        fp8_quant_cute(out_bits_cute, out_scale_cute, x)
        fp8_e4m3fn_rowwise_quant_cuda(out_bits_cuda, out_scale_cuda, x)

        torch.testing.assert_close(out_scale_cute, out_scale_cuda, rtol=1e-3, atol=1e-6)
        assert torch.equal(out_bits_cute, out_bits_cuda), f"Bits mismatch for rows={rows}, hidden={hidden}"


def test_fp8_quant_cute_cuda_graph(sm90_device):
    """Test that kernel works with CUDA graphs."""
    from kestrel_kernels.fp8_quant_cute import fp8_quant_cute

    dtype = torch.bfloat16
    hidden = 1024
    rows = 128

    x = torch.randn((rows, hidden), dtype=dtype, device=sm90_device)
    out_bits = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
    out_scale = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

    # Warmup
    fp8_quant_cute(out_bits, out_scale, x)
    torch.cuda.synchronize()

    # Capture in CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fp8_quant_cute(out_bits, out_scale, x)
    torch.cuda.synchronize()

    # Replay and check
    g.replay()
    torch.cuda.synchronize()

    expected_bits, expected_scale = _reference_fp8_rowwise_quant(x)
    torch.testing.assert_close(out_scale, expected_scale, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(out_bits.int(), expected_bits.int(), atol=1, rtol=0)


def test_fp8_quant_cute_edge_cases(sm90_device):
    """Test edge cases: zeros, very small values, very large values."""
    from kestrel_kernels.fp8_quant_cute import fp8_quant_cute

    dtype = torch.bfloat16
    hidden = 1024
    rows = 8

    # Test with zeros - should get MIN_SCALE
    x_zeros = torch.zeros((rows, hidden), dtype=dtype, device=sm90_device)
    out_bits = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
    out_scale = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

    fp8_quant_cute(out_bits, out_scale, x_zeros)

    # Scale should be clamped to MIN_SCALE (1e-6)
    assert torch.all(out_scale >= 1e-6), "Scale should be at least MIN_SCALE"

    # Test with values that saturate FP8 range
    x_large = torch.full((rows, hidden), 1000.0, dtype=dtype, device=sm90_device)
    fp8_quant_cute(out_bits, out_scale, x_large)

    expected_bits, expected_scale = _reference_fp8_rowwise_quant(x_large)
    torch.testing.assert_close(out_scale, expected_scale, rtol=1e-3, atol=1e-6)
    assert torch.equal(out_bits, expected_bits), "Large value quantization mismatch"


def test_fp8_quant_cute_negative_values(sm90_device):
    """Test that negative values are quantized correctly."""
    from kestrel_kernels.fp8_quant_cute import fp8_quant_cute

    dtype = torch.bfloat16
    hidden = 1024
    rows = 8

    # Mix of positive and negative values
    x = torch.randn((rows, hidden), dtype=dtype, device=sm90_device)
    x[:, :hidden//2] = -x[:, :hidden//2].abs()  # Force negatives in first half

    out_bits = torch.empty((rows, hidden), dtype=torch.uint8, device=sm90_device)
    out_scale = torch.empty((rows,), dtype=torch.float32, device=sm90_device)

    fp8_quant_cute(out_bits, out_scale, x)

    expected_bits, expected_scale = _reference_fp8_rowwise_quant(x)
    torch.testing.assert_close(out_scale, expected_scale, rtol=1e-3, atol=1e-6)
    assert torch.equal(out_bits, expected_bits), "Negative value quantization mismatch"
