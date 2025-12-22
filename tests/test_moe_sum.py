"""Correctness tests for moe_sum."""

import pytest
import torch

from kestrel_kernels.moe_sum import moe_sum as moe_sum_cuda


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _torch_moe_sum_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference for moe_sum (matches PyTorch reduction semantics)."""
    if x.ndim != 3:
        raise ValueError("x must have shape [num_tokens, top_k, hidden]")
    return x.sum(dim=1)


@pytest.mark.parametrize("topk", [4, 8])
def test_moe_sum_matches_reference(device: torch.device, topk: int) -> None:
    torch.manual_seed(0)
    num_tokens = 256
    hidden = 2048
    x = torch.randn((num_tokens, topk, hidden), device=device, dtype=torch.bfloat16)
    out = torch.empty((num_tokens, hidden), device=device, dtype=torch.bfloat16)

    moe_sum_cuda(x, out)

    golden = _torch_moe_sum_reference(x)
    torch.testing.assert_close(out, golden, atol=0, rtol=0)


def test_moe_sum_rejects_cpu(device: torch.device) -> None:
    x = torch.randn((4, 2, 8), device="cpu", dtype=torch.bfloat16)
    out = torch.empty((4, 8), device="cpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="input must be CUDA"):
        moe_sum_cuda(x, out)


def test_moe_sum_rejects_non_contiguous_input(device: torch.device) -> None:
    # Slice with a stride so the last dim is non-contiguous but shape stays [T, K, H].
    x_full = torch.randn((8, 8, 4096), device=device, dtype=torch.bfloat16)
    x = x_full[:, :, ::2]
    assert not x.is_contiguous()
    out = torch.empty((8, 2048), device=device, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="input must be contiguous"):
        moe_sum_cuda(x, out)


def test_moe_sum_rejects_non_contiguous_output(device: torch.device) -> None:
    x = torch.randn((8, 8, 2048), device=device, dtype=torch.bfloat16)
    out_full = torch.empty((8, 4096), device=device, dtype=torch.bfloat16)
    out = out_full[:, ::2]
    assert not out.is_contiguous()
    with pytest.raises(RuntimeError, match="output must be contiguous"):
        moe_sum_cuda(x, out)


def test_moe_sum_rejects_dtype_mismatch(device: torch.device) -> None:
    x = torch.randn((8, 8, 2048), device=device, dtype=torch.bfloat16)
    out = torch.empty((8, 2048), device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="same dtype"):
        moe_sum_cuda(x, out)
