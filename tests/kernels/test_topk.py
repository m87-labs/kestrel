"""Tests for the topk kernel."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def test_topk_correctness(device):
    """Test topk returns correct values and indices."""
    from kestrel_kernels.topk import topk_fwd

    x = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    values, indices = topk_fwd(x, k=8, softmax=True)

    # Compare with torch.topk reference
    ref_values, ref_indices = torch.topk(x, k=8, dim=-1)
    ref_softmax = torch.softmax(ref_values.float(), dim=-1).to(torch.bfloat16)

    # Values should match after softmax
    torch.testing.assert_close(values, ref_softmax, rtol=1e-2, atol=1e-2)

    # Verify indices point to valid top-k values
    # (exact ordering may differ for tied values, so we check the gathered values match)
    gathered = torch.gather(x, dim=1, index=indices.to(torch.int64))
    torch.testing.assert_close(gathered, ref_values, rtol=0, atol=0)


def test_topk_shapes(device):
    """Test topk output shapes are correct."""
    from kestrel_kernels.topk import topk_fwd

    batch_size = 16
    N = 64
    k = 8

    x = torch.randn(batch_size, N, dtype=torch.bfloat16, device=device)
    values, indices = topk_fwd(x, k=k, softmax=True)

    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)
    assert values.dtype == torch.bfloat16
    assert indices.dtype == torch.int32
