"""Shared pytest fixtures for kestrel-kernels tests."""

import pytest
import torch


@pytest.fixture
def device():
    """CUDA device fixture that skips if CUDA unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def sm90_device():
    """CUDA device fixture that requires SM90+ (Hopper)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("Requires SM90+ (Hopper or newer)")
    return torch.device("cuda", torch.cuda.current_device())
