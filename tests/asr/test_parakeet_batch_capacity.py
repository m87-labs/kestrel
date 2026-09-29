from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.models.parakeet_tdt.runtime import _batch_capacity


def test_cpu_keeps_eight() -> None:
    assert _batch_capacity(SimpleNamespace(), torch.device("cpu")) == 8


def test_configured_capacity_wins() -> None:
    assert _batch_capacity(SimpleNamespace(single_pass_batch_capacity=3), torch.device("cpu")) == 3


@pytest.mark.parametrize("value", [0, -1, 2.0])
def test_rejects_invalid_capacity(value) -> None:
    with pytest.raises(ValueError, match="single_pass_batch_capacity"):
        _batch_capacity(SimpleNamespace(single_pass_batch_capacity=value), torch.device("cpu"))
    with pytest.raises(ValueError, match="single_pass_batch_capacity"):
        RuntimeConfig(device="cpu", single_pass_batch_capacity=value)


def test_cuda_capacity_tiers_by_device_memory(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=24 * 2**30)
    )
    assert _batch_capacity(SimpleNamespace(), torch.device("cuda")) == 64
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=80 * 2**30)
    )
    assert _batch_capacity(SimpleNamespace(), torch.device("cuda")) == 128
