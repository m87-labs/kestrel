"""The bf16 default resolves to fp32 on CPUs without a native bf16 GEMM."""

import torch

from kestrel import config as kconfig


def test_cpu_default_dtype_follows_native_bf16(monkeypatch) -> None:
    monkeypatch.setattr(kconfig, "cpu_has_native_bf16", lambda: True)
    assert kconfig.cpu_default_dtype() is torch.bfloat16
    monkeypatch.setattr(kconfig, "cpu_has_native_bf16", lambda: False)
    assert kconfig.cpu_default_dtype() is torch.float32


def test_probe_never_raises() -> None:
    assert isinstance(kconfig.cpu_has_native_bf16(), bool)
