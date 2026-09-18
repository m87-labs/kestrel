"""The bf16 default resolves to fp32 on CPUs without a native bf16 GEMM."""

import torch

from kestrel import config as kconfig


def test_cpu_default_dtype_follows_the_native_bf16_probes(monkeypatch) -> None:
    for probe in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        monkeypatch.setattr(torch.cpu, probe, lambda: False, raising=False)
    assert kconfig.cpu_default_dtype() is torch.float32
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: True, raising=False)
    assert kconfig.cpu_default_dtype() is torch.bfloat16


def test_probe_never_raises() -> None:
    assert kconfig.cpu_default_dtype() in (torch.bfloat16, torch.float32)
