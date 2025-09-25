"""GPU memory debugging utilities."""

from __future__ import annotations

import os
from typing import Optional

import torch

_DEBUG_ENV = "KSTREL_DEBUG_GPU_MEM"


def _should_log() -> bool:
    return bool(os.getenv(_DEBUG_ENV))


def log_gpu_memory(label: str, device: torch.device) -> None:
    """Emit a one-line summary of current GPU memory usage if debugging is enabled."""

    if not _should_log():
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"[gpu-mem] {label}: alloc={allocated / 1e6:.1f}MB reserved={reserved / 1e6:.1f}MB "
        f"max_alloc={max_allocated / 1e6:.1f}MB max_reserved={max_reserved / 1e6:.1f}MB "
        f"free={free_bytes / 1e6:.1f}MB total={total_bytes / 1e6:.1f}MB"
    )


def reset_peak_gpu_memory(device: Optional[torch.device]) -> None:
    """Reset peak memory stats when debugging is enabled."""

    if not _should_log():
        return
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


__all__ = ["log_gpu_memory", "reset_peak_gpu_memory"]
