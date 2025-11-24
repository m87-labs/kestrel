"""Shared host/device buffer helpers."""


from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


class CpuGpuBuffer:
    """Allocate matching CPU/GPU tensors with optional NumPy view."""

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray

        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "bfloat16 torch tensors cannot be represented as NumPy arrays. "
                    "Instantiate CpuGpuBuffer(..., with_numpy=False) instead."
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """Non-blocking copy from device to host (caller must synchronize)."""

        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)
