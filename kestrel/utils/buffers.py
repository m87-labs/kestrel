"""Shared host/device buffer helpers."""


from math import prod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major strides for a given shape."""
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


class FixedBuffer:
    """Device-aware buffer that is pre-allocated once and never resized.

    After the first allocation, requesting a larger size will raise an error.
    This ensures CUDA graph replay uses stable pointers.

    Uses as_strided for faster view creation compared to slice+view.
    """

    def __init__(self, name: str = "Buffer") -> None:
        self._tensor: torch.Tensor | None = None
        self._name = name

    def get(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = prod(shape)
        if numel == 0:
            return torch.empty(shape, device=device, dtype=dtype)

        if self._tensor is None:
            # First allocation - create the buffer
            self._tensor = torch.empty(numel, device=device, dtype=dtype)
        elif self._tensor.numel() < numel:
            # Buffer too small - this is a bug, workspaces should be pre-allocated
            raise RuntimeError(
                f"{self._name} overflow: requested {numel} elements but "
                f"only {self._tensor.numel()} allocated. This indicates the buffer "
                f"was not pre-allocated to sufficient size before CUDA graph capture."
            )
        elif self._tensor.device != device or self._tensor.dtype != dtype:
            raise RuntimeError(
                f"{self._name} device/dtype mismatch: buffer is on {self._tensor.device} "
                f"with dtype {self._tensor.dtype}, but requested {device} with {dtype}."
            )
        # Use as_strided for faster view creation (avoids intermediate slice object)
        return torch.as_strided(self._tensor, shape, _compute_strides(shape))


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
