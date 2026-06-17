"""Shared host/device buffer helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
        zero: bool = True,
    ) -> None:
        # Pinned memory is a CUDA-specific allocator hint that accelerates
        # async H2D copies; on MPS torch raises ``DispatchStub: missing
        # kernel for mps`` even with ``device="cpu"`` because the pin path
        # is wired through the active device. The hint has no effect off
        # CUDA, so just drop it.
        pin_memory_effective = pin_memory and device.type == "cuda"
        if zero:
            self.cpu = torch.zeros(
                *size,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory_effective,
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
        else:
            self.cpu = torch.empty(
                *size,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory_effective,
            )
            self.gpu = torch.empty_like(self.cpu, device=device)
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

    def cpu_view(self, shape: tuple[int, ...]) -> torch.Tensor:
        numel = prod(shape)
        return self.cpu[:numel].view(shape)

    def gpu_view(self, shape: tuple[int, ...]) -> torch.Tensor:
        numel = prod(shape)
        return self.gpu[:numel].view(shape)

    def copy_view_to_gpu(self, shape: tuple[int, ...]) -> torch.Tensor:
        cpu = self.cpu_view(shape)
        gpu = self.gpu_view(shape)
        gpu.copy_(cpu, non_blocking=True)
        return gpu

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """Non-blocking copy from device to host (caller must synchronize)."""

        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


@dataclass
class PackedField:
    """A named ``cpu``/``gpu``/``np`` view into a slice of a :class:`PackedBuffer`.

    Exposes the subset of the :class:`CpuGpuBuffer` surface that metadata
    staging code reads/writes; the three views share storage with the packed
    buffer, so host-side writes (via ``cpu`` or ``np``) are picked up by the
    parent buffer's single ``copy_to_gpu``.
    """

    cpu: torch.Tensor
    gpu: torch.Tensor
    np: np.ndarray | None


class PackedBuffer:
    """Pack several named tensors of mixed dtype/shape into one CPU/GPU buffer
    pair so the whole group stages to the device in a single ``cudaMemcpyAsync``
    instead of one copy per field.

    The backing store is a flat ``uint8`` buffer; each field is an aligned,
    contiguous ``view(dtype).view(shape)`` slice exposed as an attribute
    returning a :class:`PackedField`. This is the staging-time complement to
    :class:`CpuGpuBuffer`: reach for it wherever several small per-step or
    per-batch metadata tensors are filled on the host and shipped together
    (prefill/decode metadata, gather indices, …) to cut per-launch overhead.

    Fields keep their own ``cpu``/``gpu``/``np`` views, so call sites that
    already use ``CpuGpuBuffer`` fields (``field.np[...] = ...`` on the host,
    ``field.gpu[...]`` on the device) work unchanged; only the copy collapses
    from N calls to one ``copy_to_gpu()``.

    ``copy_to_gpu`` transfers the *entire* buffer (every field, full capacity)
    in one contiguous copy. Unused tails of a field are shipped but never read
    by consumers that slice to the live length, so size the buffer to the
    working set rather than a loose upper bound.
    """

    # Every field starts at an 8-byte boundary, which satisfies the alignment
    # requirement of ``Tensor.view(dtype)`` for any packed dtype up to 8 bytes.
    _ALIGN = 8

    def __init__(
        self,
        fields: Iterable[tuple[str, Sequence[int], torch.dtype]],
        *,
        device: torch.device,
        pin_memory: bool = False,
        with_numpy: bool = True,
    ) -> None:
        specs = [
            (str(name), tuple(int(s) for s in shape), dtype)
            for name, shape, dtype in fields
        ]
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for _, shape, dtype in specs:
            elem_size = torch.empty((), dtype=dtype).element_size()
            nbytes = elem_size * prod(shape)
            cursor = -(-cursor // self._ALIGN) * self._ALIGN  # round up to align
            offsets.append((cursor, nbytes))
            cursor += nbytes
        total = max(1, -(-cursor // self._ALIGN) * self._ALIGN)

        pin = pin_memory and device.type == "cuda"
        self.device = device
        self.nbytes = total
        self.cpu = torch.zeros(total, dtype=torch.uint8, device="cpu", pin_memory=pin)
        self.gpu = torch.zeros(total, dtype=torch.uint8, device=device)
        self._fields: dict[str, PackedField] = {}

        for (name, shape, dtype), (offset, nbytes) in zip(specs, offsets):
            raw_cpu = self.cpu[offset : offset + nbytes]
            raw_gpu = self.gpu[offset : offset + nbytes]
            cpu_view = raw_cpu.view(dtype).view(shape)
            gpu_view = raw_gpu.view(dtype).view(shape)
            np_view = (
                cpu_view.numpy()
                if with_numpy and dtype != torch.bfloat16
                else None
            )
            self._fields[name] = PackedField(cpu=cpu_view, gpu=gpu_view, np=np_view)

    def __getattr__(self, name: str) -> PackedField:
        # Only consulted when normal attribute lookup misses, so the real
        # instance attributes (cpu/gpu/device/_fields) resolve directly.
        fields = self.__dict__.get("_fields")
        if fields is not None and name in fields:
            return fields[name]
        raise AttributeError(name)

    def field(self, name: str) -> PackedField:
        return self._fields[name]

    def copy_to_gpu(self) -> torch.Tensor:
        """Stage every field to the device in one contiguous H2D copy."""
        return self.gpu.copy_(self.cpu, non_blocking=True)

    def copy_to_cpu(self) -> torch.Tensor:
        """Copy every field back to the host in one D2H copy (caller syncs)."""
        return self.cpu.copy_(self.gpu, non_blocking=True)
