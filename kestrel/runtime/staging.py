"""Reusable host execution and fixed-capacity tensor staging."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

import torch

from kestrel.utils import CpuGpuBuffer


class AsyncPreprocessor:
    def __init__(self, process: Callable[[Any], Any], *, workers: int) -> None:
        self._process = process
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="kestrel-preprocess",
        )

    def submit(self, value: Any) -> Future[Any]:
        return self._executor.submit(self._process, value)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class BatchedTensorStager:
    """Own stable pinned/device buffers for a named tensor record."""

    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device,
        with_numpy: Mapping[str, bool] | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.with_numpy = dict(with_numpy or {})
        self.buffers: dict[str, CpuGpuBuffer] = {}

    def stage(
        self,
        rows: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if not rows or len(rows) > self.capacity:
            raise ValueError(
                f"staging rows must lie in [1, {self.capacity}], got {len(rows)}"
            )
        names = tuple(rows[0])
        if not names or any(tuple(row) != names for row in rows):
            raise ValueError("staged tensor records must have identical fields")

        result = {}
        for name in names:
            first = rows[0][name]
            expected_shape = (self.capacity, *first.shape)
            buffer = self.buffers.get(name)
            if buffer is None:
                buffer = CpuGpuBuffer(
                    *expected_shape,
                    dtype=first.dtype,
                    device=self.device,
                    pin_memory=True,
                    with_numpy=self.with_numpy.get(name, True),
                    zero=False,
                )
                self.buffers[name] = buffer
            elif (
                tuple(buffer.cpu.shape) != expected_shape
                or buffer.cpu.dtype != first.dtype
            ):
                raise RuntimeError(
                    f"staging field {name!r} changed shape or dtype"
                )
            for index, row in enumerate(rows):
                value = row[name]
                if value.shape != first.shape or value.dtype != first.dtype:
                    raise ValueError(
                        f"staged field {name!r} must share shape and dtype"
                    )
                buffer.cpu[index].copy_(value)
            result[name] = buffer.copy_to_gpu(len(rows))
        return result


__all__ = ["AsyncPreprocessor", "BatchedTensorStager"]
