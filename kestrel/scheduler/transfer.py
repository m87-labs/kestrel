"""Async D2H transfer management for decode outputs."""

import torch
from torch import Tensor


class TransferHandle:
    """Handle for an in-flight D2H transfer."""

    def __init__(
        self,
        event: torch.cuda.Event,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
        count: int,
    ) -> None:
        self._event = event
        self._token_ids = token_ids
        self._coord_values = coord_values
        self._size_values = size_values
        self._count = count

    def wait(self) -> tuple[Tensor, Tensor, Tensor]:
        """Block until D2H transfer completes and return the CPU tensors."""
        if self._count == 0:
            empty = self._token_ids[:0]
            return empty, self._coord_values[:0], self._size_values[:0]
        self._event.synchronize()
        return (
            self._token_ids[: self._count],
            self._coord_values[: self._count],
            self._size_values[: self._count],
        )


class RenderBuffer:
    """Pinned host buffers for sampled ids + decoded coord/size values."""

    def __init__(
        self,
        max_batch: int,
        device: torch.device,
        *,
        coord_dtype: torch.dtype,
        size_dtype: torch.dtype,
    ) -> None:
        self._token_ids = torch.empty(
            (max_batch,),
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        self._coord_values = torch.empty(
            (max_batch, 1),
            dtype=coord_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._size_values = torch.empty(
            (max_batch, 2),
            dtype=size_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event(enable_timing=False, blocking=False)

    def transfer(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
    ) -> TransferHandle:
        """Start a D2H transfer and return a handle to wait on completion."""
        count = int(token_ids.shape[0])
        if count == 0:
            return TransferHandle(
                self._event,
                self._token_ids,
                self._coord_values,
                self._size_values,
                0,
            )

        current = torch.cuda.current_stream(device=self._device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(current)
            self._token_ids[:count].copy_(token_ids, non_blocking=True)
            self._coord_values[:count].copy_(coord_values, non_blocking=True)
            self._size_values[:count].copy_(size_values, non_blocking=True)
            self._event.record(self._stream)
        return TransferHandle(
            self._event, self._token_ids, self._coord_values, self._size_values, count
        )
