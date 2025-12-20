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
        copy_stream: torch.cuda.Stream,
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
        self._stream = copy_stream
        self._event = torch.cuda.Event(enable_timing=False, blocking=False)

    def transfer(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
        *,
        ready_event: torch.cuda.Event,
    ) -> TransferHandle:
        """Start a D2H transfer and return a handle to wait on completion.

        Args:
            token_ids: GPU tensor of sampled token IDs (from per-slot staging buffer).
            coord_values: GPU tensor of coord values (from per-slot staging buffer).
            size_values: GPU tensor of size values (from per-slot staging buffer).
            ready_event: Event recorded on compute stream after all GPU writes complete.
                The copy stream will wait on this event before starting D2H copies.
                This ensures correct ordering without capturing dependencies on
                later compute stream work (which would happen with wait_stream).
        """
        count = int(token_ids.shape[0])
        if count == 0:
            return TransferHandle(
                self._event,
                self._token_ids,
                self._coord_values,
                self._size_values,
                0,
            )

        with torch.cuda.stream(self._stream):
            # Wait on the specific step's completion event (not wait_stream).
            # This anchors the dependency to exactly this step's GPU writes,
            # independent of any later work enqueued on the compute stream.
            self._stream.wait_event(ready_event)
            self._token_ids[:count].copy_(token_ids, non_blocking=True)
            self._coord_values[:count].copy_(coord_values, non_blocking=True)
            self._size_values[:count].copy_(size_values, non_blocking=True)
            self._event.record(self._stream)
        return TransferHandle(
            self._event, self._token_ids, self._coord_values, self._size_values, count
        )
