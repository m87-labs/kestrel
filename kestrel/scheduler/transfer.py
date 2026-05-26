"""Async D2H transfer management for decode outputs."""

from typing import Sequence

import torch
from torch import Tensor

from kestrel.device import make_event, stream_context
from kestrel.runtime.sampling import AuxBufferSpec


class TransferHandle:
    """Handle for an in-flight D2H transfer."""

    def __init__(
        self,
        event: torch.cuda.Event,
        token_ids: Tensor,
        aux_values: list[Tensor],
        logprobs: Tensor | None,
        count: int,
    ) -> None:
        self._event = event
        self._token_ids = token_ids
        self._aux_values = aux_values
        self._logprobs = logprobs
        self._count = count

    def wait(self) -> tuple[Tensor, list[Tensor], Tensor | None]:
        """Block until D2H transfer completes and return the CPU tensors."""
        if self._count == 0:
            empty_aux = [a[:0] for a in self._aux_values]
            logprobs = None if self._logprobs is None else self._logprobs[:0]
            return self._token_ids[:0], empty_aux, logprobs
        self._event.synchronize()
        logprobs = None
        if self._logprobs is not None:
            logprobs = self._logprobs[: self._count]
        return (
            self._token_ids[: self._count],
            [a[: self._count] for a in self._aux_values],
            logprobs,
        )


class RenderBuffer:
    """Pinned host buffers for sampled ids + runtime-declared aux values."""

    def __init__(
        self,
        max_batch: int,
        device: torch.device,
        *,
        aux_specs: Sequence[AuxBufferSpec],
        copy_stream: torch.cuda.Stream,
    ) -> None:
        # Pinned host memory accelerates async H2D/D2H copies on CUDA but
        # isn't supported (and triggers ``DispatchStub: missing kernel for
        # mps``) when an MPS device is active even with ``device="cpu"``.
        # Drop the hint off CUDA — same allocation, no pinning.
        pin = device.type == "cuda"
        self._token_ids = torch.empty(
            (max_batch,), dtype=torch.long, device="cpu", pin_memory=pin,
        )
        self._aux_values: list[Tensor] = [
            torch.empty((max_batch, spec.width), dtype=spec.dtype, device="cpu", pin_memory=pin)
            for spec in aux_specs
        ]
        self._logprobs = torch.empty(
            (max_batch,), dtype=torch.float32, device="cpu", pin_memory=pin,
        )
        self._device = device
        self._stream = copy_stream
        self._event = make_event(device, enable_timing=False, blocking=False)

    def transfer(
        self,
        token_ids: Tensor,
        aux_tensors: Sequence[Tensor],
        *,
        ready_event: torch.cuda.Event,
        logprobs: Tensor | None = None,
    ) -> TransferHandle:
        """Start a D2H transfer and return a handle to wait on completion.

        ``aux_tensors`` must match (in order, dtype, and width) the
        ``aux_specs`` this buffer was constructed with — pass an empty
        sequence when the runtime declares no aux buffers.
        """
        if len(aux_tensors) != len(self._aux_values):
            raise AssertionError(
                f"transfer expected {len(self._aux_values)} aux tensors, got {len(aux_tensors)}"
            )
        count = int(token_ids.shape[0])
        if count == 0:
            return TransferHandle(
                self._event,
                self._token_ids,
                self._aux_values,
                self._logprobs if logprobs is not None else None,
                0,
            )

        with stream_context(self._stream):
            # Wait on the specific step's completion event (not wait_stream).
            # This anchors the dependency to exactly this step's GPU writes,
            # independent of any later work enqueued on the compute stream.
            if self._stream is not None:
                self._stream.wait_event(ready_event)
            self._token_ids[:count].copy_(token_ids, non_blocking=True)
            for host_buf, dev_buf in zip(self._aux_values, aux_tensors):
                host_buf[:count].copy_(dev_buf, non_blocking=True)
            if logprobs is not None:
                self._logprobs[:count].copy_(logprobs, non_blocking=True)
            self._event.record(self._stream)
        return TransferHandle(
            self._event,
            self._token_ids,
            self._aux_values,
            self._logprobs if logprobs is not None else None,
            count,
        )
