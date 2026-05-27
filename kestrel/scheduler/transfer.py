"""Async D2H transfer for sampled token ids + logprobs.

Per-step model-specific values (e.g. Moondream's coord/size decode
output) are owned and transferred by the runtime — the scheduler-side
transfer here is generic over model and only carries the universal
sampled-id + logprob pair.
"""

import torch
from torch import Tensor

from kestrel.device import make_event, stream_context


class TransferHandle:
    """Handle for an in-flight D2H transfer of sampled ids + logprobs."""

    def __init__(
        self,
        event: torch.cuda.Event,
        token_ids: Tensor,
        logprobs: Tensor | None,
        count: int,
    ) -> None:
        self._event = event
        self._token_ids = token_ids
        self._logprobs = logprobs
        self._count = count

    def wait(self) -> tuple[Tensor, Tensor | None]:
        """Block until D2H completes and return the CPU-side ids + logprobs."""
        if self._count == 0:
            logprobs = None if self._logprobs is None else self._logprobs[:0]
            return self._token_ids[:0], logprobs
        self._event.synchronize()
        logprobs = None if self._logprobs is None else self._logprobs[: self._count]
        return self._token_ids[: self._count], logprobs


class RenderBuffer:
    """Pinned host buffers for sampled token ids + logprobs."""

    def __init__(
        self,
        max_batch: int,
        device: torch.device,
        *,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        # Pinned host memory accelerates async H2D/D2H copies on CUDA
        # but isn't supported (and triggers ``DispatchStub: missing
        # kernel for mps``) when an MPS device is active even with
        # ``device="cpu"``. Drop the hint off CUDA — same allocation,
        # no pinning.
        pin = device.type == "cuda"
        self._token_ids = torch.empty(
            (max_batch,), dtype=torch.long, device="cpu", pin_memory=pin,
        )
        self._logprobs = torch.empty(
            (max_batch,), dtype=torch.float32, device="cpu", pin_memory=pin,
        )
        self._device = device
        self._stream = copy_stream
        self._event = make_event(device, enable_timing=False, blocking=False)

    def transfer(
        self,
        token_ids: Tensor,
        *,
        ready_event: torch.cuda.Event,
        logprobs: Tensor | None = None,
    ) -> TransferHandle:
        """Start a D2H transfer of ``token_ids`` (and ``logprobs`` if given).

        ``ready_event`` is recorded on the compute stream after all GPU
        writes to ``token_ids`` complete; the copy stream waits on it
        before kicking the copy. This anchors the dependency to exactly
        this step's writes, independent of any later compute work.
        """
        count = int(token_ids.shape[0])
        if count == 0:
            return TransferHandle(
                self._event,
                self._token_ids,
                self._logprobs if logprobs is not None else None,
                0,
            )

        with stream_context(self._stream):
            if self._stream is not None:
                self._stream.wait_event(ready_event)
            self._token_ids[:count].copy_(token_ids, non_blocking=True)
            if logprobs is not None:
                self._logprobs[:count].copy_(logprobs, non_blocking=True)
            self._event.record(self._stream)
        return TransferHandle(
            self._event,
            self._token_ids,
            self._logprobs if logprobs is not None else None,
            count,
        )
