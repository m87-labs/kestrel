"""Device-agnostic wrappers for the runtime primitives Kestrel uses.

Kestrel is CUDA-first; this module centralizes the handful of
``torch.cuda.*`` calls (streams, events, synchronization, allocator hints,
capability queries) so the runtime can drive an MPS device on macOS
without sprinkling ``device.type == "cuda"`` checks throughout.

On CUDA, every primitive forwards to its ``torch.cuda.*`` counterpart with
zero overhead. On MPS, streams/events become no-ops or thin wrappers
around ``torch.mps.*`` where an equivalent exists; CUDA-graph capture is
intentionally not wired (MPS has no equivalent) — callers gate that path
on ``runtime._use_cuda_graphs`` which evaluates to False on MPS.
"""

import contextlib
from typing import Optional

import torch


def set_device(device: torch.device) -> None:
    """``torch.cuda.set_device`` on CUDA; no-op on MPS/CPU."""
    if device.type == "cuda":
        torch.cuda.set_device(device)


def synchronize(device: torch.device) -> None:
    """Block the host until pending work on ``device`` finishes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    """Hint the allocator to release cached blocks; no-op when unsupported."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device_capability(device: torch.device) -> tuple[int, int]:
    """Return SM (major, minor) on CUDA; ``(0, 0)`` elsewhere.

    Callers that gate FP8 / SM90+ paths on a specific capability tuple keep
    working — non-CUDA devices answer "no, you don't have that hardware."
    """
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device)
    return (0, 0)


def make_stream(device: torch.device) -> Optional[torch.cuda.Stream]:
    """Create a CUDA stream for explicit pipelining.

    MPS has no exposed stream abstraction in ``torch.mps`` (torch 2.11);
    return ``None`` and let callers drop into the in-line execution path
    via :func:`stream_context`.
    """
    if device.type == "cuda":
        return torch.cuda.Stream(device=device)
    return None


@contextlib.contextmanager
def stream_context(stream: Optional[torch.cuda.Stream]):
    """Run the ``with`` block on ``stream`` if non-None, else inline."""
    if stream is None:
        yield
    else:
        with torch.cuda.stream(stream):
            yield


class NoopEvent:
    """Stand-in for ``torch.cuda.Event`` on devices without async events."""

    def record(self, *_args, **_kwargs) -> None:
        pass

    def wait(self, *_args, **_kwargs) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, _other: "NoopEvent") -> float:
        return 0.0


def make_event(
    device: torch.device,
    *,
    enable_timing: bool = False,
    blocking: bool = False,
) -> "torch.cuda.Event | NoopEvent":
    """Create a CUDA event on CUDA; a no-op stand-in elsewhere."""
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=enable_timing, blocking=blocking)
    return NoopEvent()


__all__ = [
    "NoopEvent",
    "empty_cache",
    "get_device_capability",
    "make_event",
    "make_stream",
    "set_device",
    "stream_context",
    "synchronize",
]
