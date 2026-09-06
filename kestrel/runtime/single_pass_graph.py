"""Bounded CUDA-graph replay for fixed-shape single-pass operations."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from kestrel.device import resolve_device, stream_context


_InputKey = tuple[tuple[tuple[int, ...], tuple[int, ...], torch.dtype], ...]


@dataclass(slots=True)
class _GraphEntry:
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    graph: torch.cuda.CUDAGraph


class FixedShapeSinglePassGraph:
    """Capture and replay a bounded set of exact tensor-shape calls.

    Returned tensors are session-owned stable buffers.  ``launch`` therefore
    holds the session lock until its context exits; callers must submit every
    consumer of the outputs inside that context.  This makes lazy capture and
    replay safe even when a runtime can be entered from more than one thread.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        stream: torch.cuda.Stream | None,
        run_forward: Callable[..., Sequence[Tensor]],
        max_entries: int = 4,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("single-pass graph max_entries must be positive")
        self.enabled = bool(enabled)
        self.device = resolve_device(device)
        if self.enabled and self.device.type != "cuda":
            raise ValueError("single-pass graph replay requires a CUDA device")
        if self.enabled and stream is None:
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
        if stream is not None:
            if self.device.type != "cuda":
                raise ValueError("single-pass graph stream requires a CUDA device")
            with torch.cuda.device(self.device):
                if stream.device != self.device:
                    raise ValueError("single-pass graph stream must share its device")
                if self.enabled and stream == torch.cuda.default_stream(self.device):
                    raise ValueError(
                        "single-pass graph replay requires a non-default stream"
                    )
        self._stream = stream
        self._run_forward = run_forward
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[_InputKey, _GraphEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._active_lease_thread: int | None = None
        self._closed = False

    @staticmethod
    def _outputs(value: Sequence[Tensor]) -> tuple[Tensor, ...]:
        outputs = tuple(value)
        if not outputs or any(not isinstance(output, Tensor) for output in outputs):
            raise TypeError("single-pass graph forward must return tensors")
        return outputs

    def _key(self, inputs: tuple[Tensor, ...]) -> _InputKey:
        if not inputs:
            raise ValueError("single-pass graph launch requires tensor inputs")
        for value in inputs:
            if not isinstance(value, Tensor):
                raise TypeError("single-pass graph inputs must be tensors")
            if value.device != self.device:
                raise ValueError("single-pass graph inputs must share its device")
        return tuple(
            (tuple(value.shape), tuple(value.stride()), value.dtype) for value in inputs
        )

    def _capture(self, inputs: tuple[Tensor, ...]) -> _GraphEntry:
        stream = self._stream
        if stream is None:
            raise RuntimeError("single-pass graph capture has no CUDA stream")
        static_inputs = tuple(
            torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device=value.device,
            )
            for value in inputs
        )
        with (
            torch.cuda.device(self.device),
            stream_context(stream),
            torch.inference_mode(),
        ):
            for destination, source in zip(static_inputs, inputs, strict=True):
                destination.copy_(source)
            # Materialize library handles and algorithm selection outside capture.
            self._outputs(self._run_forward(*static_inputs))
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self._outputs(self._run_forward(*static_inputs))
            graph.replay()
        return _GraphEntry(static_inputs, outputs, graph)

    @contextmanager
    def _ordered_stream(self) -> Iterator[None]:
        """Order the ambient CUDA stream before the configured session stream."""
        stream = self._stream
        if stream is None:
            yield
            return
        with torch.cuda.device(self.device):
            source_stream = torch.cuda.current_stream(self.device)
            ready = None
            if source_stream != stream:
                ready = torch.cuda.Event()
                ready.record(source_stream)
                stream.wait_event(ready)
            with stream_context(stream):
                yield
            del ready

    @contextmanager
    def launch(self, *inputs: Tensor) -> Iterator[tuple[Tensor, ...]]:
        """Stage one exact-shape call and lease its outputs to the caller.

        CUDA inputs must be ready on the ambient stream at context entry.  The
        session orders that stream before staging without synchronizing the host.
        """
        values = tuple(inputs)
        with self._lock:
            if self._closed:
                raise RuntimeError("single-pass graph session is shut down")
            thread_id = threading.get_ident()
            if self._active_lease_thread == thread_id:
                raise RuntimeError("single-pass graph leases cannot be nested")
            self._active_lease_thread = thread_id
            try:
                # Keep the caller in the configured stream context for the entire
                # lease, so eager and replay consumers share identical ordering.
                with self._ordered_stream():
                    if not self.enabled:
                        yield self._outputs(self._run_forward(*values))
                    else:
                        stream = self._stream
                        if stream is None:
                            raise RuntimeError(
                                "single-pass graph launch has no CUDA stream"
                            )
                        key = self._key(values)
                        entry = self._entries.get(key)
                        if entry is None:
                            if len(self._entries) >= self._max_entries:
                                stream.synchronize()
                                self._entries.popitem(last=False)
                            entry = self._capture(values)
                            self._entries[key] = entry
                        else:
                            self._entries.move_to_end(key)
                            for destination, source in zip(
                                entry.inputs, values, strict=True
                            ):
                                destination.copy_(source)
                            entry.graph.replay()
                        yield entry.outputs
            finally:
                self._active_lease_thread = None

    def shutdown(self) -> None:
        """Synchronize the configured stream and release retained state."""
        with self._lock:
            if self._closed:
                return
            if self._active_lease_thread == threading.get_ident():
                raise RuntimeError(
                    "single-pass graph cannot shut down during an active lease"
                )
            self._closed = True
            try:
                stream = self._stream
                if stream is not None:
                    stream.synchronize()
            finally:
                self._entries.clear()
                self._run_forward = lambda *_args: ()
                self._stream = None


__all__ = ["FixedShapeSinglePassGraph"]
