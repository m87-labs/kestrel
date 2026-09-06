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
        self._stream = stream
        self._run_forward = run_forward
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[_InputKey, _GraphEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        if self.enabled and (self.device.type != "cuda" or self._stream is None):
            raise ValueError("single-pass graph replay requires a CUDA stream")

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
    def launch(self, *inputs: Tensor) -> Iterator[tuple[Tensor, ...]]:
        """Stage one exact-shape call and lease its outputs to the caller."""
        values = tuple(inputs)
        with self._lock:
            if self._closed:
                raise RuntimeError("single-pass graph session is shut down")
            if not self.enabled:
                yield self._outputs(self._run_forward(*values))
                return

            stream = self._stream
            if stream is None:
                raise RuntimeError("single-pass graph launch has no CUDA stream")
            # Keep the caller in the owned stream context for the entire lease,
            # so consumers are ordered after replay before stable outputs can be
            # staged again by another caller.
            with torch.cuda.device(self.device), stream_context(stream):
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
                    for destination, source in zip(entry.inputs, values, strict=True):
                        destination.copy_(source)
                    entry.graph.replay()
                yield entry.outputs

    def shutdown(self) -> None:
        """Synchronize the owned stream and release graph pools and buffers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.enabled:
                stream = self._stream
                if stream is None:
                    raise RuntimeError("single-pass graph shutdown has no CUDA stream")
                stream.synchronize()
            self._entries.clear()
            self._run_forward = lambda *_args: ()


__all__ = ["FixedShapeSinglePassGraph"]
