"""Shared CUDA graph management for autoregressive decode paths."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from typing import Any, Generic, TypeVar

import torch

from kestrel.device import resolve_device


logger = logging.getLogger(__name__)

SlotT = TypeVar("SlotT")


def make_decode_graph_batch_sizes(max_batch: int) -> list[int]:
    """Return the decode batch buckets captured for CUDA graph replay."""
    max_batch = max(1, max_batch)
    seeds = [size for size in (1, 2, 4, 8) if size <= max_batch]
    ramps = list(range(16, max_batch + 1, 16))
    return sorted({*seeds, *ramps, max_batch})


class DecodeGraphManager(Generic[SlotT]):
    """Own CUDA graph capture/replay for fixed-buffer decode slots.

    The manager is model-agnostic: runtimes provide callbacks for slot-specific
    staging, per-step metadata construction, and the actual decode forward.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        max_batch: int,
        graph_capture_lock: Any,
        compute_stream: torch.cuda.Stream | None,
        run_forward: Callable[[SlotT, int], None],
        prepare_step: Callable[[SlotT, int], None],
        zero_padding: Callable[[SlotT, int, int], None] | None = None,
        zero_for_capture: Callable[[SlotT], None] | None = None,
        eager_batch_sizes: Collection[int] = (),
    ) -> None:
        self.enabled = enabled
        self.device = resolve_device(device)
        self.max_batch = max(1, max_batch)
        self._graph_capture_lock = graph_capture_lock
        self._compute_stream = compute_stream
        self._run_forward = run_forward
        self._prepare_step = prepare_step
        self._zero_padding = zero_padding or _noop_padding
        self._zero_for_capture = zero_for_capture
        # Batch buckets served by a runtime-owned eager path. They do not need a
        # redundant CUDA graph; all other buckets retain native graph replay.
        self._eager_batch_sizes = frozenset(int(size) for size in eager_batch_sizes)
        self._batch_sizes: list[int] = []
        self._graphs: dict[int, dict[int, Any]] = {}

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._batch_sizes)

    def clear(self) -> None:
        """Drop all captured decode graphs."""
        self._graphs.clear()
        self._batch_sizes = []

    def ensure_ready(self, slots: Sequence[SlotT]) -> None:
        """Capture missing graph sets for ``slots`` when graph replay is enabled."""
        if not self.enabled:
            return
        if not slots:
            return
        if not self._batch_sizes:
            self._batch_sizes = make_decode_graph_batch_sizes(self.max_batch)
        if self._zero_for_capture is None:
            raise RuntimeError("CUDA graph capture requires zero_for_capture")

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        for slot in slots:
            # Decode slots are runtime-owned, long-lived objects; clear() drops
            # graph state before any slot rebuild, so object identity is stable.
            key = id(slot)
            if key not in self._graphs:
                self._graphs[key] = self._capture_slot_graphs(slot)

    def run(self, slot: SlotT, batch_size: int) -> None:
        """Prepare one decode step and execute via graph replay or eager forward."""
        graphs = self._graphs.get(id(slot)) if self.enabled else None
        if graphs is None:
            graph_batch_size = batch_size
        else:
            graph_batch_size = self._select_batch_size(batch_size)
            if graph_batch_size is None:
                raise RuntimeError(
                    f"Batch size {batch_size} exceeds max graph capacity "
                    f"{self._batch_sizes[-1] if self._batch_sizes else 0}"
                )
            # A bucket owned by an eager path has no captured graph by design.
            if (
                graph_batch_size not in graphs
                and graph_batch_size not in self._eager_batch_sizes
            ):
                raise RuntimeError(
                    f"No CUDA graph captured for batch size {graph_batch_size}"
                )

        if graph_batch_size > batch_size:
            self._zero_padding(slot, batch_size, graph_batch_size)

        self._prepare_step(slot, graph_batch_size)

        if graphs is None or graph_batch_size in self._eager_batch_sizes:
            # No graph globally, or this bucket belongs to an eager path.
            self._run_forward(slot, graph_batch_size)
        else:
            graphs[graph_batch_size].replay()

    def _select_batch_size(self, batch_size: int) -> int | None:
        for size in self._batch_sizes:
            if size >= batch_size:
                return size
        return None

    def _capture_slot_graphs(self, slot: SlotT) -> dict[int, Any]:
        cuda_graphs: dict[int, Any] = {}

        with self._graph_capture_lock:
            stream = self._compute_stream
            if stream is None:
                raise RuntimeError("CUDA graph capture requires a decode compute stream")

            # Capture on the same stream used for replay so kernels see metadata
            # writes in the same order as eager decode.
            with torch.cuda.stream(stream):
                zero_for_capture = self._zero_for_capture
                if zero_for_capture is None:
                    raise RuntimeError("CUDA graph capture requires zero_for_capture")
                zero_for_capture(slot)
                try:
                    torch.cuda.synchronize(device=self.device)
                    reserved_before = (
                        torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
                    )
                    for batch_size in reversed(self._batch_sizes):
                        if batch_size in self._eager_batch_sizes:
                            # The runtime serves this bucket eagerly, so it needs no graph pool.
                            continue
                        graph = torch.cuda.CUDAGraph()
                        with torch.inference_mode():
                            self._prepare_step(slot, batch_size)

                            self._run_forward(slot, batch_size)
                            torch.cuda.synchronize(device=self.device)

                            with torch.cuda.graph(graph):
                                self._run_forward(slot, batch_size)

                        cuda_graphs[batch_size] = graph
                        torch.cuda.synchronize(device=self.device)
                finally:
                    zero_for_capture(slot)

        if self._eager_batch_sizes:
            reserved_after = (
                torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            )
            logger.info(
                "decode-graph capture: %d eager bucket(s) %s omitted; captured %d bucket(s) "
                "using %.1f MiB of graph pool",
                len(self._eager_batch_sizes), sorted(self._eager_batch_sizes), len(cuda_graphs),
                max(0, reserved_after - reserved_before) / (1024 * 1024),
            )

        return cuda_graphs


def _noop_padding(_slot: object, _batch_size: int, _graph_batch_size: int) -> None:
    pass
