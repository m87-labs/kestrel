"""Shared CUDA graph management for autoregressive decode paths."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
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
        plan_eager: Callable[[SlotT, list[int]], set[int]] | None = None,
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
        # Optional: given a slot and the batch buckets about to be captured, return the SUBSET this
        # runtime will serve EAGERLY instead of a captured native graph (the whole-model decode
        # megakernel). Those buckets are NOT captured (no graph, no graph-pool memory) and ``run``
        # routes them to the eager ``run_forward`` instead of a graph replay. The callback also
        # WARMS the eager path (builds its session) and MUST exclude any bucket whose warmup fails,
        # so a build failure transparently falls back to native capture.
        self._plan_eager = plan_eager
        self._batch_sizes: list[int] = []
        self._graphs: dict[int, dict[int, Any]] = {}
        # Per-slot buckets served eagerly (megakernel), so no native graph was captured for them.
        self._eager_buckets: dict[int, frozenset[int]] = {}

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._batch_sizes)

    def clear(self) -> None:
        """Drop all captured decode graphs."""
        self._graphs.clear()
        self._eager_buckets.clear()
        self._batch_sizes = []

    def eager_buckets(self, slot: SlotT) -> frozenset[int]:
        """Buckets served eagerly (megakernel) for ``slot`` -- i.e. buckets for which NO native decode
        graph was captured. Empty when there is no eager plan or graphs are disabled."""
        return self._eager_buckets.get(id(slot), frozenset())

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
        eager = self._eager_buckets.get(id(slot), frozenset())
        if graphs is None:
            graph_batch_size = batch_size
        else:
            graph_batch_size = self._select_batch_size(batch_size)
            if graph_batch_size is None:
                raise RuntimeError(
                    f"Batch size {batch_size} exceeds max graph capacity "
                    f"{self._batch_sizes[-1] if self._batch_sizes else 0}"
                )
            # A bucket served eagerly (megakernel) has NO captured graph by design; run it eager.
            if graph_batch_size not in graphs and graph_batch_size not in eager:
                raise RuntimeError(
                    f"No CUDA graph captured for batch size {graph_batch_size}"
                )

        if graph_batch_size > batch_size:
            self._zero_padding(slot, batch_size, graph_batch_size)

        self._prepare_step(slot, graph_batch_size)

        if graphs is None or graph_batch_size in eager:
            # Eager: no graph (globally disabled) or an eager-served bucket -> run the forward, which
            # routes the megakernel-served bucket through the megakernel each step.
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
                    # Decide (and WARM) the eager megakernel buckets BEFORE any capture. This runs
                    # OUTSIDE ``torch.cuda.graph`` (it builds a session + does one eager launch, both
                    # illegal under capture) and at the SAME warmup point capture used to touch these
                    # buckets. Any bucket whose warmup fails is excluded -> it is captured native.
                    eager = self._plan_eager(slot, list(self._batch_sizes)) if self._plan_eager else set()
                    eager = frozenset(int(b) for b in eager)
                    self._eager_buckets[id(slot)] = eager
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(device=self.device)
                    reserved_before = (
                        torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
                    )
                    for batch_size in reversed(self._batch_sizes):
                        if batch_size in eager:
                            # Served eagerly by the megakernel: capture no native graph, allocate no
                            # graph pool for this bucket.
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

        if eager:
            reserved_after = (
                torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            )
            logger.info(
                "decode-graph capture: %d bucket(s) served eagerly by the megakernel %s "
                "(no native graph captured); captured %d native bucket(s) using %.1f MiB of graph "
                "pool. Skipping capture for the eager buckets avoids their graph pools entirely.",
                len(eager), sorted(eager), len(cuda_graphs),
                max(0, reserved_after - reserved_before) / (1024 * 1024),
            )

        return cuda_graphs


def _noop_padding(_slot: object, _batch_size: int, _graph_batch_size: int) -> None:
    pass
