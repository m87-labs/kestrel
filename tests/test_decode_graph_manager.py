from __future__ import annotations

import contextlib
from dataclasses import dataclass

import pytest
import torch

from kestrel.runtime.decode_graph import (
    DecodeGraphManager,
    make_decode_graph_batch_sizes,
)


@dataclass
class FakeSlot:
    slot_id: int


class FakeGraph:
    def __init__(self, events: list[tuple], slot_id: int, batch_size: int) -> None:
        self._events = events
        self._slot_id = slot_id
        self._batch_size = batch_size

    def replay(self) -> None:
        self._events.append(("replay", self._slot_id, self._batch_size))


class FakeDecodeGraphManager(DecodeGraphManager[FakeSlot]):
    def __init__(self, *args, events: list[tuple], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events = events

    def _capture_slot_graphs(self, slot: FakeSlot) -> dict[int, FakeGraph]:
        # Mirror the real method's capture-skip: buckets the plan serves eagerly get no captured
        # graph (the real path also runs the CUDA graph mechanics + memory log, which need a GPU).
        eager = (
            frozenset(int(b) for b in self._plan_eager(slot, list(self.batch_sizes)))
            if self._plan_eager is not None
            else frozenset()
        )
        self._eager_buckets[id(slot)] = eager
        captured = tuple(b for b in self.batch_sizes if b not in eager)
        self.events.append(("capture", slot.slot_id, captured))
        return {
            batch_size: FakeGraph(self.events, slot.slot_id, batch_size)
            for batch_size in captured
        }


def _make_manager(
    events: list[tuple],
    *,
    enabled: bool,
    max_batch: int,
    capture: bool = False,
    plan_eager=None,
) -> DecodeGraphManager[FakeSlot]:
    cls = FakeDecodeGraphManager if capture else DecodeGraphManager
    kwargs = {"events": events} if capture else {}
    return cls(
        enabled=enabled,
        device=torch.device("cpu"),
        max_batch=max_batch,
        graph_capture_lock=contextlib.nullcontext(),
        compute_stream=None,
        prepare_step=lambda slot, batch_size: events.append(
            ("prepare", slot.slot_id, batch_size)
        ),
        run_forward=lambda slot, batch_size: events.append(
            ("forward", slot.slot_id, batch_size)
        ),
        zero_for_capture=lambda _slot: None,
        zero_padding=lambda slot, batch_size, graph_batch_size: events.append(
            ("pad", slot.slot_id, batch_size, graph_batch_size)
        ),
        plan_eager=plan_eager,
        **kwargs,
    )


def test_make_decode_graph_batch_sizes() -> None:
    assert make_decode_graph_batch_sizes(1) == [1]
    assert make_decode_graph_batch_sizes(3) == [1, 2, 3]
    assert make_decode_graph_batch_sizes(16) == [1, 2, 4, 8, 16]
    assert make_decode_graph_batch_sizes(33) == [1, 2, 4, 8, 16, 32, 33]


def test_run_without_graphs_prepares_and_forwards_exact_batch() -> None:
    events: list[tuple] = []
    manager = _make_manager(events, enabled=False, max_batch=8)

    manager.run(FakeSlot(slot_id=0), 3)

    assert events == [
        ("prepare", 0, 3),
        ("forward", 0, 3),
    ]


def test_run_replays_padded_graph_batch() -> None:
    events: list[tuple] = []
    manager = _make_manager(events, enabled=True, max_batch=8, capture=True)
    slot = FakeSlot(slot_id=0)

    manager.ensure_ready([slot])
    manager.run(slot, 3)

    assert events == [
        ("capture", 0, (1, 2, 4, 8)),
        ("pad", 0, 3, 4),
        ("prepare", 0, 4),
        ("replay", 0, 4),
    ]


def test_enabled_capture_requires_zero_for_capture() -> None:
    events: list[tuple] = []
    manager = DecodeGraphManager[FakeSlot](
        enabled=True,
        device=torch.device("cpu"),
        max_batch=2,
        graph_capture_lock=contextlib.nullcontext(),
        compute_stream=None,
        prepare_step=lambda _slot, _batch_size: None,
        run_forward=lambda _slot, _batch_size: None,
    )

    with pytest.raises(RuntimeError, match="requires zero_for_capture"):
        manager.ensure_ready([FakeSlot(slot_id=0)])


def test_clear_forces_recapture() -> None:
    events: list[tuple] = []
    manager = _make_manager(events, enabled=True, max_batch=2, capture=True)
    slot = FakeSlot(slot_id=0)

    manager.ensure_ready([slot])
    manager.ensure_ready([slot])
    manager.clear()
    manager.ensure_ready([slot])

    assert events == [
        ("capture", 0, (1, 2)),
        ("capture", 0, (1, 2)),
    ]


def test_plan_eager_skips_capture_and_routes_bucket_to_forward() -> None:
    # An eager-served bucket (the megakernel B1) is NOT captured, and run() routes it to the eager
    # forward instead of a graph replay; a non-eager bucket still replays its captured graph.
    events: list[tuple] = []
    planned: list[tuple] = []

    def plan_eager(slot, batch_sizes):
        planned.append((slot.slot_id, tuple(batch_sizes)))
        return {1}

    manager = _make_manager(
        events, enabled=True, max_batch=8, capture=True, plan_eager=plan_eager
    )
    slot = FakeSlot(slot_id=0)
    manager.ensure_ready([slot])

    # Bucket 1 was warmed + skipped: no captured graph, tracked as eager.
    assert planned == [(0, (1, 2, 4, 8))]
    assert manager.eager_buckets(slot) == frozenset({1})
    assert ("capture", 0, (2, 4, 8)) in events

    # B1 decode step routes eager (forward), NOT a replay.
    events.clear()
    manager.run(slot, 1)
    assert events == [("prepare", 0, 1), ("forward", 0, 1)]

    # A native bucket still replays its captured graph.
    events.clear()
    manager.run(slot, 4)
    assert events == [("prepare", 0, 4), ("replay", 0, 4)]


def test_plan_eager_exclusion_falls_back_to_native_capture() -> None:
    # A bucket the plan does NOT serve (e.g. warmup build failed, or ineligible B8) keeps the
    # captured native path exactly as before.
    events: list[tuple] = []
    manager = _make_manager(
        events, enabled=True, max_batch=8, capture=True, plan_eager=lambda s, bs: set()
    )
    slot = FakeSlot(slot_id=0)
    manager.ensure_ready([slot])

    assert manager.eager_buckets(slot) == frozenset()
    assert ("capture", 0, (1, 2, 4, 8)) in events
    events.clear()
    manager.run(slot, 8)
    assert events == [("prepare", 0, 8), ("replay", 0, 8)]


def test_clear_drops_eager_buckets() -> None:
    events: list[tuple] = []
    manager = _make_manager(
        events, enabled=True, max_batch=2, capture=True, plan_eager=lambda s, bs: {1}
    )
    slot = FakeSlot(slot_id=0)
    manager.ensure_ready([slot])
    assert manager.eager_buckets(slot) == frozenset({1})
    manager.clear()
    assert manager.eager_buckets(slot) == frozenset()


def test_run_raises_when_batch_exceeds_graph_capacity() -> None:
    events: list[tuple] = []
    manager = _make_manager(events, enabled=True, max_batch=2, capture=True)
    slot = FakeSlot(slot_id=0)

    manager.ensure_ready([slot])

    with pytest.raises(RuntimeError, match="exceeds max graph capacity"):
        manager.run(slot, 3)
