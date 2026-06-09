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
        self.events.append(("capture", slot.slot_id, tuple(self.batch_sizes)))
        return {
            batch_size: FakeGraph(self.events, slot.slot_id, batch_size)
            for batch_size in self.batch_sizes
        }


def _make_manager(
    events: list[tuple],
    *,
    enabled: bool,
    max_batch: int,
    capture: bool = False,
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


def test_run_raises_when_batch_exceeds_graph_capacity() -> None:
    events: list[tuple] = []
    manager = _make_manager(events, enabled=True, max_batch=2, capture=True)
    slot = FakeSlot(slot_id=0)

    manager.ensure_ready([slot])

    with pytest.raises(RuntimeError, match="exceeds max graph capacity"):
        manager.run(slot, 3)
