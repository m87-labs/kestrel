from __future__ import annotations

import pytest

from kestrel.kv_cache import PageTable
from kestrel.models.moondream import runtime as runtime_mod
from kestrel.runtime import SequenceState


def _runtime(
    *, max_seq_length: int = 64, page_size: int = 1
) -> runtime_mod.MoondreamRuntime:
    runtime = runtime_mod.MoondreamRuntime.__new__(runtime_mod.MoondreamRuntime)
    runtime.image_prefix_length = 0
    runtime.max_seq_length = max_seq_length
    runtime.prefix_cache = None
    runtime.page_table = PageTable(
        n_pages=128,
        page_size=page_size,
        max_batch_size=4,
        device="cpu",
    )
    return runtime


def test_prepare_sequence_reserves_initial_window_but_clamps_max_length() -> None:
    runtime = _runtime(max_seq_length=32)

    prepared = runtime.prepare_sequence(
        [runtime_mod.TextToken(1), runtime_mod.TextToken(2)],
        max_new_tokens=100,
    )

    assert prepared.state.max_length == 32
    assert prepared.state.length == 2
    assert runtime.page_table.capacity[prepared.state.batch_idx] == 18


def test_prepare_sequence_rejects_prompt_that_exceeds_context() -> None:
    runtime = _runtime(max_seq_length=2)

    with pytest.raises(ValueError, match="Prompt length 3 exceeds max_seq_length=2"):
        runtime.prepare_sequence(
            [
                runtime_mod.TextToken(1),
                runtime_mod.TextToken(2),
                runtime_mod.TextToken(3),
            ],
            max_new_tokens=1,
        )


def test_prepare_sequence_rejects_full_context_generation() -> None:
    runtime = _runtime(max_seq_length=2)

    with pytest.raises(ValueError, match="leaves no room for generation"):
        runtime.prepare_sequence(
            [
                runtime_mod.TextToken(1),
                runtime_mod.TextToken(2),
            ],
            max_new_tokens=1,
        )


def test_expand_kv_reservation_grows_page_table_row() -> None:
    runtime = _runtime(max_seq_length=8, page_size=2)
    batch_idx = runtime.page_table.allocate()
    runtime.page_table.reserve(batch_idx, 2)
    state = SequenceState(
        batch_idx=batch_idx,
        length=2,
        max_length=8,
        prompt_length=1,
    )

    assert runtime.expand_kv_reservation(state, tokens=1) is True

    assert runtime.page_table.capacity[batch_idx] == 4
    assert batch_idx in runtime.page_table._dirty_rows


def test_expand_kv_reservation_can_reserve_completion_runway() -> None:
    runtime = _runtime(max_seq_length=12, page_size=2)
    batch_idx = runtime.page_table.allocate()
    runtime.page_table.reserve(batch_idx, 4)
    state = SequenceState(
        batch_idx=batch_idx,
        length=4,
        max_length=12,
        prompt_length=1,
    )

    assert runtime.expand_kv_reservation(state, tokens=8) is True

    assert runtime.page_table.capacity[batch_idx] == 12
