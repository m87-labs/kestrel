"""Host policy bookkeeping for Parakeet TDT decode."""

from types import SimpleNamespace
from typing import cast

from kestrel.models.parakeet_tdt.config import ParakeetTdtConfig
from kestrel.models.parakeet_tdt.model import _TdtBatchDecodeState


def _config(*, max_symbols_per_step: int = 2) -> ParakeetTdtConfig:
    return cast(
        ParakeetTdtConfig,
        SimpleNamespace(
            blank_token_id=0,
            durations=(0, 1, 2),
            max_symbols_per_step=max_symbols_per_step,
        ),
    )


def test_commit_reports_only_new_host_policy_stops() -> None:
    state = _TdtBatchDecodeState.create(
        _config(), valid_lengths=[3, 1, 4], max_tokens=1
    )

    stopped = state.commit(
        decisions=[[1, 0], [0, 1], [0, 0]],
        active=state.active(),
    )

    assert stopped == [0]
    assert state.active() == [False, False, True]
    assert state.commit([[2, 0], [2, 0], [2, 0]], state.active()) == [2]
    assert state.commit([[2, 0], [2, 0], [2, 0]], state.active()) == []


def test_commit_reports_step_budget_stop_before_encoder_exhaustion() -> None:
    state = _TdtBatchDecodeState.create(
        _config(max_symbols_per_step=1), valid_lengths=[2], max_tokens=None
    )

    assert state.commit([[1, 0]], state.active()) == []
    assert state.commit([[1, 0]], state.active()) == [0]
    assert state.frames == [0]
    assert state.active() == [False]
