import os
from types import SimpleNamespace

import pytest

from kestrel.runtime.preprocessing import (
    derive_image_insertion_offset,
    derive_preprocessing_workers,
)


@pytest.mark.parametrize(
    ("batch_capacity", "host_parallelism", "expected"),
    [
        (1, 64, 1),
        (8, 64, 8),
        (16, 8, 8),
        (16, 16, 16),
    ],
)
def test_preprocessing_workers_follow_capacity_and_host(
    batch_capacity: int,
    host_parallelism: int,
    expected: int,
) -> None:
    assert (
        derive_preprocessing_workers(
            batch_capacity,
            host_parallelism=host_parallelism,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("batch_capacity", "host_parallelism"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_preprocessing_workers_reject_invalid_domains(
    batch_capacity: int,
    host_parallelism: int,
) -> None:
    with pytest.raises(ValueError):
        derive_preprocessing_workers(
            batch_capacity,
            host_parallelism=host_parallelism,
        )


def test_preprocessing_workers_honor_process_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)

    assert derive_preprocessing_workers(8) == 3


def test_preprocessing_workers_fall_back_to_affinity(monkeypatch) -> None:
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    assert derive_preprocessing_workers(8) == 2


def test_preprocessing_workers_handle_unknown_cpu_count(monkeypatch) -> None:
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert derive_preprocessing_workers(8) == 1


def test_image_insertion_follows_user_turn_after_system_turn() -> None:
    prompt = [
        SimpleNamespace(token_id=2),
        SimpleNamespace(token_id=105),
        SimpleNamespace(token_id=9731),
        SimpleNamespace(token_id=107),
        SimpleNamespace(token_id=98),
        SimpleNamespace(token_id=106),
        SimpleNamespace(token_id=105),
        SimpleNamespace(token_id=2364),
        SimpleNamespace(token_id=107),
        SimpleNamespace(token_id=42),
    ]

    assert derive_image_insertion_offset(
        prompt,
        user_turn_opener=(105, 2364, 107),
        fallback_offset=4,
    ) == 9


def test_image_insertion_uses_validated_fallback() -> None:
    prompt = [SimpleNamespace(token_id=2), SimpleNamespace(token_id=3)]

    assert derive_image_insertion_offset(
        prompt,
        user_turn_opener=(10, 11),
        fallback_offset=1,
    ) == 1
    with pytest.raises(ValueError):
        derive_image_insertion_offset(
            prompt,
            user_turn_opener=(10, 11),
            fallback_offset=3,
        )


def test_image_insertion_accepts_bare_integer_tokens() -> None:
    assert derive_image_insertion_offset(
        [2, 99, 10, 11, 42],
        user_turn_opener=(10, 11),
        fallback_offset=1,
    ) == 4
