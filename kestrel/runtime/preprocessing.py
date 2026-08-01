"""Host preprocessing policy shared by private model runtimes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


def _process_parallelism() -> int | None:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        parallelism = process_cpu_count()
        if parallelism is not None:
            return int(parallelism)

    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def derive_preprocessing_workers(
    batch_capacity: int,
    *,
    host_parallelism: int | None = None,
) -> int:
    """Derive useful host concurrency from serving capacity and the host."""
    capacity = int(batch_capacity)
    if capacity < 1:
        raise ValueError("batch_capacity must be positive")

    parallelism = (
        _process_parallelism()
        if host_parallelism is None
        else int(host_parallelism)
    )
    if parallelism is None:
        parallelism = 1
    if parallelism < 1:
        raise ValueError("host_parallelism must be positive")
    return min(capacity, parallelism)


def derive_image_insertion_offset(
    prompt_tokens: Sequence[Any],
    *,
    user_turn_opener: Sequence[int],
    fallback_offset: int,
) -> int:
    """Place a query image after the first complete user-turn opener."""

    opener = tuple(int(token_id) for token_id in user_turn_opener)
    if not opener:
        raise ValueError("user_turn_opener must not be empty")

    token_ids = tuple(
        token if isinstance(token, int) else getattr(token, "token_id", None)
        for token in prompt_tokens
    )
    last_start = len(token_ids) - len(opener)
    for start in range(last_start + 1):
        if token_ids[start : start + len(opener)] == opener:
            return start + len(opener)

    fallback = int(fallback_offset)
    if not 0 <= fallback <= len(prompt_tokens):
        raise ValueError(
            f"fallback_offset={fallback} is outside prompt length {len(prompt_tokens)}"
        )
    return fallback
