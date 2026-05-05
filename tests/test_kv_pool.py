"""Tests for :class:`kestrel.kv_cache.KVMemoryPool`.

Exercises both the diagnostics-only (unbudgeted) path and the
budget-enforcing path that lets multiple consumers share one pool
without exceeding a total cap.
"""

from __future__ import annotations

import pytest
import torch

from kestrel.kv_cache import KVMemoryPool, PageTable, PagedKVCache


def _make_page_table(*, page_size: int = 1, n_pages: int = 4) -> PageTable:
    return PageTable(
        n_pages=n_pages,
        page_size=page_size,
        max_batch_size=2,
        device="cpu",
    )


def test_pool_tracks_allocated_bytes_without_budget() -> None:
    pool = KVMemoryPool(device="cpu")
    page_table = _make_page_table()
    cache = PagedKVCache(
        page_table,
        n_heads=2,
        head_dim=8,
        dtype=torch.float32,
        pool=pool,
    )
    expected = 2 * 4 * 2 * 1 * 8 * 4  # 2x for K+V, fp32 = 4 bytes
    assert pool.allocated_bytes == expected
    assert tuple(cache.k_cache.shape) == (4, 2, 1, 8)


def test_pool_serves_multiple_caches_against_one_budget() -> None:
    """Two caches sharing one pool count against the same allocated_bytes."""

    pool = KVMemoryPool(device="cpu", budget_bytes=10_000)
    page_table = _make_page_table()
    PagedKVCache(page_table, n_heads=2, head_dim=8, dtype=torch.float32, pool=pool)
    after_first = pool.allocated_bytes
    PagedKVCache(page_table, n_heads=2, head_dim=8, dtype=torch.float32, pool=pool)
    assert pool.allocated_bytes == 2 * after_first
    assert pool.allocated_bytes <= pool.budget_bytes  # type: ignore[operator]


def test_pool_raises_when_budget_exceeded() -> None:
    pool = KVMemoryPool(device="cpu", budget_bytes=128)
    page_table = _make_page_table()
    with pytest.raises(MemoryError, match="budget exceeded"):
        PagedKVCache(
            page_table,
            n_heads=4,
            head_dim=16,
            dtype=torch.float32,
            pool=pool,
        )
    # A failed allocation must not advance the counter.
    assert pool.allocated_bytes == 0


def test_pool_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        KVMemoryPool(device="cpu", budget_bytes=-1)


def test_pool_normalizes_string_device() -> None:
    pool = KVMemoryPool(device="cpu")
    assert pool.device == torch.device("cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
def test_pool_canonicalizes_indexless_cuda_device() -> None:
    """``KVMemoryPool(device='cuda')`` must compare equal to the same
    device built from ``RuntimeConfig`` so shared-pool wiring doesn't
    raise on default single-GPU setups."""

    pool = KVMemoryPool(device="cuda")
    assert pool.device.type == "cuda"
    assert pool.device.index == torch.cuda.current_device()
    assert pool.device == torch.device("cuda", torch.cuda.current_device())
