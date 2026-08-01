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


def test_paged_cache_update_accepts_fused_projection_value_slice() -> None:
    pool = KVMemoryPool(device="cpu")
    page_table = _make_page_table(page_size=4, n_pages=3)
    cache = PagedKVCache(
        page_table,
        n_heads=2,
        head_dim=4,
        dtype=torch.float32,
        pool=pool,
    )

    batch = 2
    seq_len = 3
    q_size = 16
    kv_size = 8
    key = torch.arange(batch * seq_len * kv_size, dtype=torch.float32).reshape(
        batch, seq_len, 2, 4
    )
    fused = torch.arange(
        batch * seq_len * (q_size + 2 * kv_size),
        dtype=torch.float32,
    ).reshape(batch, seq_len, q_size + 2 * kv_size)
    value = fused[..., q_size + kv_size :].reshape(batch, seq_len, 2, 4)

    assert value.stride(1) > value.shape[2] * value.shape[3]
    assert value.stride(2) == value.shape[3]
    assert value.stride(3) == 1

    slot_mapping = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int64)
    input_pos = torch.arange(batch * seq_len).reshape(batch, seq_len)

    cache.update(
        input_pos=input_pos,
        k_val=key,
        v_val=value,
        slot_mapping=slot_mapping,
    )

    expected_k = torch.zeros_like(cache.k_cache)
    expected_v = torch.zeros_like(cache.v_cache)
    key_flat = key.view(-1, 2, 4)
    value_flat = value.view(-1, 2, 4)
    for row, slot in enumerate(slot_mapping.view(-1).tolist()):
        if slot < 0:
            continue
        block = slot // page_table.page_size
        offset = slot % page_table.page_size
        expected_k[block, :, offset, :] = key_flat[row]
        expected_v[block, :, offset, :] = value_flat[row]

    torch.testing.assert_close(cache.k_cache, expected_k)
    torch.testing.assert_close(cache.v_cache, expected_v)


def test_paged_cache_update_accepts_batched_decode_value_slice() -> None:
    pool = KVMemoryPool(device="cpu")
    page_table = _make_page_table(page_size=4, n_pages=3)
    cache = PagedKVCache(
        page_table,
        n_heads=2,
        head_dim=4,
        dtype=torch.float32,
        pool=pool,
    )

    batch = 2
    seq_len = 1
    q_size = 16
    kv_size = 8
    fused = torch.arange(
        batch * seq_len * (q_size + 2 * kv_size),
        dtype=torch.float32,
    ).reshape(batch, seq_len, q_size + 2 * kv_size)
    key = fused[..., q_size : q_size + kv_size].reshape(batch, seq_len, 2, 4)
    value = fused[..., q_size + kv_size :].reshape(batch, seq_len, 2, 4)

    assert value.stride(0) > value.stride(1)
    assert value.stride(2) == value.shape[3]
    assert value.stride(3) == 1

    slot_mapping = torch.tensor([[2], [5]], dtype=torch.int64)
    input_pos = torch.tensor([[0], [0]], dtype=torch.int64)

    cache.update(
        input_pos=input_pos,
        k_val=key,
        v_val=value,
        slot_mapping=slot_mapping,
    )

    expected_k = torch.zeros_like(cache.k_cache)
    expected_v = torch.zeros_like(cache.v_cache)
    key_flat = key[:, 0]
    value_flat = value[:, 0]
    for row, slot in enumerate(slot_mapping.view(-1).tolist()):
        block = slot // page_table.page_size
        offset = slot % page_table.page_size
        expected_k[block, :, offset, :] = key_flat[row]
        expected_v[block, :, offset, :] = value_flat[row]

    torch.testing.assert_close(cache.k_cache, expected_k)
    torch.testing.assert_close(cache.v_cache, expected_v)


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


def test_pool_releases_bytes_when_tensors_are_collected() -> None:
    """Discarding a cache must return its bytes to the pool so a
    partial-init failure of one runtime doesn't permanently shrink the
    budget for other runtimes sharing the same pool."""

    import gc

    pool = KVMemoryPool(device="cpu")
    page_table = _make_page_table()
    cache = PagedKVCache(
        page_table, n_heads=2, head_dim=8, dtype=torch.float32, pool=pool
    )
    assert pool.allocated_bytes > 0

    del cache
    gc.collect()

    assert pool.allocated_bytes == 0


def test_pool_budget_serializes_concurrent_allocations() -> None:
    """Two threads sharing one pool must not both pass the precheck and
    bust the cap. Exactly one allocation may succeed when the budget
    only fits one."""

    import threading

    layer_bytes = 4 * 2 * 1 * 8 * 4  # n_pages=4 n_heads=2 page=1 head_dim=8 fp32
    layer_total = 2 * layer_bytes  # K + V
    pool = KVMemoryPool(device="cpu", budget_bytes=layer_total)
    page_table = _make_page_table()

    successes: list[PagedKVCache] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(2)

    def alloc() -> None:
        barrier.wait()
        try:
            cache = PagedKVCache(
                page_table, n_heads=2, head_dim=8, dtype=torch.float32, pool=pool
            )
            successes.append(cache)
        except MemoryError as exc:
            failures.append(exc)

    threads = [threading.Thread(target=alloc) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert pool.allocated_bytes == layer_total
