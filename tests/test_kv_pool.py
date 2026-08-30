"""Tests for :class:`kestrel.kv_cache.KVMemoryPool`.

Exercises both the diagnostics-only (unbudgeted) path and the
budget-enforcing path that lets multiple consumers share one pool
without exceeding a total cap.
"""

from __future__ import annotations

from contextlib import contextmanager
import weakref

import pytest
import torch

from kestrel.kv_cache import (
    KVMemoryPool,
    PageTable,
    PagedKVCache,
    PagedKVLayerSpec,
    allocate_paged_kv_layers,
    allocate_paged_kv_storage,
    fit_and_allocate_paged_kv_storage,
    paged_kv_bytes_per_page,
    paged_kv_storage_bytes,
)


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


def test_paged_kv_bytes_per_page_counts_sparse_layers_and_page_size() -> None:
    specs = (
        PagedKVLayerSpec(n_heads=2, head_dim=8),
        None,
        PagedKVLayerSpec(n_heads=1, head_dim=16),
    )

    assert paged_kv_bytes_per_page(
        specs,
        page_size=4,
        dtype=torch.bfloat16,
    ) == 2 * (2 * 8 + 1 * 16) * 4 * 2


def test_fitted_paged_kv_storage_respects_consumed_shared_pool_budget() -> None:
    specs = (PagedKVLayerSpec(n_heads=2, head_dim=8), None)
    bytes_per_page = paged_kv_bytes_per_page(
        specs,
        page_size=4,
        dtype=torch.bfloat16,
    )
    storage_bytes = paged_kv_storage_bytes(
        8,
        layer_specs=specs,
        page_size=4,
        dtype=torch.bfloat16,
    )
    pool = KVMemoryPool(
        device="cpu",
        budget_bytes=2 * bytes_per_page + storage_bytes,
    )
    pool.allocated_bytes = 2 * bytes_per_page

    storage, additional = fit_and_allocate_paged_kv_storage(
        100,
        layer_specs=specs,
        page_size=4,
        dtype=torch.bfloat16,
        pool=pool,
        stream=None,
    )

    assert storage.n_pages == 8
    assert additional is None
    assert pool.allocated_bytes == pool.budget_bytes


def test_fitted_paged_kv_storage_retains_first_successful_fragmented_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.kv_cache as kv_cache

    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    pool = KVMemoryPool(device="cpu")
    original_allocate = allocate_paged_kv_storage
    attempts = []
    resources = []
    compute_stream = object()
    stream_contexts = []

    @contextmanager
    def use_stream(stream):
        stream_contexts.append(("enter", stream))
        try:
            yield
        finally:
            stream_contexts.append(("exit", stream))

    def allocate_storage(n_pages, **kwargs):
        attempts.append(n_pages)
        if n_pages > 5:
            raise torch.OutOfMemoryError("synthetic fragmented allocator")
        return original_allocate(n_pages, **kwargs)

    def allocate_additional(n_pages):
        value = torch.empty(n_pages, dtype=torch.int32)
        resources.append(value)
        return value

    monkeypatch.setattr(kv_cache, "allocate_paged_kv_storage", allocate_storage)
    monkeypatch.setattr(kv_cache, "stream_context", use_stream)

    storage, additional = fit_and_allocate_paged_kv_storage(
        8,
        layer_specs=specs,
        page_size=2,
        dtype=torch.bfloat16,
        pool=pool,
        stream=compute_stream,
        allocate_additional=allocate_additional,
    )

    assert attempts == [8, 7, 6, 5]
    assert storage.n_pages == 5
    assert additional is resources[-1]
    assert additional.numel() == 5
    assert stream_contexts == [
        item
        for _attempt in attempts
        for item in (("enter", compute_stream), ("exit", compute_stream))
    ]


def test_fitted_paged_kv_storage_materializes_fixed_resources_before_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.kv_cache as kv_cache

    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    pool = KVMemoryPool(device="cuda:0")
    observed_free = 4096
    events = []

    @contextmanager
    def cuda_device(_device):
        yield

    def materialize_fixed():
        nonlocal observed_free
        assert torch.is_inference_mode_enabled()
        events.append("materialize")
        observed_free = 2048

    def mem_get_info(_device):
        events.append(("observe", observed_free))
        return observed_free, 8192

    def largest(upper, available_bytes, **_kwargs):
        events.append(("size", available_bytes))
        return min(int(upper), 3)

    storage = type("Storage", (), {"n_pages": 3})()
    monkeypatch.setattr(kv_cache.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(kv_cache.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(kv_cache.torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(kv_cache, "_largest_storage_pages", largest)
    monkeypatch.setattr(
        kv_cache,
        "allocate_paged_kv_storage",
        lambda *_args, **_kwargs: storage,
    )

    fitted, additional = fit_and_allocate_paged_kv_storage(
        8,
        layer_specs=specs,
        page_size=1,
        dtype=torch.bfloat16,
        pool=pool,
        stream=None,
        materialize_fixed=materialize_fixed,
    )

    assert fitted is storage
    assert additional is None
    assert events[0] == "materialize"
    assert events.count("materialize") == 1
    assert all(event != ("observe", 4096) for event in events)
    assert ("observe", 2048) in events
    assert ("size", 2048) in events


def test_fitted_paged_kv_storage_retries_failed_transient_without_retaining_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.kv_cache as kv_cache

    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    pool = KVMemoryPool(device="cpu")
    original_allocate = allocate_paged_kv_storage
    backing_refs = []
    additional_refs = []
    probe_refs = []
    probe_attempts = 0

    def allocate_storage(n_pages, **kwargs):
        storage = original_allocate(n_pages, **kwargs)
        backing_refs.append(weakref.ref(storage._kv_backing))
        return storage

    def allocate_additional(n_pages):
        value = torch.empty(n_pages, dtype=torch.int32)
        additional_refs.append(weakref.ref(value))
        return value

    def validate_transient():
        nonlocal probe_attempts
        assert torch.is_inference_mode_enabled()
        probe_attempts += 1
        if probe_attempts == 1:
            raise torch.OutOfMemoryError("synthetic live-workspace failure")
        value = torch.empty(1)
        probe_refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(kv_cache, "allocate_paged_kv_storage", allocate_storage)

    storage, additional = fit_and_allocate_paged_kv_storage(
        4,
        layer_specs=specs,
        page_size=2,
        dtype=torch.bfloat16,
        pool=pool,
        stream=None,
        allocate_additional=allocate_additional,
        validate_transient=validate_transient,
    )

    assert storage.n_pages == 3
    assert probe_attempts == 2
    assert backing_refs[0]() is None
    assert backing_refs[1]() is storage._kv_backing
    assert additional_refs[0]() is None
    assert additional_refs[1]() is additional
    assert probe_refs[0]() is None
    assert pool.allocated_bytes == paged_kv_storage_bytes(
        3,
        layer_specs=specs,
        page_size=2,
        dtype=torch.bfloat16,
    )


def test_fitted_paged_kv_storage_keeps_transient_alive_through_stream_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.kv_cache as kv_cache

    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    pool = KVMemoryPool(device="cuda:0")
    probe_ref = None

    class Probe:
        pass

    class Stream:
        def synchronize(self):
            assert probe_ref is not None and probe_ref() is not None

    @contextmanager
    def cuda_device(_device):
        yield

    @contextmanager
    def use_stream(_stream):
        yield

    def validate_transient():
        nonlocal probe_ref
        assert torch.is_inference_mode_enabled()
        probe = Probe()
        probe_ref = weakref.ref(probe)
        return probe

    storage = type("Storage", (), {"n_pages": 3})()
    monkeypatch.setattr(kv_cache.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(kv_cache.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        kv_cache.torch.cuda,
        "mem_get_info",
        lambda _device: (1 << 30, 1 << 31),
    )
    monkeypatch.setattr(kv_cache, "stream_context", use_stream)
    monkeypatch.setattr(
        kv_cache,
        "allocate_paged_kv_storage",
        lambda *_args, **_kwargs: storage,
    )

    fitted, additional = fit_and_allocate_paged_kv_storage(
        3,
        layer_specs=specs,
        page_size=1,
        dtype=torch.bfloat16,
        pool=pool,
        stream=Stream(),
        validate_transient=validate_transient,
    )

    assert fitted is storage
    assert additional is None
    assert probe_ref is not None and probe_ref() is None


def test_fitted_paged_kv_storage_rejects_capacity_without_serving_page() -> None:
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    two_page_bytes = paged_kv_storage_bytes(
        2,
        layer_specs=specs,
        page_size=1,
        dtype=torch.float32,
    )
    pool = KVMemoryPool(device="cpu", budget_bytes=two_page_bytes)

    with pytest.raises(MemoryError, match="serving K/V pages"):
        fit_and_allocate_paged_kv_storage(
            8,
            layer_specs=specs,
            page_size=1,
            dtype=torch.float32,
            pool=pool,
            stream=None,
        )


def test_grouped_paged_kv_storage_alignment_zero_page_and_pointer_reuse() -> None:
    specs = (
        PagedKVLayerSpec(n_heads=3, head_dim=5),
        None,
        PagedKVLayerSpec(n_heads=1, head_dim=7),
    )
    pool = KVMemoryPool(device="cpu")
    storage = allocate_paged_kv_storage(
        5,
        layer_specs=specs,
        page_size=3,
        dtype=torch.bfloat16,
        pool=pool,
    )
    before = tuple(
        None if layer is None else (layer.k_cache.data_ptr(), layer.v_cache.data_ptr())
        for layer in storage.layers
    )
    page_table = _make_page_table(page_size=3, n_pages=5)
    caches = allocate_paged_kv_layers(
        layer_specs=specs,
        page_table=page_table,
        pool=pool,
        dtype=torch.bfloat16,
        storage=storage,
    )

    after = tuple(
        None if cache is None else (cache.k_cache.data_ptr(), cache.v_cache.data_ptr())
        for cache in caches
    )
    assert after == before
    assert all(
        pointer % 256 == 0 for pair in before if pair is not None for pointer in pair
    )
    for cache in caches:
        if cache is not None:
            torch.testing.assert_close(
                cache.k_cache[0], torch.zeros_like(cache.k_cache[0])
            )
            torch.testing.assert_close(
                cache.v_cache[0], torch.zeros_like(cache.v_cache[0])
            )
    assert pool.allocated_bytes == paged_kv_storage_bytes(
        5,
        layer_specs=specs,
        page_size=3,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "mismatched_specs",
    [
        (
            PagedKVLayerSpec(
                n_heads=1,
                head_dim=8,
                k_scale=0.75,
                v_scale=0.25,
            ),
            None,
        ),
        (
            None,
            PagedKVLayerSpec(
                n_heads=1,
                head_dim=8,
                k_scale=0.5,
                v_scale=0.25,
            ),
        ),
    ],
)
def test_grouped_paged_kv_storage_rejects_scale_or_topology_reuse(
    mismatched_specs,
) -> None:
    specs = (
        PagedKVLayerSpec(
            n_heads=1,
            head_dim=8,
            k_scale=0.5,
            v_scale=0.25,
        ),
        None,
    )
    pool = KVMemoryPool(device="cpu")
    storage = allocate_paged_kv_storage(
        4,
        layer_specs=specs,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        pool=pool,
    )

    with pytest.raises(ValueError, match="does not match"):
        allocate_paged_kv_layers(
            layer_specs=mismatched_specs,
            page_table=_make_page_table(n_pages=4),
            pool=pool,
            dtype=torch.float8_e4m3fn,
            storage=storage,
        )


def test_grouped_paged_kv_storage_has_one_accounting_owner() -> None:
    import gc
    import weakref

    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    pool = KVMemoryPool(device="cpu")
    storage = allocate_paged_kv_storage(
        4,
        layer_specs=specs,
        page_size=1,
        dtype=torch.float32,
        pool=pool,
    )
    backing = weakref.ref(storage._kv_backing)
    caches = allocate_paged_kv_layers(
        layer_specs=specs,
        page_table=_make_page_table(n_pages=4),
        pool=pool,
        dtype=torch.float32,
        storage=storage,
    )
    allocated = pool.allocated_bytes
    del storage
    gc.collect()
    assert backing() is not None
    assert pool.allocated_bytes == allocated

    del caches
    gc.collect()
    assert backing() is None
    assert pool.allocated_bytes == 0


def test_grouped_paged_kv_storage_budget_failure_rolls_back() -> None:
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)
    required = paged_kv_storage_bytes(
        3,
        layer_specs=specs,
        page_size=1,
        dtype=torch.float32,
    )
    pool = KVMemoryPool(device="cpu", budget_bytes=required - 1)

    with pytest.raises(MemoryError, match="budget exceeded"):
        allocate_paged_kv_storage(
            3,
            layer_specs=specs,
            page_size=1,
            dtype=torch.float32,
            pool=pool,
        )
    assert pool.allocated_bytes == 0


def test_paged_kv_storage_bytes_includes_alignment_padding() -> None:
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=1),)

    assert paged_kv_storage_bytes(
        3,
        layer_specs=specs,
        page_size=1,
        dtype=torch.float32,
    ) == 520


def test_fitted_storage_requires_three_requested_pages() -> None:
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)

    with pytest.raises(ValueError, match="reserved, padding, and serving"):
        fit_and_allocate_paged_kv_storage(
            2,
            layer_specs=specs,
            page_size=1,
            dtype=torch.float32,
            pool=KVMemoryPool(device="cpu"),
            stream=None,
        )


def test_paged_kv_bytes_per_page_excludes_fixed_alignment_padding() -> None:
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=8),)

    assert paged_kv_bytes_per_page(
        specs,
        page_size=1,
        dtype=torch.float32,
    ) == 64


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
