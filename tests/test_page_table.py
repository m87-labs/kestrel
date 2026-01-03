"""Tests for PageTable prefix cache integration."""

from __future__ import annotations

import pytest
import torch

from kestrel.kv_cache import PageTable
from kestrel.prefix_cache import CacheNamespace, RadixPrefixCache


class MockToken:
    """Mock token for cache testing."""

    def __init__(self, id: int, kv_len: int = 1):
        self.id = id
        self._kv_len = kv_len

    def cache_key(self) -> tuple[int, int]:
        return (0, self.id)

    def kv_length(self) -> int:
        return self._kv_len


# =============================================================================
# Basic PageTable Tests (no prefix cache)
# =============================================================================


class TestAllocatePagesBasic:
    """Tests for allocate_pages without prefix cache."""

    def test_allocate_pages_returns_requested_count(self) -> None:
        """allocate_pages returns the requested number of pages."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available

        pages = page_table.allocate_pages(5)

        assert len(pages) == 5
        assert page_table.pages_available == initial_available - 5

    def test_allocate_pages_returns_unique_pages(self) -> None:
        """allocate_pages returns distinct page indices."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )

        pages = page_table.allocate_pages(10)

        assert len(set(pages)) == 10  # All unique

    def test_allocate_pages_removes_from_free_pool(self) -> None:
        """Allocated pages are no longer in free pool."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )

        pages = page_table.allocate_pages(5)

        for p in pages:
            assert p not in page_table.free_pages

    def test_allocate_pages_insufficient_raises(self) -> None:
        """allocate_pages raises when not enough pages available."""
        page_table = PageTable(
            n_pages=10, page_size=1, max_batch_size=5, device="cpu"
        )
        # Page 0 is reserved, so only 9 available

        with pytest.raises(RuntimeError, match="Cannot allocate"):
            page_table.allocate_pages(20)

    def test_allocate_pages_exact_available(self) -> None:
        """allocate_pages succeeds when requesting exactly available pages."""
        page_table = PageTable(
            n_pages=10, page_size=1, max_batch_size=5, device="cpu"
        )
        available = page_table.pages_available

        pages = page_table.allocate_pages(available)

        assert len(pages) == available
        assert page_table.pages_available == 0


class TestMapPagesBasic:
    """Tests for map_pages."""

    def test_map_pages_updates_page_table(self) -> None:
        """map_pages correctly maps physical pages to logical positions."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)

        page_table.map_pages(batch_idx, 0, pages)

        # Verify mapping via get_pages
        mapped = page_table.get_pages(batch_idx, 0, 5)
        assert mapped == pages

    def test_map_pages_updates_capacity(self) -> None:
        """map_pages updates batch capacity."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)

        page_table.map_pages(batch_idx, 0, pages)

        assert page_table.capacity[batch_idx] == 5

    def test_map_pages_sequential_extension(self) -> None:
        """map_pages can extend mapping sequentially."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages1 = page_table.allocate_pages(3)
        pages2 = page_table.allocate_pages(2)

        page_table.map_pages(batch_idx, 0, pages1)
        page_table.map_pages(batch_idx, 3, pages2)

        assert page_table.get_pages(batch_idx, 0, 5) == pages1 + pages2
        assert page_table.capacity[batch_idx] == 5

    def test_map_pages_non_sequential_raises(self) -> None:
        """map_pages raises if logical_start doesn't match current length."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(3)

        # Try to map at position 5 without mapping 0-4 first
        with pytest.raises(AssertionError, match="sequentially"):
            page_table.map_pages(batch_idx, 5, pages)

    def test_map_pages_tracks_in_page_table_cpu(self) -> None:
        """map_pages updates page_table_cpu for erase tracking."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)

        page_table.map_pages(batch_idx, 0, pages)

        assert page_table.page_table_cpu[batch_idx] == pages


class TestGetPages:
    """Tests for get_pages."""

    def test_get_pages_returns_mapped_pages(self) -> None:
        """get_pages returns physical pages for logical range."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch_idx, 0, pages)

        result = page_table.get_pages(batch_idx, 2, 7)

        assert result == pages[2:7]

    def test_get_pages_empty_range(self) -> None:
        """get_pages returns empty list for empty range."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)
        page_table.map_pages(batch_idx, 0, pages)

        result = page_table.get_pages(batch_idx, 3, 3)

        assert result == []


class TestFreePagesToPool:
    """Tests for free_pages_to_pool."""

    def test_free_pages_to_pool_returns_pages(self) -> None:
        """free_pages_to_pool adds pages back to free pool."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available
        pages = page_table.allocate_pages(5)

        page_table.free_pages_to_pool(pages)

        assert page_table.pages_available == initial_available

    def test_free_pages_to_pool_accepts_tuple(self) -> None:
        """free_pages_to_pool accepts tuple (from TreeNode.physical_pages)."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        pages = page_table.allocate_pages(3)
        pages_tuple = tuple(pages)

        page_table.free_pages_to_pool(pages_tuple)

        for p in pages:
            assert p in page_table.free_pages


# =============================================================================
# Erase Tests
# =============================================================================


class TestEraseWithoutCache:
    """Tests for erase without prefix cache (backward compatibility)."""

    def test_erase_frees_all_pages_default(self) -> None:
        """erase() with no cached_page_count frees all pages."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)
        page_table.map_pages(batch_idx, 0, pages)

        page_table.erase(batch_idx)

        assert page_table.pages_available == initial_available

    def test_erase_clears_page_table_cpu(self) -> None:
        """erase clears the page_table_cpu for the batch."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)
        page_table.map_pages(batch_idx, 0, pages)

        page_table.erase(batch_idx)

        assert page_table.page_table_cpu[batch_idx] == []

    def test_erase_returns_batch_idx_to_pool(self) -> None:
        """erase returns batch_idx to free_batch_idx pool."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()

        page_table.erase(batch_idx)

        assert batch_idx in page_table.free_batch_idx


class TestEraseWithCachedPageCount:
    """Tests for erase with cached_page_count parameter."""

    def test_erase_skips_cached_pages(self) -> None:
        """erase with cached_page_count skips freeing first N pages."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch_idx, 0, pages)
        cached_pages = pages[:7]
        private_pages = pages[7:]

        page_table.erase(batch_idx, cached_page_count=7)

        # Only private pages should be freed
        assert page_table.pages_available == initial_available - 10 + 3
        for p in private_pages:
            assert p in page_table.free_pages
        for p in cached_pages:
            assert p not in page_table.free_pages

    def test_erase_cached_page_count_zero(self) -> None:
        """erase with cached_page_count=0 frees all pages (same as default)."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)
        page_table.map_pages(batch_idx, 0, pages)

        page_table.erase(batch_idx, cached_page_count=0)

        assert page_table.pages_available == initial_available

    def test_erase_all_cached(self) -> None:
        """erase with cached_page_count=len(pages) frees nothing."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial_available = page_table.pages_available
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(5)
        page_table.map_pages(batch_idx, 0, pages)

        page_table.erase(batch_idx, cached_page_count=5)

        # No pages freed - they're all cache-owned
        assert page_table.pages_available == initial_available - 5


# =============================================================================
# Prefix Cache Integration Tests (cache disabled)
# =============================================================================


class TestPrefixCacheDisabled:
    """Tests with prefix cache disabled (None)."""

    def test_prefix_cache_defaults_to_none(self) -> None:
        """prefix_cache defaults to None when not passed to constructor."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )

        assert page_table.prefix_cache is None

    def test_allocate_pages_no_eviction_when_disabled(self) -> None:
        """allocate_pages doesn't try to evict when cache is None."""
        page_table = PageTable(
            n_pages=10, page_size=1, max_batch_size=5, device="cpu"
        )
        # No prefix_cache - default is None

        # Use up all pages
        available = page_table.pages_available
        page_table.allocate_pages(available)

        # Should raise, not try to evict
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            page_table.allocate_pages(1)

    def test_can_reserve_with_eviction_disabled(self) -> None:
        """can_reserve_with_eviction falls back to free-pool check when disabled."""
        page_table = PageTable(
            n_pages=20, page_size=1, max_batch_size=5, device="cpu"
        )
        # No prefix_cache - default is None

        # Should be able to reserve what's available
        assert page_table.can_reserve_with_eviction(10) is True

        # Use up all pages
        page_table.allocate_pages(page_table.pages_available)

        # Now should return False (no eviction possible)
        assert page_table.can_reserve_with_eviction(1) is False


# =============================================================================
# Prefix Cache Integration Tests (cache enabled)
# =============================================================================


class TestPrefixCacheEnabled:
    """Tests with prefix cache enabled."""

    def test_prefix_cache_via_constructor(self) -> None:
        """prefix_cache constructor arg attaches cache and wires sink."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu",
            prefix_cache=cache,
        )

        assert page_table.prefix_cache is cache
        # Verify sink was wired (bound methods compare by __self__ and __func__)
        assert cache._free_pages_sink is not None
        assert cache._free_pages_sink.__self__ is page_table
        assert cache._free_pages_sink.__func__ is PageTable.free_pages_to_pool

    def test_prefix_cache_rejects_page_size_not_one(self) -> None:
        """Constructor asserts if page_size != 1 with prefix_cache."""
        cache = RadixPrefixCache()

        with pytest.raises(AssertionError, match="page_size=1"):
            PageTable(
                n_pages=100, page_size=16, max_batch_size=10, device="cpu",
                prefix_cache=cache,
            )

    def test_allocate_pages_with_eviction(self) -> None:
        """allocate_pages evicts from cache when free pool insufficient."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=30, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        # Insert multiple separate sequences so eviction can be partial
        # (eviction works by node, not by individual page)
        tokens1 = [MockToken(i) for i in range(10)]
        pages1 = page_table.allocate_pages(10)
        cache.insert(tokens1, pages1)

        tokens2 = [MockToken(i + 100) for i in range(10)]
        pages2 = page_table.allocate_pages(10)
        cache.insert(tokens2, pages2)

        initial_cached = cache.total_cached_pages
        assert initial_cached == 20

        # Use remaining free pages
        remaining = page_table.allocate_pages(page_table.pages_available)
        assert page_table.pages_available == 0

        # Return those to pool, then clear it to force eviction
        page_table.free_pages_to_pool(remaining)
        page_table.free_pages.clear()

        # allocate_pages should evict from cache
        allocated = page_table.allocate_pages(5)

        assert len(allocated) == 5
        # At least one node should have been evicted (10 pages)
        assert cache.total_cached_pages < initial_cached

    def test_allocate_pages_all_locked_raises(self) -> None:
        """allocate_pages raises when all cached pages are locked."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=20, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        # Insert and lock
        available = page_table.pages_available
        tokens = [MockToken(i) for i in range(available)]
        pages = list(range(1, available + 1))
        result = cache.insert(tokens, pages)
        cache.lock(result.node)
        page_table.free_pages.clear()

        # Should raise - all pages locked
        with pytest.raises(RuntimeError, match="locked"):
            page_table.allocate_pages(1)

    def test_allocate_pages_partial_eviction(self) -> None:
        """allocate_pages evicts only what's needed."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=30, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        # Insert tokens into cache
        tokens = [MockToken(i) for i in range(20)]
        pages = page_table.allocate_pages(20)
        cache.insert(tokens, pages)
        initial_cached = cache.total_cached_pages

        # Request more pages than free pool has
        free_before = page_table.pages_available
        needed = free_before + 5  # Need 5 from eviction

        allocated = page_table.allocate_pages(needed)

        assert len(allocated) == needed
        # Should have evicted at least 5 pages
        assert cache.total_cached_pages <= initial_cached - 5

    def test_can_reserve_with_eviction_considers_cache(self) -> None:
        """can_reserve_with_eviction includes evictable cache pages."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=30, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        # Insert into cache
        tokens = [MockToken(i) for i in range(20)]
        pages = page_table.allocate_pages(20)
        cache.insert(tokens, pages)

        free_pages = page_table.pages_available
        evictable = cache.evictable_page_count()

        # Should be able to reserve free + evictable
        assert page_table.can_reserve_with_eviction(free_pages + evictable) is True
        # But not more
        assert page_table.can_reserve_with_eviction(free_pages + evictable + 1) is False

    def test_can_reserve_with_eviction_locked_not_evictable(self) -> None:
        """can_reserve_with_eviction doesn't count locked pages."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=30, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        # Insert and lock
        tokens = [MockToken(i) for i in range(20)]
        pages = page_table.allocate_pages(20)
        result = cache.insert(tokens, pages)
        cache.lock(result.node)

        free_pages = page_table.pages_available
        evictable = cache.evictable_page_count()
        assert evictable == 0  # All locked

        # Can only reserve what's in free pool
        assert page_table.can_reserve_with_eviction(free_pages) is True
        assert page_table.can_reserve_with_eviction(free_pages + 1) is False


# =============================================================================
# Page Sharing Tests
# =============================================================================


class TestPageSharing:
    """Tests for multiple batches sharing cached pages."""

    def test_multiple_batches_map_same_pages(self) -> None:
        """Multiple batches can map the same physical pages."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu",
            prefix_cache=cache,
        )

        # First batch: full prefill, insert into cache
        batch1 = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch1, 0, pages)
        tokens = [MockToken(i) for i in range(10)]
        result = cache.insert(tokens, pages)
        cache.lock(result.node)

        # Second batch: cache hit, map same pages
        batch2 = page_table.allocate()
        match = cache.match_prefix(tokens)
        cached_pages = match.matched_pages
        page_table.map_pages(batch2, 0, cached_pages)
        cache.lock(match.last_node)

        # Both batches should see same physical pages
        assert page_table.get_pages(batch1, 0, 10) == page_table.get_pages(batch2, 0, 10)

    def test_shared_pages_not_double_freed(self) -> None:
        """Shared pages are not freed when either batch erases."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu",
            prefix_cache=cache,
        )

        initial_available = page_table.pages_available

        # First batch
        batch1 = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch1, 0, pages)
        tokens = [MockToken(i) for i in range(10)]
        result = cache.insert(tokens, pages)
        cache.lock(result.node)

        # Second batch shares cached pages
        batch2 = page_table.allocate()
        match = cache.match_prefix(tokens)
        page_table.map_pages(batch2, 0, match.matched_pages)
        cache.lock(match.last_node)

        # Release batch1 - pages should NOT go to free pool (cache-owned)
        cache.unlock(result.node)
        page_table.erase(batch1, cached_page_count=10)

        # Pages still not in free pool
        for p in pages:
            assert p not in page_table.free_pages

        # Release batch2 - still not freed
        cache.unlock(match.last_node)
        page_table.erase(batch2, cached_page_count=10)

        for p in pages:
            assert p not in page_table.free_pages

        # Only after cache eviction do pages return to pool
        # (cache sends freed pages to page_table.free_pages_to_pool via sink)
        cache.evict(10)
        assert page_table.pages_available == initial_available


# =============================================================================
# Page Ownership Invariant Tests
# =============================================================================


class TestPageOwnershipInvariants:
    """Tests for page ownership invariants."""

    def test_page_conservation_miss_insert_release(self) -> None:
        """Pages conserved through miss → insert → release cycle."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=50, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        total_pages = page_table.pages_available
        assert cache.total_cached_pages == 0

        # Cache miss: allocate, map, insert
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch_idx, 0, pages)
        tokens = [MockToken(i) for i in range(10)]
        result = cache.insert(tokens, pages)
        cache.lock(result.node)

        # Invariant: free + cached = total
        assert page_table.pages_available + cache.total_cached_pages == total_pages

        # Release sequence
        cache.unlock(result.node)
        page_table.erase(batch_idx, cached_page_count=result.inserted_pages)

        # Invariant still holds
        assert page_table.pages_available + cache.total_cached_pages == total_pages

    def test_page_conservation_hit_extend_release(self) -> None:
        """Pages conserved through hit → extend → release cycle."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=50, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        total_pages = page_table.pages_available

        # First: populate cache
        batch1 = page_table.allocate()
        pages1 = page_table.allocate_pages(5)
        page_table.map_pages(batch1, 0, pages1)
        tokens = [MockToken(i) for i in range(5)]
        result1 = cache.insert(tokens, pages1)
        cache.lock(result1.node)
        cache.unlock(result1.node)
        page_table.erase(batch1, cached_page_count=5)

        # Second: partial hit, extend
        batch2 = page_table.allocate()
        extended_tokens = tokens + [MockToken(5), MockToken(6), MockToken(7)]
        match = cache.match_prefix(extended_tokens)
        skip_positions = match.matched_kv_length
        assert skip_positions == 5

        # Map cached pages, allocate suffix
        page_table.map_pages(batch2, 0, match.matched_pages)
        cache.lock(match.last_node)
        suffix_pages = page_table.allocate_pages(3)
        page_table.map_pages(batch2, skip_positions, suffix_pages)

        # Insert extended sequence
        all_pages = match.matched_pages + suffix_pages
        result2 = cache.insert(
            extended_tokens,
            all_pages,
            from_node=match.last_node,
            from_token_idx=5,
            from_page_idx=skip_positions,
        )

        # Lock transfer
        cache.lock(result2.node)
        cache.unlock(match.last_node)

        # Invariant
        assert page_table.pages_available + cache.total_cached_pages == total_pages

        # Release
        cache_owned = skip_positions + result2.inserted_pages
        cache.unlock(result2.node)
        page_table.erase(batch2, cached_page_count=cache_owned)

        # Invariant
        assert page_table.pages_available + cache.total_cached_pages == total_pages

    def test_page_conservation_full_match_private_suffix(self) -> None:
        """Pages conserved when full match forces private suffix page."""
        cache = RadixPrefixCache()
        page_table = PageTable(
            n_pages=50, page_size=1, max_batch_size=5, device="cpu",
            prefix_cache=cache,
        )

        total_pages = page_table.pages_available

        # Populate cache with full prompt
        batch1 = page_table.allocate()
        pages1 = page_table.allocate_pages(5)
        page_table.map_pages(batch1, 0, pages1)
        tokens = [MockToken(i) for i in range(5)]
        result1 = cache.insert(tokens, pages1)
        cache.lock(result1.node)
        cache.unlock(result1.node)
        page_table.erase(batch1, cached_page_count=5)

        # Second request: full match
        batch2 = page_table.allocate()
        match = cache.match_prefix(tokens)
        prompt_len = 5
        assert match.matched_kv_length >= prompt_len  # Full match

        # Cap skip_positions to ensure at least one suffix token
        skip_positions = min(match.matched_kv_length, prompt_len - 1)
        assert skip_positions == 4

        # Map only reused pages
        cached_pages = match.matched_pages[:skip_positions]
        page_table.map_pages(batch2, 0, cached_pages)
        cache.lock(match.last_node)

        # Allocate private page for recomputed last token
        private_pages = page_table.allocate_pages(1)
        page_table.map_pages(batch2, skip_positions, private_pages)

        # No insert needed - full prompt already cached
        cache_owned = skip_positions  # NOT prompt_len

        # Invariant (private page is "in flight", not free or cached)
        in_flight = 1  # The private page
        assert page_table.pages_available + cache.total_cached_pages + in_flight == total_pages

        # Release
        cache.unlock(match.last_node)
        page_table.erase(batch2, cached_page_count=cache_owned)

        # Private page returned to pool
        assert private_pages[0] in page_table.free_pages

        # Invariant restored
        assert page_table.pages_available + cache.total_cached_pages == total_pages


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    def test_reserve_still_works(self) -> None:
        """Existing reserve() method still works."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        batch_tensor = torch.tensor([batch_idx], dtype=torch.int32)

        success = page_table.reserve(batch_idx, batch_tensor, 10)

        assert success is True
        assert page_table.capacity[batch_idx] == 10

    def test_can_reserve_still_works(self) -> None:
        """Existing can_reserve() method still works."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )

        assert page_table.can_reserve(10) is True
        assert page_table.can_reserve(1000) is False

    def test_erase_without_cached_page_count(self) -> None:
        """erase() without cached_page_count still works (backward compat)."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        initial = page_table.pages_available
        batch_idx = page_table.allocate()
        batch_tensor = torch.tensor([batch_idx], dtype=torch.int32)
        page_table.reserve(batch_idx, batch_tensor, 10)

        # Old-style erase (no cached_page_count)
        page_table.erase(batch_idx)

        assert page_table.pages_available == initial

    def test_build_slot_mapping_still_works(self) -> None:
        """Existing build_slot_mapping() still works with new mapping."""
        page_table = PageTable(
            n_pages=100, page_size=1, max_batch_size=10, device="cpu"
        )
        batch_idx = page_table.allocate()
        pages = page_table.allocate_pages(10)
        page_table.map_pages(batch_idx, 0, pages)

        # build_slot_mapping expects 2D positions when batch_idx is 1D
        # batch_idx: [batch_size], positions: [batch_size, seq_len]
        batch_tensor = torch.tensor([batch_idx], device="cpu")
        positions = torch.arange(10, device="cpu", dtype=torch.int32).unsqueeze(0)  # [1, 10]

        slots = page_table.build_slot_mapping(batch_tensor, positions)

        # Slots should map to physical pages (with page_size=1, slot == page)
        expected = torch.tensor(pages, dtype=torch.int64).unsqueeze(0)  # [1, 10]
        assert torch.equal(slots, expected)
