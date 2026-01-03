"""Tests for RadixPrefixCache."""

from __future__ import annotations

import time

import pytest

from kestrel.prefix_cache import (
    CacheNamespace,
    InsertResult,
    MatchResult,
    RadixPrefixCache,
    TreeNode,
)


class MockToken:
    """Mock token for testing."""

    def __init__(self, id: int, kv_len: int = 1):
        self.id = id
        self._kv_len = kv_len

    def cache_key(self) -> tuple[int, int]:
        return (0, self.id)

    def kv_length(self) -> int:
        return self._kv_len

    def __repr__(self) -> str:
        return f"MockToken({self.id}, kv_len={self._kv_len})"


class TestTreeNode:
    """Tests for TreeNode dataclass."""

    def test_is_leaf_with_no_children(self) -> None:
        node = TreeNode()
        assert node.is_leaf()

    def test_is_leaf_with_children(self) -> None:
        node = TreeNode()
        node.children[(0, 1)] = TreeNode()
        assert not node.is_leaf()

    def test_total_kv_length(self) -> None:
        node = TreeNode(physical_pages=(0, 1, 2))
        assert node.total_kv_length == 3

    def test_total_kv_length_empty(self) -> None:
        node = TreeNode()
        assert node.total_kv_length == 0


class TestMatchPrefix:
    """Tests for match_prefix operation."""

    def test_empty_cache_returns_empty_match(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2)]
        result = cache.match_prefix(tokens)

        assert result.matched_kv_length == 0
        assert result.matched_token_count == 0
        assert result.last_node is None
        assert len(result.unmatched_tokens) == 2

    def test_full_match(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2)]
        cache.insert(tokens, [0, 1])

        result = cache.match_prefix(tokens)

        assert result.matched_kv_length == 2
        assert result.matched_token_count == 2
        assert result.last_node is not None
        assert len(result.unmatched_tokens) == 0

    def test_partial_match(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2), MockToken(3)]
        cache.insert(tokens, [0, 1, 2])

        query = [MockToken(1), MockToken(2), MockToken(99)]
        result = cache.match_prefix(query)

        assert result.matched_token_count == 2
        assert result.matched_pages == [0, 1]
        assert len(result.unmatched_tokens) == 1

    def test_no_match(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2)]
        cache.insert(tokens, [0, 1])

        query = [MockToken(99)]
        result = cache.match_prefix(query)

        assert result.matched_kv_length == 0
        assert result.matched_token_count == 0

    def test_match_updates_access_time(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1)]
        insert_result = cache.insert(tokens, [0])
        original_time = insert_result.node.last_access_time

        time.sleep(0.01)
        cache.match_prefix(tokens)

        assert insert_result.node.last_access_time > original_time


class TestNodeSplitting:
    """Tests for node splitting behavior."""

    def test_partial_match_triggers_split(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(i) for i in range(4)]
        cache.insert(tokens, [0, 1, 2, 3])

        query = [MockToken(0), MockToken(1), MockToken(99)]
        result = cache.match_prefix(query)

        assert result.matched_token_count == 2
        assert result.matched_pages == [0, 1]

    def test_split_preserves_pages(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(i) for i in range(4)]
        cache.insert(tokens, [10, 20, 30, 40])

        query = [MockToken(0), MockToken(1), MockToken(99)]
        result = cache.match_prefix(query)

        parent = result.last_node
        assert parent is not None
        assert parent.physical_pages == (10, 20)

        # Find child (suffix node)
        assert len(parent.children) == 1
        child = list(parent.children.values())[0]
        assert child.physical_pages == (30, 40)

    def test_split_maintains_lock_refs(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(i) for i in range(4)]
        insert_result = cache.insert(tokens, [0, 1, 2, 3])
        cache.lock(insert_result.node)

        query = [MockToken(0), MockToken(1), MockToken(99)]
        match_result = cache.match_prefix(query)

        # Both parent and child should have lock_ref=1
        assert match_result.last_node is not None
        assert match_result.last_node.lock_ref == 1
        child = list(match_result.last_node.children.values())[0]
        assert child.lock_ref == 1

    def test_split_with_variable_kv_length(self) -> None:
        cache = RadixPrefixCache()
        # Token with kv_length=3
        tokens = [MockToken(0), MockToken(1, kv_len=3), MockToken(2)]
        pages = list(range(5))  # 1 + 3 + 1
        cache.insert(tokens, pages)

        # Split after first token
        query = [MockToken(0), MockToken(99)]
        result = cache.match_prefix(query)

        assert result.matched_token_count == 1
        assert result.matched_pages == [0]

        parent = result.last_node
        assert parent is not None
        child = list(parent.children.values())[0]
        assert child.physical_pages == (1, 2, 3, 4)  # kv_len=3 + kv_len=1


class TestInsert:
    """Tests for insert operation."""

    def test_insert_creates_node(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2)]
        result = cache.insert(tokens, [0, 1])

        assert result.node is not None
        assert result.inserted_pages == 2
        assert cache.total_cached_pages == 2

    def test_insert_same_sequence_twice(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(1), MockToken(2)]

        result1 = cache.insert(tokens, [0, 1])
        result2 = cache.insert(tokens, [0, 1])

        assert result2.node is result1.node
        assert result2.inserted_pages == 0
        assert cache.total_cached_pages == 2  # Not doubled

    def test_insert_with_variable_kv_length(self) -> None:
        cache = RadixPrefixCache()
        # Token with kv_length=729 (like image)
        tokens = [MockToken(0), MockToken(1, kv_len=729), MockToken(2)]
        pages = list(range(731))  # 1 + 729 + 1

        result = cache.insert(tokens, pages)
        assert result.inserted_pages == 731

    def test_insert_from_node(self) -> None:
        cache = RadixPrefixCache()
        prefix = [MockToken(0), MockToken(1)]
        cache.insert(prefix, [0, 1])

        # Match prefix, then insert extension
        match = cache.match_prefix(prefix)
        extended = prefix + [MockToken(2), MockToken(3)]
        result = cache.insert(
            extended,
            [0, 1, 2, 3],
            from_node=match.last_node,
            from_token_idx=2,
            from_page_idx=2,
        )

        assert result.inserted_pages == 2  # Only new pages
        assert cache.total_cached_pages == 4

    def test_insert_extending_existing(self) -> None:
        cache = RadixPrefixCache()
        prefix = [MockToken(0), MockToken(1)]
        cache.insert(prefix, [0, 1])

        # Insert longer sequence
        extended = [MockToken(0), MockToken(1), MockToken(2)]
        result = cache.insert(extended, [0, 1, 2])

        assert result.inserted_pages == 1  # Only the new token
        assert cache.total_cached_pages == 3

    def test_insert_branching(self) -> None:
        cache = RadixPrefixCache()
        seq1 = [MockToken(0), MockToken(1)]
        seq2 = [MockToken(0), MockToken(2)]

        cache.insert(seq1, [0, 1])
        cache.insert(seq2, [0, 2])

        # Both should be accessible
        result1 = cache.match_prefix(seq1)
        result2 = cache.match_prefix(seq2)

        assert result1.matched_token_count == 2
        assert result2.matched_token_count == 2
        assert result1.matched_pages == [0, 1]
        assert result2.matched_pages == [0, 2]


class TestLocking:
    """Tests for lock/unlock operations."""

    def test_lock_increments_ancestor_refs(self) -> None:
        cache = RadixPrefixCache()
        # Create tree: root -> [0] -> [0,1] -> [0,1,2]
        cache.insert([MockToken(0)], [0])
        cache.insert([MockToken(0), MockToken(1)], [0, 1])
        result = cache.insert([MockToken(0), MockToken(1), MockToken(2)], [0, 1, 2])

        cache.lock(result.node)

        # Walk up and check lock_refs
        node: TreeNode | None = result.node
        while node is not None:
            assert node.lock_ref == 1
            node = node.parent

    def test_unlock_decrements(self) -> None:
        cache = RadixPrefixCache()
        result = cache.insert([MockToken(0)], [0])
        cache.lock(result.node)
        cache.unlock(result.node)

        assert result.node.lock_ref == 0

    def test_multiple_locks(self) -> None:
        cache = RadixPrefixCache()
        result = cache.insert([MockToken(0)], [0])

        cache.lock(result.node)
        cache.lock(result.node)

        assert result.node.lock_ref == 2

        cache.unlock(result.node)
        assert result.node.lock_ref == 1

        cache.unlock(result.node)
        assert result.node.lock_ref == 0

    def test_unlock_underflow_raises(self) -> None:
        cache = RadixPrefixCache()
        result = cache.insert([MockToken(0)], [0])

        with pytest.raises(AssertionError):
            cache.unlock(result.node)

    def test_lock_none_is_noop(self) -> None:
        cache = RadixPrefixCache()
        cache.lock(None)  # Should not raise

    def test_unlock_none_is_noop(self) -> None:
        cache = RadixPrefixCache()
        cache.unlock(None)  # Should not raise


class TestEviction:
    """Tests for eviction operation."""

    def test_evict_frees_unlocked_leaf(self) -> None:
        cache = RadixPrefixCache()
        cache.insert([MockToken(0), MockToken(1)], [0, 1])

        freed_count, freed_pages = cache.evict(2)

        assert freed_count == 2
        assert set(freed_pages) == {0, 1}
        assert cache.total_cached_pages == 0

    def test_evict_skips_locked_nodes(self) -> None:
        cache = RadixPrefixCache()
        result = cache.insert([MockToken(0)], [0])
        cache.lock(result.node)

        freed_count, freed_pages = cache.evict(1)

        assert freed_count == 0
        assert freed_pages == []
        assert cache.total_cached_pages == 1

    def test_evict_lru_order(self) -> None:
        cache = RadixPrefixCache()
        # Insert old then new
        cache.insert([MockToken(0)], [0])
        time.sleep(0.02)  # Ensure different timestamps
        cache.insert([MockToken(1)], [1])

        # Evict one - should be older
        freed_count, freed_pages = cache.evict(1)

        assert freed_count == 1
        assert freed_pages == [0]

    def test_evict_lru_order_after_match(self) -> None:
        """Matching a prefix should update LRU order."""
        cache = RadixPrefixCache()
        # Insert A (old) then B (newer)
        cache.insert([MockToken(0)], [0])
        time.sleep(0.02)
        cache.insert([MockToken(1)], [1])
        time.sleep(0.02)

        # Match A - should update A's access time to now
        cache.match_prefix([MockToken(0)])

        # Evict one - should evict B (older access), not A (recently matched)
        freed_count, freed_pages = cache.evict(1)

        assert freed_count == 1
        assert freed_pages == [1], "Should evict B (page 1), not A (page 0) which was recently matched"

    def test_evict_lru_updates_ancestor_access_time(self) -> None:
        """Matching a path should update access time for ancestors too.

        When an internal node becomes a leaf (after children evicted), it should
        retain the access time from when it was last part of a matched path.
        """
        cache = RadixPrefixCache()
        # Insert [A] (will become internal node) and [A, B] (leaf)
        cache.insert([MockToken(0)], [0])
        cache.insert([MockToken(0), MockToken(1)], [0, 1])
        time.sleep(0.02)

        # Insert [C] as another leaf
        cache.insert([MockToken(2)], [2])
        time.sleep(0.02)

        # Match [A, B] - updates access time for both A and B nodes
        cache.match_prefix([MockToken(0), MockToken(1)])

        # Evict one - should evict C (older than recently matched [A,B] path)
        freed1, pages1 = cache.evict(1)
        assert freed1 == 1
        assert pages1 == [2], "Should evict C first (older than recently matched [A,B])"

        # Now evict B - A becomes a leaf with access time from the match
        freed2, pages2 = cache.evict(1)
        assert freed2 == 1
        assert pages2 == [1], "Should evict B"

        # Insert [D] - this happens BEFORE we match again
        cache.insert([MockToken(3)], [3])
        time.sleep(0.02)

        # Match [A] again - this should update A's access time
        cache.match_prefix([MockToken(0)])

        # Now evict - should evict D (older), not A (just matched)
        freed3, pages3 = cache.evict(1)
        assert freed3 == 1
        assert pages3 == [3], "Should evict D, not A which was just matched"

    def test_evict_makes_parent_eligible(self) -> None:
        cache = RadixPrefixCache()
        # Insert [A] and [A, B]
        cache.insert([MockToken(0)], [0])
        cache.insert([MockToken(0), MockToken(1)], [0, 1])

        # Evict [A, B] leaf
        freed1, _ = cache.evict(1)

        # Now [A] is a leaf and evictable
        freed2, _ = cache.evict(1)

        assert freed1 == 1
        assert freed2 == 1
        assert cache.total_cached_pages == 0

    def test_evict_partial(self) -> None:
        cache = RadixPrefixCache()
        cache.insert([MockToken(0)], [0])
        cache.insert([MockToken(1)], [1])

        freed_count, freed_pages = cache.evict(1)

        assert freed_count == 1
        assert len(freed_pages) == 1
        assert cache.total_cached_pages == 1

    def test_evict_more_than_available(self) -> None:
        cache = RadixPrefixCache()
        cache.insert([MockToken(0)], [0])

        freed_count, freed_pages = cache.evict(10)

        assert freed_count == 1
        assert freed_pages == [0]
        assert cache.total_cached_pages == 0

    def test_evictable_page_count(self) -> None:
        cache = RadixPrefixCache()

        # Empty cache
        assert cache.evictable_page_count() == 0

        # Insert some nodes
        cache.insert([MockToken(0)], [0])
        cache.insert([MockToken(1), MockToken(2)], [1, 2])
        assert cache.evictable_page_count() == 3

        # Lock one node - its pages become non-evictable
        result = cache.insert([MockToken(3)], [3])
        cache.lock(result.node)
        assert cache.evictable_page_count() == 3  # Only unlocked leaves

        # Unlock - now all are evictable
        cache.unlock(result.node)
        assert cache.evictable_page_count() == 4


class TestNamespace:
    """Tests for namespace isolation."""

    def test_separate_namespaces(self) -> None:
        cache = RadixPrefixCache()
        ns1 = CacheNamespace(lora_id=1)
        ns2 = CacheNamespace(lora_id=2)

        tokens = [MockToken(0)]
        cache.insert(tokens, [0], namespace=ns1)
        cache.insert(tokens, [1], namespace=ns2)

        result1 = cache.match_prefix(tokens, namespace=ns1)
        result2 = cache.match_prefix(tokens, namespace=ns2)

        assert result1.matched_pages == [0]
        assert result2.matched_pages == [1]

    def test_miss_in_wrong_namespace(self) -> None:
        cache = RadixPrefixCache()
        ns1 = CacheNamespace(lora_id=1)
        ns2 = CacheNamespace(lora_id=2)

        tokens = [MockToken(0)]
        cache.insert(tokens, [0], namespace=ns1)

        result = cache.match_prefix(tokens, namespace=ns2)
        assert result.matched_kv_length == 0

    def test_default_namespace(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(0)]
        cache.insert(tokens, [0])

        # Both None and default namespace should work
        result1 = cache.match_prefix(tokens)
        result2 = cache.match_prefix(tokens, namespace=None)
        result3 = cache.match_prefix(tokens, namespace=CacheNamespace())

        assert result1.matched_kv_length == 1
        assert result2.matched_kv_length == 1
        assert result3.matched_kv_length == 1

    def test_image_hash_namespace(self) -> None:
        cache = RadixPrefixCache()
        ns1 = CacheNamespace(image_hash=12345)
        ns2 = CacheNamespace(image_hash=67890)

        tokens = [MockToken(0)]
        cache.insert(tokens, [0], namespace=ns1)

        result1 = cache.match_prefix(tokens, namespace=ns1)
        result2 = cache.match_prefix(tokens, namespace=ns2)

        assert result1.matched_kv_length == 1
        assert result2.matched_kv_length == 0

    def test_match_prefix_does_not_create_namespace_root(self) -> None:
        """match_prefix() should not create namespace roots on miss."""
        cache = RadixPrefixCache()
        ns = CacheNamespace(lora_id=42)

        tokens = [MockToken(0), MockToken(1)]

        # match_prefix on non-existent namespace should not create a root
        result = cache.match_prefix(tokens, namespace=ns)
        assert result.matched_kv_length == 0
        assert ns not in cache._trees, "match_prefix() should not create namespace root on miss"

        # Verify _trees is still empty
        assert len(cache._trees) == 0

    def test_empty_namespace_root_pruned(self) -> None:
        """Empty namespace roots should be removed after last child evicted."""
        cache = RadixPrefixCache()
        ns = CacheNamespace(lora_id=42)

        tokens = [MockToken(0)]
        cache.insert(tokens, [0], namespace=ns)

        # Namespace should exist
        assert ns in cache._trees

        # Evict the only node
        cache.evict(1)

        # Namespace root should be pruned
        assert ns not in cache._trees, "Empty namespace root should be pruned"

    def test_namespace_root_prune_asserts_on_nonzero_lock_ref(self) -> None:
        """Pruning namespace root with lock_ref > 0 should assert (catches lock bugs)."""
        cache = RadixPrefixCache()
        ns = CacheNamespace(lora_id=42)

        tokens = [MockToken(0)]
        cache.insert(tokens, [0], namespace=ns)

        # Simulate a bug: manually corrupt the namespace root's lock_ref
        root = cache._trees[ns]
        root.lock_ref = 1  # Should be 0 for unlocked root

        # Eviction should trigger assertion when trying to prune corrupted root
        with pytest.raises(AssertionError, match="lock_ref"):
            cache.evict(1)


class TestInvariants:
    """Tests for cache invariants."""

    def test_insert_rejects_misaligned_pages(self) -> None:
        """Insert should reject page lists that don't match token kv_lengths."""
        cache = RadixPrefixCache()
        tokens = [MockToken(0), MockToken(1)]  # kv_length = 2

        # Too few pages
        with pytest.raises(AssertionError):
            cache.insert(tokens, [0])

        # Too many pages
        with pytest.raises(AssertionError):
            cache.insert(tokens, [0, 1, 2])

    def test_insert_rejects_misaligned_from_page_idx(self) -> None:
        """Insert should reject from_page_idx that doesn't match token kv_lengths."""
        cache = RadixPrefixCache()
        # Insert prefix first
        prefix = [MockToken(0), MockToken(1, kv_len=3)]  # kv_length = 1 + 3 = 4
        cache.insert(prefix, [0, 1, 2, 3])

        # Match and try to extend with wrong from_page_idx
        match = cache.match_prefix(prefix)
        extended = prefix + [MockToken(2)]

        # Correct from_page_idx should be 4 (1 + 3), but pass 2
        with pytest.raises(AssertionError):
            cache.insert(
                extended,
                [0, 1, 2, 3, 4],
                from_node=match.last_node,
                from_token_idx=2,
                from_page_idx=2,  # Wrong! Should be 4
            )

    def test_page_token_alignment_on_insert(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(0), MockToken(1, kv_len=729)]
        pages = list(range(730))  # 1 + 729

        result = cache.insert(tokens, pages)

        assert result.node.total_kv_length == len(result.node.physical_pages)

    def test_page_token_alignment_after_split(self) -> None:
        cache = RadixPrefixCache()
        tokens = [MockToken(i) for i in range(4)]
        cache.insert(tokens, [0, 1, 2, 3])

        # Split
        cache.match_prefix([MockToken(0), MockToken(1), MockToken(99)])

        # Verify all nodes maintain invariant
        def check_invariant(node: TreeNode) -> None:
            expected = sum(t.kv_length() for t in node.tokens)
            assert len(node.physical_pages) == expected
            for child in node.children.values():
                check_invariant(child)

        for root in cache._trees.values():
            check_invariant(root)

    def test_total_cached_pages_consistency(self) -> None:
        cache = RadixPrefixCache()

        cache.insert([MockToken(0)], [0])
        assert cache.total_cached_pages == 1

        cache.insert([MockToken(0), MockToken(1)], [0, 1])
        assert cache.total_cached_pages == 2

        cache.insert([MockToken(2)], [2])
        assert cache.total_cached_pages == 3

        cache.evict(1)
        assert cache.total_cached_pages == 2

    def test_bidirectional_parent_child(self) -> None:
        cache = RadixPrefixCache()
        cache.insert([MockToken(0), MockToken(1)], [0, 1])
        cache.insert([MockToken(0), MockToken(2)], [0, 2])

        # Trigger split
        cache.match_prefix([MockToken(0), MockToken(99)])

        def verify_tree(node: TreeNode) -> None:
            for child in node.children.values():
                assert child.parent is node
                verify_tree(child)

        for root in cache._trees.values():
            verify_tree(root)
