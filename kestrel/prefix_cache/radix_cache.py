"""Radix tree prefix cache for KV cache sharing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import BasePrefixCache, CacheToken
from .eviction import LRUEvictionPolicy
from .namespace import CacheNamespace

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class TreeNode:
    """Radix tree node representing a cached prefix segment.

    Invariant: len(physical_pages) == sum(tok.kv_length() for tok in tokens)
    """

    children: dict[tuple[Any, ...], TreeNode] = field(default_factory=dict)
    parent: TreeNode | None = None
    tokens: tuple[CacheToken, ...] = ()
    physical_pages: tuple[int, ...] = ()
    lock_ref: int = 0
    prefill_lock_ref: int = 0
    last_access_time: float = 0.0
    heap_version: int = 0
    namespace: CacheNamespace | None = None  # Only set on root nodes

    @property
    def total_kv_length(self) -> int:
        """Total KV positions in this node. O(1) via invariant."""
        return len(self.physical_pages)

    def is_leaf(self) -> bool:
        """Check if this node has no children."""
        return len(self.children) == 0


@dataclass(slots=True)
class MatchResult:
    """Result of prefix matching operation."""

    matched_pages: list[int]
    matched_kv_length: int
    matched_token_count: int
    last_node: TreeNode | None
    unmatched_tokens: list[CacheToken]


@dataclass(slots=True)
class InsertResult:
    """Result of cache insertion."""

    node: TreeNode
    inserted_pages: int  # Pages actually claimed by cache


class RadixPrefixCache(BasePrefixCache):
    """Radix tree for KV cache prefix sharing.

    Stores computed KV cache pages keyed by token sequences. Enables sharing
    of cached pages across requests with matching prefixes.

    page_size must be 1 (enforced by design: len(physical_pages) == sum of
    token kv_lengths).
    """

    def __init__(self, free_pages_sink=None) -> None:
        """Initialize empty prefix cache.

        Args:
            free_pages_sink: Optional callback to receive freed pages during eviction.
                Should accept a list[int] of physical page indices.
                Typically set automatically by PageTable constructor.
        """
        self._trees: dict[CacheNamespace, TreeNode] = {}
        self._default_namespace = CacheNamespace()
        self._total_cached_pages = 0
        self._evictable_page_count = 0  # Incrementally tracked for O(1) lookup
        self._eviction_policy = LRUEvictionPolicy(self._collect_unlocked_leaves)
        self._free_pages_sink = free_pages_sink

    @property
    def total_cached_pages(self) -> int:
        """Total pages currently owned by cache."""
        return self._total_cached_pages

    def match_prefix(
        self,
        tokens: list[CacheToken],
        namespace: CacheNamespace | None = None,
    ) -> MatchResult:
        """Find longest cached prefix matching tokens.

        Updates access times for entire matched path. Splits nodes on partial
        match within a node.

        Args:
            tokens: Sequence of tokens to match.
            namespace: Cache namespace (None uses default).

        Returns:
            MatchResult with matched pages, KV length, token count, terminal node,
            and unmatched tokens.
        """
        if namespace is None:
            namespace = self._default_namespace

        if namespace not in self._trees:
            return MatchResult(
                matched_pages=[],
                matched_kv_length=0,
                matched_token_count=0,
                last_node=None,
                unmatched_tokens=list(tokens),
            )

        root = self._trees[namespace]
        matched_pages: list[int] = []
        matched_token_count = 0
        current_node = root
        path: list[TreeNode] = [root]

        token_idx = 0
        while token_idx < len(tokens):
            token = tokens[token_idx]
            key = token.cache_key()

            if key not in current_node.children:
                break

            child = current_node.children[key]
            child_tokens = child.tokens

            # Match tokens within this child node
            match_len = 0
            while (
                match_len < len(child_tokens)
                and token_idx + match_len < len(tokens)
                and tokens[token_idx + match_len].cache_key()
                == child_tokens[match_len].cache_key()
            ):
                match_len += 1

            if match_len < len(child_tokens):
                # Partial match - split the node unless it's locked.
                if child.prefill_lock_ref > 0:
                    break
                child = self._split_node(child, match_len)

            # Accumulate matched pages
            matched_pages.extend(child.physical_pages)
            matched_token_count += len(child.tokens)
            token_idx += len(child.tokens)
            current_node = child
            path.append(child)

        # Update access times and re-enqueue evictable nodes for entire path.
        # Internal nodes won't be added (fail is_leaf() check), but updating
        # their access time is useful when they become leaves later.
        now = time.monotonic()
        for node in path:
            node.last_access_time = now
            self._eviction_policy.add(node)  # No-op for non-leaves

        last_node = current_node if current_node is not root else None

        return MatchResult(
            matched_pages=matched_pages,
            matched_kv_length=len(matched_pages),
            matched_token_count=matched_token_count,
            last_node=last_node,
            unmatched_tokens=list(tokens[token_idx:]),
        )

    def insert(
        self,
        tokens: list[CacheToken],
        pages: list[int],
        namespace: CacheNamespace | None = None,
        from_node: TreeNode | None = None,
        from_token_idx: int = 0,
        from_page_idx: int = 0,
    ) -> InsertResult:
        """Insert prefix into cache.

        If from_node is provided, starts insertion from that point (optimization
        for extending after a cache hit).

        Args:
            tokens: Full token sequence.
            pages: Physical page indices for the sequence.
            namespace: Cache namespace (None uses default).
            from_node: Start insertion from this node.
            from_token_idx: Token index to start from when using from_node.
            from_page_idx: Page index to start from when using from_node.

        Returns:
            InsertResult with terminal node and count of pages actually inserted.
        """
        # Validate token/page alignment
        expected_pages = sum(t.kv_length() for t in tokens)
        assert len(pages) == expected_pages, (
            f"Page count mismatch: got {len(pages)} pages, "
            f"expected {expected_pages} for {len(tokens)} tokens"
        )

        if namespace is None:
            namespace = self._default_namespace

        # Get or create root
        if namespace not in self._trees:
            root = TreeNode(namespace=namespace)
            self._trees[namespace] = root
        root = self._trees[namespace]

        # Start from from_node or root
        if from_node is not None:
            current_node = from_node
            token_idx = from_token_idx
            page_idx = from_page_idx

            # Validate from_page_idx alignment (debug only to avoid O(N) re-summation)
            if __debug__:
                expected_page_idx = sum(t.kv_length() for t in tokens[:from_token_idx])
                assert page_idx == expected_page_idx, (
                    f"from_page_idx mismatch: got {page_idx}, "
                    f"expected {expected_page_idx} for {from_token_idx} tokens"
                )
        else:
            current_node = root
            token_idx = 0
            page_idx = 0

        # Traverse existing path
        while token_idx < len(tokens):
            token = tokens[token_idx]
            key = token.cache_key()

            if key not in current_node.children:
                break

            child = current_node.children[key]
            child_tokens = child.tokens

            # Check how many tokens match
            match_len = 0
            while (
                match_len < len(child_tokens)
                and token_idx + match_len < len(tokens)
                and tokens[token_idx + match_len].cache_key()
                == child_tokens[match_len].cache_key()
            ):
                match_len += 1

            if match_len < len(child_tokens):
                # Partial match - split the node unless it's locked.
                if child.prefill_lock_ref > 0:
                    return InsertResult(node=current_node, inserted_pages=0)
                child = self._split_node(child, match_len)

            # Guard against inserting when the caller didn't map the cache's
            # existing prefix pages into its own page table (prevents corruption).
            if tuple(pages[page_idx : page_idx + child.total_kv_length]) != child.physical_pages:
                return InsertResult(node=current_node, inserted_pages=0)

            # Advance past matched tokens
            page_idx += child.total_kv_length
            token_idx += len(child.tokens)
            current_node = child

        # If we've consumed all tokens, the sequence already exists
        if token_idx >= len(tokens):
            return InsertResult(node=current_node, inserted_pages=0)

        # Create new node for remaining tokens
        remaining_tokens = tuple(tokens[token_idx:])
        remaining_pages = tuple(pages[page_idx:])

        # Check if parent was an evictable leaf before adding child
        parent_was_evictable_leaf = (
            current_node.is_leaf()
            and self._is_evictable(current_node)
            and current_node.parent is not None
        )

        new_node = TreeNode(
            parent=current_node,
            tokens=remaining_tokens,
            physical_pages=remaining_pages,
            last_access_time=time.monotonic(),
        )

        first_key = remaining_tokens[0].cache_key()
        current_node.children[first_key] = new_node

        inserted_pages = len(remaining_pages)
        self._total_cached_pages += inserted_pages
        # New node is unlocked leaf, so it's evictable
        self._evictable_page_count += inserted_pages
        # Parent is no longer a leaf, so subtract its pages if it was evictable
        if parent_was_evictable_leaf:
            self._evictable_page_count -= len(current_node.physical_pages)

        # Add to eviction heap (will be skipped if locked immediately after)
        self._eviction_policy.add(new_node)

        return InsertResult(node=new_node, inserted_pages=inserted_pages)

    def lock(self, node: TreeNode | None) -> None:
        """Prevent eviction of node and all ancestors.

        Increments lock_ref up the ancestor chain.

        Args:
            node: Node to lock (None is a no-op).
        """
        if node is None:
            return

        # If node was an evictable leaf, it's no longer evictable after lock
        if node.is_leaf() and self._is_evictable(node) and node.parent is not None:
            self._evictable_page_count -= len(node.physical_pages)

        current: TreeNode | None = node
        while current is not None:
            current.lock_ref += 1
            current = current.parent

    def lock_prefill(self, node: TreeNode | None) -> None:
        """Prevent mutation of node and all ancestors during prefill.

        Increments prefill_lock_ref up the ancestor chain.
        """
        if node is None:
            return

        # If node was an evictable leaf, it's no longer evictable after lock
        if node.is_leaf() and self._is_evictable(node) and node.parent is not None:
            self._evictable_page_count -= len(node.physical_pages)

        current: TreeNode | None = node
        while current is not None:
            current.prefill_lock_ref += 1
            current = current.parent

    def unlock(self, node: TreeNode | None) -> None:
        """Release eviction lock on node and all ancestors.

        Decrements lock_ref up the ancestor chain. Asserts no underflow.
        Adds leaf to eviction heap when lock_ref reaches 0.

        Args:
            node: Node to unlock (None is a no-op).
        """
        if node is None:
            return

        current: TreeNode | None = node
        while current is not None:
            assert current.lock_ref > 0, "Lock ref underflow"
            current.lock_ref -= 1
            current = current.parent

        # Add leaf to eviction heap if now evictable
        if node.is_leaf() and self._is_evictable(node) and node.parent is not None:
            self._evictable_page_count += len(node.physical_pages)
            self._eviction_policy.add(node)

    def unlock_prefill(self, node: TreeNode | None) -> None:
        """Release prefill mutation lock on node and all ancestors.

        Decrements prefill_lock_ref up the ancestor chain. Asserts no underflow.
        Adds leaf to eviction heap when fully unlocked.
        """
        if node is None:
            return

        current: TreeNode | None = node
        while current is not None:
            assert current.prefill_lock_ref > 0, "Prefill lock ref underflow"
            current.prefill_lock_ref -= 1
            current = current.parent

        if node.is_leaf() and self._is_evictable(node) and node.parent is not None:
            self._evictable_page_count += len(node.physical_pages)
            self._eviction_policy.add(node)

    def evict(self, needed_pages: int) -> int:
        """Evict unlocked leaves to free pages using LRU policy.

        Freed pages are sent to the free_pages_sink callback (if configured).

        Args:
            needed_pages: Number of pages to free.

        Returns:
            Number of pages actually freed.
        """
        freed_count = 0

        while freed_count < needed_pages:
            candidate = self._eviction_policy.pop_candidate()

            if candidate is None:
                # Heap exhausted, try rebuilding
                self._eviction_policy.rebuild()
                candidate = self._eviction_policy.pop_candidate()
                if candidate is None:
                    break  # No evictable nodes

            # Evict this node
            node_pages = list(candidate.physical_pages)
            freed_count += len(node_pages)
            self._total_cached_pages -= len(node_pages)
            # Node was evictable, so subtract from evictable count
            self._evictable_page_count -= len(node_pages)

            # Send freed pages to sink (e.g., PageTable.free_pages_to_pool)
            if self._free_pages_sink is not None:
                self._free_pages_sink(node_pages)

            # Remove from tree
            parent = candidate.parent
            if parent is not None and candidate.tokens:
                first_key = candidate.tokens[0].cache_key()
                if parent.children.get(first_key) is candidate:
                    del parent.children[first_key]

                # If parent becomes a leaf, add to eviction heap
                if parent.is_leaf() and self._is_evictable(parent) and parent.parent is not None:
                    self._evictable_page_count += len(parent.physical_pages)
                    self._eviction_policy.add(parent)

                # Prune empty namespace roots
                if parent.parent is None and parent.is_leaf() and parent.namespace is not None:
                    assert self._is_evictable(parent), (
                        "Namespace root has non-zero lock_ref/prefill_lock_ref during prune"
                    )
                    del self._trees[parent.namespace]

        return freed_count

    def evictable_page_count(self) -> int:
        """Return number of pages that can be evicted.

        Used by PageTable.can_reserve_with_eviction() to check if enough
        pages can be freed without actually evicting.

        Returns:
            Number of pages in unlocked leaf nodes.
        """
        return self._evictable_page_count

    def _split_node(self, node: TreeNode, split_at: int) -> TreeNode:
        """Split a node into prefix parent and suffix child.

        Args:
            node: Node to split.
            split_at: Token index to split at.

        Returns:
            The new prefix parent node.
        """
        assert 0 < split_at < len(node.tokens), "Invalid split position"
        assert node.prefill_lock_ref == 0, "Cannot split prefill-locked node"

        # Check if node was evictable before split (for incremental tracking)
        was_evictable = node.is_leaf() and self._is_evictable(node) and node.parent is not None

        # Calculate page split point
        page_split = sum(t.kv_length() for t in node.tokens[:split_at])

        prefix_tokens = node.tokens[:split_at]
        suffix_tokens = node.tokens[split_at:]
        prefix_pages = node.physical_pages[:page_split]
        suffix_pages = node.physical_pages[page_split:]

        # Create new parent with prefix
        new_parent = TreeNode(
            parent=node.parent,
            tokens=prefix_tokens,
            physical_pages=prefix_pages,
            lock_ref=node.lock_ref,  # Inherit lock refs
            prefill_lock_ref=node.prefill_lock_ref,
            last_access_time=node.last_access_time,
        )

        # Update original node to be suffix child
        node.tokens = suffix_tokens
        node.physical_pages = suffix_pages
        node.parent = new_parent

        # Move node to be child of new parent
        if suffix_tokens:
            suffix_key = suffix_tokens[0].cache_key()
            new_parent.children[suffix_key] = node

        # Update grandparent's child pointer
        if new_parent.parent is not None:
            prefix_key = prefix_tokens[0].cache_key()
            new_parent.parent.children[prefix_key] = new_parent

        # If node was evictable, prefix pages move to non-leaf parent (not evictable)
        if was_evictable:
            self._evictable_page_count -= len(prefix_pages)

        # Invalidate stale heap entries for suffix node
        node.heap_version += 1

        # Add suffix to eviction heap if eligible
        if node.is_leaf() and self._is_evictable(node):
            self._eviction_policy.add(node)

        return new_parent

    def _collect_unlocked_leaves(self) -> list[TreeNode]:
        """Collect all unlocked leaf nodes across all namespaces.

        Used for eviction heap rebuild. Uses iterative stack to avoid
        recursion limit issues with deep trees.

        Returns:
            List of unlocked leaf nodes.
        """
        leaves: list[TreeNode] = []
        stack: list[TreeNode] = list(self._trees.values())

        while stack:
            node = stack.pop()
            if node.is_leaf():
                if self._is_evictable(node) and node.parent is not None:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())

        return leaves

    @staticmethod
    def _is_evictable(node: TreeNode) -> bool:
        return node.lock_ref == 0 and node.prefill_lock_ref == 0
