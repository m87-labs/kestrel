"""LRU eviction policy for prefix cache."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .radix_cache import TreeNode


class LRUEvictionPolicy:
    """LRU eviction using lazy-deletion min-heap.

    Tracks evictable leaf nodes. Uses version numbers to handle
    lazy deletion when nodes are re-accessed or become locked.

    The heap contains tuples of (last_access_time, counter, node, version).
    When a node is re-added to the heap, its heap_version is incremented,
    invalidating any stale entries with older versions.
    """

    def __init__(self, collect_leaves_fn: Callable[[], list[TreeNode]]):
        """Initialize the eviction policy.

        Args:
            collect_leaves_fn: Callback to collect all unlocked leaves for heap rebuild.
        """
        self._heap: list[tuple[float, int, TreeNode, int]] = []
        self._counter = 0
        self._collect_leaves = collect_leaves_fn

    def add(self, node: TreeNode) -> None:
        """Add node to eviction heap if evictable.

        A node is evictable if it is an unlocked leaf (not root).

        Args:
            node: Node to potentially add to heap.
        """
        if node.is_leaf() and node.lock_ref == 0 and node.parent is not None:
            node.heap_version += 1
            heapq.heappush(
                self._heap,
                (node.last_access_time, self._counter, node, node.heap_version),
            )
            self._counter += 1

    def pop_candidate(self) -> TreeNode | None:
        """Pop next valid eviction candidate, skipping stale entries.

        Returns:
            Valid eviction candidate, or None if no valid candidates.
        """
        while self._heap:
            _, _, node, version = heapq.heappop(self._heap)
            if self._is_valid_candidate(node, version):
                return node
        return None

    def _is_valid_candidate(self, node: TreeNode, version: int) -> bool:
        """Check if heap entry is still valid for eviction.

        Args:
            node: Node from heap entry.
            version: Version from heap entry.

        Returns:
            True if node is still a valid eviction candidate.
        """
        if node.heap_version != version:
            return False
        if not node.is_leaf() or node.lock_ref != 0:
            return False
        if node.parent is None:
            return False
        if not node.tokens:
            return False
        first_key = node.tokens[0].cache_key()
        if node.parent.children.get(first_key) is not node:
            return False
        return True

    def rebuild(self) -> None:
        """Rebuild heap from scratch.

        Called as fallback when heap becomes too stale or exhausted.
        """
        self._heap.clear()
        for leaf in self._collect_leaves():
            self.add(leaf)
