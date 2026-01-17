"""Abstract base class for prefix cache implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .namespace import CacheNamespace
    from .radix_cache import InsertResult, MatchResult, TreeNode


@runtime_checkable
class CacheToken(Protocol):
    """Protocol for tokens that can be cached."""

    def cache_key(self) -> tuple[Any, ...]:
        """Return hashable cache key for this token."""
        ...

    def kv_length(self) -> int:
        """Return number of KV positions this token occupies."""
        ...


class BasePrefixCache(ABC):
    """Abstract base class for prefix cache implementations."""

    @abstractmethod
    def match_prefix(
        self,
        tokens: list[CacheToken],
        namespace: CacheNamespace | None = None,
    ) -> MatchResult:
        """Find longest cached prefix matching tokens.

        Args:
            tokens: Sequence of tokens to match.
            namespace: Cache namespace (None uses default).

        Returns:
            MatchResult with matched pages, KV length, token count, terminal node,
            and unmatched tokens.
        """
        ...

    @abstractmethod
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

        Args:
            tokens: Full token sequence.
            pages: Physical page indices for the sequence.
            namespace: Cache namespace (None uses default).
            from_node: Start insertion from this node (optimization for extending).
            from_token_idx: Token index to start from when using from_node.
            from_page_idx: Page index to start from when using from_node.

        Returns:
            InsertResult with terminal node and count of pages actually inserted.
        """
        ...

    @abstractmethod
    def lock(self, node: TreeNode | None) -> None:
        """Prevent eviction of node and all ancestors.

        Args:
            node: Node to lock (None is a no-op).
        """
        ...

    @abstractmethod
    def unlock(self, node: TreeNode | None) -> None:
        """Release eviction lock on node and all ancestors.

        Args:
            node: Node to unlock (None is a no-op).
        """
        ...

    @abstractmethod
    def lock_prefill(self, node: TreeNode | None) -> None:
        """Prevent mutation of node and all ancestors during prefill.

        Args:
            node: Node to lock (None is a no-op).
        """
        ...

    @abstractmethod
    def unlock_prefill(self, node: TreeNode | None) -> None:
        """Release prefill lock on node and all ancestors.

        Args:
            node: Node to unlock (None is a no-op).
        """
        ...

    @abstractmethod
    def evict(self, needed_pages: int) -> int:
        """Evict unlocked leaves to free pages using LRU policy.

        Freed pages are sent to the free_pages_sink callback (if configured).

        Args:
            needed_pages: Number of pages to free.

        Returns:
            Number of pages actually freed.
        """
        ...

    @abstractmethod
    def evictable_page_count(self) -> int:
        """Return number of pages that can be evicted.

        Used by PageTable.can_reserve_with_eviction() to check if enough
        pages can be freed without actually evicting.

        Returns:
            Number of pages in unlocked leaf nodes.
        """
        ...
