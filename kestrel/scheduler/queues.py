"""Lightweight FIFO queues used by the scheduler."""


from collections import deque
from typing import Deque, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class FIFOQueue(Generic[T]):
    """Simple typed wrapper around ``deque`` for clarity in scheduling code."""

    def __init__(self) -> None:
        self._items: Deque[T] = deque()

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.popleft()

    def peek(self) -> Optional[T]:
        return self._items[0] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def extend(self, items: Iterable[T]) -> None:
        self._items.extend(items)

    def clear(self) -> None:
        self._items.clear()

    def take_all(self) -> list[T]:
        items = list(self._items)
        self._items.clear()
        return items

    def remove(self, item: T) -> None:
        """Remove an item from the queue. Raises ValueError if not found."""
        self._items.remove(item)


RequestQueue = FIFOQueue
RunningQueue = FIFOQueue
