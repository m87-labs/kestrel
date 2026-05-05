"""Typed tokens exchanged between runtimes, the prefix cache, and the scheduler.

Each token kind reports its own ``cache_key()`` (used by the prefix cache
to disambiguate token kinds at the same vocabulary position) and a
``kv_length()`` (the number of KV positions the token occupies). All
current token kinds occupy exactly one position, but the abstraction
exists so future token kinds can carry multi-position state.
"""

from __future__ import annotations

from typing import NamedTuple


class TextToken(NamedTuple):
    """Discrete text token represented by its vocabulary id."""

    token_id: int

    def cache_key(self) -> tuple:
        """Cache key: (0, token_id) - 0 discriminates from other token types."""
        return (0, self.token_id)

    def kv_length(self) -> int:
        return 1


class CoordToken(NamedTuple):
    """Normalized positional token emitted or consumed by the region model."""

    pos: float

    def cache_key(self) -> tuple:
        """Cache key: (1, pos) - 1 discriminates from text tokens."""
        return (1, self.pos)

    def kv_length(self) -> int:
        return 1


class SizeToken(NamedTuple):
    """Normalized width/height token emitted or consumed by the region model."""

    width: float
    height: float

    def cache_key(self) -> tuple:
        """Cache key: (2, width, height) - 2 discriminates from coord tokens."""
        return (2, self.width, self.height)

    def kv_length(self) -> int:
        return 1


Token = TextToken | CoordToken | SizeToken
