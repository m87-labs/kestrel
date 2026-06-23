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


class ImageMarker(NamedTuple):
    """Sentinel marking where an image is spliced into the prompt.

    Deliberately NOT a vocabulary id, so user text can never collide with it.
    A model's runtime replaces it before the forward pass — Qwen expands it to
    ``<|vision_start|><|image_pad|>×N<|vision_end|>``; Moondream injects the
    image-embedding block. ``index`` is the image's position in the request's
    ordered image list, so each marker maps to a specific image.
    """

    index: int

    def cache_key(self) -> tuple:
        """Cache key: (4, index) - 4 discriminates from other token types."""
        return (4, self.index)

    def kv_length(self) -> int:
        # A single placeholder slot; the runtime expands it to the image's
        # real token/embedding length during sequence preparation.
        return 1


Token = TextToken | CoordToken | SizeToken | ImageMarker
