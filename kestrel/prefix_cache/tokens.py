"""Cache token implementations for prefix caching."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageToken:
    """Token representing image content in the prefix cache.

    Images produce multiple KV positions (typically 729 for ViT encoder output).
    The cache key is the image content hash, ensuring different images are cached
    separately even if they appear at the same position in the prompt.
    """

    content_hash: int  # 128-bit hash from SHA256 of image bytes
    kv_length_: int  # Number of KV positions (e.g., 729 for Moondream)

    def cache_key(self) -> tuple:
        """Cache key: (3, content_hash, kv_length) - 3 discriminates from text/coord/size tokens."""
        return (3, self.content_hash, self.kv_length_)

    def kv_length(self) -> int:
        """Number of KV positions this image occupies."""
        return self.kv_length_
