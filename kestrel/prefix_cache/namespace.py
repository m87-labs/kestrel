"""Cache namespace for prefix cache isolation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Namespace for cache isolation (LoRA adapters, image hashes).

    Frozen to be hashable for use as dict keys. Uses slots for memory efficiency.

    Attributes:
        lora_id: LoRA adapter name, or None for base model. Uses the stable
            adapter identity (not slot index) to avoid cross-adapter cache hits
            when slots are reused.
        image_hash: 128-bit hash of image content, or None for text-only.
    """

    lora_id: str | None = None
    image_hash: int | None = None
