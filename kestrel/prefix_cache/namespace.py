"""Cache namespace for prefix cache isolation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Namespace for cache isolation (LoRA adapters, image hashes).

    Frozen to be hashable for use as dict keys. Uses slots for memory efficiency.

    Attributes:
        lora_id: LoRA adapter ID, or None for base model.
        image_hash: 128-bit hash of image content, or None for text-only.
    """

    lora_id: int | None = None
    image_hash: int | None = None
