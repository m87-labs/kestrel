"""Cache namespace for prefix cache isolation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Namespace for cache isolation (runtime, LoRA adapters, image hashes).

    Frozen to be hashable for use as dict keys. Uses slots for memory efficiency.

    Attributes:
        runtime_id: Stable identifier for the runtime that owns this cache
            entry, or None for runtime-agnostic callers. When two runtimes
            (e.g., different model variants) share one prefix cache,
            ``runtime_id`` keeps their trees separate so a token-id sequence
            valid in one model can't surface as a hit for the other.
        lora_id: LoRA adapter name, or None for base model. Uses the stable
            adapter identity (not slot index) to avoid cross-adapter cache hits
            when slots are reused.
        image_hash: 128-bit hash of image content, or None for text-only.
    """

    runtime_id: str | None = None
    lora_id: str | None = None
    image_hash: int | None = None
