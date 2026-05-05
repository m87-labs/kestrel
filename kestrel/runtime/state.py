"""Sequence-state and prefill-preparation dataclasses used by runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor

from kestrel.prefix_cache import CacheNamespace, CacheToken, MatchResult, TreeNode

from kestrel.runtime.tokens import Token


class RuntimeDecodeResult(NamedTuple):
    logits: Tensor
    hidden: Tensor


@dataclass
class SequenceState:
    """Metadata for an active text request."""

    batch_idx: int
    length: int
    max_length: int
    prompt_length: int | None = None
    # DEPRECATED: Use image_regions instead. Kept for backward compatibility with
    # scheduler code. Will be removed once scheduler is migrated to image_regions.
    image_length: int = 0
    last_hidden: Tensor | None = None
    lora_slot: int = 0  # 0 = no LoRA, >0 = slot in TextLoRAWorkspace

    # Prefix cache fields
    cache_tokens: list[CacheToken] | None = None
    cache_lock_node: TreeNode | None = None
    cache_owned_page_count: int = 0  # Pages belonging to cache (not freed on release)
    reused_page_count: int = 0  # Pages reused from cache hit (for metrics)
    # List of (start, end) KV positions for bidirectional attention (image regions).
    # For single-image: [(1, 1+image_kv_length)]. Empty for text-only.
    # This will replace image_length once multi-image is supported.
    image_regions: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.prompt_length is None:
            self.prompt_length = self.length
        # Validate consistency between image_length and image_regions
        if self.image_regions:
            computed_length = sum(end - start for start, end in self.image_regions)
            assert self.image_length == computed_length, (
                f"image_length ({self.image_length}) inconsistent with "
                f"image_regions ({self.image_regions}, computed={computed_length})"
            )

    def advance(self, tokens: int = 1) -> None:
        self.length += tokens

    @property
    def output_length(self) -> int:
        return self.length - (self.prompt_length or 0)


@dataclass
class _CacheLookupResult:
    """Result of prefix cache lookup during prefill preparation."""

    match: MatchResult | None
    skip_positions: int
    temp_lock_node: TreeNode | None
    can_reuse: bool
    namespace: CacheNamespace | None


@dataclass(frozen=True, slots=True)
class PrefillClassification:
    """Read-only classification of how a request would prefill if launched now."""

    prompt_length: int
    skip_positions: int
    can_reuse: bool
    use_prefix_attn: bool

    @property
    def query_length(self) -> int:
        return self.prompt_length - self.skip_positions if self.can_reuse else self.prompt_length


@dataclass(slots=True)
class PreparedSequence:
    """Prepared prefill state for a sequence before GPU prefill is launched.

    This bundles the CPU-side work and KV/prefix-cache bookkeeping needed to
    safely launch the GPU prefill later without stalling the GPU on admission
    work (e.g., cache lookup, page allocation, and reservation).

    Lifecycle:
    - created by the runtime's ``prepare_sequence(...)``
    - consumed by the runtime's ``launch_prepared_batch(...)`` (GPU enqueue)
    - finalized by the runtime's ``finalize_prepared_sequence_after_prefill(...)``
    - aborted by the runtime's ``abort_prepared_sequence(...)`` on error/pause

    Note: PreparedSequence is decoupled from the prefill slot. The slot is
    acquired at launch time, allowing preparation to proceed even when all
    prefill slots are occupied by in-flight prefills.
    """

    state: "SequenceState"
    tokens_list: list["Token"]
    cache_tokens: list[CacheToken]
    cache_result: _CacheLookupResult
    adapter_id: str | None
    image_hash: bytes | None
