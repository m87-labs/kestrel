"""Protocol describing the runtime contract used by engine + scheduler.

The :class:`~kestrel.engine.InferenceEngine` and
:class:`~kestrel.scheduler.scheduler.GenerationScheduler` are generic
over runtime implementations: they manage admission, queues, prefill
batching, and decode pacing without knowing which model is actually
running. Concrete runtimes (today: ``MoondreamRuntime``) implement the
methods declared here.

Some attributes are typed as ``Any`` because they expose model-specific
state the engine + scheduler currently reach into directly (config,
region, spatial decoding tables, prefix cache, slot containers).
Tightening those — or refactoring callers to stop reaching into them —
is a follow-up; declaring them here keeps the protocol an honest
record of the surface a runtime must satisfy today.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from kestrel.runtime.state import (
    PrefillClassification,
    PreparedSequence,
    SequenceState,
)
from kestrel.runtime.tokens import Token

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import Tensor


class Runtime(Protocol):
    """Surface that the engine and scheduler call on a runtime."""

    # Identity. Stable across the runtime's lifetime; used to scope
    # prefix-cache entries and (later) to route requests in a
    # multi-runtime engine.
    model_name: str

    # Capacity / shape
    max_batch_size: int
    max_batch_slots: int
    max_seq_length: int
    image_prefix_length: int

    # Device + streams
    device: torch.device
    primary_stream: Any
    copy_stream: Any

    # Slot containers + sequence registry
    prefill_slots: Sequence[Any]
    decode_slots: Sequence[Any]
    active_sequences: Mapping[int, SequenceState]

    # Engine + scheduler infrastructure
    prefix_cache: Any  # ``RadixPrefixCache | None`` on MoondreamRuntime
    graph_capture_lock: Any  # context manager serializing CUDA-graph capture

    # Model-specific state callers reach into today.
    # Narrowing these is a follow-up.
    config: Any
    region: Any
    spatial_tables: Any
    page_table: Any

    # Capacity queries
    def can_reserve(self, total_length: int) -> bool: ...

    def prefill_budget(self) -> tuple[int, int]: ...

    # Image preprocessing. The engine receives raw images (np.ndarray
    # or bytes) and needs them converted to whatever opaque payload the
    # runtime threads through ``GenerationRequest.image_crops`` and
    # back into ``launch_prepared_batch``. Different runtimes need
    # different preprocessing (Moondream's overlap-crop pipeline vs.
    # Gemma's resize+normalize), so the dispatch lives here. Async so
    # the engine can hide preprocessing latency behind admission /
    # other work.
    def preprocess_image_async(
        self, image: np.ndarray | bytes
    ) -> Future[Any]: ...

    # Called once when the engine is shutting down so the runtime can
    # tear down its preprocessing thread pool (if any).
    def shutdown_image_preprocessor(self) -> None: ...

    # Hash an image into the bytes used as the prefix-cache key. Lets
    # runtimes pick a hashing scheme that matches how their preprocessor
    # canonicalises the input.
    def image_hash(self, image: np.ndarray | bytes) -> bytes: ...

    # Slot lifecycle
    def acquire_prefill_slot(self, slot_id: int | None = ...) -> Any: ...

    def release_prefill_slot(self, slot: Any) -> None: ...

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int: ...

    def release_adapter_slot(self, slot: int) -> None: ...

    # Prefill / decode
    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool,
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> PrefillClassification: ...

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: np.ndarray | None = ...,
        image_crops: Any | None = ...,
        max_new_tokens: int | None = ...,
        lora_slot: int = ...,
        image_hash: bytes | None = ...,
        adapter_id: str | None = ...,
    ) -> PreparedSequence: ...

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: Any,
        *,
        images: Sequence[np.ndarray | None] | None = ...,
        image_crops_list: Sequence[Any] | None = ...,
    ) -> Tensor: ...

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None: ...

    def abort_prepared_sequence(self, prepared: PreparedSequence) -> None: ...

    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None: ...

    def release_sequence(self, state: SequenceState) -> None: ...

    def decode_with_slot(self, slot: Any, batch_size: int) -> None: ...
