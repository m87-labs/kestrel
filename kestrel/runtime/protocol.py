"""Protocol describing the scheduler↔runtime contract.

The :class:`~kestrel.scheduler.scheduler.GenerationScheduler` is generic
over runtime implementations: it manages admission, queues, prefill
batching, and decode pacing without knowing which model is actually
running. Concrete runtimes (today: ``MoondreamRuntime``) implement the
methods declared here.

Model-specific parameter types — image crops, adapter objects, prefill
and decode slots — are typed as ``Any`` so the protocol stays
model-agnostic. Concrete runtimes narrow them.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
from torch import Tensor

from kestrel.runtime.state import (
    PrefillClassification,
    PreparedSequence,
    SequenceState,
)
from kestrel.runtime.tokens import Token


class Runtime(Protocol):
    """Surface that :class:`GenerationScheduler` calls on a runtime."""

    max_batch_size: int
    max_seq_length: int
    image_prefix_length: int

    def can_reserve(self, total_length: int) -> bool: ...

    def prefill_budget(self) -> tuple[int, int]: ...

    def acquire_prefill_slot(self, slot_id: int | None = ...) -> Any: ...

    def release_prefill_slot(self, slot: Any) -> None: ...

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int: ...

    def release_adapter_slot(self, slot: int) -> None: ...

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

    def release_sequence(self, state: SequenceState) -> None: ...

    def decode_with_slot(self, slot: Any, batch_size: int) -> None: ...
