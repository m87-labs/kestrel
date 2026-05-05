"""In-memory ``Runtime`` implementation for scheduler tests.

The :class:`~kestrel.scheduler.scheduler.GenerationScheduler` is typed
against the :class:`~kestrel.runtime.Runtime` protocol. ``FakeRuntime``
gives the scheduler test suite a second consumer of that protocol so
scheduler logic can be driven without instantiating ``MoondreamRuntime``
(which requires real model weights, GPU resources, and vision/text
encoders).

Behaviour:
- every Runtime method is implemented as a lightweight bookkeeping
  call. Calls are recorded into public lists/dicts so tests can assert
  ordering and arguments.
- attributes default to small, plausible values (``max_batch_size=1``,
  ``page_size=1``, etc.) but are mutable so individual tests can
  override.
- ``prepare_sequence`` and ``launch_prepared_batch`` accept a
  pre-built return value or an exception so failure-path tests can
  pin behaviour without constructing a real ``PreparedSequence``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping, Sequence

import torch

from kestrel.device import NoopEvent
from kestrel.runtime import (
    PrefillClassification,
    PreparedSequence,
    SequenceState,
)
from kestrel.runtime.tokens import Token


class _FakePageTable:
    """Stand-in for :class:`kestrel.kv_cache.PageTable`.

    Exposes the attributes the scheduler reads (``page_size``) plus a
    ``commit_block_table`` no-op so decode-step tests can run.
    """

    def __init__(self, *, page_size: int = 1) -> None:
        self.page_size = page_size
        self.commit_block_table_calls: list[list[int] | None] = []

    def commit_block_table(self, batch_indices: list[int] | None = None) -> None:
        self.commit_block_table_calls.append(batch_indices)


class _FakePrefillSlot:
    """Minimal stand-in for a prefill slot.

    Carries the attributes the scheduler reaches into during the
    success path (``step_done_event`` / ``commit_done_event`` to
    record + synchronize, ``batch_idx`` to slice). Moondream-specific
    state like ``scratch`` is left as ``None``.
    """

    def __init__(self, *, slot_id: int, max_batch_size: int) -> None:
        self.slot_id = slot_id
        self.batch_idx = torch.zeros(max_batch_size, dtype=torch.int32)
        self.step_done_event = NoopEvent()
        self.commit_done_event = NoopEvent()
        self.scratch = None


class _FakeDecodeSlot:
    """Minimal stand-in for a decode slot.

    Carries the attributes the scheduler reaches into on the decode
    success path (``step_done_event`` / ``commit_done_event``). Tests
    that drive richer DecodeSlot state (``meta``, ``sampled_ids``,
    etc.) should extend this stub.
    """

    def __init__(self, *, slot_id: int) -> None:
        self.slot_id = slot_id
        self.step_done_event = NoopEvent()
        self.commit_done_event = NoopEvent()


class FakeRuntime:
    """Minimal :class:`Runtime` impl for scheduler tests."""

    def __init__(
        self,
        *,
        model_name: str = "fake",
        max_batch_size: int = 1,
        max_batch_slots: int = 2,
        max_seq_length: int = 1024,
        image_prefix_length: int = 0,
        device: torch.device | str = "cpu",
        page_size: int = 1,
        n_prefill_slots: int | None = None,
        n_decode_slots: int | None = None,
        prepare_exc: Exception | None = None,
        prepare_result: PreparedSequence | None = None,
        launch_logits: torch.Tensor | None = None,
    ) -> None:
        # Identity
        self.model_name = model_name

        # Capacity / shape
        self.max_batch_size = max_batch_size
        self.max_batch_slots = max_batch_slots
        self.max_seq_length = max_seq_length
        self.image_prefix_length = image_prefix_length

        # Device + streams
        self.device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.primary_stream: Any = None
        self.copy_stream: Any = None

        # Slot containers + sequence registry. The scheduler treats these
        # as fixed-capacity arrays — sizing its staging pool from
        # ``len(prefill_slots)`` and indexing ``decode_slots[slot_id]`` —
        # so they must be pre-populated for the scheduler to construct
        # against this fake at all.
        prefill_capacity = (
            n_prefill_slots if n_prefill_slots is not None else max_batch_slots
        )
        decode_capacity = (
            n_decode_slots if n_decode_slots is not None else max_batch_slots
        )
        self.prefill_slots: list[Any] = [
            _FakePrefillSlot(slot_id=i, max_batch_size=max_batch_size)
            for i in range(prefill_capacity)
        ]
        self.decode_slots: list[Any] = [
            _FakeDecodeSlot(slot_id=i) for i in range(decode_capacity)
        ]
        self.active_sequences: dict[int, SequenceState] = {}

        # Model-specific surfaces — left as ``None``/empty for fake purposes;
        # tests that need richer values assign attributes directly.
        self.config: Any = None
        self.region: Any = None
        self.spatial_tables: Any = None
        self.page_table = _FakePageTable(page_size=page_size)

        # Engine + scheduler infrastructure
        self.prefix_cache: Any = None
        self.graph_capture_lock = contextlib.nullcontext()

        # Knobs
        self._prepare_exc = prepare_exc
        self._prepare_result = prepare_result
        self._launch_logits = launch_logits

        # Call log
        self.acquired_prefill_slots: list[int | None] = []
        self.released_prefill_slots: list[Any] = []
        self.acquired_adapter_slots: list[tuple[str, Any]] = []
        self.released_adapter_slots: list[int] = []
        self.classify_calls: list[Sequence[Token]] = []
        self.prepare_calls: list[dict[str, Any]] = []
        self.launch_calls: list[Sequence[PreparedSequence]] = []
        self.finalized_prepared: list[PreparedSequence] = []
        self.aborted_prepared: list[PreparedSequence] = []
        self.released_sequences: list[SequenceState] = []
        self.decode_calls: list[tuple[Any, int]] = []

    # Capacity queries -------------------------------------------------
    def can_reserve(self, total_length: int) -> bool:
        return total_length <= self.max_seq_length

    def prefill_budget(self) -> tuple[int, int]:
        return (self.max_seq_length, self.max_batch_slots)

    # Slot lifecycle ---------------------------------------------------
    def acquire_prefill_slot(self, slot_id: int | None = None) -> Any:
        self.acquired_prefill_slots.append(slot_id)
        index = 0 if slot_id is None else slot_id
        return self.prefill_slots[index]

    def release_prefill_slot(self, slot: Any) -> None:
        self.released_prefill_slots.append(slot)

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int:
        self.acquired_adapter_slots.append((adapter_id, adapter))
        return len(self.acquired_adapter_slots)

    def release_adapter_slot(self, slot: int) -> None:
        self.released_adapter_slots.append(slot)

    # Prefill / decode -------------------------------------------------
    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool,
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> PrefillClassification:
        self.classify_calls.append(prompt_tokens)
        prompt_length = len(prompt_tokens) + (
            self.image_prefix_length if has_image else 0
        )
        return PrefillClassification(
            prompt_length=prompt_length,
            skip_positions=0,
            can_reuse=False,
            use_prefix_attn=False,
        )

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: Any = None,
        image_crops: Any = None,
        max_new_tokens: int | None = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> PreparedSequence:
        self.prepare_calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "image": image,
                "image_crops": image_crops,
                "max_new_tokens": max_new_tokens,
                "lora_slot": lora_slot,
                "image_hash": image_hash,
                "adapter_id": adapter_id,
            }
        )
        if self._prepare_exc is not None:
            raise self._prepare_exc
        if self._prepare_result is not None:
            return self._prepare_result
        raise AssertionError(
            "prepare_sequence called but FakeRuntime has no prepare_result configured"
        )

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: Any,
        *,
        images: Sequence[Any] | None = None,
        image_crops_list: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        self.launch_calls.append(prepared_sequences)
        if self._launch_logits is not None:
            return self._launch_logits
        return torch.zeros(len(prepared_sequences), 1)

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None:
        self.finalized_prepared.append(prepared)

    def abort_prepared_sequence(self, prepared: PreparedSequence) -> None:
        self.aborted_prepared.append(prepared)

    def release_sequence(self, state: SequenceState) -> None:
        self.released_sequences.append(state)
        self.active_sequences.pop(state.batch_idx, None)

    def decode_with_slot(self, slot: Any, batch_size: int) -> None:
        self.decode_calls.append((slot, batch_size))
