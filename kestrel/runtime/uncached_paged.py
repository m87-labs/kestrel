"""Admission and sequence lifecycle for paged runtimes without prefix caching."""

from __future__ import annotations

from typing import Any, Sequence

from kestrel.runtime.state import (
    PreparedSequence,
    PrefillClassification,
    SequenceState,
    _CacheLookupResult,
)
from kestrel.runtime.tokens import Token


class UncachedPagedRuntime:
    """Own page reservation and sequence state for uncached paged inference."""

    prefix_cache = None
    decode_path = "auto"
    generated_decode = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def compute_stream(self) -> Any:
        return self._compute_stream

    @property
    def copy_stream(self) -> Any:
        return self._copy_stream

    @property
    def kv_pool(self) -> Any:
        return self._kv_pool

    def skills(self) -> Any:
        from kestrel.models.registry import get_spec

        return get_spec(self.model_name).skills()

    def tasks(self) -> tuple[str, ...]:
        return self.skills().names()

    def preprocess_image_async(self, image: Any) -> Any:
        return self._image_preprocessor.submit(image)

    def shutdown(self) -> None:
        self._image_preprocessor.shutdown()

    def can_reserve(self, total_length: int) -> bool:
        return (
            total_length <= self.max_seq_length
            and self.page_table.can_reserve_with_eviction(total_length)
            and self._available_batch_slots() > 0
        )

    def prefill_budget(self) -> tuple[int, int]:
        return (self.page_table.pages_available, self._available_batch_slots())

    def _available_batch_slots(self) -> int:
        active_headroom = max(0, self.max_batch_size - len(self.active_sequences))
        return min(active_headroom, len(self.page_table.free_batch_idx))

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int:
        raise NotImplementedError("LoRA adapters are not supported")

    def release_adapter_slot(self, slot: int) -> None:
        raise NotImplementedError("LoRA adapters are not supported")

    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool = False,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> PrefillClassification:
        return PrefillClassification(
            prompt_length=len(prompt_tokens),
            skip_positions=0,
            can_reuse=False,
            use_prefix_attn=False,
        )

    def _prepare_uncached_sequence(
        self,
        *,
        tokens: list[Token],
        target_length: int,
        image_length: int,
        lora_slot: int,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> PreparedSequence:
        prompt_length = len(tokens)
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds "
                f"max_seq_length={self.max_seq_length}"
            )
        if self._available_batch_slots() <= 0:
            raise RuntimeError("Cannot reserve batch slot")
        batch_idx = self.page_table.allocate()
        try:
            self.page_table.reserve(batch_idx, target_length)
        except Exception:
            self.page_table.erase(batch_idx, 0)
            raise
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_length,
            max_length=target_length,
            prompt_length=prompt_length,
            image_length=image_length,
            lora_slot=lora_slot,
        )
        return PreparedSequence(
            state=state,
            tokens_list=tokens,
            cache_tokens=[],
            cache_result=_CacheLookupResult(
                match=None,
                skip_positions=0,
                temp_lock_node=None,
                can_reuse=False,
                namespace=None,
            ),
            adapter_id=adapter_id,
            image_hash=image_hash,
            use_prefix_attn=False,
        )

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None:
        self.active_sequences[prepared.state.batch_idx] = prepared.state

    def abort_prepared_sequence(self, prepared: PreparedSequence) -> None:
        self._release_batch_idx(prepared.state.batch_idx)

    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None:
        del state, generated_tokens, adapter_id, image_hash

    def release_sequence(self, state: SequenceState) -> None:
        self._release_batch_idx(state.batch_idx)

    def _release_runtime_state(self, batch_idx: int) -> None:
        del batch_idx

    def _release_batch_idx(self, batch_idx: int) -> None:
        self.active_sequences.pop(batch_idx, None)
        self._release_runtime_state(batch_idx)
        if batch_idx not in self.page_table.free_batch_idx:
            self.page_table.erase(batch_idx, 0)


__all__ = ["UncachedPagedRuntime"]
