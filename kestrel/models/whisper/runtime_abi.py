"""Resident buffer contracts shared by Whisper execution components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

from .config import WhisperTurboConfig


@dataclass(frozen=True, slots=True)
class WhisperCrossArenas:
    """Pointer-stable global cross-attention K/V storage."""

    keys: Tensor
    values: Tensor

    @classmethod
    def allocate(
        cls,
        config: WhisperTurboConfig,
        state_rows: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WhisperCrossArenas":
        shape = (
            config.decoder_layers,
            state_rows,
            config.max_source_positions,
            config.decoder_attention_heads,
            config.decoder_head_dim,
        )
        return cls(
            keys=torch.empty(shape, device=device, dtype=dtype),
            values=torch.empty(shape, device=device, dtype=dtype),
        )

    def validate(
        self,
        config: WhisperTurboConfig,
        state_rows: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        expected = (
            config.decoder_layers,
            state_rows,
            config.max_source_positions,
            config.decoder_attention_heads,
            config.decoder_head_dim,
        )
        for name, tensor in (("keys", self.keys), ("values", self.values)):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Whisper cross {name} must have shape {expected}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.device != device or tensor.dtype is not dtype:
                raise ValueError(f"Whisper cross {name} must be {dtype} on {device}")
            if not tensor.is_contiguous():
                raise ValueError(f"Whisper cross {name} must be contiguous")
            if tensor.requires_grad:
                raise ValueError(f"Whisper cross {name} must not require gradients")


@dataclass(frozen=True, slots=True)
class WhisperSelfKVArenas:
    """Engine-pool-owned paged self-K/V, with the size-one page axis removed."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]

    def validate(
        self,
        config: WhisperTurboConfig,
        *,
        n_pages: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if len(self.keys) != config.decoder_layers or len(self.values) != len(
            self.keys
        ):
            raise ValueError(
                "Whisper self K/V must own one key/value arena per decoder layer"
            )
        expected = (
            n_pages,
            config.decoder_attention_heads,
            config.decoder_head_dim,
        )
        for layer, (key, value) in enumerate(zip(self.keys, self.values)):
            for kind, tensor in (("key", key), ("value", value)):
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Whisper self {kind} layer {layer} must have shape "
                        f"{expected}, got {tuple(tensor.shape)}"
                    )
                if tensor.device != device or tensor.dtype is not dtype:
                    raise ValueError(
                        f"Whisper self {kind} layer {layer} must be {dtype} on {device}"
                    )
                if not tensor.is_contiguous():
                    raise ValueError(
                        f"Whisper self {kind} layer {layer} must be contiguous"
                    )


@dataclass(frozen=True, slots=True)
class WhisperPrefillBuffers:
    """Resident inputs/outputs bound by one custom-prefill slot."""

    slot_id: int
    input_features: Tensor
    control_token_ids: Tensor
    prefix_lengths: Tensor
    batch_idx: Tensor
    slot_mapping: Tensor
    logits_out: Tensor


@dataclass(frozen=True, slots=True)
class WhisperDecodeBuffers:
    """Resident inputs/outputs bound by one generated-decode slot."""

    slot_id: int
    token_ids: Tensor
    input_pos: Tensor
    batch_idx: Tensor
    logits_out: Tensor


@dataclass(frozen=True, slots=True)
class WhisperExecutionBindings:
    """Resident buffers shared by prefill and injected test sessions."""

    cross_kv: WhisperCrossArenas
    self_kv: WhisperSelfKVArenas
    prefill_buffers: tuple[WhisperPrefillBuffers, ...]
    decode_buffers: tuple[WhisperDecodeBuffers, ...]
    max_batch_size: int
    compute_stream: Any


class WhisperPrefillSession(Protocol):
    """Complete custom encoder/control-prefix session.

    ``launch`` must enqueue the audio stem, all encoder blocks, global cross-K/V
    row writes, decoder control-prefix self-K/V writes, and first-token logits
    before returning.  The public scheduler may enqueue optimistic decode before
    CPU-side prefill commit, so installing conditioning in a later finalizer is
    invalid.
    """

    @property
    def artifact_receipts(self) -> tuple[dict[str, object], ...]: ...

    def warmup(self) -> None: ...

    def launch(self, slot_id: int, batch_size: int) -> None: ...

    def shutdown(self) -> None: ...


def validate_resident_buffers(
    *,
    config: WhisperTurboConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    prefill_buffers: Sequence[WhisperPrefillBuffers],
    decode_buffers: Sequence[WhisperDecodeBuffers],
) -> None:
    """Validate the fixed pointer-bearing part of the backend ABI."""

    if len(prefill_buffers) != 2 or len(decode_buffers) != 2:
        raise ValueError("Whisper runtime requires two prefill and two decode slots")
    for expected_slot_id, slot in enumerate(prefill_buffers):
        if slot.slot_id != expected_slot_id:
            raise ValueError("Whisper prefill slot IDs must be dense from zero")
        expected = {
            "input_features": (
                slot.input_features,
                (max_batch_size, config.num_mel_bins, config.max_source_positions * 2),
                dtype,
            ),
            "control_token_ids": (
                slot.control_token_ids,
                (max_batch_size, 4),
                torch.int64,
            ),
            "prefix_lengths": (
                slot.prefix_lengths,
                (max_batch_size,),
                torch.int32,
            ),
            "batch_idx": (
                slot.batch_idx,
                (max_batch_size,),
                torch.int64,
            ),
            "slot_mapping": (
                slot.slot_mapping,
                (max_batch_size, 4),
                torch.int64,
            ),
            "logits_out": (
                slot.logits_out,
                (max_batch_size, config.vocab_size),
                dtype,
            ),
        }
        for name, (tensor, shape, expected_dtype) in expected.items():
            if (
                tuple(tensor.shape) != shape
                or tensor.device != device
                or tensor.dtype is not expected_dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"Whisper prefill {name} must be contiguous "
                    f"{expected_dtype} {shape} on {device}"
                )

    for expected_slot_id, slot in enumerate(decode_buffers):
        if slot.slot_id != expected_slot_id:
            raise ValueError("Whisper decode slot IDs must be dense from zero")
        rows = int(slot.token_ids.shape[0])
        expected = {
            "token_ids": (slot.token_ids, (rows,), torch.int64),
            "input_pos": (slot.input_pos, (rows,), torch.int32),
            "batch_idx": (slot.batch_idx, (rows,), torch.int64),
            "logits_out": (
                slot.logits_out,
                (rows, config.vocab_size),
                dtype,
            ),
        }
        if rows < max_batch_size:
            raise ValueError("Whisper decode slots have insufficient batch capacity")
        for name, (tensor, shape, expected_dtype) in expected.items():
            if (
                tuple(tensor.shape) != shape
                or tensor.device != device
                or tensor.dtype is not expected_dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"Whisper decode {name} must be contiguous "
                    f"{expected_dtype} {shape} on {device}"
                )


__all__ = [
    "WhisperExecutionBindings",
    "WhisperCrossArenas",
    "WhisperDecodeBuffers",
    "WhisperPrefillBuffers",
    "WhisperPrefillSession",
    "WhisperSelfKVArenas",
    "validate_resident_buffers",
]
