"""Backend contract for Whisper prefill and generated decode.

The public model owns preprocessing, weights, scheduling, and the complete
Whisper runtime. Optimized distributions supply the two execution sessions:

* prefill composes public Kestrel kernels for the complete encoder and decoder
  control-prefix launch;
* the generated megakernel session owns only one-token decoder launches.

Neither protocol permits an eager serving fallback. Keeping this small boundary
lets a packaged backend provide architecture-specific artifacts without moving
model behavior or customer-facing code out of Kestrel.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import torch
from torch import Tensor

from .config import WhisperTurboConfig
from .weights import WhisperModelWeights


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
class WhisperBackendBindings:
    """Immutable construction inputs shared by the two backend sessions."""

    config: WhisperTurboConfig
    weights: WhisperModelWeights
    cross_kv: WhisperCrossArenas
    self_kv: WhisperSelfKVArenas
    page_table: Tensor
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


class WhisperDecodeSession(Protocol):
    """Generated single-token decoder session; logits remain unsampled."""

    @property
    def artifact_identities(self) -> tuple[dict[str, object], ...]: ...

    def warmup(self) -> None: ...

    def launch(self, slot_id: int, batch_size: int) -> None: ...

    def shutdown(self) -> None: ...


class WhisperBackendFactory(Protocol):
    """Prepack weights and bind stable buffers without retaining eager code."""

    def create_prefill(
        self, bindings: WhisperBackendBindings
    ) -> WhisperPrefillSession: ...

    def create_decode(
        self, bindings: WhisperBackendBindings
    ) -> WhisperDecodeSession: ...

    def native_provenance(
        self,
        bindings: WhisperBackendBindings,
        decode_session: WhisperDecodeSession,
    ) -> dict[str, Any]:
        """Return the JSON-safe native artifact contract for these bindings."""
        ...


WhisperBackendProvider = Callable[[], WhisperBackendFactory]

_BACKEND_LOCK = threading.Lock()
_BACKEND_PROVIDER: WhisperBackendProvider | None = None


def register_backend(provider: WhisperBackendProvider) -> None:
    """Register the process-wide optimized Whisper backend provider.

    Kestrel owns the model implementation, while a separately packaged backend
    owns generated device programs. Registration is explicit so importing the
    public model never scans plugins or silently selects an implementation.
    """

    if not callable(provider):
        raise TypeError("Whisper backend provider must be callable")
    global _BACKEND_PROVIDER
    with _BACKEND_LOCK:
        if _BACKEND_PROVIDER is not None and _BACKEND_PROVIDER is not provider:
            raise RuntimeError("A different Whisper backend is already registered")
        _BACKEND_PROVIDER = provider


def create_backend() -> WhisperBackendFactory:
    """Create the registered backend or fail before allocating model state."""

    with _BACKEND_LOCK:
        provider = _BACKEND_PROVIDER
    if provider is None:
        raise RuntimeError(
            "No optimized Whisper backend is registered. Install and import the "
            "backend package supplied with this Kestrel distribution before "
            "creating openai/whisper-large-v3-turbo."
        )
    backend = provider()
    required = ("create_prefill", "create_decode", "native_provenance")
    if any(not callable(getattr(backend, name, None)) for name in required):
        raise TypeError("Whisper backend does not implement the required contract")
    return backend


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
    "WhisperBackendBindings",
    "WhisperBackendFactory",
    "WhisperBackendProvider",
    "WhisperCrossArenas",
    "WhisperDecodeBuffers",
    "WhisperDecodeSession",
    "WhisperPrefillBuffers",
    "WhisperPrefillSession",
    "WhisperSelfKVArenas",
    "create_backend",
    "register_backend",
    "validate_resident_buffers",
]
