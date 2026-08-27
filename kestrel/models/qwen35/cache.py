"""Hybrid attention and recurrent state for Qwen 3.5/3.6."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from kestrel.kv_cache import PagedKVCache, PagedKVLayerSpec

from .gdn_state import LinearAttentionState

if TYPE_CHECKING:
    from kestrel.runtime.carried_state import StatePhysicalForm


def qwen_paged_kv_specs(
    config: Any,
) -> tuple[PagedKVLayerSpec | None, ...]:
    """Describe which hybrid layers own ordinary paged K/V storage."""

    head_dim = int(config.head_dim)
    specs: list[PagedKVLayerSpec | None] = []
    for layer_type in config.layer_types:
        if layer_type == "linear_attention":
            specs.append(None)
        else:
            specs.append(
                PagedKVLayerSpec(
                    n_heads=int(config.num_key_value_heads),
                    head_dim=head_dim,
                )
            )
    return tuple(specs)


class Qwen35InferenceCache:
    """Per-forward GDN state over runtime-owned Kestrel paged K/V."""

    def __init__(
        self,
        *,
        config: Any,
        paged_kv: Sequence[PagedKVCache | None],
        replay_capacity: int,
        prepare_gdn_replay_state: bool = True,
    ) -> None:
        layer_types = tuple(config.layer_types)
        if len(paged_kv) != len(layer_types):
            raise ValueError("paged_kv layout must match layer_types")
        layers: list[Any] = []
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                layers.append(
                    LinearAttentionState(replay_capacity=replay_capacity)
                )
                continue
            layer = paged_kv[layer_idx]
            if layer is None:
                raise ValueError(
                    f"full-attention layer {layer_idx} has no paged K/V producer"
                )
            layers.append(layer)
        self.layers = tuple(layers)
        self.seq_length = 0
        self.prepare_gdn_replay_state = bool(prepare_gdn_replay_state)

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if layer_idx is None:
            return (
                any(
                    isinstance(layer, LinearAttentionState)
                    and bool(layer.has_previous_state)
                    for layer in self.layers
                )
                or self.seq_length > 0
            )
        layer = self.layers[layer_idx]
        if isinstance(layer, LinearAttentionState):
            return bool(layer.has_previous_state)
        return self.seq_length > 0

    def get_seq_length(self) -> int:
        return int(self.seq_length)

    def advance_to(self, seq_length: int) -> None:
        self.seq_length = max(self.seq_length, int(seq_length))


class Qwen35LinearStatePool:
    """Runtime-owned GDN state indexed by Kestrel batch slot."""

    _VALUE_MAJOR_RECURRENT_AXES = ("state_row", "value_head", "value", "key")
    _GENERATED_RECURRENT_STORAGE_DTYPE = "bf16"

    def __init__(
        self,
        *,
        config: Any,
        max_batch_slots: int,
        device: torch.device,
        replay_capacity: int,
    ) -> None:
        self.max_batch_slots = int(max_batch_slots)
        self.device = device
        self.replay_capacity = int(replay_capacity)
        self._conv_shape = (
            self.max_batch_slots,
            2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            + int(config.linear_num_value_heads) * int(config.linear_value_head_dim),
            int(config.linear_conv_kernel_dim),
        )
        self._key_major_recurrent_shape = (
            self.max_batch_slots,
            int(config.linear_num_value_heads),
            int(config.linear_key_head_dim),
            int(config.linear_value_head_dim),
        )
        self._value_major_recurrent_shape = (
            self.max_batch_slots,
            int(config.linear_num_value_heads),
            int(config.linear_value_head_dim),
            int(config.linear_key_head_dim),
        )
        self._recurrent_mode: str | None = None
        self.layers: list[LinearAttentionState | None] = [
            (
                LinearAttentionState(replay_capacity=self.replay_capacity)
                if layer_type == "linear_attention"
                else None
            )
            for layer_type in config.layer_types
        ]

    def initialize_from_config(self, config: Any, *, dtype: torch.dtype) -> None:
        """Allocate shared convolution state before decode representation selection."""

        expected_conv_shape = (
            self.max_batch_slots,
            2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            + int(config.linear_num_value_heads) * int(config.linear_value_head_dim),
            int(config.linear_conv_kernel_dim),
        )
        if expected_conv_shape != self._conv_shape:
            raise RuntimeError("Qwen GDN convolution geometry changed")
        for storage in self.layers:
            if storage is None:
                continue
            tensor = storage.conv_states
            if tensor is None:
                storage.conv_states = torch.zeros(
                    self._conv_shape, dtype=dtype, device=self.device)
            elif tuple(tensor.shape) != self._conv_shape or tensor.dtype != dtype:
                raise RuntimeError("Qwen GDN convolution state contract changed")

    def initialize_native_recurrent(self) -> None:
        """Select the FP32 replay representation for the native decode path."""

        if self._recurrent_mode not in (None, "native"):
            raise RuntimeError(
                "Qwen generated recurrent state cannot switch to native replay")
        self._recurrent_mode = "native"
        for storage in self.layers:
            if storage is None:
                continue
            tensor = storage.recurrent_states
            if tensor is None:
                storage.recurrent_states = torch.zeros(
                    self._key_major_recurrent_shape,
                    dtype=torch.float32,
                    device=self.device,
                )
            elif (
                tuple(tensor.shape) != self._key_major_recurrent_shape
                or tensor.dtype != torch.float32
            ):
                raise RuntimeError("Qwen native recurrent state contract changed")
            storage._ensure_replay_state(storage.recurrent_states)

    def _initialize_generated_recurrent(self) -> None:
        if self._recurrent_mode not in (None, "generated"):
            raise RuntimeError(
                "Qwen native replay state cannot switch to generated decode")
        self._recurrent_mode = "generated"
        for storage in self.layers:
            if storage is None:
                continue
            tensor = storage.recurrent_states
            if tensor is None:
                storage.recurrent_states = torch.zeros(
                    self._value_major_recurrent_shape,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            elif (
                tuple(tensor.shape) != self._value_major_recurrent_shape
                or tensor.dtype != torch.bfloat16
            ):
                raise RuntimeError("Qwen generated recurrent state contract changed")

    def zero_all(self) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            storage.clear()

    def capture_batch_from_cache(
        self,
        batch_indices: Sequence[int],
        cache: Qwen35InferenceCache,
        *,
        batch_size: int,
        copy_replay_payload: bool = True,
    ) -> torch.Tensor:
        rows = tuple(batch_indices)
        if any(type(row) is not int for row in rows):
            raise ValueError("Qwen GDN state rows must be host integers")
        if len(rows) != batch_size:
            raise ValueError("Qwen GDN state rows must match packed batch size")
        if len(set(rows)) != len(rows):
            raise ValueError("Qwen GDN state rows must be unique")
        if any(row < 0 or row >= self.max_batch_slots for row in rows):
            raise ValueError("Qwen GDN state row is outside the runtime pool")
        indices = torch.tensor(rows, device=self.device, dtype=torch.long)
        if self._recurrent_mode not in ("generated", "native"):
            raise RuntimeError("Qwen recurrent decode representation is not selected")
        captures: list[tuple[LinearAttentionState, LinearAttentionState]] = []
        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            src_layer = cache.layers[layer_idx]
            if not isinstance(src_layer, LinearAttentionState):
                raise ValueError("Cannot capture mismatched Qwen linear state")
            if not src_layer.has_previous_state:
                raise RuntimeError("Cannot capture uninitialized Qwen GDN state")
            self._validate_conv_rows(storage, src_layer, batch_size=batch_size)
            if self._recurrent_mode == "generated":
                self._validate_generated_recurrent_rows(
                    storage, src_layer, batch_size=batch_size)
            else:
                self._validate_prefill_recurrent(src_layer, batch_size=batch_size)
            captures.append((storage, src_layer))

        # Validate every layer before the first persistent state write so a
        # malformed later layer cannot leave a partially committed batch.
        for storage, src_layer in captures:
            if self._recurrent_mode == "generated":
                self._capture_conv_rows(storage, src_layer, indices)
                self._capture_generated_recurrent_rows(
                    storage, src_layer, indices)
            else:
                storage.copy_rows_from(
                    src_layer, indices, copy_replay_payload=copy_replay_payload)
        return indices

    def bind_to_cache(self, cache: Qwen35InferenceCache) -> None:
        """Bind cache linear layers directly to runtime-owned persistent state."""

        if self._recurrent_mode != "native":
            raise RuntimeError("Qwen generated recurrent state cannot bind native replay")

        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            layer = cache.layers[layer_idx]
            if not isinstance(layer, LinearAttentionState):
                raise ValueError("Cannot bind mismatched Qwen linear state")
            if storage.conv_states is None or storage.recurrent_states is None:
                raise RuntimeError("Qwen GDN state pool is not initialized")
            layer.conv_states = storage.conv_states
            layer.recurrent_states = storage.recurrent_states
            layer.replay_checkpoint_states = storage.replay_checkpoint_states
            layer.replay_k = storage.replay_k
            layer.replay_u = storage.replay_u
            layer.replay_g = storage.replay_g
            layer.replay_lengths = storage.replay_lengths
            layer.replay_capacity = self.replay_capacity
            layer.has_previous_state = True

    def clear(self, batch_idx: int) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            storage.clear(batch_idx)

    def recurrent_tensors_for_form(
        self,
        form: "StatePhysicalForm",
    ) -> list[torch.Tensor | None]:
        """Resolve compiler-selected recurrence storage without naming a path."""

        if (
            form.representation != "materialized"
            or form.storage_axis_order != self._VALUE_MAJOR_RECURRENT_AXES
            or form.storage_dtype != self._GENERATED_RECURRENT_STORAGE_DTYPE
        ):
            raise ValueError(
                "generated Qwen recurrent state requires materialized BF16 "
                "value-major storage"
            )
        self._initialize_generated_recurrent()
        return [
            None if storage is None else storage.recurrent_states
            for storage in self.layers
        ]

    def _capture_conv_rows(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionState,
        indices: torch.Tensor,
    ) -> None:
        assert storage.conv_states is not None
        assert src_layer.conv_states is not None
        storage.conv_states.index_copy_(0, indices, src_layer.conv_states)

    def _validate_conv_rows(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionState,
        *,
        batch_size: int,
    ) -> None:
        source = src_layer.conv_states
        target = storage.conv_states
        expected_source = (batch_size, *self._conv_shape[1:])
        if (
            source is None
            or tuple(source.shape) != expected_source
            or source.device != self.device
            or not source.is_contiguous()
        ):
            raise RuntimeError(
                "Qwen GDN prefill convolution batch must match capture batch"
            )
        if (
            target is None
            or tuple(target.shape) != self._conv_shape
            or target.dtype != source.dtype
            or target.device != self.device
            or not target.is_contiguous()
        ):
            raise RuntimeError("Qwen GDN convolution pool is incomplete")

    def _capture_generated_recurrent_rows(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionState,
        indices: torch.Tensor,
    ) -> None:
        assert src_layer.recurrent_states is not None
        assert storage.recurrent_states is not None
        value_major = (
            src_layer.recurrent_states.transpose(-1, -2)
            .to(
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
        )
        storage.recurrent_states.index_copy_(0, indices, value_major)

    def _validate_generated_recurrent_rows(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionState,
        *,
        batch_size: int,
    ) -> None:
        source = src_layer.recurrent_states
        target = storage.recurrent_states
        expected_source = (batch_size, *self._key_major_recurrent_shape[1:])
        if (
            source is None
            or tuple(source.shape) != expected_source
            or source.dtype != torch.float32
            or source.device != self.device
            or not source.is_contiguous()
        ):
            raise RuntimeError(
                "Qwen generated prefill requires contiguous sequence-major FP32 state"
            )
        if (
            target is None
            or tuple(target.shape) != self._value_major_recurrent_shape
            or target.dtype != torch.bfloat16
            or target.device != self.device
            or not target.is_contiguous()
        ):
            raise RuntimeError("Qwen generated recurrent state pool is incomplete")

    @staticmethod
    def _validate_prefill_recurrent(
        src_layer: LinearAttentionState, *, batch_size: int
    ) -> None:
        recurrent_states = src_layer.recurrent_states
        if recurrent_states is None or recurrent_states.shape[0] != batch_size:
            raise RuntimeError(
                "Qwen GDN prefill recurrent batch must match capture batch")


__all__ = [
    "qwen_paged_kv_specs",
    "Qwen35InferenceCache",
    "Qwen35LinearStatePool",
]
