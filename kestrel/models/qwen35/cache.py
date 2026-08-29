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
    ) -> None:
        layer_types = tuple(config.layer_types)
        if len(paged_kv) != len(layer_types):
            raise ValueError("paged_kv layout must match layer_types")
        layers: list[Any] = []
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                layers.append(LinearAttentionState())
                continue
            layer = paged_kv[layer_idx]
            if layer is None:
                raise ValueError(
                    f"full-attention layer {layer_idx} has no paged K/V producer"
                )
            layers.append(layer)
        self.layers = tuple(layers)
        self.seq_length = 0

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

    _RECURRENT_AXES = ("state_row", "value_head", "value", "key")
    _RECURRENT_STORAGE_DTYPE = "bf16"

    def __init__(
        self,
        *,
        config: Any,
        max_batch_slots: int,
        device: torch.device,
    ) -> None:
        self.max_batch_slots = int(max_batch_slots)
        self.device = device
        self._conv_shape = (
            self.max_batch_slots,
            2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            + int(config.linear_num_value_heads) * int(config.linear_value_head_dim),
            int(config.linear_conv_kernel_dim),
        )
        self._recurrent_shape = (
            self.max_batch_slots,
            int(config.linear_num_value_heads),
            int(config.linear_value_head_dim),
            int(config.linear_key_head_dim),
        )
        self.layers: list[LinearAttentionState | None] = [
            (
                LinearAttentionState()
                if layer_type == "linear_attention"
                else None
            )
            for layer_type in config.layer_types
        ]

    def initialize_from_config(self, config: Any, *, dtype: torch.dtype) -> None:
        """Allocate the runtime-owned convolution state."""

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

    def _initialize_recurrent(self) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            tensor = storage.recurrent_states
            if tensor is None:
                storage.recurrent_states = torch.zeros(
                    self._recurrent_shape,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            elif (
                tuple(tensor.shape) != self._recurrent_shape
                or tensor.dtype != torch.bfloat16
            ):
                raise RuntimeError("Qwen recurrent state contract changed")

    def capture_batch_from_cache(
        self,
        batch_idx: torch.Tensor,
        cache: Qwen35InferenceCache,
        *,
        batch_size: int,
    ) -> None:
        indices = batch_idx[:batch_size].to(device=self.device, dtype=torch.long)
        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            src_layer = cache.layers[layer_idx]
            if not isinstance(src_layer, LinearAttentionState):
                raise ValueError("Cannot capture mismatched Qwen linear state")
            if not src_layer.has_previous_state:
                raise RuntimeError("Cannot capture uninitialized Qwen GDN state")
            self._capture_conv_rows(
                storage, src_layer, indices, batch_size=batch_size)
            target = storage.recurrent_states
            if target is None or src_layer.recurrent_states is not target:
                raise RuntimeError(
                    "Qwen prefill did not write the recurrent state pool")

    def bind_prefill_state(self, cache: Qwen35InferenceCache) -> None:
        """Expose the authoritative BF16 pool as packed-prefill final state."""

        self._initialize_recurrent()
        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            target = storage.recurrent_states
            if target is None:
                raise RuntimeError("Qwen recurrent state is incomplete")
            layer = cache.layers[layer_idx]
            if not isinstance(layer, LinearAttentionState):
                raise ValueError("Cannot bind mismatched Qwen linear state")
            layer.recurrent_states = target

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
            or form.storage_axis_order != self._RECURRENT_AXES
            or form.storage_dtype != self._RECURRENT_STORAGE_DTYPE
        ):
            raise ValueError(
                "generated Qwen recurrent state requires materialized BF16 "
                "value-major storage"
            )
        self._initialize_recurrent()
        return [
            None if storage is None else storage.recurrent_states
            for storage in self.layers
        ]

    def _capture_conv_rows(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionState,
        indices: torch.Tensor,
        *,
        batch_size: int = 1,
    ) -> None:
        conv_states = src_layer.conv_states
        if conv_states is None or conv_states.shape[0] != batch_size:
            raise RuntimeError(
                "Qwen GDN prefill convolution batch must match capture batch"
            )
        if storage.conv_states is None:
            raise RuntimeError("Qwen GDN convolution pool is not initialized")
        storage.conv_states.index_copy_(0, indices, conv_states)

__all__ = [
    "qwen_paged_kv_specs",
    "Qwen35InferenceCache",
    "Qwen35LinearStatePool",
]
