"""Hybrid attention and recurrent state for Qwen 3.5/3.6."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from kestrel.kv_cache import LayeredPagedKV, PagedKVLayerSpec

from .inference_ops import LinearAttentionLayer, LinearAttentionState

if TYPE_CHECKING:
    from kestrel.runtime.carried_state import StatePhysicalForm


def qwen_kv_layout(
    config: Any,
) -> tuple[tuple[PagedKVLayerSpec | None, ...], tuple[int, ...]]:
    """Describe which hybrid layers own ordinary paged K/V storage."""

    head_dim = int(config.head_dim)
    specs: list[PagedKVLayerSpec | None] = []
    sources: list[int] = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            specs.append(None)
            sources.append(-1)
        else:
            specs.append(
                PagedKVLayerSpec(
                    n_heads=int(config.num_key_value_heads),
                    head_dim=head_dim,
                )
            )
            sources.append(layer_idx)
    return tuple(specs), tuple(sources)


class Qwen35InferenceCache:
    """Per-forward GDN state over runtime-owned Kestrel paged K/V."""

    def __init__(
        self,
        *,
        config: Any,
        paged_kv: LayeredPagedKV,
        replay_capacity: int,
    ) -> None:
        layer_types = tuple(config.layer_types)
        if len(paged_kv.layers) != len(layer_types):
            raise ValueError("paged_kv layout must match layer_types")
        layers: list[Any] = []
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                layers.append(
                    LinearAttentionLayer(
                        config,
                        replay_capacity=replay_capacity,
                    )
                )
                continue
            layer = paged_kv.producer(layer_idx)
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
                    isinstance(layer, LinearAttentionLayer)
                    and bool(layer.has_previous_state)
                    for layer in self.layers
                )
                or self.seq_length > 0
            )
        layer = self.layers[layer_idx]
        if isinstance(layer, LinearAttentionLayer):
            return bool(layer.has_previous_state)
        return self.seq_length > 0

    def get_seq_length(self) -> int:
        return int(self.seq_length)

    def advance_to(self, seq_length: int) -> None:
        self.seq_length = max(self.seq_length, int(seq_length))


class Qwen35LinearStatePool:
    """Runtime-owned GDN state indexed by Kestrel batch slot."""

    _KEY_MAJOR_RECURRENT_AXES = ("state_row", "value_head", "key", "value")
    _VALUE_MAJOR_RECURRENT_AXES = ("state_row", "value_head", "value", "key")
    _RECURRENT_STORAGE_DTYPE = "fp32"

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
        self.layers: list[LinearAttentionState | None] = [
            (
                LinearAttentionState(replay_capacity=self.replay_capacity)
                if layer_type == "linear_attention"
                else None
            )
            for layer_type in config.layer_types
        ]

    def initialize_from_config(self, config: Any, *, dtype: torch.dtype) -> None:
        """Allocate zero GDN state rows without waiting for first prefill."""

        conv_dim = 2 * int(config.linear_num_key_heads) * int(
            config.linear_key_head_dim
        ) + int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
        conv_shape = (
            self.max_batch_slots,
            conv_dim,
            int(config.linear_conv_kernel_dim),
        )
        recurrent_shape = (
            self.max_batch_slots,
            int(config.linear_num_value_heads),
            int(config.linear_key_head_dim),
            int(config.linear_value_head_dim),
        )
        for storage in self.layers:
            if storage is None:
                continue
            storage.allocate_zeroed(
                conv_shape=conv_shape,
                recurrent_shape=recurrent_shape,
                conv_dtype=dtype,
                recurrent_dtype=torch.float32,
                device=self.device,
            )

    def zero_all(self) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            storage.clear()

    def capture_batch_from_cache(
        self,
        batch_idx: torch.Tensor,
        cache: Qwen35InferenceCache,
        *,
        batch_size: int,
        copy_replay_payload: bool = True,
    ) -> None:
        indices = batch_idx[:batch_size].to(device=self.device, dtype=torch.long)
        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            src_layer = cache.layers[layer_idx]
            if not isinstance(src_layer, LinearAttentionLayer):
                raise ValueError("Cannot capture mismatched Qwen linear state")
            if not src_layer.has_previous_state:
                raise RuntimeError("Cannot capture uninitialized Qwen GDN state")
            self._ensure_storage(storage, src_layer, batch_size=batch_size)
            storage.copy_rows_from(
                src_layer,
                indices,
                copy_replay_payload=copy_replay_payload,
            )

    def bind_to_cache(self, cache: Qwen35InferenceCache) -> None:
        """Bind cache linear layers directly to runtime-owned persistent state."""

        for layer_idx, storage in enumerate(self.layers):
            if storage is None:
                continue
            layer = cache.layers[layer_idx]
            if not isinstance(layer, LinearAttentionLayer):
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
            # Mirror ``_install_linear_conv_state``'s metadata so the multi-token
            # (chunked / spec-verify) GDN conv path -- which reads
            # ``conv_kernel_size`` -- works against the bound persistent state,
            # not just the single-token decode path that goes through the
            # indexed conv-update kernel.
            layer.dtype = storage.conv_states.dtype
            layer.device = storage.conv_states.device
            layer.max_batch_size = storage.conv_states.shape[0]
            layer.conv_kernel_size = storage.conv_states.shape[-1]
            layer.is_conv_states_initialized = True
            layer.is_recurrent_states_initialized = True
            layer.has_previous_state = True

    def clear(self, batch_idx: int) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            storage.clear(batch_idx)

    @property
    def replay_recurrent_form(self) -> "StatePhysicalForm":
        """Return the native replay path's authoritative checkpoint form."""

        from kestrel.runtime.carried_state import StatePhysicalForm

        return StatePhysicalForm(
            representation="replay",
            storage_axis_order=self._VALUE_MAJOR_RECURRENT_AXES,
            storage_dtype=self._RECURRENT_STORAGE_DTYPE,
        )

    def recurrent_tensors_for_form(
        self,
        form: "StatePhysicalForm",
    ) -> list[torch.Tensor | None]:
        """Resolve compiler-selected recurrence storage without naming a path."""

        field = self._recurrent_field_for_form(form)
        return [
            None if storage is None else getattr(storage, field)
            for storage in self.layers
        ]

    @classmethod
    def _recurrent_field_for_form(
        cls,
        form: "StatePhysicalForm",
    ) -> str:
        if form.storage_dtype != cls._RECURRENT_STORAGE_DTYPE:
            raise ValueError(
                "recurrent state requires fp32 physical storage, got "
                f"{form.storage_dtype!r}"
            )
        fields_by_order = {
            cls._KEY_MAJOR_RECURRENT_AXES: "recurrent_states",
            cls._VALUE_MAJOR_RECURRENT_AXES: "replay_checkpoint_states",
        }
        try:
            return fields_by_order[form.storage_axis_order]
        except KeyError as exc:
            raise ValueError(
                "unsupported recurrent-state storage axis order "
                f"{form.storage_axis_order!r}"
            ) from exc

    def transition_recurrent_form(
        self,
        source: "StatePhysicalForm",
        target: "StatePhysicalForm",
        rows: tuple[int, ...],
    ) -> None:
        """Convert selected rows between native replay and materialized state."""

        if source == target or not rows:
            return
        source_representation = source.representation
        target_representation = target.representation
        if source_representation == target_representation:
            raise ValueError(
                "recurrent-state converter does not support changing physical "
                f"form within representation {source_representation!r}"
            )
        source_field = self._recurrent_field_for_form(source)
        target_field = self._recurrent_field_for_form(target)
        checkpoint_field = "replay_checkpoint_states"
        if (source_representation == "replay" and source_field != checkpoint_field) or (
            target_representation == "replay" and target_field != checkpoint_field
        ):
            raise ValueError(
                "replay recurrence requires value-major checkpoint storage"
            )
        if (source_representation, target_representation) not in {
            ("replay", "materialized"),
            ("materialized", "replay"),
        }:
            raise ValueError(
                "unsupported recurrent-state transition "
                f"{source_representation!r} -> {target_representation!r}"
            )
        row_indices = torch.tensor(rows, dtype=torch.long, device=self.device)
        for storage in self.layers:
            if storage is None or storage.recurrent_states is None:
                continue
            if source_representation == "replay":
                storage.materialize_recurrent_from_replay(
                    row_indices,
                    write_recurrent=(target_field == "recurrent_states"),
                )
            elif source_field == checkpoint_field:
                storage.reset_replay_tail(row_indices)
            else:
                storage.seed_replay_rows(row_indices)

    def _ensure_storage(
        self,
        storage: LinearAttentionState,
        src_layer: LinearAttentionLayer,
        *,
        batch_size: int = 1,
    ) -> None:
        conv_states = src_layer.conv_states
        recurrent_states = src_layer.recurrent_states
        if conv_states is None or recurrent_states is None:
            raise RuntimeError("Initialized Qwen GDN state tensor is missing")
        if (
            conv_states.shape[0] != batch_size
            or recurrent_states.shape[0] != batch_size
        ):
            raise RuntimeError(
                "Qwen GDN prefill state batch dimension must match capture batch"
            )

        conv_shape = (self.max_batch_slots, *conv_states.shape[1:])
        recurrent_shape = (self.max_batch_slots, *recurrent_states.shape[1:])
        storage.allocate_zeroed(
            conv_shape=conv_shape,
            recurrent_shape=recurrent_shape,
            conv_dtype=conv_states.dtype,
            recurrent_dtype=recurrent_states.dtype,
            device=self.device,
        )


__all__ = [
    "qwen_kv_layout",
    "Qwen35InferenceCache",
    "Qwen35LinearStatePool",
]
