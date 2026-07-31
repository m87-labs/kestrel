"""Hybrid cache helpers for Qwen 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from kestrel.kv_cache import (
    KVMemoryPool,
    LayeredPagedKV,
    PageTable,
    PagedKVCache,
    PagedKVLayerSpec,
)

from .inference_ops import LinearAttentionLayer

if TYPE_CHECKING:
    from mkl.megakernel.state_runtime import StatePhysicalForm


def allocate_qwen35_paged_kv(
    *,
    config: Any,
    page_table: PageTable,
    pool: KVMemoryPool,
    dtype: torch.dtype,
) -> LayeredPagedKV:
    """Allocate full-attention storage through Kestrel's shared cache owner."""

    head_dim = _head_dim(config)
    specs: list[PagedKVLayerSpec | None] = []
    sources: list[int] = []
    for layer_idx, layer_type in enumerate(_layer_types(config)):
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
    return LayeredPagedKV.allocate(
        layer_specs=specs,
        source_layer_idx=sources,
        page_table=page_table,
        pool=pool,
        dtype=dtype,
    )


class Qwen35InferenceCache:
    """Per-forward GDN state over runtime-owned Kestrel paged K/V."""

    def __init__(
        self,
        *,
        config: Any,
        paged_kv: LayeredPagedKV,
        replay_capacity: int,
    ) -> None:
        layer_types = _layer_types(config)
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
        self.paged_kv = paged_kv
        self.seq_length = 0

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].update_conv_state(
            conv_states,
            state_indices,
        )

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].update_recurrent_state(
            recurrent_states,
            state_indices,
        )

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        if layer_idx is None:
            return any(
                isinstance(layer, LinearAttentionLayer)
                and bool(layer.has_previous_state)
                for layer in self.layers
            ) or self.seq_length > 0
        layer = self.layers[layer_idx]
        if isinstance(layer, LinearAttentionLayer):
            return bool(layer.has_previous_state)
        return self.seq_length > 0

    def get_paged_layer(self, layer_idx: int) -> PagedKVCache | None:
        return self.paged_kv.producer(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.seq_length)

    def advance_to(self, seq_length: int) -> None:
        self.seq_length = max(self.seq_length, int(seq_length))


@dataclass
class _LinearStateStorage:
    conv_states: torch.Tensor | None = None
    recurrent_states: torch.Tensor | None = None
    replay_checkpoint_states: torch.Tensor | None = None
    replay_k: torch.Tensor | None = None
    replay_u: torch.Tensor | None = None
    replay_g: torch.Tensor | None = None
    replay_lengths: torch.Tensor | None = None


class Qwen35LinearStatePool:
    """Runtime-owned GDN state indexed by Kestrel batch slot."""

    _KEY_MAJOR_RECURRENT_AXES = (
        "state_row", "value_head", "key", "value")
    _VALUE_MAJOR_RECURRENT_AXES = (
        "state_row", "value_head", "value", "key")
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
        self.num_k_heads = int(config.linear_num_key_heads)
        self.layers: list[_LinearStateStorage | None] = [
            _LinearStateStorage() if layer_type == "linear_attention" else None
            for layer_type in _layer_types(config)
        ]

    def initialize_from_config(self, config: Any, *, dtype: torch.dtype) -> None:
        """Allocate zero GDN state rows without waiting for first prefill."""

        conv_dim = (
            2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            + int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
        )
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
            self._ensure_storage_tensors(
                storage,
                conv_shape=conv_shape,
                recurrent_shape=recurrent_shape,
                conv_dtype=dtype,
                recurrent_dtype=torch.float32,
            )

    def zero_all(self) -> None:
        for storage in self.layers:
            if storage is None:
                continue
            if storage.conv_states is not None:
                storage.conv_states.zero_()
            if storage.recurrent_states is not None:
                storage.recurrent_states.zero_()
            if storage.replay_checkpoint_states is not None:
                storage.replay_checkpoint_states.zero_()
            if storage.replay_k is not None:
                storage.replay_k.zero_()
            if storage.replay_u is not None:
                storage.replay_u.zero_()
            if storage.replay_g is not None:
                storage.replay_g.zero_()
            if storage.replay_lengths is not None:
                storage.replay_lengths.zero_()

    def capture_from_cache(
        self,
        batch_idx: int,
        cache: Qwen35InferenceCache,
    ) -> None:
        batch_idx_tensor = torch.empty((1,), dtype=torch.long, device=self.device)
        batch_idx_tensor.fill_(int(batch_idx))
        self.capture_batch_from_cache(
            batch_idx_tensor,
            cache,
            batch_size=1,
        )

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
            self._copy_rows_from_layer(
                storage,
                indices,
                src_layer,
                batch_size,
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
            if storage.conv_states is not None:
                storage.conv_states[int(batch_idx)].zero_()
            if storage.recurrent_states is not None:
                storage.recurrent_states[int(batch_idx)].zero_()
            if storage.replay_checkpoint_states is not None:
                storage.replay_checkpoint_states[int(batch_idx)].zero_()
            if storage.replay_k is not None:
                storage.replay_k[int(batch_idx)].zero_()
            if storage.replay_u is not None:
                storage.replay_u[int(batch_idx)].zero_()
            if storage.replay_g is not None:
                storage.replay_g[int(batch_idx)].zero_()
            if storage.replay_lengths is not None:
                storage.replay_lengths[int(batch_idx)].zero_()

    @property
    def replay_recurrent_form(self) -> "StatePhysicalForm":
        """Return the native replay path's authoritative checkpoint form."""

        from mkl.megakernel.state_runtime import StatePhysicalForm

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
                f"{form.storage_dtype!r}")
        fields_by_order = {
            cls._KEY_MAJOR_RECURRENT_AXES: "recurrent_states",
            cls._VALUE_MAJOR_RECURRENT_AXES: "replay_checkpoint_states",
        }
        try:
            return fields_by_order[form.storage_axis_order]
        except KeyError as exc:
            raise ValueError(
                "unsupported recurrent-state storage axis order "
                f"{form.storage_axis_order!r}") from exc

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
                f"form within representation {source_representation!r}")
        source_field = self._recurrent_field_for_form(source)
        target_field = self._recurrent_field_for_form(target)
        checkpoint_field = "replay_checkpoint_states"
        if (
            source_representation == "replay"
            and source_field != checkpoint_field
        ) or (
            target_representation == "replay"
            and target_field != checkpoint_field
        ):
            raise ValueError(
                "replay recurrence requires value-major checkpoint storage")
        if (source_representation, target_representation) not in {
            ("replay", "materialized"),
            ("materialized", "replay"),
        }:
            raise ValueError(
                "unsupported recurrent-state transition "
                f"{source_representation!r} -> {target_representation!r}"
            )
        row_indices = torch.tensor(
            rows, dtype=torch.long, device=self.device)
        for storage in self.layers:
            if storage is None or storage.recurrent_states is None:
                continue
            if source_representation == "replay":
                LinearAttentionLayer.materialize_recurrent_from_replay(
                    storage,
                    row_indices,
                    write_recurrent=(target_field == "recurrent_states"),
                )
            elif source_field == checkpoint_field:
                self._reset_replay_tail(storage, row_indices)
            else:
                self._seed_replay_rows(storage, row_indices)

    @staticmethod
    def _reset_replay_tail(
        storage: _LinearStateStorage,
        row_indices: torch.Tensor,
    ) -> None:
        if (
            storage.replay_k is None
            or storage.replay_u is None
            or storage.replay_g is None
            or storage.replay_lengths is None
        ):
            raise RuntimeError("Qwen replay state is not initialized")
        # replay_lengths is the validity boundary. Payload slots are overwritten
        # before the cursor advances, so clearing inaccessible K/U/G bytes only
        # adds three launches per recurrent layer.
        storage.replay_lengths.index_fill_(0, row_indices, 0)

    @classmethod
    def _seed_replay_rows(
        cls,
        storage: _LinearStateStorage,
        row_indices: torch.Tensor,
    ) -> None:
        if (
            storage.recurrent_states is None
            or storage.replay_checkpoint_states is None
        ):
            raise RuntimeError("Qwen recurrent state is not initialized")
        checkpoint_rows = storage.recurrent_states.index_select(
            0, row_indices).transpose(-1, -2).contiguous()
        storage.replay_checkpoint_states.index_copy_(
            0, row_indices, checkpoint_rows)
        cls._reset_replay_tail(storage, row_indices)

    def _ensure_storage(
        self,
        storage: _LinearStateStorage,
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
        self._ensure_storage_tensors(
            storage,
            conv_shape=conv_shape,
            recurrent_shape=recurrent_shape,
            conv_dtype=conv_states.dtype,
            recurrent_dtype=recurrent_states.dtype,
        )

    def _ensure_storage_tensors(
        self,
        storage: _LinearStateStorage,
        *,
        conv_shape: tuple[int, ...],
        recurrent_shape: tuple[int, ...],
        conv_dtype: torch.dtype,
        recurrent_dtype: torch.dtype,
    ) -> None:
        if storage.conv_states is None:
            storage.conv_states = torch.zeros(
                conv_shape,
                dtype=conv_dtype,
                device=self.device,
            )
        elif tuple(storage.conv_states.shape) != conv_shape:
            raise RuntimeError("Qwen GDN conv state shape changed")

        if storage.recurrent_states is None:
            storage.recurrent_states = torch.zeros(
                recurrent_shape,
                dtype=recurrent_dtype,
                device=self.device,
            )
        elif tuple(storage.recurrent_states.shape) != recurrent_shape:
            raise RuntimeError("Qwen GDN recurrent state shape changed")

        if len(recurrent_shape) != 4:
            return

        slots, value_heads, key_dim, value_dim = recurrent_shape
        # Size the replay key ring by VALUE head to match the per-layer
        # ``LinearAttentionLayer._ensure_replay_state`` allocation (and the
        # native verify/decode/materialize GQA kernels, which index the ring by
        # v-head and apply the k->v fan-out internally). Sizing by num_k_heads
        # here makes the pool ring (16-head) inconsistent with the captured
        # layer ring (32-head), so ``_copy_rows_from_layer`` fails to copy a
        # [*, cap, 32, D] source into a [*, cap, 16, D] pool row at the very
        # first prefill-cache capture on k16/v32 Qwen3.5-4B.
        checkpoint_shape = (slots, value_heads, value_dim, key_dim)
        replay_k_shape = (slots, self.replay_capacity, value_heads, key_dim)
        replay_u_shape = (slots, self.replay_capacity, value_heads, value_dim)
        replay_g_shape = (slots, self.replay_capacity, value_heads)
        replay_lengths_shape = (slots,)
        replay_k_dtype = (
            conv_dtype if self.device.type == "mps" else torch.bfloat16
        )

        if storage.replay_checkpoint_states is None:
            storage.replay_checkpoint_states = torch.zeros(
                checkpoint_shape,
                dtype=torch.float32,
                device=self.device,
            )
        elif tuple(storage.replay_checkpoint_states.shape) != checkpoint_shape:
            raise RuntimeError("Qwen GDN replay checkpoint shape changed")

        if storage.replay_k is None:
            storage.replay_k = torch.zeros(
                replay_k_shape,
                dtype=replay_k_dtype,
                device=self.device,
            )
        elif tuple(storage.replay_k.shape) != replay_k_shape:
            raise RuntimeError("Qwen GDN replay key buffer shape changed")
        elif storage.replay_k.dtype != replay_k_dtype:
            raise RuntimeError("Qwen GDN replay key buffer dtype changed")

        if storage.replay_u is None:
            storage.replay_u = torch.zeros(
                replay_u_shape,
                dtype=torch.float32,
                device=self.device,
            )
        elif tuple(storage.replay_u.shape) != replay_u_shape:
            raise RuntimeError("Qwen GDN replay update buffer shape changed")

        if storage.replay_g is None:
            storage.replay_g = torch.zeros(
                replay_g_shape,
                dtype=torch.float32,
                device=self.device,
            )
        elif tuple(storage.replay_g.shape) != replay_g_shape:
            raise RuntimeError("Qwen GDN replay gate buffer shape changed")

        if storage.replay_lengths is None:
            storage.replay_lengths = torch.zeros(
                replay_lengths_shape,
                dtype=torch.int32,
                device=self.device,
            )
        elif tuple(storage.replay_lengths.shape) != replay_lengths_shape:
            raise RuntimeError("Qwen GDN replay length shape changed")

    @staticmethod
    def _copy_rows_from_layer(
        storage: _LinearStateStorage,
        batch_idx: torch.Tensor,
        src_layer: LinearAttentionLayer,
        batch_size: int,
        *,
        copy_replay_payload: bool = True,
    ) -> None:
        assert storage.conv_states is not None
        assert storage.recurrent_states is not None
        assert src_layer.conv_states is not None
        assert src_layer.recurrent_states is not None
        storage.conv_states.index_copy_(
            0,
            batch_idx[:batch_size],
            src_layer.conv_states[:batch_size],
        )
        storage.recurrent_states.index_copy_(
            0,
            batch_idx[:batch_size],
            src_layer.recurrent_states[:batch_size],
        )
        if storage.replay_checkpoint_states is None:
            return
        assert storage.replay_k is not None
        assert storage.replay_u is not None
        assert storage.replay_g is not None
        assert storage.replay_lengths is not None

        src_indices = batch_idx[:batch_size]
        if src_layer.replay_checkpoint_states is not None:
            storage.replay_checkpoint_states.index_copy_(
                0,
                src_indices,
                src_layer.replay_checkpoint_states[:batch_size],
            )
        else:
            checkpoint_rows = (
                src_layer.recurrent_states[:batch_size]
                .transpose(-1, -2)
                .contiguous()
            )
            storage.replay_checkpoint_states.index_copy_(
                0,
                src_indices,
                checkpoint_rows,
            )

        if (
            src_layer.replay_k is not None
            and src_layer.replay_u is not None
            and src_layer.replay_g is not None
            and src_layer.replay_lengths is not None
        ):
            if copy_replay_payload:
                storage.replay_k.index_copy_(
                    0,
                    src_indices,
                    src_layer.replay_k[:batch_size],
                )
                storage.replay_u.index_copy_(
                    0,
                    src_indices,
                    src_layer.replay_u[:batch_size],
                )
                storage.replay_g.index_copy_(
                    0,
                    src_indices,
                    src_layer.replay_g[:batch_size],
                )
                storage.replay_lengths.index_copy_(
                    0,
                    src_indices,
                    src_layer.replay_lengths[:batch_size],
                )
            else:
                storage.replay_lengths.index_fill_(0, src_indices, 0)
        else:
            storage.replay_lengths.index_fill_(0, src_indices, 0)

def _layer_types(config: Any) -> list[str]:
    return list(config.layer_types)


def _head_dim(config: Any) -> int:
    return int(config.head_dim)


__all__ = [
    "allocate_qwen35_paged_kv",
    "Qwen35InferenceCache",
    "Qwen35LinearStatePool",
]
