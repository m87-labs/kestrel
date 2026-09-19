"""Hybrid attention and recurrent state for Qwen 3.5/3.6."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, Sequence

import torch

from kestrel.kv_cache import PagedKVCache, PagedKVLayerSpec

from .gdn_state import LinearAttentionState

if TYPE_CHECKING:
    from kestrel.runtime.carried_state import StatePhysicalForm
    from .qwen_model import _RecurrentPrefixRecord


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
        self._prefix_source: Qwen35InferenceCache | None = None
        self._prefix_start = 0
        self._prefix_row = 0
        self._prefix_records: dict[int, _RecurrentPrefixRecord] = {}
        self._conv_sequence_layout = None

    def conv_sequence_indices(self, lengths: Sequence[int], prefix: int, device: torch.device) -> torch.Tensor:
        """Reuse one immutable convolution layout across compatible layers."""
        lengths = tuple(lengths)
        stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
        key = (lengths, prefix, device, stream)
        if self._conv_sequence_layout is None or self._conv_sequence_layout[0] != key:
            indices = torch.cat([
                torch.full((1, length + prefix), index, device=device, dtype=torch.int32)
                for index, length in enumerate(lengths)
            ], dim=-1)
            self._conv_sequence_layout = (key, indices)
        return self._conv_sequence_layout[1]

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

    def fork_recurrent_state(self, *, capture_prefix: bool = False) -> Qwen35InferenceCache:
        """Copy recurrent state while sharing append-only paged K/V storage.

        The caller must restrict attention to the branch's sequence length;
        rejected K/V suffixes remain allocated but are not committed context.
        Siblings must be explored serially and discarded before another sibling
        writes the shared suffix; this is not concurrent branch storage.
        """
        branch = copy(self)
        if capture_prefix and (self._prefix_source is not None or self.seq_length <= 0):
            raise ValueError("prefix capture requires a committed nonempty cache")
        branch._prefix_source = self if capture_prefix else None
        branch._prefix_start = self.seq_length if capture_prefix else 0
        branch._prefix_records = {}
        branch._prefix_row = 0
        layers = []
        sources, destinations = [], []
        for layer in self.layers:
            if isinstance(layer, LinearAttentionState):
                layer = copy(layer)
                for name in ("conv_states", "recurrent_states"):
                    source = getattr(layer, name)
                    if source is not None:
                        destination = torch.empty_like(source)
                        sources.append(source)
                        destinations.append(destination)
                        setattr(layer, name, destination)
            layers.append(layer)
        if sources:
            torch._foreach_copy_(destinations, sources)
        branch.layers = tuple(layers)
        return branch

    @staticmethod
    def fork_packed_recurrent_state(caches):
        """Copy independent committed rows once into packed verification storage."""
        if not caches or len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("packed verification requires distinct caches")
        if any(cache._prefix_source is not None or cache.seq_length <= 0 for cache in caches):
            raise ValueError("prefix capture requires committed nonempty caches")
        if len({len(cache.layers) for cache in caches}) != 1:
            raise ValueError("packed verification layer counts must match")
        branches = []
        for row, source in enumerate(caches):
            branch = copy(source)
            branch._prefix_source = source
            branch._prefix_start = source.seq_length
            branch._prefix_records = {}
            branch._prefix_row = row
            branch.layers = list(source.layers)
            branches.append(branch)
        packed = copy(branches[0])
        packed._prefix_records = {}
        packed_layers = list(packed.layers)
        for index, first in enumerate(caches[0].layers):
            owners = [cache.layers[index] for cache in caches]
            if any(type(layer) is not type(first) for layer in owners):
                raise ValueError("packed verification layer kinds must match")
            if not isinstance(first, LinearAttentionState):
                continue
            layer = copy(first)
            for name in ("conv_states", "recurrent_states"):
                tensors = [getattr(owner, name) for owner in owners]
                if any(tensor is None or tensor.shape[0] != 1 for tensor in tensors):
                    raise ValueError("packed verification requires initialized single-row state")
                setattr(layer, name, torch.cat(tensors, dim=0))
            packed_layers[index] = layer
            for row, branch in enumerate(branches):
                owned = copy(owners[row])
                owned.conv_states = layer.conv_states[row:row + 1]
                owned.recurrent_states = layer.recurrent_states[row:row + 1]
                branch.layers[index] = owned
        packed.layers = tuple(packed_layers)
        for branch in branches:
            branch.layers = tuple(branch.layers)
        return packed, branches

    @staticmethod
    def commit_recurrent_prefixes(caches, lengths, *, finalizer=None):
        """Finalize one packed verification before publishing any cache state."""
        from contextlib import nullcontext
        from kestrel_kernels import get_runtime

        caches, lengths = tuple(caches), tuple(lengths)
        if not caches or len(caches) != len(lengths):
            raise ValueError("packed prefix owners and lengths must match")
        records = caches[0]._prefix_records
        indices = tuple(i for i, layer in enumerate(caches[0].layers)
                        if isinstance(layer, LinearAttentionState))
        if not indices or set(records) != set(indices):
            raise RuntimeError("packed prefix is missing recurrent layers")
        contexts = tuple(records[index].prefix_context for index in indices)
        if any(context is None for context in contexts):
            raise RuntimeError("packed prefix has no retained finalization context")
        sources = tuple(cache._prefix_source for cache in caches)
        if any(source is None for source in sources) or len({id(source) for source in sources}) != len(sources):
            raise RuntimeError("packed prefix requires distinct committed sources")
        count = len(caches)
        for row, (cache, source, length) in enumerate(zip(caches, sources, lengths)):
            if (cache._prefix_records is not records or cache._prefix_row != row
                    or source.seq_length != cache._prefix_start
                    or cache.seq_length != cache._prefix_start + 16):
                raise RuntimeError("packed prefix source or row ownership changed")
            if type(length) is not int or not 1 <= length <= 16:
                raise ValueError("accepted prefix length must be in [1,16]")
            if tuple(i for i, layer in enumerate(source.layers)
                     if isinstance(layer, LinearAttentionState)) != indices:
                raise ValueError("packed prefix recurrent layer layout changed")
            for index in indices:
                record, layer = records[index], source.layers[index]
                if (record.qkv.shape[1] != count * 16
                        or record.initial_state.shape[0] != count
                        or not isinstance(layer, LinearAttentionState)
                        or layer.recurrent_states.shape[0] != 1
                        or layer.conv_states.shape[0] != 1):
                    raise ValueError("packed prefix state geometry changed")
        templates = tuple(records[index].initial_state for index in indices)
        accepted = torch.tensor(lengths, device=templates[0].device, dtype=torch.int32)
        if finalizer is None:
            outputs = tuple(torch.empty_like(template) for template in templates)
            get_runtime().gated_delta.finalize_packed_gated_delta_prefix(
                contexts, accepted, out_states=outputs)
            lease = nullcontext(outputs)
        else:
            lease = finalizer(records, accepted)
        with lease as outputs:
            # Allocate after submission to overlap host work with finalization.
            # Destinations remain detached until every state/history is written.
            results = []
            for source in sources:
                result = copy(source)
                result._prefix_source = None
                result._prefix_start = result._prefix_row = 0
                result._prefix_records = {}
                layers = list(source.layers)
                # Both fields are fully overwritten before publication.
                for index in indices:
                    layer = copy(layers[index])
                    layer.conv_states = torch.empty_like(layer.conv_states)
                    layer.recurrent_states = torch.empty_like(layer.recurrent_states)
                    layers[index] = layer
                result.layers = tuple(layers)
                results.append(result)
            results = tuple(results)
            state_destinations, states = [], []
            conv_destinations, histories = [], []
            for index, state in zip(indices, outputs, strict=True):
                record = records[index]
                width = record.module.conv_kernel_size
                for row, (result, length) in enumerate(zip(results, lengths)):
                    layer = result.layers[index]
                    start = row * (16 + width - 1) + length - 1
                    state_destinations.append(layer.recurrent_states)
                    states.append(state[row:row + 1])
                    conv_destinations.append(layer.conv_states)
                    histories.append(record.conv_input[..., start:start + width])
                    layer.has_previous_state = True
            # Strided histories must not disable the recurrent copies' fast path.
            torch._foreach_copy_(state_destinations, states)
            torch._foreach_copy_(conv_destinations, histories)
        for cache, result, length in zip(caches, results, lengths):
            result.advance_to(cache._prefix_start + length)
            cache._prefix_records = {}
            cache._prefix_source = None
            cache._prefix_start = cache._prefix_row = 0
        return results

    def commit_recurrent_prefix(self, length: int, *, replay_graph=None) -> Qwen35InferenceCache:
        """Commit one verified prefix without repeating its dense projections.

        Records belong to a single serial verification fork. K/V suffix storage
        is shared, as for ordinary forks; callers retain only committed features
        and never expose attention positions beyond the returned sequence length.
        """
        source = self._prefix_source
        if source is None:
            raise RuntimeError("cache has no captured verification prefix")
        if source.seq_length != self._prefix_start:
            raise RuntimeError("captured prefix source has advanced")
        expected = {i for i, layer in enumerate(self.layers)
                    if isinstance(layer, LinearAttentionState)}
        if not expected or set(self._prefix_records) != expected:
            raise RuntimeError("captured prefix is missing recurrent layers")
        lengths = {record.qkv.shape[1] for record in self._prefix_records.values()}
        if len(lengths) != 1 or self.seq_length != self._prefix_start + next(iter(lengths)):
            raise RuntimeError("captured prefix does not match verified sequence length")
        total = next(iter(lengths))
        if type(length) is not int or not 1 <= length <= total:
            raise ValueError("prefix length must be within the verified token range")
        if length == total:
            result = self
        else:
            from kestrel_kernels import get_runtime

            result = source.fork_recurrent_state()
            # Tried cross-request grouping: C8 requests 2.28-2.46s vs 2.13-2.30s,
            # including GC; keeping per-request groups.
            groups = {}
            for index, record in self._prefix_records.items():
                groups.setdefault(record.replay_geometry, []).append((record, result.layers[index]))
            for group in groups.values():
                if len(group) == 1:
                    record, layer = group[0]
                    cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
                        sequence_lengths=(length,), device=record.qkv.device)
                    record.replay_into(layer, length, cu, topology)
                else:
                    group[0][0].replay_group(group, length, graph=replay_graph)
            result.advance_to(self._prefix_start + length)
        self._prefix_records = {}
        self._prefix_source = None
        self._prefix_start = 0
        return result


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
