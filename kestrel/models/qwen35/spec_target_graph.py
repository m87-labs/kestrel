"""Tensor-only native verification over leased fixed-shape graph buffers."""

from contextlib import contextmanager
from dataclasses import fields
import torch

from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph
from kestrel_kernels import get_runtime
from .cache import Qwen35InferenceCache
from .qwen_model import _RecurrentPrefixRecord, _TextModelOutput


_INPUTS = ('input_ids', 'position_ids', 'cache_position_ids', 'slot_mapping',
           'page_table', 'paged_kv_seqlens_k', 'seq_idx', 'gdn_state_indices')
_RECORDS = ('qkv', 'a', 'b', 'conv_input', 'initial_state', 'state_indices')


class Qwen35TargetGraph:
    def __init__(self, runtime, text, capture_layers, block_size):
        self._runtime = runtime
        self._text = text
        self._capture_layers = tuple(capture_layers)
        self._block_size = block_size
        self._linear = tuple(i for i, kind in enumerate(text.config.layer_types)
                             if kind == 'linear_attention')
        self._layouts = {}
        self._graphs = FixedShapeSinglePassGraph(
            enabled=True, device=runtime.device, stream=runtime._compute_stream,
            run_forward=self._forward, max_entries=runtime.max_batch_size)

    def _layout(self, count, device):
        if count not in self._layouts:
            if not 1 <= count <= self._runtime.max_batch_size:
                raise ValueError('verification graph exceeds sequence capacity')
            lengths = (self._block_size,) * count
            cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
                sequence_lengths=lengths, device=device)
            cache = Qwen35InferenceCache(config=self._text.config, paged_kv=self._runtime._paged_kv)
            # This is an execution workspace, not a committed session. Absolute
            # positions and actual state values are always supplied as tensors.
            source = Qwen35InferenceCache(config=self._text.config, paged_kv=self._runtime._paged_kv)
            source.seq_length = cache.seq_length = cache._prefix_start = 1
            cache._prefix_source = source
            self._layouts[count] = cache, lengths, cu, topology
        return self._layouts[count]

    def _forward(self, *values):
        inputs = dict(zip(_INPUTS, values[:len(_INPUTS)], strict=True))
        count = inputs['gdn_state_indices'].numel()
        cache, lengths, cu, topology = self._layout(count, inputs['input_ids'].device)
        cache._prefix_records = {}
        states = iter(values[len(_INPUTS):])
        for index in self._linear:
            layer = cache.layers[index]
            layer.conv_states, layer.recurrent_states = next(states), next(states)
            layer.has_previous_state = True
        output = self._text(**inputs, past_key_values=cache, cu_seq_lens_q=cu,
            sequence_lengths=lengths, topology_token=topology,
            gdn_state_indices_allocator_owned=True, capture_layers=self._capture_layers)
        tensors = [output.last_hidden_state, *output.layer_hidden_states]
        for index in self._linear:
            layer, record = cache.layers[index], cache._prefix_records[index]
            tensors.extend((layer.conv_states, layer.recurrent_states))
            tensors.extend(getattr(record, name) for name in _RECORDS)
        # Owner-only outputs tie scratch lifetime to the actual graph entry,
        # including when another C replaces the model's workspace cache.
        for index in self._linear:
            workspace = self._text.layers[index].linear_attn._prefill_workspace_cache.workspace
            tensors.extend(value for field in fields(workspace)
                           if isinstance(value := getattr(workspace, field.name), torch.Tensor))
            cache.layers[index].conv_states = cache.layers[index].recurrent_states = None
        cache._prefix_records = {}
        tensors.append(cu)
        return tuple(tensors)

    @contextmanager
    def launch(self, **kwargs):
        lengths = tuple(kwargs['sequence_lengths'])
        if not lengths or any(length != self._block_size for length in lengths):
            raise ValueError('verification graph requires complete speculative blocks')
        cache = kwargs['past_key_values']
        if cache._prefix_source is None or cache._prefix_records:
            raise RuntimeError('verification graph requires a fresh owned recurrent fork')
        if (cache.seq_length != cache._prefix_start
                or cache._prefix_source.seq_length != cache._prefix_start):
            raise RuntimeError('verification graph source has advanced')
        if (kwargs['gdn_state_indices_allocator_owned'] is not True
                or tuple(kwargs['capture_layers']) != self._capture_layers):
            raise ValueError('verification graph requires owned indices and matching feature taps')
        inputs = [kwargs[name] for name in _INPUTS]
        if inputs[0].shape != (1, sum(lengths)) or inputs[-1].numel() != len(lengths):
            raise ValueError('verification graph metadata does not match packed sequences')
        for index in self._linear:
            layer = cache.layers[index]
            if not layer.has_previous_state:
                raise ValueError('verification graph requires committed recurrent state')
            inputs.extend((layer.conv_states, layer.recurrent_states))
        with self._graphs.launch(*inputs) as values:
            hidden, taps = values[0], values[1:1 + len(self._capture_layers)]
            remaining = iter(values[1 + len(self._capture_layers):])
            destinations, sources = [], []
            for index in self._linear:
                layer = cache.layers[index]
                destinations.extend((layer.conv_states, layer.recurrent_states))
                sources.extend((next(remaining), next(remaining)))
                record = _RecurrentPrefixRecord(self._text.layers[index].linear_attn,
                    *(next(remaining) for _ in _RECORDS))
                cache._prefix_records[index] = record
            if destinations:
                torch._foreach_copy_(destinations, sources)
            yield _TextModelOutput(hidden, cache, tuple(taps))

    def shutdown(self):
        try:
            self._graphs.shutdown()
        finally:
            self._layouts.clear()
