"""Tensor-only native verification over leased fixed-shape graph buffers."""

from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import fields
import torch

from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph
from kestrel_kernels import get_runtime
from .cache import Qwen35InferenceCache, _finalize_recurrent_prefixes
from .qwen_model import _RecurrentPrefixRecord, _TextModelOutput


_INPUTS = ('input_ids', 'position_ids', 'cache_position_ids', 'slot_mapping',
           'page_table', 'paged_kv_seqlens_k', 'seq_idx', 'gdn_state_indices')
_RECORDS = ('qkv', 'a', 'b', 'conv_input', 'initial_state', 'state_indices')


class Qwen35TargetGraph:
    def __init__(self, runtime, text, capture_layers, block_size, *, finalize_stream=None):
        self._runtime = runtime
        self._text = text
        self._capture_layers = tuple(capture_layers)
        self._block_size = block_size
        self._finalize_stream = runtime._compute_stream if finalize_stream is None else finalize_stream
        self._linear = tuple(i for i, kind in enumerate(text.config.layer_types)
                             if kind == 'linear_attention')
        self._layouts = {}
        self._prefix_bindings = {}
        self._bound_outputs = OrderedDict()
        self._finalizers = OrderedDict()
        self._graphs = FixedShapeSinglePassGraph(
            enabled=True, device=runtime.device, stream=runtime._compute_stream,
            run_forward=self._forward, max_entries=runtime.max_batch_size)

    def _layout(self, count, device):
        stream = torch.cuda.current_stream(device)
        previous = self._layouts.get(count)
        if previous is None or previous[0] != stream:
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
            # Separate graph entries own separate capture streams. Sharing a
            # topology would record its storage against a stream that can retire
            # before that storage, and import external waits during capture.
            self._layouts[count] = stream, (cache, lengths, cu, topology)
        return self._layouts[count][1]

    def _forward(self, *values):
        inputs = dict(zip(_INPUTS, values[:len(_INPUTS)], strict=True))
        count = inputs['gdn_state_indices'].numel()
        cache, lengths, cu, topology = self._layout(count, inputs['input_ids'].device)
        cache._prefix_records = {}
        states = iter(values[len(_INPUTS):])
        row_inputs = len(values) != len(_INPUTS) + 2 * len(self._linear)
        for index in self._linear:
            layer = cache.layers[index]
            if row_inputs:
                layer.conv_states = torch.cat([next(states) for _ in range(count)], dim=0)
                layer.recurrent_states = torch.cat([next(states) for _ in range(count)], dim=0)
            else:
                layer.conv_states, layer.recurrent_states = next(states), next(states)
            layer.has_previous_state = True
        output = self._text(**inputs, past_key_values=cache, cu_seq_lens_q=cu,
            sequence_lengths=lengths, topology_token=topology,
            gdn_state_indices_allocator_owned=True, capture_layers=self._capture_layers)
        tensors = [output.last_hidden_state, *output.layer_hidden_states]
        bindings = []
        for index in self._linear:
            layer, record = cache.layers[index], cache._prefix_records[index]
            tensors.extend((layer.conv_states, layer.recurrent_states))
            tensors.extend(getattr(record, name) for name in _RECORDS)
            context = record.prefix_context
            if context is None:
                bindings.append(None)
            else:
                owners = context.owned_tensors
                tensors.extend(owners)
                bindings.append((context.binding_spec, len(owners)))
        # These schemas contain no tensor owners. Each launch binds the actual
        # leased outputs, including after an entry with the same C is recaptured.
        self._prefix_bindings[count] = tuple(bindings)
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
    def launch(self, *, state_sources=None, **kwargs):
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
        if state_sources is not None and len(state_sources) != len(lengths):
            raise ValueError('verification graph requires one committed source per sequence')
        for index in self._linear:
            layer = cache.layers[index]
            if not layer.has_previous_state:
                raise ValueError('verification graph requires committed recurrent state')
            if state_sources is None:
                inputs.extend((layer.conv_states, layer.recurrent_states))
            else:
                for name in ('conv_states', 'recurrent_states'):
                    inputs.extend(getattr(source.layers[index], name) for source in state_sources)
        with self._graphs.launch(*inputs) as values:
            hidden, taps = values[0], values[1:1 + len(self._capture_layers)]
            _, records, sources, _ = self._bind_outputs(values, len(lengths))
            cache._prefix_records = records
            # Prefix finalization consumes the leased records, not a copied full
            # block. Full-block commits detach these states before publication.
            states = iter(sources)
            for index in self._linear:
                layer = cache.layers[index]
                layer.conv_states, layer.recurrent_states = next(states), next(states)
            cache._borrowed_recurrent_state = True
            yield _TextModelOutput(hidden, cache, tuple(taps))

    def _bind_outputs(self, values, count):
        # The graph owns one immutable output tuple per entry. Retaining it
        # prevents id reuse and binds opaque contexts only once per capture.
        key = id(values)
        entry = self._bound_outputs.get(key)
        if entry is None:
            if len(self._bound_outputs) >= self._runtime.max_batch_size:
                _, (_, retired, _, _) = self._bound_outputs.popitem(last=False)
                finalizer = self._finalizers.pop(id(retired), None)
                if finalizer is not None:
                    finalizer[1].shutdown()
            remaining = iter(values[1 + len(self._capture_layers):])
            records, sources = {}, []
            for index, binding in zip(self._linear, self._prefix_bindings[count], strict=True):
                sources.extend((next(remaining), next(remaining)))
                record_values = tuple(next(remaining) for _ in _RECORDS)
                context = None
                if binding is not None:
                    spec, size = binding
                    context = spec.bind(tuple(next(remaining) for _ in range(size)))
                records[index] = _RecurrentPrefixRecord(
                    self._text.layers[index].linear_attn, *record_values, context)
            # Retain the topology's private canonical offsets as well as its
            # public tensor when a same-count capture replaces the warmup layout.
            entry = (values, records, tuple(sources), self._layouts.get(count))
            self._bound_outputs[key] = entry
        self._bound_outputs.move_to_end(key)
        return entry

    @contextmanager
    def finalize_prefixes(self, records, accepted_lengths):
        """Bind finalization to the producing entry, not merely its batch size."""
        key = id(records)
        entry = self._finalizers.get(key)
        if entry is None:
            if len(self._finalizers) >= self._runtime.max_batch_size:
                _, oldest = self._finalizers.popitem(last=False)
                oldest[1].shutdown()
            indices = tuple(sorted(records))
            contexts = tuple(records[index].prefix_context for index in indices)
            def forward(lengths):
                return _finalize_recurrent_prefixes(records, lengths)

            graph = FixedShapeSinglePassGraph(
                enabled=all(context.supports_graph_capture for context in contexts),
                device=self._runtime.device,
                stream=self._finalize_stream, run_forward=forward, max_entries=1)
            entry = (records, graph)
            self._finalizers[key] = entry
        self._finalizers.move_to_end(key)
        with entry[1].launch(accepted_lengths) as outputs:
            yield outputs

    def shutdown(self):
        try:
            self._graphs.shutdown()
        finally:
            for _, graph in self._finalizers.values():
                graph.shutdown()
            self._finalizers.clear()
            self._bound_outputs.clear()
            self._layouts.clear()
            self._prefix_bindings.clear()
