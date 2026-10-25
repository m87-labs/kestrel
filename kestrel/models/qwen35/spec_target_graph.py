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
        self._layouts = OrderedDict()
        self._prefix_bindings = {}
        self._bound_outputs = OrderedDict()
        self._finalizers = OrderedDict()
        self._graphs = FixedShapeSinglePassGraph(
            enabled=True, device=runtime.device, stream=runtime._compute_stream,
            run_forward=self._forward, max_entries=runtime.max_batch_size,
            allocate_inputs=self._allocate_inputs)

    def _allocate_inputs(self, inputs):
        count = inputs[_INPUTS.index('gdn_state_indices')].numel()
        prefix = len(_INPUTS)
        row_inputs = len(inputs) != prefix + 2 * len(self._linear)
        if not row_inputs:
            prefix = len(inputs)
        result = [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                  for x in inputs[:prefix]]
        for offset in range(prefix, len(inputs), count):
            rows = inputs[offset:offset + count]
            first = rows[0]
            if (len(rows) != count or first.shape[0] != 1
                    or any(not x.is_contiguous() or x.shape != first.shape
                           or x.dtype != first.dtype or x.device != first.device for x in rows)):
                raise ValueError('packed verification needs matching contiguous state rows')
            owner = first.new_empty((count, *first.shape[1:]))
            result.extend(owner[row:row + 1] for row in range(count))
        return tuple(result)

    def _layout(self, count, device, *, width=None):
        width = self._block_size if width is None else width
        if (type(width) is not int or not 1 <= width <= self._block_size
                or type(count) is not int or not 1 <= count <= self._runtime.max_batch_size):
            raise ValueError("verification shape exceeds sequence or proposal capacity")
        key = (count, width)
        stream = torch.cuda.current_stream(device)
        previous = self._layouts.get(key)
        if previous is None or previous[0] != stream:
            if not 1 <= count <= self._runtime.max_batch_size:
                raise ValueError('verification graph exceeds sequence capacity')
            lengths = (width,) * count
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
            if key not in self._layouts and len(self._layouts) >= self._runtime.max_batch_size:
                self._layouts.pop(next(iter(self._layouts)))
            self._layouts[key] = stream, (cache, lengths, cu, topology)
        # Leased output bindings retain evicted topology owners until replay and
        # prefix finalization finish; this cache only owns reusable preparation.
        value = self._layouts.pop(key)
        self._layouts[key] = value
        return value[1]

    def _forward(self, *values):
        inputs = dict(zip(_INPUTS, values[:len(_INPUTS)], strict=True))
        count = inputs['gdn_state_indices'].numel()
        rows = inputs['input_ids'].shape[1]
        if count < 1 or rows % count:
            raise ValueError('verification rows must form equal proposal spans')
        width = rows // count
        cache, lengths, cu, topology = self._layout(count, inputs['input_ids'].device, width=width)
        cache._prefix_records = {}
        states = iter(values[len(_INPUTS):])
        row_inputs = len(values) != len(_INPUTS) + 2 * len(self._linear)
        cache._prefix_initial_states = {} if row_inputs else None
        if row_inputs:
            # Direct sources are supplied in sequence order, not pool-slot order.
            inputs['gdn_state_indices'] = torch.arange(count, device=inputs['input_ids'].device)
        for index in self._linear:
            layer = cache.layers[index]
            if row_inputs:
                conv_rows = [next(states) for _ in range(count)]
                state_rows = [next(states) for _ in range(count)]
                # The input allocator gives each cohort one contiguous owner;
                # staging writes directly into its rows before graph replay.
                layer.conv_states = conv_rows[0].as_strided(
                    (count, *conv_rows[0].shape[1:]), conv_rows[0].stride())
                initial = state_rows[0].as_strided(
                    (count, *state_rows[0].shape[1:]), state_rows[0].stride())
                cache._prefix_initial_states[index] = initial
                layer.recurrent_states = torch.empty_like(initial)
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
        self._prefix_bindings[(count, width)] = tuple(bindings)
        # Owner-only outputs tie scratch lifetime to the actual graph entry,
        # including when another C replaces the model's workspace cache.
        for index in self._linear:
            workspace = self._text.layers[index].linear_attn._prefill_workspace_cache.workspace
            tensors.extend(value for field in fields(workspace)
                           if isinstance(value := getattr(workspace, field.name), torch.Tensor))
            cache.layers[index].conv_states = cache.layers[index].recurrent_states = None
        cache._prefix_records = {}
        cache._prefix_initial_states = None
        tensors.append(cu)
        tensors.append(self._runtime.model.lm_head(output.last_hidden_state).argmax(-1))
        return tuple(tensors)

    @contextmanager
    def launch(self, *, state_sources=None, **kwargs):
        """Lease verification outputs; direct sources are in sequence order."""
        lengths = tuple(kwargs['sequence_lengths'])
        if (not lengths or any(type(length) is not int or not 1 <= length <= self._block_size
                               for length in lengths) or len(set(lengths)) != 1):
            raise ValueError('verification graph requires equal positive proposal spans within capacity')
        if len(lengths) > self._runtime.max_batch_size:
            raise ValueError('verification graph exceeds sequence capacity')
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
            _, records, sources, _ = self._bind_outputs(values, len(lengths), width=lengths[0])
            cache._prefix_records = records
            # Prefix finalization consumes the leased records, not a copied full
            # block. Full-block commits detach these states before publication.
            states = iter(sources)
            for index in self._linear:
                layer = cache.layers[index]
                layer.conv_states, layer.recurrent_states = next(states), next(states)
            cache._borrowed_recurrent_state = True
            yield _TextModelOutput(hidden, cache, tuple(taps)), values[-1]

    def _bind_outputs(self, values, count, *, width=None):
        width = self._block_size if width is None else width
        shape = (count, width)
        # The graph owns one immutable output tuple per entry. Retaining it
        # prevents id reuse and binds opaque contexts only once per capture.
        key = id(values)
        entry = self._bound_outputs.get(key)
        if entry is None:
            if len(self._bound_outputs) >= self._runtime.max_batch_size:
                _, (_, retired, _, _) = self._bound_outputs.popitem(last=False)
                for finalizer_key in tuple(self._finalizers):
                    if finalizer_key[0] == id(retired):
                        self._finalizers.pop(finalizer_key)[1].shutdown()
            remaining = iter(values[1 + len(self._capture_layers):])
            records, sources = {}, []
            for index, binding in zip(self._linear, self._prefix_bindings[shape], strict=True):
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
            entry = (values, records, tuple(sources), self._layouts.get(shape))
            self._bound_outputs[key] = entry
        self._bound_outputs.move_to_end(key)
        return entry

    def prepare_output_bank(self, bank):
        """Bound graph-retained payload before allocating a new destination bank.

        Live cache views own their storage independently; this budget covers graph
        retention only, not total device memory or allocator overhead.
        """
        if bank is None:
            self._prepared_output_bank = None
            return
        geometry = bank.geometry
        count = geometry.shapes[0][0]
        if count > self._runtime.max_batch_size:
            raise ValueError('prefix output cohort exceeds declared capacity')
        signature = (tuple(shape[1:] for shape in geometry.shapes), geometry.dtypes,
                     geometry.device)
        previous = getattr(self, '_output_bank_signature', None)
        if previous is not None and signature != previous:
            raise ValueError('prefix output geometry changed within target owner')
        self._output_bank_signature = signature
        self._output_bank_budget = 2 * self._runtime.max_batch_size * (geometry.payload_bytes // count)
        for key, entry in tuple(self._finalizers.items()):
            owner = entry[2]
            if owner is not None and not owner.is_current():
                self._finalizers.pop(key)[1].shutdown()
        def retained_bytes():
            owners = {id(entry[2]): entry[2] for entry in self._finalizers.values()
                      if entry[2] is not None}
            owners[id(bank)] = bank
            return sum(owner.geometry.payload_bytes for owner in owners.values())
        while retained_bytes() > self._output_bank_budget:
            _, entry = self._finalizers.popitem(last=False)
            entry[1].shutdown()
        self._prepared_output_bank = bank

    def retire_output_banks(self, caches):
        """Quiesce captured writers, then compact survivors outside the decode hot path.

        All replacement allocations/copies complete before any cache publication.
        External snapshots remain valid and may independently retain old storage.
        """
        caches = tuple(caches)
        self._prepared_output_bank = None
        owners = {id(owner): owner for cache in caches
                  if (owner := getattr(cache, '_prefix_output_bank', None)) is not None}
        for key, entry in tuple(self._finalizers.items()):
            owner = entry[2]
            if owner is not None and (id(owner) in owners or not owner.is_current()):
                self._finalizers.pop(key)[1].shutdown()
                owners[id(owner)] = owner
        replacements = []
        def clone_survivors():
            for owner in owners.values():
                for reference in owner.owners:
                    cache = reference()
                    if (cache is None or any(cache is retired for retired in caches)
                            or getattr(cache, '_prefix_output_bank', None) is not owner):
                        continue
                    states = tuple(cache.layers[index].recurrent_states.clone(
                        memory_format=torch.contiguous_format) for index in owner.indices)
                    replacements.append((cache, owner.indices, states))
        if owners and next(iter(owners.values())).geometry.device.type == 'cuda':
            with torch.cuda.stream(self._finalize_stream):
                clone_survivors()
            self._finalize_stream.synchronize()
        else:
            clone_survivors()
        for cache, indices, states in replacements:
            for index, state in zip(indices, states):
                cache.layers[index].recurrent_states = state
            cache._prefix_output_bank = None
        for owner in owners.values():
            for reference in owner.owners:
                cache = reference()
                if cache is not None and getattr(cache, '_prefix_output_bank', None) is owner:
                    cache._prefix_output_bank = None
            owner.failed = True
            owner.states, owner.row_states, owner.row_descriptors = None, (), ()
        if getattr(self, '_prepared_output_bank', None) in owners.values():
            self._prepared_output_bank = None

    @contextmanager
    def finalize_prefixes(self, records, accepted_lengths, *, out_states=None, output_owner=None):
        """Bind finalization to the producing entry, not merely its batch size."""
        out_states = None if out_states is None else tuple(out_states)
        if output_owner is not None:
            if out_states is None or len(out_states) != len(output_owner.states) or any(
                    actual is not expected for actual, expected in zip(out_states, output_owner.states)):
                raise ValueError('finalizer output tensors differ from prepared owner')
            if getattr(self, '_prepared_output_bank', None) is not output_owner:
                raise RuntimeError('output bank has no current allocation reservation')
            self._prepared_output_bank = None
        key = (id(records), None if out_states is None else tuple(id(t) for t in out_states))
        entry = self._finalizers.get(key)
        if entry is None:
            if len(self._finalizers) >= 2 * self._runtime.max_batch_size:
                _, oldest = self._finalizers.popitem(last=False)
                oldest[1].shutdown()
            indices = tuple(sorted(records))
            contexts = tuple(records[index].prefix_context for index in indices)
            def forward(lengths):
                return _finalize_recurrent_prefixes(records, lengths, out_states=out_states)

            graph = FixedShapeSinglePassGraph(
                enabled=all(context.supports_graph_capture for context in contexts),
                device=self._runtime.device,
                stream=self._finalize_stream, run_forward=forward, max_entries=1)
            entry = (records, graph, output_owner)
            self._finalizers[key] = entry
        self._finalizers.move_to_end(key)
        with entry[1].launch(accepted_lengths) as outputs:
            yield outputs

    def shutdown(self):
        try:
            self._graphs.shutdown()
        finally:
            for _, graph, _ in self._finalizers.values():
                graph.shutdown()
            self._finalizers.clear()
            self._prepared_output_bank = None
            self._bound_outputs.clear()
            self._layouts.clear()
            self._prefix_bindings.clear()
