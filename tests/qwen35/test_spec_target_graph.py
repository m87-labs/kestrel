from contextlib import contextmanager
from dataclasses import dataclass
from collections import OrderedDict
from types import SimpleNamespace
import weakref

import torch
import pytest

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topology_owners_survive_capture_replacement_and_retirement():
    from kestrel_kernels import runtime as runtime_api

    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=2, _paged_kv=(None,))
    target._text = SimpleNamespace(config=SimpleNamespace(layer_types=("linear_attention",)))
    target._block_size = 16
    target._layouts, target._prefix_bindings = {}, {(2, 16): ()}
    target._bound_outputs, target._finalizers = OrderedDict(), OrderedDict()
    target._capture_layers, target._linear = (), ()
    owners = []

    def forward(value):
        _, lengths, offsets, token = target._layout(2, device)
        authority = runtime_api._resolve_packed_prefill_topology_authority(token, offsets, lengths)
        canonical = runtime_api._prepare_packed_prefill_topology_launch(authority, value)
        # No peer-stream registration should be necessary for capture metadata.
        assert authority.allocation_stream == torch.cuda.current_stream(device)
        return value + canonical.sum(), offsets

    target._graphs = FixedShapeSinglePassGraph(
        enabled=True, device=device, stream=stream, run_forward=forward, max_entries=2)
    try:
        for index, size in enumerate((1, 2, 1, 3, 2)):
            value = torch.ones(size, device=device)
            with target._graphs.launch(value) as outputs:
                binding = target._bind_outputs(outputs, 2)
                owners.append(weakref.ref(binding[3][1][3]))
                torch.testing.assert_close(outputs[0], torch.full_like(value, 49))
            del binding, outputs
            if index == 1:
                assert owners[0]() is not None
                assert owners[0]() is not owners[1]()
            if index == 2:
                assert owners[0]() is owners[2]()
            if index == 3:
                # Retiring a graph returns its stream to the pool. A new graph
                # on that same stream may reuse its still-valid topology.
                assert owners[1]() is None or owners[1]() is owners[3]()
            live = [entry[3][1][3] for entry in target._bound_outputs.values()]
            assert all(ref() is None or any(ref() is token for token in live)
                       for ref in owners)
            del live
        assert owners[4]() is not None
    finally:
        target.shutdown()
    torch.cuda.synchronize(device)
    assert all(ref() is None for ref in owners)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_captured_row_packing_matches_packed_inputs_without_mutating_sources(count):
    config = SimpleNamespace(layer_types=("linear_attention", "linear_attention"))
    cache = Qwen35InferenceCache(config=config, paged_kv=(None, None))
    rows = [torch.full((1, 2, 4), float(index)) for index in range(4 * count)]
    seen = []

    @dataclass
    class Workspace:
        owner: torch.Tensor

    class Text:
        layers = [SimpleNamespace(linear_attn=SimpleNamespace(
            _prefill_workspace_cache=SimpleNamespace(workspace=Workspace(torch.zeros(1)))))
            for _ in range(2)]

        def __call__(self, **kwargs):
            packed = kwargs['past_key_values']
            initial = packed._prefix_initial_states
            seen.append(tuple(value.clone() for index, layer in enumerate(packed.layers)
                              for value in (layer.conv_states,
                                  initial[index] if initial is not None else layer.recurrent_states)))
            for index, layer in enumerate(packed.layers):
                state = initial[index] if initial is not None else layer.recurrent_states.clone()
                packed._prefix_records[index] = SimpleNamespace(
                    qkv=torch.zeros(1, 16 * count, 2), a=torch.zeros(1), b=torch.zeros(1),
                    conv_input=torch.zeros(1), initial_state=state,
                    state_indices=torch.arange(count), prefix_context=None)
                layer.conv_states.add_(10)
                layer.recurrent_states.copy_(state + 10)
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 16 * count, 2), layer_hidden_states=())

    graph = object.__new__(Qwen35TargetGraph)
    graph._linear, graph._capture_layers, graph._prefix_bindings = (0, 1), (), {}
    graph._layout = lambda count, device, *, width: (cache, (width,)*count, torch.arange(count + 1) * 16, None)
    graph._text = Text()
    graph._runtime = SimpleNamespace(model=SimpleNamespace(lm_head=lambda hidden: hidden))
    metadata = [torch.zeros(1, 16 * count, dtype=torch.long) for _ in range(7)] + [torch.arange(count)]
    staged = graph._allocate_inputs(tuple(metadata + rows))
    for destination, source in zip(staged, metadata + rows, strict=True):
        destination.copy_(source)
    if count > 1:
        assert staged[9].data_ptr() == staged[8].data_ptr() + rows[0].numel() * rows[0].element_size()
    output = graph._forward(*staged)
    assert torch.equal(output[-1], torch.zeros(1, 16 * count, dtype=torch.long))
    assert cache._prefix_initial_states is None
    assert all(torch.all(value == index) for index, value in enumerate(rows))
    for offset, start in ((0, count), (8, 3 * count)):
        torch.testing.assert_close(output[7 + offset], torch.cat(rows[start:start + count]))
        assert output[2 + offset].data_ptr() != output[7 + offset].data_ptr()
    retained_initials = [output[7 + 8 * index].clone() for index in range(2)]
    packed_inputs = [torch.cat(rows[i:i+count]) for i in range(0, 4 * count, count)]
    packed_initials = [value.clone() for value in packed_inputs[1::2]]
    packed_output = graph._forward(*metadata, *packed_inputs)
    for index, (initial, original) in enumerate(zip(packed_inputs[1::2], packed_initials)):
        if count == 1:
            torch.testing.assert_close(initial, original)
            assert packed_output[2 + 8 * index].data_ptr() != initial.data_ptr()
            assert packed_output[7 + 8 * index] is initial
        else:
            torch.testing.assert_close(initial, original + 10)
            assert packed_output[2 + 8 * index] is initial
    assert all(torch.equal(a, b) for a, b in zip(*seen, strict=True))
    # A later forward replaces workspace owners, not the retained entry snapshots.
    for index, initial in enumerate(retained_initials):
        torch.testing.assert_close(output[7 + 8 * index], initial)


@pytest.mark.parametrize("width", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("retain_prefix", [False, True])
@pytest.mark.parametrize("direct_rows", [False, True])
def test_target_state_is_leased_until_full_commit_detaches(monkeypatch, retain_prefix, direct_rows, width):
    config = SimpleNamespace(layer_types=("linear_attention", "linear_attention"))
    source = Qwen35InferenceCache(config=config, paged_kv=(None, None))
    source.seq_length = 3
    for layer in source.layers:
        layer.conv_states = torch.ones(1, 8, 4, dtype=torch.bfloat16)
        layer.recurrent_states = torch.ones(1, 2, 4, 4, dtype=torch.bfloat16)
        layer.has_previous_state = True
    branch = source.fork_recurrent_state(capture_prefix=True)
    tensors = [torch.zeros(1, width, 8)]
    graph_states = []
    bindings, prefix_owners = [], []
    for index in range(2):
        states = [torch.full_like(branch.layers[index].conv_states, index + 2),
                  torch.full_like(branch.layers[index].recurrent_states, index + 4)]
        graph_states.extend(states)
        tensors.extend(states)
        tensors.extend([torch.zeros(1, width, 8)] * 4)
        tensors.extend((torch.ones(1, 2, 4, 4), torch.zeros(1, dtype=torch.long)))
        if retain_prefix:
            owner = torch.full((3,), index)
            prefix_owners.append(owner)
            tensors.append(owner)
            bindings.append((SimpleNamespace(bind=lambda values: SimpleNamespace(owned_tensors=values)), 1))
        else:
            bindings.append(None)

    @contextmanager
    def launch(*inputs):
        expected = source if direct_rows else branch
        assert all(actual is wanted for actual, wanted in zip(inputs[8:],
            (value for layer in expected.layers
             for value in (layer.conv_states, layer.recurrent_states)), strict=True))
        yield tuple(tensors)
        for value in graph_states:
            value.zero_()

    graph = object.__new__(Qwen35TargetGraph)
    graph._block_size = 16
    graph._capture_layers = ()
    graph._linear = (0, 1)
    graph._text = SimpleNamespace(layers=[SimpleNamespace(linear_attn=object()) for _ in range(2)])
    graph._graphs = SimpleNamespace(launch=launch)
    graph._runtime = SimpleNamespace(max_batch_size=1)
    graph._bound_outputs = OrderedDict()
    graph._layouts = {}
    graph._finalizers = OrderedDict()
    graph._prefix_bindings = {(1, width): tuple(bindings)}
    copies = []
    original = torch._foreach_copy_

    def copy_many(destinations, sources):
        copies.append(len(destinations))
        return original(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", copy_many)
    row = torch.zeros(1, width, dtype=torch.long)
    with graph.launch(state_sources=[source] if direct_rows else None,
                      input_ids=row, position_ids=row, cache_position_ids=row,
                      slot_mapping=row, page_table=row, paged_kv_seqlens_k=torch.tensor([3 + width]),
                      seq_idx=row, gdn_state_indices=torch.zeros(1, dtype=torch.long),
                      past_key_values=branch, sequence_lengths=(width,),
                      gdn_state_indices_allocator_owned=True, capture_layers=()):
        assert copies == []
        assert branch._borrowed_recurrent_state
        if retain_prefix:
            for index, record in branch._prefix_records.items():
                assert record.prefix_context.owned_tensors[0] is prefix_owners[index]
        branch.advance_to(3 + width)
        committed = branch.commit_recurrent_prefix(width)
        assert copies == [4]
        assert not committed._borrowed_recurrent_state
    for index, (old, owned) in enumerate(zip(source.layers, committed.layers)):
        assert torch.all(old.conv_states == 1) and torch.all(old.recurrent_states == 1)
        assert torch.all(owned.conv_states == index + 2)
        assert torch.all(owned.recurrent_states == index + 4)
        assert owned.conv_states.data_ptr() != graph_states[index * 2].data_ptr()
        assert torch.all(branch.layers[index].recurrent_states == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_short_width_topologies_are_separate_bounded_and_replayable():
    from kestrel_kernels import runtime as runtime_api

    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=2, _paged_kv=(None,))
    target._text = SimpleNamespace(config=SimpleNamespace(layer_types=('linear_attention',)))
    target._block_size = 16
    target._layouts, target._prefix_bindings = OrderedDict(), {}
    target._bound_outputs, target._finalizers = OrderedDict(), OrderedDict()
    target._capture_layers, target._linear = (), ()
    owners = []

    def forward(value):
        width = value.numel()
        _, lengths, offsets, token = target._layout(2, device, width=width)
        assert lengths == (width, width)
        target._prefix_bindings[(2, width)] = ()
        authority = runtime_api._resolve_packed_prefill_topology_authority(token, offsets, lengths)
        canonical = runtime_api._prepare_packed_prefill_topology_launch(authority, value)
        return value + canonical.sum(), offsets

    target._graphs = FixedShapeSinglePassGraph(
        enabled=True, device=device, stream=stream, run_forward=forward, max_entries=2)
    try:
        for iteration, width in enumerate((1, 3, 3, 16, 3, 8, 1)):
            if iteration == 2:
                # Evict the preparation cache without retiring the captured
                # graph. Its leased output binding must retain the old owner.
                target._layout(1, device, width=5)
                target._layout(1, device, width=7)
                assert (2, width) not in target._layouts
            value = torch.ones(width, device=device)
            with target._graphs.launch(value) as outputs:
                binding = target._bind_outputs(outputs, 2, width=width)
                owners.append(weakref.ref(binding[3][1][3]))
                torch.testing.assert_close(outputs[0], torch.full_like(value, 1 + 3*width))
                assert binding[3][1][1] == (width, width)
            del binding, outputs
            assert len(target._layouts) <= 2
            assert len(target._bound_outputs) <= 2
    finally:
        target.shutdown()
    torch.cuda.synchronize(device)
    assert all(ref() is None for ref in owners)


@pytest.mark.parametrize('lengths', [(), (0,), (17,), (True,), (1, 2), (1, 1, 1)])
def test_invalid_proposal_shape_rejected_before_staging(lengths):
    target = object.__new__(Qwen35TargetGraph)
    target._block_size = 16
    target._runtime = SimpleNamespace(max_batch_size=2)
    with pytest.raises(ValueError):
        with target.launch(sequence_lengths=lengths):
            pytest.fail('invalid proposal shape was accepted')


@pytest.mark.parametrize("count,state_rows", [(2, 2), (1, 2)])
def test_packed_snapshot_domain_preserves_indexed_gather(count, state_rows):
    cache = Qwen35InferenceCache(config=SimpleNamespace(layer_types=("linear_attention",)), paged_kv=(None,))
    graph = object.__new__(Qwen35TargetGraph)
    graph._linear = (0,)
    graph._capture_layers = ()
    graph._layout = lambda *args, **kwargs: (cache, (2,) * count, None, None)
    state = torch.arange(state_rows, dtype=torch.float32).reshape(state_rows, 1, 1, 1)
    indices = torch.tensor([1, 0]) if count == 2 else torch.tensor([0])
    class Observed(Exception):
        pass
    def text(**kwargs):
        # Nonrow C2 must still let the model gather by the supplied indices.
        assert cache._prefix_initial_states is None
        assert cache.layers[0].recurrent_states is state
        torch.testing.assert_close(state[kwargs['gdn_state_indices']], torch.tensor([1., 0.]).reshape(2, 1, 1, 1))
        raise Observed
    graph._text = text
    metadata = [torch.zeros(1, 2 * count, dtype=torch.long) for _ in range(7)] + [indices]
    if count == 2:
        with pytest.raises(Observed):
            graph._forward(*metadata, torch.zeros(count, 1, 2), state)
    else:
        with pytest.raises(ValueError, match="one state row"):
            graph._forward(*metadata, torch.zeros(count, 1, 2), state)
