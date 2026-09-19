from contextlib import contextmanager
from collections import OrderedDict
from types import SimpleNamespace

import torch
import pytest

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph


@pytest.mark.parametrize("retain_prefix", [False, True])
def test_target_state_copies_are_batched_and_remain_owned(monkeypatch, retain_prefix):
    config = SimpleNamespace(layer_types=("linear_attention", "linear_attention"))
    source = Qwen35InferenceCache(config=config, paged_kv=(None, None))
    source.seq_length = 3
    for layer in source.layers:
        layer.conv_states = torch.ones(1, 8, 4, dtype=torch.bfloat16)
        layer.recurrent_states = torch.ones(1, 2, 4, 4, dtype=torch.bfloat16)
        layer.has_previous_state = True
    branch = source.fork_recurrent_state(capture_prefix=True)
    tensors = [torch.zeros(1, 16, 8)]
    graph_states = []
    bindings, prefix_owners = [], []
    for index in range(2):
        states = [torch.full_like(branch.layers[index].conv_states, index + 2),
                  torch.full_like(branch.layers[index].recurrent_states, index + 4)]
        graph_states.extend(states)
        tensors.extend(states)
        tensors.extend([torch.zeros(1, 16, 8)] * 4)
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
    graph._finalizers = OrderedDict()
    graph._prefix_bindings = {1: tuple(bindings)}
    copies = []
    original = torch._foreach_copy_

    def copy_many(destinations, sources):
        copies.append(len(destinations))
        return original(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", copy_many)
    row = torch.zeros(1, 16, dtype=torch.long)
    with graph.launch(input_ids=row, position_ids=row, cache_position_ids=row,
                      slot_mapping=row, page_table=row, paged_kv_seqlens_k=torch.tensor([19]),
                      seq_idx=row, gdn_state_indices=torch.zeros(1, dtype=torch.long),
                      past_key_values=branch, sequence_lengths=(16,),
                      gdn_state_indices_allocator_owned=True, capture_layers=()):
        assert copies == [4]
        if retain_prefix:
            for index, record in branch._prefix_records.items():
                assert record.prefix_context.owned_tensors[0] is prefix_owners[index]
    for index, (old, owned) in enumerate(zip(source.layers, branch.layers)):
        assert torch.all(old.conv_states == 1) and torch.all(old.recurrent_states == 1)
        assert torch.all(owned.conv_states == index + 2)
        assert torch.all(owned.recurrent_states == index + 4)
        assert owned.conv_states.data_ptr() != graph_states[index * 2].data_ptr()
