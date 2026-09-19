"""CPU ownership model for finalizers bound to leased target-entry tensors."""

from collections import OrderedDict
from contextlib import contextmanager
from asyncio import CancelledError
from types import SimpleNamespace
import weakref

import pytest
import torch

from kestrel.models.qwen35 import spec_target_graph


def _record(context, template):
    count = template.shape[0]
    return SimpleNamespace(prefix_context=context, initial_state=template,
        conv_input=torch.arange(count * 2 * 19, dtype=template.dtype).reshape(1, 2, count * 19),
        module=SimpleNamespace(conv_kernel_size=4))


class _Graph:
    """Exercise host cache/closure ownership without pretending to test CUDA."""

    instances = []

    def __init__(self, *, run_forward, **kwargs):
        self.forward = run_forward
        self.enabled = kwargs["enabled"]
        self.closed = False
        self.calls = 0
        self.instances.append(self)

    @contextmanager
    def launch(self, *inputs):
        assert not self.closed
        self.calls += 1
        yield self.forward(*inputs)

    def shutdown(self):
        self.closed = True
        self.forward = None


@pytest.mark.parametrize("device,capture", [
    ("cpu", False),
    pytest.param("cuda", True, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required")),
])
def test_row_output_lease_reuse_and_cancellation(monkeypatch, device, capture):
    graph = object.__new__(spec_target_graph.Qwen35TargetGraph)
    graph._runtime = SimpleNamespace(max_batch_size=1, device=torch.device(device),
        _compute_stream=torch.cuda.Stream(device=device) if capture else None)
    graph._finalizers, graph._bound_outputs = OrderedDict(), OrderedDict()
    graph._layouts, graph._prefix_bindings = {}, {}
    graph._graphs = SimpleNamespace(shutdown=lambda: None)
    template = torch.zeros(8, 2, 3, 3, device=device)
    records = {i: _record(SimpleNamespace(supports_graph_capture=capture), template)
               for i in range(48)}
    for record in records.values():
        record.conv_input = record.conv_input.to(device)

    def finalize(contexts, lengths, *, out_states):
        for state in out_states:
            state.copy_(lengths[:, None, None, None].expand_as(state))

    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    lengths = torch.arange(1, 9, device=device, dtype=torch.int32)
    try:
        with graph.finalize_prefixes(records, lengths) as first:
            assert len(first) == 768
            identities = tuple(map(id, first))
            owned = tuple(value.clone() for value in first)
            assert torch.all(first[0] == 1) and torch.all(first[14] == 8)
            assert first[0].untyped_storage().data_ptr() == first[2].untyped_storage().data_ptr()
            assert first[0].data_ptr() != first[2].data_ptr()
        with pytest.raises(CancelledError):
            with graph.finalize_prefixes(records, lengths.flip(0)) as second:
                assert torch.all(second[0] == 8) and torch.all(second[14] == 1)
                if capture:
                    assert tuple(map(id, second)) == identities
                else:
                    assert not torch._C._overlaps(first[0], second[0])
                raise CancelledError("consumer cancelled")
        with graph.finalize_prefixes(records, lengths) as third:
            assert all(torch.equal(left, right) for left, right in zip(third, owned))
            assert torch.count_nonzero(template) == 0
    finally:
        graph.shutdown()


@pytest.mark.parametrize("capture", (False, True))
def test_same_shape_target_recapture_rebinds_finalizer_and_releases_evicted_owners(monkeypatch, capture):
    _Graph.instances = []
    monkeypatch.setattr(spec_target_graph, "FixedShapeSinglePassGraph", _Graph)
    calls = []

    def finalize(contexts, lengths, *, out_states):
        calls.append(tuple(context.owned_tensors[0].data_ptr() for context in contexts))
        for context, output in zip(contexts, out_states):
            output.copy_(context.owned_tensors[0] + lengths[:, None, None, None])

    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    graph = object.__new__(spec_target_graph.Qwen35TargetGraph)
    graph._runtime = SimpleNamespace(max_batch_size=1, device=torch.device("cpu"),
                                     _compute_stream=None)
    graph._finalizers = OrderedDict()
    graph._bound_outputs = OrderedDict()
    graph._graphs = SimpleNamespace(shutdown=lambda: None)
    graph._layouts = {}
    graph._prefix_bindings = {}
    templates = (torch.zeros((2, 1, 2, 2)),) * 2
    lengths = torch.tensor([1, 16], dtype=torch.int32)
    first = tuple(SimpleNamespace(owned_tensors=(torch.full_like(template, 10 + index),),
                                  supports_graph_capture=capture)
                  for index, template in enumerate(templates))
    references = tuple(weakref.ref(context.owned_tensors[0]) for context in first)
    pointers = tuple(context.owned_tensors[0].data_ptr() for context in first)
    first_records = {index: _record(context, template)
                     for index, (context, template) in enumerate(zip(first, templates))}
    history_owners = tuple(weakref.ref(record.conv_input) for record in first_records.values())
    with graph.finalize_prefixes(first_records, lengths) as outputs:
        assert torch.all(outputs[0] == 11) and torch.all(outputs[2] == 26)
    del outputs
    with graph.finalize_prefixes(first_records, lengths.flip(0)) as outputs:
        assert torch.all(outputs[4] == 27) and torch.all(outputs[6] == 12)
        assert outputs[1].is_contiguous() and outputs[3].is_contiguous()
    del outputs
    assert len(_Graph.instances) == 1 and _Graph.instances[0].calls == 2
    assert _Graph.instances[0].enabled is capture
    del first, first_records
    # A finalizer keeps its producer tensors alive after the target entry retires.
    assert all(reference() is not None for reference in references)
    assert all(reference() is not None for reference in history_owners)

    # Same C/shape, but a recaptured target owns different buffers. It must not
    # reuse a closure bound to the retired target's tensors.
    second = tuple(SimpleNamespace(owned_tensors=(torch.full_like(template, 50 + index),),
                                   supports_graph_capture=capture)
                   for index, template in enumerate(templates))
    assert all(context.owned_tensors[0].data_ptr() != pointer
               for context, pointer in zip(second, pointers))
    second_records = {index: _record(context, template)
                      for index, (context, template) in enumerate(zip(second, templates))}
    with graph.finalize_prefixes(second_records, lengths) as outputs:
        assert torch.all(outputs[0] == 51) and torch.all(outputs[6] == 67)
    del outputs
    assert len(_Graph.instances) == 2 and _Graph.instances[0].closed
    assert _Graph.instances[1].enabled is capture
    assert calls[:2] == [pointers, pointers]
    assert calls[2] == tuple(context.owned_tensors[0].data_ptr() for context in second)
    assert all(reference() is None for reference in references)
    assert all(reference() is None for reference in history_owners)
    if not capture:
        snapshots = tuple(context.owned_tensors[0].clone() for context in second)

        def fail(contexts, lengths, *, out_states):
            out_states[0].fill_(999)
            raise RuntimeError("eager finalizer failed after first layer")

        monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
            gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=fail)))
        with pytest.raises(RuntimeError, match="eager finalizer failed"):
            with graph.finalize_prefixes(second_records, lengths):
                raise AssertionError("partially finalized outputs must not escape")
        for context, before in zip(second, snapshots):
            assert torch.equal(context.owned_tensors[0], before)
        del context
    remaining = tuple(weakref.ref(context.owned_tensors[0]) for context in second)
    del second, second_records
    graph.shutdown()
    assert _Graph.instances[1].closed
    assert not graph._finalizers
    assert all(reference() is None for reference in remaining)


def test_bound_target_entry_reuses_records_and_evicts_matching_finalizer(monkeypatch):
    _Graph.instances = []
    monkeypatch.setattr(spec_target_graph, "FixedShapeSinglePassGraph", _Graph)
    bound, constructed = [], []
    original_record = spec_target_graph._RecurrentPrefixRecord

    def record(*args):
        constructed.append(1)
        return original_record(*args)

    def bind(owners):
        bound.append(owners[0].data_ptr())
        return SimpleNamespace(owned_tensors=owners, supports_graph_capture=True)

    def finalize(contexts, lengths, *, out_states):
        out_states[0].copy_(contexts[0].owned_tensors[0])

    monkeypatch.setattr(spec_target_graph, "_RecurrentPrefixRecord", record)
    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    graph = object.__new__(spec_target_graph.Qwen35TargetGraph)
    graph._runtime = SimpleNamespace(max_batch_size=1, device=torch.device("cpu"),
                                     _compute_stream=None)
    graph._finalizers = OrderedDict()
    graph._bound_outputs = OrderedDict()
    graph._capture_layers = ()
    graph._linear = (0,)
    graph._text = SimpleNamespace(layers=[SimpleNamespace(linear_attn=SimpleNamespace(conv_kernel_size=4))])
    graph._prefix_bindings = {2: ((SimpleNamespace(bind=bind), 1),)}
    graph._graphs = SimpleNamespace(shutdown=lambda: None)
    graph._layouts = {}

    def outputs(value):
        hidden = torch.zeros((1, 32, 2))
        state = torch.zeros((2, 1, 2, 2))
        return (hidden, torch.zeros((2, 2, 4)), state,
                hidden, hidden, hidden, torch.zeros((1, 2, 38)),
                state.clone(), torch.arange(2), torch.full_like(state, value))

    first = outputs(7)
    initial_owner = weakref.ref(first[-1])
    entry = graph._bind_outputs(first, 2)
    assert graph._bind_outputs(first, 2) is entry
    assert len(bound) == len(constructed) == 1
    with graph.finalize_prefixes(entry[1], torch.tensor([1, 16], dtype=torch.int32)) as result:
        assert torch.all(result[0] == 7)
    del result, first, entry
    assert initial_owner() is not None
    second = outputs(19)
    replacement = graph._bind_outputs(second, 2)
    assert len(bound) == len(constructed) == 2
    assert _Graph.instances[0].closed and not graph._finalizers
    assert initial_owner() is None
    assert graph._bind_outputs(second, 2) is replacement
    assert len(bound) == len(constructed) == 2
    with graph.finalize_prefixes(replacement[1], torch.tensor([16, 3], dtype=torch.int32)) as result:
        assert torch.all(result[0] == 19)
    del result
    graph.shutdown()
    assert _Graph.instances[1].closed and not graph._bound_outputs
