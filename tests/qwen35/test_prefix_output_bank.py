from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from kestrel.models.qwen35.prefix_output_bank import PrefixOutputBank, PrefixOutputGeometry, validate_disjoint_outputs


def fixtures(count=2, layers=2):
    class Cache:pass
    caches=[]
    for _ in range(count):
        cache=Cache();cache.layers=[SimpleNamespace(recurrent_states=torch.zeros(1,2,3,3))for _ in range(layers)]
        caches.append(cache)
    templates=tuple(torch.zeros(count,2,3,3)for _ in range(layers))
    return caches,templates


@pytest.mark.parametrize('count',[1,2,8])
@pytest.mark.parametrize('layers',[1,3])
def test_bank_reuse_payload_and_detached_lifetime(count,layers):
    caches,templates=fixtures(count,layers)
    bank=PrefixOutputBank(caches,range(layers),templates)
    assert bank.geometry.payload_bytes==count*layers*2*3*3*4
    with bank.lease(tuple(t for cache in caches for t in (cache.layers[0].recurrent_states,))):
        for state in bank.states:state.fill_(7)
        bank.publish(caches)
    assert bank.matches(caches,range(layers),templates)
    reference=weakref.ref(bank)
    del bank
    assert reference() is not None
    assert torch.all(caches[-1].layers[-1].recurrent_states==7)
    # Cached row views and scalar descriptors are reused, not rebuilt each step.
    retained=reference();views=retained.row_states
    assert retained.matches(caches,range(layers),templates)
    assert retained.row_states is views


def test_exclusive_writer_failure_and_publication_contract():
    caches,templates=fixtures();bank=PrefixOutputBank(caches,(0,1),templates)
    with pytest.raises(RuntimeError,match='publication'):
        bank.publish(caches)
    with bank.lease(templates):
        with pytest.raises(RuntimeError,match='exclusive'):
            with bank.lease(templates):pass
        bank.publish(caches)
    with pytest.raises(RuntimeError,match='submission'):
        with bank.lease(templates):raise RuntimeError('submission failure')
    assert bank.failed and not bank.matches(caches,(0,1),templates)
    with pytest.raises(RuntimeError,match='retired'):
        with bank.lease(templates):pass


def test_stream_switch_refused_before_writer_submission(monkeypatch):
    caches,templates=fixtures();bank=PrefixOutputBank(caches,(0,1),templates)
    bank.geometry=replace(bank.geometry,device=torch.device('cuda:0'))
    stream=[11]
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:SimpleNamespace(cuda_stream=stream[0]))
    with bank.lease(templates):pass
    stream[0]=12
    with pytest.raises(RuntimeError,match='switch writer streams'):
        with bank.lease(templates):raise AssertionError('submitted')
    assert not bank._active


@pytest.mark.parametrize('source_slice',[(0,4),(4,8),(7,10),(12,16)])
@pytest.mark.parametrize('output_slice',[(0,4),(4,8),(8,12),(12,16)])
def test_interval_audit_matches_independent_address_sets(source_slice,output_slice):
    storage=torch.empty(16)
    source=storage[slice(*source_slice)];out=storage[slice(*output_slice)]
    overlap=bool(set(range(*source_slice))&set(range(*output_slice)))
    if overlap:
        with pytest.raises(ValueError,match='overlaps'):validate_disjoint_outputs((out,),(source,))
    else:validate_disjoint_outputs((out,),(source,))


def test_cross_layer_and_strided_holes_are_conservative():
    storage=torch.empty(32)
    with pytest.raises(ValueError,match='overlaps'):
        validate_disjoint_outputs((storage[1:2],torch.empty(2)),(torch.empty(2),storage[::2]))
    validate_disjoint_outputs((storage[16:],),(storage[:16],torch.empty(3)))


def test_returned_state_views_require_bound_storage():
    caches,templates=fixtures();bank=PrefixOutputBank(caches,(0,1),templates)
    rows=tuple(t for layer in bank.row_states for t in layer)
    bank.validate_returned_states(rows)
    for bad in (rows[::-1],tuple(t.clone()for t in rows),rows[:-1]):
        with pytest.raises(ValueError,match='finalizer'):bank.validate_returned_states(bad)


def test_geometry_refuses_empty_cohort_and_layout_mismatch():
    for templates in ((),(torch.empty(0,2,3,3),),(torch.empty(2,3),),
                      (torch.empty(2,2,3,3).transpose(2,3),),
                      (torch.empty(1,2,3,3),torch.empty(2,2,3,3))):
        with pytest.raises(ValueError):PrefixOutputGeometry.from_templates(templates)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('count',[1,2,4,8])
def test_fixed_finalizer_outputs_replay_alternate_and_survive_graph_shutdown(count,monkeypatch):
    from kestrel.models.qwen35.cache import _finalize_recurrent_prefixes
    from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph
    device=torch.device('cuda:0')
    stream=torch.cuda.Stream(device=device)
    caches,unused=fixtures(count,1)
    initial=torch.ones(count,2,3,3,device=device,dtype=torch.bfloat16)
    history=torch.arange(count*2*11,device=device,dtype=torch.bfloat16).reshape(1,2,count*11)
    record=SimpleNamespace(initial_state=initial,conv_input=history,qkv=torch.empty(1,count*8,2,device=device),
                           module=SimpleNamespace(conv_kernel_size=4),prefix_context=object())
    def finalize(contexts,lengths,*,out_states):
        # Independent fixture math exercises fixed external destination ownership;
        # the installed full-model gate supplies actual CuTe recurrence coverage.
        out_states[0].copy_(initial+lengths[:,None,None,None])
    monkeypatch.setattr('kestrel_kernels.get_runtime',lambda:SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    banks=[];graphs=[]
    for unused in range(2):
        owners,_=fixtures(count,1)
        bank=PrefixOutputBank(owners,(0,),(initial,));banks.append((bank,owners))
        graph=FixedShapeSinglePassGraph(enabled=True,device=device,stream=stream,
            run_forward=lambda lengths,bank=bank:_finalize_recurrent_prefixes({0:record},lengths,out_states=bank.states),max_entries=1)
        graphs.append(graph)
    observed=[]
    stream.wait_stream(torch.cuda.current_stream(device))
    try:
        with torch.cuda.stream(stream):
            for step in range(6):
                bank,owners=banks[step%2]
                lengths=torch.tensor([(row+step)%8+1 for row in range(count)],device=device,dtype=torch.int32)
                with bank.lease((initial,history)):
                    with graphs[step%2].launch(lengths) as outputs:
                        bank.validate_returned_states(outputs[::2])
                        bank.publish(owners)
                        observed.append((torch.cat([owner.layers[0].recurrent_states for owner in owners]).clone(),
                                         (initial+lengths[:,None,None,None]).clone()))
            stream.synchronize()
        for actual,expected in observed:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        other=torch.cuda.Stream(device=device)
        with torch.cuda.stream(other),pytest.raises(RuntimeError,match='switch writer streams'):
            with banks[0][0].lease((initial,)):pass
    finally:
        for graph in graphs:graph.shutdown()
    # Cache row views still own their explicit bank after all graph owners retire.
    for bank,owners in banks:
        assert bank.matches(owners,(0,),(initial,))
        assert all(torch.isfinite(owner.layers[0].recurrent_states).all() for owner in owners)


def test_graph_retained_bank_budget_churn_and_retirement():
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=8)
    target._finalizers = OrderedDict()
    class Graph:
        def __init__(self): self.closed = 0
        def shutdown(self): self.closed += 1
    retained_views, graphs, live_owners = [], [], []
    # Keep old owners alive so pruning cannot accidentally rely on garbage collection.
    for serial, count in enumerate([8] * 20 + [1, 4, 2, 8, 3, 8]):
        caches, templates = fixtures(count)
        bank = PrefixOutputBank(caches, (0, 1), templates, allocate=False)
        target.prepare_output_bank(bank)
        assert bank.states is None  # eviction precedes the new allocation
        assert target._output_bank_budget == 2 * 8 * 2 * 2 * 3 * 3 * 4
        existing = {id(entry[2]): entry[2] for entry in target._finalizers.values()}
        assert sum(b.geometry.payload_bytes for b in existing.values()) + bank.geometry.payload_bytes <= target._output_bank_budget
        bank.materialize()
        with bank.lease(templates):
            for state in bank.states: state.fill_(serial)
            bank.publish(caches)
        graph = Graph(); graphs.append(graph)
        target._finalizers[(serial, serial)] = (object(), graph, bank)
        retained_views.append((caches[0].layers[0].recurrent_states, serial))
        live_owners.append(caches)
    assert any(graph.closed for graph in graphs)
    assert all(graph.closed <= 1 for graph in graphs)
    # Removing graph closures must not invalidate independently retained cache views.
    for view, value in retained_views: assert torch.all(view == value)
    target.retire_output_banks(live_owners[-1])
    assert bank.failed and not any(entry[2] is bank for entry in target._finalizers.values())
    assert torch.all(retained_views[-1][0] == retained_views[-1][1])


def test_bank_budget_rejects_changed_geometry_before_allocation():
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=8)
    target._finalizers = OrderedDict()
    caches, templates = fixtures(8)
    target.prepare_output_bank(PrefixOutputBank(caches, (0, 1), templates, allocate=False))
    smaller, small_templates = fixtures(1)
    target.prepare_output_bank(PrefixOutputBank(smaller, (0, 1), small_templates, allocate=False))
    original_budget = target._output_bank_budget
    bad = PrefixOutputBank(smaller, (0, 1), tuple(t.double() for t in small_templates), allocate=False)
    with pytest.raises(ValueError, match='geometry changed'):
        target.prepare_output_bank(bad)
    assert bad.states is None and target._output_bank_budget == original_budget


def test_capacity_one_alternating_banks_reuses_two_graphs(monkeypatch):
    from collections import OrderedDict
    from contextlib import contextmanager
    from kestrel.models.qwen35 import spec_target_graph as module
    target = object.__new__(module.Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=1, device=torch.device('cpu'))
    target._finalize_stream = None
    target._finalizers = OrderedDict()
    created = []
    class Graph:
        def __init__(self, *, run_forward, **kwargs):
            self.forward = run_forward
            self.closed = 0
            created.append(self)
        @contextmanager
        def launch(self, lengths): yield self.forward(lengths)
        def shutdown(self): self.closed += 1
    monkeypatch.setattr(module, 'FixedShapeSinglePassGraph', Graph)
    monkeypatch.setattr(module, '_finalize_recurrent_prefixes',
                        lambda records, lengths, *, out_states: out_states)
    owners, banks = [], []
    for _ in range(2):
        caches, templates = fixtures(1)
        owners.append(caches)
        banks.append(PrefixOutputBank(caches, (0, 1), templates, allocate=False))
    records = {0: SimpleNamespace(prefix_context=SimpleNamespace(supports_graph_capture=True))}
    for step in range(6):
        bank, caches = banks[step % 2], owners[step % 2]
        target.prepare_output_bank(bank)
        bank.materialize()
        with bank.lease(templates):
            with target.finalize_prefixes(records, torch.ones(1, dtype=torch.int32),
                                          out_states=bank.states, output_owner=bank):
                bank.publish(caches)
    assert len(created) == 2 and not any(graph.closed for graph in created)
    assert len(target._finalizers) == 2
    with pytest.raises(RuntimeError, match='reservation'):
        with target.finalize_prefixes(records, torch.ones(1, dtype=torch.int32),
                                      out_states=bank.states, output_owner=bank): pass


def test_retirement_compacts_live_rows_and_releases_private_bank_storage():
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=8)
    target._finalizers = OrderedDict()
    snapshots = []
    for iteration in range(20):
        caches, templates = fixtures(8)
        bank = PrefixOutputBank(caches, (0, 1), templates)
        with bank.lease(templates):
            for state in bank.states: state.fill_(iteration)
            bank.publish(caches)
        snapshots.append(caches[0].layers[0].recurrent_states)
        target.retire_output_banks(caches[1:])
        survivor = caches[0]
        assert survivor._prefix_output_bank is None
        assert bank.states is None and not bank.row_states
        storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                    for tensor in (layer.recurrent_states for layer in survivor.layers)}
        assert sum(storages.values()) == 2 * 1 * 2 * 3 * 3 * 4
        assert all(torch.all(layer.recurrent_states == iteration) for layer in survivor.layers)
    # Snapshots deliberately retained outside the bank remain intact.
    for value, snapshot in enumerate(snapshots): assert torch.all(snapshot == value)


def test_retirement_clone_failure_does_not_publish_partial_replacements(monkeypatch):
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._finalizers = OrderedDict()
    caches, templates = fixtures(8)
    bank = PrefixOutputBank(caches, (0, 1), templates)
    with bank.lease(templates): bank.publish(caches)
    before = tuple(layer.recurrent_states for layer in caches[0].layers)
    original = torch.Tensor.clone
    calls = []
    def fail_second(tensor, **kwargs):
        calls.append(tensor)
        if len(calls) == 2: raise RuntimeError('simulated allocation failure')
        return original(tensor, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'clone', fail_second)
    with pytest.raises(RuntimeError, match='allocation failure'):
        target.retire_output_banks(caches[1:])
    assert all(layer.recurrent_states is tensor for layer, tensor in zip(caches[0].layers, before))
    assert caches[0]._prefix_output_bank is bank and bank.states is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_retirement_compaction_quiesces_writer_and_exposes_complete_survivor():
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._finalizers = OrderedDict()
    target._finalize_stream = torch.cuda.Stream()
    caches, _ = fixtures(8)
    templates = tuple(torch.zeros(8, 2, 3, 3, device='cuda') for _ in range(2))
    bank = PrefixOutputBank(caches, (0, 1), templates)
    with torch.cuda.stream(target._finalize_stream):
        with bank.lease(templates):
            for state in bank.states: state.fill_(23)
            bank.publish(caches)
    snapshot = caches[0].layers[0].recurrent_states
    target.retire_output_banks(caches[1:])
    # No caller wait is necessary: retirement completed its explicit writer stream.
    for layer in caches[0].layers:
        assert torch.equal(layer.recurrent_states, torch.full_like(layer.recurrent_states, 23))
        assert layer.recurrent_states.untyped_storage().nbytes() == layer.recurrent_states.numel() * 4
    assert bank.states is None and not bank.row_states
    assert torch.all(snapshot == 23)


def test_retirement_releases_failed_unpublished_bank_and_reservation():
    from collections import OrderedDict
    from kestrel.models.qwen35.spec_target_graph import Qwen35TargetGraph
    target = object.__new__(Qwen35TargetGraph)
    target._runtime = SimpleNamespace(max_batch_size=8)
    target._finalizers = OrderedDict()
    caches, templates = fixtures(8)
    bank = PrefixOutputBank(caches, (0, 1), templates)
    reference, tensor_reference = weakref.ref(bank), weakref.ref(bank.states[0])
    graph = SimpleNamespace(shutdown=lambda: None)
    target._finalizers[(1, 2)] = (object(), graph, bank)
    target._prepared_output_bank = bank
    with pytest.raises(RuntimeError, match='finalizer failed'):
        with bank.lease(templates): raise RuntimeError('finalizer failed before publish')
    del bank
    assert reference() is not None and tensor_reference() is not None
    target.retire_output_banks(caches)
    assert not target._finalizers and target._prepared_output_bank is None
    assert reference() is None and tensor_reference() is None
