from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.draft_owner_cache import DraftOwnerCache


class Owner:
    def __init__(self, size=10):
        self.active = self.closed = self.failed = False
        self.size, self.closes = size, 0

    def known_owned_tensor_bytes(self):
        return self.size

    def shutdown(self):
        assert not self.active
        self.closes += 1
        self.closed = True


def test_reuse_width_capacity_keys_and_exactly_once_close():
    cache = DraftOwnerCache(max_entries=4, known_tensor_bytes=40)
    made = []
    def factory():
        owner = Owner(); made.append(owner); return owner
    owners = []
    for key in ((1,7,128),(2,7,128),(1,7,128),(1,3,128),(1,7,256)):
        with cache.lease(key, 10, factory) as owner: owners.append(owner)
    assert owners[0] is owners[2]
    assert len(made) == 4 and cache.reserved_bytes == 40
    cache.shutdown(); cache.shutdown()
    assert all(o.closes == 1 for o in made)


def test_live_lease_and_active_replay_cannot_be_evicted():
    cache = DraftOwnerCache(max_entries=1, known_tensor_bytes=10)
    with cache.lease('a',10,Owner) as a:
        with pytest.raises(RuntimeError,match='live leases'):
            with cache.lease('b',10,Owner): pass
        with pytest.raises(RuntimeError,match='live leases'): cache.shutdown()
    a.active = True
    with pytest.raises(RuntimeError,match='live leases'):
        with cache.lease('b',10,Owner): pass
    a.active = False
    with cache.lease('b',10,Owner): pass
    assert a.closes == 1
    with pytest.raises(RuntimeError,match='no longer retained'):
        with cache.pin(a): pass
    cache.shutdown()


def test_budget_evicts_before_construction_and_oom_propagates():
    cache = DraftOwnerCache(max_entries=4, known_tensor_bytes=10)
    with cache.lease('a',10,Owner) as a: pass
    def oom():
        assert a.closed
        raise torch.OutOfMemoryError('injected')
    with pytest.raises(torch.OutOfMemoryError):
        with cache.lease('b',10,oom): pass
    assert cache.reserved_bytes == 0 and a.closes == 1
    with pytest.raises(RuntimeError,match='byte budget'):
        with cache.lease('c',11,Owner): pass


def test_capture_growth_fails_closed_and_releases_owner():
    cache = DraftOwnerCache(max_entries=4,known_tensor_bytes=40)
    with pytest.raises(RuntimeError,match='reservation'):
        with cache.lease('a',10,Owner) as owner:
            owner.size=11
            cache.validate_retained_bytes(owner)
    assert owner.failed and owner.closes == 1 and cache.reserved_bytes == 0


def test_initial_construction_overreservation_is_retired():
    cache = DraftOwnerCache(max_entries=4,known_tensor_bytes=40)
    owner=Owner(11)
    with pytest.raises(RuntimeError,match='construction'):
        with cache.lease('a',10,lambda:owner): pass
    assert owner.closes == 1 and cache.reserved_bytes == 0


def test_decoder_cohort_roundtrip_reuses_and_rebinds(monkeypatch):
    from kestrel.models.qwen35.spec_decoder import Qwen35DFlashDecoder
    from kestrel.models.qwen35 import draft_workspace
    made=[]
    class FakeSession(Owner):
        @staticmethod
        def known_tensor_reservation(model,caches,proposal_count,**kwargs): return 10
        def __init__(self,model,caches,**kwargs):
            super().__init__(); self.caches=tuple(caches); self.lengths=tuple(c.length for c in caches)
            self.rebinds=0; made.append(self)
        def rebind(self,caches):
            self.rebinds+=1; self.caches=tuple(caches); self.lengths=tuple(c.length for c in caches)
            self.snapshot=tuple(getattr(c, "content", 0) for c in caches)
        @contextmanager
        def launch(self,*args):
            self.active=True
            try: yield self
            finally: self.active=False
    monkeypatch.setattr(draft_workspace,'DFlashDraftGraphSession',FakeSession)
    decoder=Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    parameter=torch.zeros(1)
    decoder.draft=SimpleNamespace(config=SimpleNamespace(block_size=8),parameters=lambda:iter([parameter]))
    decoder.runtime=SimpleNamespace(model=SimpleNamespace(lm_head=object()))
    decoder._generated_verification=None; decoder._draft_graph_enabled=True; decoder._draft_graph=None
    decoder._draft_owners=DraftOwnerCache(max_entries=4,known_tensor_bytes=40)
    a=SimpleNamespace(capacity=128,length=1,layers=[1]);b=SimpleNamespace(capacity=128,length=1,layers=[1])
    selected=[]
    for caches in ([a],[a,b],[a]):
        with decoder._draft_tokens([], [torch.zeros(1,1,1) for _ in caches], [], caches) as owner:
            selected.append(owner)
        a.length+=1
    assert selected[0] is selected[2] and len(made)==2
    assert selected[0].rebinds==2
    # Another cohort updates the same source cache without changing its length.
    with decoder._draft_tokens([], [torch.zeros(1,1,1)]*2, [], [a,b]): pass
    a.content=123
    with decoder._draft_tokens([], [torch.zeros(1,1,1)], [], [a]) as owner:
        assert owner is selected[0] and owner.snapshot == (123,)
    a.capacity=256
    with decoder._draft_tokens([], [torch.zeros(1,1,1)], [], [a]) as owner:
        assert owner is not selected[0]
    decoder._draft_owners.shutdown()
    assert all(o.closes==1 for o in made)


def test_owned_tensor_accounting_deduplicates_views():
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    value=torch.zeros(20)
    owner=DFlashDraftGraphSession.__new__(DFlashDraftGraphSession)
    owner.targets=value[:10];owner.positions=value[10:]
    owner.workspaces=[SimpleNamespace(keys=value,values=value,scale=value[:1])]
    owner.graph=SimpleNamespace(retained_io_tensors=lambda:(value[2:5],))
    assert owner.known_owned_tensor_bytes()==80


@pytest.mark.parametrize('field', ['max_entries','known_tensor_bytes'])
@pytest.mark.parametrize('value',[True,0,-1,1.5])
def test_invalid_bounds(field,value):
    args=dict(max_entries=4,known_tensor_bytes=40);args[field]=value
    with pytest.raises(ValueError): DraftOwnerCache(**args)


def test_partial_constructor_failure_quiesces_copy_stream(monkeypatch):
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    events=[]; partial=[]
    def fail(self,*args,**kwargs):
        partial.append(self)
        self.workspaces=[object()]
        self._construction_stream=SimpleNamespace(synchronize=lambda:events.append('copy_sync'))
        raise torch.OutOfMemoryError('allocation failed after queued copies')
    monkeypatch.setattr(DFlashDraftGraphSession,'_initialize',fail)
    with pytest.raises(torch.OutOfMemoryError): DFlashDraftGraphSession(None,[])
    assert events==['copy_sync'] and partial[0].workspaces==[]


def test_pin_rejects_failed_owner():
    cache=DraftOwnerCache(max_entries=1,known_tensor_bytes=10)
    with cache.lease('a',10,Owner) as owner: pass
    owner.failed=True
    with pytest.raises(RuntimeError):
        with cache.pin(owner): pass
    cache.shutdown(); assert owner.closes==1


def test_retirement_detaches_borrowed_request_storage():
    cache=DraftOwnerCache(max_entries=2,known_tensor_bytes=20)
    request=object()
    class Bound(Owner):
        def __init__(self): super().__init__(); self.caches=(request,); self.detaches=0
        def release_bindings(self): self.caches=(); self.detaches+=1
    with cache.lease('a',10,Bound) as owner:
        with pytest.raises(RuntimeError,match='leased'): cache.detach_cache_owner(request)
    cache.detach_cache_owner(request)
    assert owner.caches==() and owner.detaches==1 and owner.closes==0
    cache.shutdown(); assert owner.closes==1


def test_decoder_fastpath_failure_clears_alias_and_retires_once():
    from kestrel.models.qwen35.spec_decoder import Qwen35DFlashDecoder
    decoder=Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    decoder._draft_owners=DraftOwnerCache(max_entries=1,known_tensor_bytes=10)
    with decoder._draft_owners.lease('a',10,Owner) as owner: pass
    decoder._draft_graph=owner
    with pytest.raises(RuntimeError,match='replay failure'):
        with decoder._pin_draft_owner(owner):
            owner.failed=True
            raise RuntimeError('replay failure')
    assert decoder._draft_graph is None and owner.closes==1
    decoder._draft_owners.shutdown()
    assert owner.closes==1


def test_reservation_matches_distinct_owned_storage_shapes():
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession, DraftLayerKVWorkspace
    c,b,h,capacity,proposals=2,4,8,16,3
    config=SimpleNamespace(block_size=b,hidden_size=h,target_layer_ids=(1,3),
                           num_key_value_heads=2,head_dim=4)
    model=SimpleNamespace(config=config,layers=[None,None],parameters=lambda:iter([torch.zeros(1,dtype=torch.bfloat16)]))
    caches=[SimpleNamespace(capacity=capacity)]*c
    owner=DFlashDraftGraphSession.__new__(DFlashDraftGraphSession)
    owner.workspaces=[DraftLayerKVWorkspace(slots=c,capacity=capacity,context_rows=b,
        query_rows=b,heads=2,head_dim=4,device='cpu') for _ in model.layers]
    owner.targets=torch.empty(c,b,2*h,dtype=torch.bfloat16)
    owner.positions=torch.empty(c,2*b,dtype=torch.int64)
    graph_tensors=(torch.empty(c,b,h,dtype=torch.bfloat16),torch.empty_like(owner.targets),
        torch.empty_like(owner.positions),torch.empty(c*2*b,dtype=torch.int64),
        torch.empty(c,dtype=torch.int32),torch.empty(c,dtype=torch.int64),
        torch.empty(c,proposals,dtype=torch.int64))
    owner.graph=SimpleNamespace(retained_io_tensors=lambda:graph_tensors)
    assert owner.known_owned_tensor_bytes()==DFlashDraftGraphSession.known_tensor_reservation(model,caches,proposals)


def test_config_default_retains_four_owners_with_explicit_budget():
    from kestrel.config import RuntimeConfig
    cfg=RuntimeConfig(device="cpu")
    assert cfg.draft_graph_cache_entries==4
    assert cfg.draft_graph_known_tensor_bytes==4*1024**3
    with pytest.raises(ValueError,match='cache bounds'):
        RuntimeConfig(draft_graph_cache_entries=True)



def test_overlapping_same_owner_lease_and_pin_preserve_first_user():
    from concurrent.futures import ThreadPoolExecutor
    cache=DraftOwnerCache(max_entries=1,known_tensor_bytes=10)
    with cache.lease('a',10,Owner) as owner:
        def overlapping():
            with pytest.raises(RuntimeError,match='exclusive lease'):
                with cache.lease('a',10,Owner): pass
            with pytest.raises(RuntimeError,match='exclusive lease'):
                with cache.pin(owner): pass
        overlapping()
        with ThreadPoolExecutor(max_workers=1) as pool: pool.submit(overlapping).result()
        assert not owner.failed and not owner.closed
    with cache.pin(owner):
        with pytest.raises(RuntimeError,match='exclusive lease'):
            with cache.pin(owner): pass
        assert not owner.failed
    with cache.lease('a',10,Owner) as same: assert same is owner
    cache.shutdown(); assert owner.closes==1
