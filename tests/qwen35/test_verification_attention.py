"""Model scope and execution-cache ownership without device implementation mocks leaking into dispatch."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import weakref

import pytest
import torch
from kestrel.models.qwen35.verification_attention import VerificationAttention
from kestrel.models.qwen35.verification_linear import NativeVerificationProjections


class Attention(torch.nn.Module, VerificationAttention):
    def __init__(self, index):
        super().__init__()
        self.layer_idx = index


@pytest.fixture
def fake_native(monkeypatch):
    class Layout:
        @classmethod
        def bind_graph_inputs(cls, geometry, **kwargs):
            result = cls()
            result.geometry = geometry
            result.inputs = kwargs
            return result
    class Plan:
        def __init__(self, q, k, v, layout, **kwargs):
            self.q, self.layout = q, layout
            self.workspace = kwargs["workspace"]
        def __call__(self):
            return self.q.clone()
    class Workspace:
        def __init__(self, q, **kwargs):
            self.shape = tuple(q.shape)
    monkeypatch.setitem(sys.modules, 'kestrel_kernels.flash_attn.cute.multiquery',
                        SimpleNamespace(PreparedMultiqueryLayout=Layout, PreparedMultiqueryAttention=Plan, MultiqueryAttentionWorkspace=Workspace))
    monkeypatch.setitem(sys.modules, 'kestrel_kernels.verification_dense', SimpleNamespace(PreparedDense=None))
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)


def test_attention_scope_and_cache_own_bounded_plans(fake_native):
    a, b = Attention(0), Attention(1)
    owner = NativeVerificationProjections([a, b], attention_splits=8)
    query = torch.randn(1, 2, 3, 64)
    kv = SimpleNamespace(k_cache=torch.randn(8, 1, 1, 64), v_cache=torch.randn(8, 1, 1, 64))
    cache = SimpleNamespace(verification_attention_context=None)
    args = dict(cache=cache, paged_kv_layer=kv, page_table=torch.zeros(2, 8, dtype=torch.int32),
                used_k=torch.zeros(2, dtype=torch.int32), sequence_lengths=(1, 2), scaling=.125)
    assert a.verification_attention(query, **args) is None
    with owner.activate():
        torch.testing.assert_close(a.verification_attention(query, **args), query.transpose(1, 2))
        first = cache.verification_attention_context
        a.verification_attention(query + 1, **args)
        b.verification_attention(query, **args)
        assert len(first.plans) == 2
        assert len(first.workspaces) == 1
        assert first.plans[0][1].workspace is first.plans[1][1].workspace
        assert all(plan.layout is first.layout for _, plan in first.plans.values())
        old = weakref.ref(first)
        del first
        args['page_table'] = args['page_table'].clone()
        a.verification_attention(query, **args)
        assert old() is None
        assert len(cache.verification_attention_context.plans) == 1
    owner.close()
    assert a._verification_scope is b._verification_scope is None
    rebound = NativeVerificationProjections([a, b], attention_splits=4)
    rebound.close()


def test_other_model_scope_cannot_activate_attention(fake_native):
    a, b = Attention(0), Attention(0)
    sa = NativeVerificationProjections([a], attention_splits=8)
    sb = NativeVerificationProjections([b], attention_splits=8)
    with sa.activate():
        assert b.verification_attention(None, cache=None, paged_kv_layer=None,
            page_table=None, used_k=None, sequence_lengths=None, scaling=1) is None
    sa.close(); sb.close()
