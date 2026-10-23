import sys
from types import SimpleNamespace
import pytest
import torch
from kestrel.models.qwen35.verification_linear import (
    NativeVerificationProjections, VerificationLinear,
)


@pytest.fixture
def fake_plans(monkeypatch):
    class Plan:
        def __init__(self,weight): self.weight=weight
        def __call__(self,x): return torch.nn.functional.linear(x,self.weight)+1
    monkeypatch.setitem(sys.modules,'kestrel_kernels.verification_dense',SimpleNamespace(PreparedDense=Plan))
    return Plan


def test_scopes_are_model_owned_and_exception_safe(fake_plans):
    a,b=VerificationLinear(4,3,bias=False),VerificationLinear(4,3,bias=False)
    sa,sb=NativeVerificationProjections([a]),NativeVerificationProjections([b])
    x=torch.ones(2,4);ra,rb=a(x),b(x)
    with pytest.raises(RuntimeError):
        with sa.activate():
            torch.testing.assert_close(a(x),ra+1)
            torch.testing.assert_close(b(x),rb)
            with sb.activate():
                torch.testing.assert_close(a(x),ra+1)
                torch.testing.assert_close(b(x),rb+1)
            torch.testing.assert_close(b(x),rb)
            raise RuntimeError('scope failure')
    torch.testing.assert_close(a(x),ra)
    torch.testing.assert_close(b(x),rb)


def test_prepare_rejects_rebinding_owner(fake_plans):
    module=VerificationLinear(4,3,bias=False)
    NativeVerificationProjections([module])
    with pytest.raises(ValueError,match='already belong'):
        NativeVerificationProjections([module])


def test_inventory_failure_does_not_publish_partial_bindings(monkeypatch):
    count=0
    def prepare(weight):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('missing inventory')
        return object()
    monkeypatch.setitem(sys.modules,'kestrel_kernels.verification_dense',SimpleNamespace(PreparedDense=prepare))
    a,b=VerificationLinear(4,3,bias=False),VerificationLinear(4,3,bias=False)
    with pytest.raises(RuntimeError,match='missing inventory'):
        NativeVerificationProjections([a,b])
    assert a._verification_scope is b._verification_scope is None


def test_close_releases_weights_and_permits_rebinding(fake_plans):
    module=VerificationLinear(4,3,bias=False)
    owner=NativeVerificationProjections([module])
    owner.close();owner.close()
    assert module._verification_scope is None
    assert owner.plans == ()
    assert not hasattr(module,'_verification_projection')
    second=NativeVerificationProjections([module])
    assert module._verification_scope is second
    with pytest.raises(RuntimeError,match='closed'):
        with owner.activate(): pass
    second.close()


def test_close_never_unbinds_newer_owner(fake_plans):
    module=VerificationLinear(4,3,bias=False)
    owner=NativeVerificationProjections([module])
    newer=object()
    module._verification_scope=newer
    owner.close()
    assert module._verification_scope is newer


def test_decoder_constructor_failure_releases_projection_owner(fake_plans,monkeypatch):
    from kestrel.models.qwen35 import spec_decoder, spec_target_graph
    text=torch.nn.Module()
    text.proj=VerificationLinear(4,4,bias=False)
    text.config=SimpleNamespace(hidden_size=4,num_hidden_layers=1,vocab_size=16)
    model=SimpleNamespace(model=SimpleNamespace(language_model=text),lm_head=VerificationLinear(4,16,bias=False))
    runtime=SimpleNamespace(device=torch.device('cuda'),dtype=torch.bfloat16,model=model,
        _cfg=SimpleNamespace(enable_cuda_graphs=True,verification_path='native',verification_attention_splits=None, verification_attention_implementation="simt"),
        decode_path='generated',max_batch_size=1)
    draft=SimpleNamespace(config=SimpleNamespace(hidden_size=4,target_layer_ids=(0,),
        mask_token_id=0,vocab_size=16,selector_rank=0,block_size=4))
    monkeypatch.setattr(spec_decoder,'load_dflash_drafter',lambda *a,**k:draft)
    monkeypatch.setattr(torch.cuda,'Stream',lambda **k:SimpleNamespace(synchronize=lambda:None))
    monkeypatch.setattr(torch.cuda,'Event',lambda **k:object())
    def fail(*a,**k):raise RuntimeError('capture setup failed')
    monkeypatch.setattr(spec_target_graph,'Qwen35TargetGraph',fail)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='capture setup failed'):
            spec_decoder.Qwen35DFlashDecoder(runtime,'draft')
        assert text.proj._verification_scope is None
        assert model.lm_head._verification_scope is None
