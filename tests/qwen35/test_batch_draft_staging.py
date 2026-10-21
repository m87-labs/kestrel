import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('count', [2, 4])
def test_batched_staged_graph_matches_normal_tokens_confidence_and_cache(count):
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    from kestrel.models.qwen35.dflash import DFlashConfig, DFlashDraftModel, DFlashContextCache, _LayerContextBuffer
    torch.manual_seed(20261020)
    config = DFlashConfig(128, 256, 1, 4, 1, 128, 1e-6, 10000, 16, 0, (0,), ('full_attention',), None)
    model = DFlashDraftModel(config).cuda().bfloat16().requires_grad_(False)
    head = torch.nn.Linear(128, 257, bias=False).cuda().bfloat16().requires_grad_(False)
    embedding = torch.randn(257,128,device='cuda',dtype=torch.bfloat16)
    def owners():
        caches=[]
        for slot in range(count):
            cache=DFlashContextCache(256);cache.length=slot+3
            key=torch.full((1,256,1,128),slot+1,device='cuda',dtype=torch.bfloat16)
            cache.layers=[_LayerContextBuffer(key,key.clone())];caches.append(cache)
        return caches
    caches, references=owners(),owners()
    confidence=torch.empty(count,15,device='cuda',dtype=torch.float32)
    expected_confidence=torch.empty_like(confidence)
    direct=DFlashDraftGraphSession(model,caches,lm_head=head,confidence=confidence)
    normal=DFlashDraftGraphSession(model,references,lm_head=head,confidence=expected_confidence)
    try:
        for iteration in range(6):
            lengths=[(0,1,6,16)[(iteration+i)%4] for i in range(count)]
            features=[torch.randn(1,n,128,device='cuda',dtype=torch.bfloat16) for n in lengths]
            bonuses=[7+i+iteration for i in range(count)]
            ids=torch.full((count,16),0,device='cuda',dtype=torch.long)
            ids[:,0]=torch.tensor(bonuses,device='cuda')
            noise=torch.nn.functional.embedding(ids,embedding).split(1)
            positions=[torch.arange(c.length,c.length+n+16,device='cuda')[None] for c,n in zip(caches,lengths)]
            with normal.launch(noise,features,positions,ids[:,0]) as output:
                expected=output.clone()
            lease=(direct.launch(noise,features,positions,ids[:,0]) if iteration==0 else
                   direct.launch_tokens_many(features,embedding,bonuses=bonuses,
                       target_positions=[c.length+n for c,n in zip(caches,lengths)]))
            with lease as actual:
                assert torch.equal(actual,expected)
                assert torch.equal(confidence,expected_confidence)
            for a,b in zip(caches,references):
                assert a.length==b.length
                for x,y in zip(a.layers,b.layers):
                    assert torch.equal(x.keys[:,:a.length],y.keys[:,:b.length])
                    assert torch.equal(x.values[:,:a.length],y.values[:,:b.length])
        starts=[cache.length for cache in caches]
        retained=[(layer.keys.clone(),layer.values.clone()) for cache in caches for layer in cache.layers]
        features=[torch.randn(1,1,128,device='cuda',dtype=torch.bfloat16) for _ in caches]
        positions=[start+1 for start in starts]
        with pytest.raises(ValueError,match='ownership or capacity'):
            with direct.launch_tokens_many(features,embedding,bonuses=[7]*count,
                    target_positions=[*positions[:-1],positions[-1]+1]):
                pass
        assert not direct.failed and not direct.active
        with pytest.raises(RuntimeError,match='consumer failure'):
            with direct.launch_tokens_many(features,embedding,bonuses=[7]*count,target_positions=positions):
                raise RuntimeError('consumer failure')
        assert direct.failed and not direct.active
        assert [cache.length for cache in caches]==starts
        for layer,(keys,values) in zip((layer for cache in caches for layer in cache.layers),retained):
            assert torch.equal(layer.keys,keys) and torch.equal(layer.values,values)
        with pytest.raises(RuntimeError,match='failed or retired'):
            with direct.launch_tokens_many(features,embedding,bonuses=[7]*count,target_positions=positions):
                pass
    finally:
        direct.shutdown();normal.shutdown()
