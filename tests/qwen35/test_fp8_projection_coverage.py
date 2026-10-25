from types import SimpleNamespace
import pytest
import torch
from kestrel.models.qwen35.qwen_config import Qwen3_5TextConfig
from kestrel.models.qwen35.generated_verification import validate_fp8_projection_coverage


def config():
    return Qwen3_5TextConfig(vocab_size=248320,hidden_size=5120,num_hidden_layers=2,
        intermediate_size=17408,num_attention_heads=24,num_key_value_heads=4,
        max_position_embeddings=262144,rms_norm_eps=1e-6,tie_word_embeddings=False,
        rope_theta=10000000,partial_rotary_factor=.25,mrope_section=(11,11,10),head_dim=256,
        linear_conv_kernel_dim=4,linear_key_head_dim=128,linear_value_head_dim=128,
        linear_num_key_heads=16,linear_num_value_heads=48,
        layer_types=('linear_attention','full_attention'),dense_weight_format='fp8_e4m3')


def test_exact_storage_projection_shapes():
    rows=config().block_scaled_projections()
    assert {(r.name,r.n,r.k,r.parts,r.gated_output) for r in rows}=={
        ('recurrent.in',16384,5120,1,False),('recurrent.out',5120,6144,1,False),
        ('attention.qkv',14336,5120,1,False),('attention.out',5120,6144,1,False),
        ('mlp.gate_up',34816,5120,2,True),('mlp.down',5120,17408,1,False)}

@pytest.mark.parametrize('capacity',[1,2,4,8])
def test_all_cohorts_and_prompt_domain_before_weights(monkeypatch,capacity):
    from kestrel_kernels import block_scaled_coverage as provider
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda _: (9,0))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(multi_processor_count=132))
    seen=[]
    def missing(rows,**kwargs):
        seen.append((rows,kwargs));raise RuntimeError('missing exact FP8 artifact')
    monkeypatch.setattr(provider,'require_hopper_projection_coverage',missing)
    runtime=SimpleNamespace(device=torch.device('cuda'),dtype=torch.bfloat16,max_batch_size=capacity)
    with pytest.raises(RuntimeError,match='missing exact'):validate_fp8_projection_coverage(runtime,config())
    assert seen[0][1]=={'max_rows':capacity*262144,'num_sms':132}
    assert len(seen[0][0])==6


def test_loader_checks_fp8_before_constructor_and_weight_lookup(tmp_path,monkeypatch):
    from kestrel.models.qwen35 import qwen_loader
    path=tmp_path/'config.json';path.write_text('{}')
    def resolve(source,name,**kw):
        assert name=='config.json','weights accessed before coverage'
        return path
    monkeypatch.setattr(qwen_loader,'_resolve_checkpoint_file',resolve)
    monkeypatch.setattr(qwen_loader.Qwen3_5Config,'from_dict',lambda _:SimpleNamespace(text_config=config()))
    monkeypatch.setattr(qwen_loader,'Qwen3_5ForConditionalGeneration',lambda *a,**k:pytest.fail('model allocated'))
    def coverage(cfg):raise RuntimeError('missing FP8 coverage')
    with pytest.raises(RuntimeError,match='missing FP8'):
        qwen_loader.load_qwen35_model(tmp_path,device=torch.device('cpu'),dtype=torch.bfloat16,validate_config=coverage)
