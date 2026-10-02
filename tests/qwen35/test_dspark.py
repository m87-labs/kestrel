from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.dflash import DFlashConfig, DFlashDraftModel, _VanillaMarkov
from kestrel.ops.rotary import yarn_inv_freq, MultidimensionalRotaryEmbedding


def metadata():
    return dict(architectures=["DSparkDraftModel"], hidden_size=128, intermediate_size=256,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1,
                head_dim=128, rms_norm_eps=1e-6, vocab_size=257, block_size=7,
                dflash_config=dict(mask_token_id=0, target_layer_ids=[0], projector_type="dspark"),
                markov_rank=16, markov_head_type="vanilla", enable_confidence_head=True,
                confidence_head_with_markov=True,
                rope_parameters=dict(rope_type="yarn", rope_theta=10000000, factor=32,
                                     original_max_position_embeddings=8192, beta_fast=32, beta_slow=1))


def test_dspark_separates_query_and_verification_extents():
    config = DFlashConfig.from_dict(metadata())
    assert config.query_rows == 7
    assert config.block_size == 8
    assert config.sample_from_anchor
    assert config.attention(0) == (False, None)
    assert config.markov_rank == 16
    model = DFlashDraftModel(config)
    assert model.confidence_head.proj.weight.shape == (1, 144)
    assert model.markov_head.markov_w2.weight.shape == (257, 16)


def test_dspark_loader_consumes_all_checkpoint_heads(tmp_path):
    import json
    from safetensors.torch import save_file
    from kestrel.models.qwen35.dflash import load_dflash_drafter
    data = metadata()
    model = DFlashDraftModel(DFlashConfig.from_dict(data)).bfloat16()
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(data))
    actual = load_dflash_drafter(tmp_path, device=torch.device("cpu"))
    assert actual.config.query_rows == 7
    for name, value in model.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value)
    assert all(not p.requires_grad for p in actual.parameters())
    assert actual.rotary_emb.inv_freq.dtype == torch.float32


def test_dspark_reuses_existing_generated_verification(monkeypatch):
    from kestrel.models.qwen35.generated_verification import Qwen35GeneratedVerification
    data = metadata()
    data.update(hidden_size=5120, intermediate_size=17408, vocab_size=248320,
                num_target_layers=64)
    data["dflash_config"]["target_layer_ids"] = [5, 19, 33, 47, 61]
    draft = SimpleNamespace(config=DFlashConfig.from_dict(data))
    target = SimpleNamespace(hidden_size=5120, num_hidden_layers=64,
                             intermediate_size=17408, vocab_size=248320,
                             dense_weight_format="fp8_e4m3")
    runtime = SimpleNamespace(max_batch_size=2, page_size=1, device=torch.device("cuda"),
                              model=SimpleNamespace(model=SimpleNamespace(
                                  language_model=SimpleNamespace(config=target))))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    assert Qwen35GeneratedVerification.supports(runtime, draft)


@pytest.mark.parametrize("field,value", [
    ("markov_rank", 0), ("markov_rank", -1), ("markov_head_type", "rnn"),
    ("draft_vocab_size", 256), ("enable_confidence_head", "true"),
])
def test_dspark_rejects_unsupported_heads(field, value):
    data = metadata()
    data[field] = value
    with pytest.raises(ValueError, match="DSpark"):
        DFlashConfig.from_dict(data)


def test_markov_chain_uses_anchor_then_its_own_previous_selection():
    head = _VanillaMarkov(SimpleNamespace(vocab_size=3, markov_rank=1))
    with torch.no_grad():
        head.markov_w1.weight.copy_(torch.tensor([[1.], [-1.], [0.]]))
        head.markov_w2.weight.copy_(torch.tensor([[0.], [2.], [-2.]]))
    logits = torch.zeros(2, 3, 3)
    assert head(logits, torch.tensor([0, 1])).tolist() == [[1, 2, 0], [2, 0, 1]]
    with pytest.raises(ValueError, match="predecessor"):
        head(logits, None)


def test_dspark_selects_anchor_row_instead_of_discarding_it():
    model = DFlashDraftModel(DFlashConfig.from_dict(metadata()))
    with torch.no_grad():
        model.markov_head.markov_w2.weight.zero_()
    hidden = torch.zeros(1, 7, 128)
    hidden[0, :, 0] = torch.arange(7)
    def head(value):
        result = torch.full((*value.shape[:-1], 257), -100.)
        result.scatter_(-1, value[..., :1].long(), 100.)
        return result
    assert model.select_tokens(hidden, head, torch.tensor([10])).tolist() == [list(range(7))]


def test_yarn_matches_transformers_schedule_and_tables():
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    data = metadata()
    config = Qwen3Config(**data)
    reference = Qwen3RotaryEmbedding(config)
    actual = MultidimensionalRotaryEmbedding(128, 10000000, dimensions=1)
    actual.inv_freq, actual.attention_factor = yarn_inv_freq(128, 10000000, data["rope_parameters"])
    torch.testing.assert_close(actual.inv_freq, reference.inv_freq)
    assert actual.attention_factor == reference.attention_scaling
    hidden = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    positions = torch.tensor([[0, 8, 8192, 131072]])
    for a, b in zip(actual(hidden, positions[..., None]), reference(hidden, positions)):
        torch.testing.assert_close(a, b, atol=0.004, rtol=0.004)


def test_yarn_meta_construction_and_invalid_schedule():
    parameters = metadata()["rope_parameters"]
    with torch.device("meta"):
        model = DFlashDraftModel(DFlashConfig.from_dict(metadata()))
        assert model.rotary_emb.inv_freq.is_meta
        assert model.rotary_emb.inv_freq.shape == (64,)
    for change in ({"factor": 0}, {"beta_slow": -1}, {"attention_factor": float("nan")}):
        with pytest.raises(ValueError):
            yarn_inv_freq(128, 10000000, parameters | change)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dspark_graph_matches_eager_with_changing_anchors():
    from kestrel.models.qwen35.dflash import DFlashContextCache, _LayerContextBuffer
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    torch.manual_seed(42)
    model = DFlashDraftModel(DFlashConfig.from_dict(metadata())).cuda().bfloat16().requires_grad_(False)
    head = torch.nn.Linear(128, 257, bias=False).cuda().bfloat16().requires_grad_(False)
    caches = []
    for _ in range(2):
        cache = DFlashContextCache(64)
        cache.length = 3
        keys = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
        cache.layers = [_LayerContextBuffer(keys, keys.clone())]
        caches.append(cache)
    session = DFlashDraftGraphSession(model, caches, lm_head=head)
    try:
        assert session.query_rows == 7 and session.context_rows == 8
        for anchors in ([1, 2], [3, 4], [2, 1]):
            noise = [torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
            features = [value[:, :0] for value in noise]
            positions = [torch.arange(3, 10, device="cuda")[None] for _ in caches]
            anchors = torch.tensor(anchors, device="cuda")
            hidden = torch.cat([model(n, f, p, context_cache=c) for n, f, p, c in
                                zip(noise, features, positions, caches)])
            expected = model.select_tokens(hidden, head, anchors)
            with session.launch(noise, features, positions, anchors) as actual:
                assert torch.equal(actual, expected)
    finally:
        session.shutdown()
