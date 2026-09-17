import pytest
import torch

from kestrel.models.qwen35.dflash import DFlashConfig, _Attention


def test_speculative_recurrent_branch_does_not_mutate_committed_state():
    from types import SimpleNamespace
    from kestrel.models.qwen35.cache import Qwen35InferenceCache

    paged = object()
    cache = Qwen35InferenceCache(
        config=SimpleNamespace(layer_types=("linear_attention", "full_attention")),
        paged_kv=(None, paged))
    layer = cache.layers[0]
    layer.conv_states = torch.ones(1, 8, 4, dtype=torch.bfloat16)
    layer.recurrent_states = torch.ones(2, 2, 4, 4, dtype=torch.bfloat16)
    layer.has_previous_state = True
    cache.advance_to(31)
    branch = cache.fork_recurrent_state()
    branch.layers[0].conv_states.fill_(2)
    branch.layers[0].recurrent_states.fill_(3)
    branch.advance_to(47)
    assert torch.all(layer.conv_states == 1)
    assert torch.all(layer.recurrent_states == 1)
    assert branch.layers[0].has_previous_state
    assert branch.layers[1] is paged
    assert cache.get_seq_length() == 31
    assert branch.get_seq_length() == 47


def _tap_model(monkeypatch):
    from types import SimpleNamespace
    from kestrel.models.qwen35 import qwen_model

    class Layer(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment
            self.input_layernorm = SimpleNamespace(weight=None, eps=1e-6)

        def _forward_from_normalized(self, hidden, normalized, **kwargs):
            hidden.add_(self.increment)
            return hidden, hidden + 100

    model = qwen_model.Qwen3_5TextModel.__new__(qwen_model.Qwen3_5TextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=3, layer_types=["full_attention"] * 3)
    model.layers = torch.nn.ModuleList([Layer(1), Layer(2), Layer(3)])
    model.norm = SimpleNamespace(weight=None, eps=1e-6)
    model.rotary_emb = lambda hidden, positions: ()
    monkeypatch.setattr(qwen_model, "_kestrel_rmsnorm", lambda value, *args: value + 100)
    return model


def test_target_taps_are_post_layer_residuals_and_own_storage(monkeypatch):
    model = _tap_model(monkeypatch)
    result = model(inputs_embeds=torch.zeros(1, 1, 4), capture_layers=(0, 2))
    torch.testing.assert_close(result.layer_hidden_states[0], torch.ones(1, 1, 4))
    torch.testing.assert_close(result.layer_hidden_states[1], torch.full((1, 1, 4), 6.0))
    torch.testing.assert_close(result.last_hidden_state, torch.full((1, 1, 4), 106.0))
    assert model(inputs_embeds=torch.zeros(1, 1, 4)).layer_hidden_states == ()


@pytest.mark.parametrize("layers", [(2, 0), (1, 1), (-1,), (3,), (True,)])
def test_target_taps_reject_invalid_layer_indices(monkeypatch, layers):
    with pytest.raises(ValueError, match="zero-based layer indices"):
        _tap_model(monkeypatch)(inputs_embeds=torch.zeros(1, 1, 4), capture_layers=layers)


def test_target_taps_pass_through_multimodal_model(monkeypatch):
    from kestrel.models.qwen35.qwen_model import Qwen3_5Model

    wrapper = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = _tap_model(monkeypatch)
    wrapper.language_model.embed_tokens = torch.nn.Embedding(4, 4)
    with torch.no_grad():
        wrapper.language_model.embed_tokens.weight.zero_()
    result = wrapper(torch.zeros((1, 1), dtype=torch.long), None, capture_layers=(1,))
    torch.testing.assert_close(result.layer_hidden_states[0], torch.full((1, 1, 4), 3.0))


def _metadata():
    return {
        "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 6, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 128,
        "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "sliding_window": 4,
        "dflash_config": {"block_size": 16, "mask_token_id": 248077,
                          "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61]},
    }


def test_q27_draft_uses_mixed_causality():
    config = DFlashConfig.from_dict(_metadata())
    assert [config.attention(i) for i in range(6)] == [(True, 4)] * 5 + [(False, None)]
    assert config.target_layer_ids == (1, 10, 18, 27, 35, 44, 52, 61)
    assert config.block_size == 16


def test_draft_causality_override_is_explicit():
    metadata = _metadata()
    metadata["dflash_config"]["causal"] = False
    config = DFlashConfig.from_dict(metadata)
    assert config.attention(0) == (False, 4)
    metadata["is_causal"] = True
    assert DFlashConfig.from_dict(metadata).attention(5) == (True, None)


@pytest.mark.parametrize("field,value", [
    ("sliding_window", None), ("sliding_window", 0),
    ("layer_types", ["sliding_attention"]),
    ("is_causal", "false"),
])
def test_invalid_attention_metadata_rejected(field, value):
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        DFlashConfig.from_dict(metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layer", [0, 5])
@pytest.mark.filterwarnings("error:.*[Ff]allback.*:RuntimeWarning")
def test_draft_native_attention_matches_bottom_right_mask(layer):
    torch.manual_seed(2715)
    config = DFlashConfig.from_dict(_metadata())
    module = _Attention(config, layer).cuda().bfloat16().eval()
    hidden = torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16)
    cos = torch.ones(1, 10, 128, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    with torch.inference_mode():
        actual = module(hidden, context, cos, sin)
        joined = torch.cat((context, hidden), dim=1)
        q = module.q_norm(module.q_proj(hidden).reshape(1, 3, 4, 128)).transpose(1, 2)
        k = module.k_norm(module.k_proj(joined).reshape(1, 10, 1, 128)).transpose(1, 2)
        v = module.v_proj(joined).reshape(1, 10, 1, 128).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * 128 ** -0.5
        query_position = torch.arange(7, 10, device="cuda")[:, None]
        key_position = torch.arange(10, device="cuda")[None, :]
        allowed = torch.ones(3, 10, device="cuda", dtype=torch.bool)
        if module.causal:
            allowed &= key_position <= query_position
        if module.window is not None:
            allowed &= key_position >= query_position - module.window + 1
        scores.masked_fill_(~allowed, float("-inf"))
        out = torch.matmul(scores.softmax(-1).to(v.dtype), v)
        expected = module.o_proj(out.transpose(1, 2).reshape(1, 3, 512))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.filterwarnings("error:.*[Ff]allback.*:RuntimeWarning")
@pytest.mark.parametrize("position_base", [0, 1000])
def test_draft_context_cache_excludes_queries_and_appends_verified_context(monkeypatch, position_base):
    from kestrel.models.qwen35.dflash import DFlashContextCache, DFlashDraftModel

    torch.manual_seed(907)
    config = DFlashConfig.from_dict(_metadata())
    model = DFlashDraftModel(config).cuda().bfloat16().eval()
    target = torch.randn(1, 7, 8 * 128, device="cuda", dtype=torch.bfloat16)
    cache = DFlashContextCache(16)
    consumed = 0
    pointers = None
    with torch.inference_mode():
        for added in (5, 2, 0):
            noise = torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16)
            end = consumed + added
            positions = torch.arange(position_base + consumed, position_base + end + 3,
                                     device="cuda")[None]
            if added == 2:
                prefix = [(layer.keys[:, :consumed].clone(), layer.values[:, :consumed].clone())
                          for layer in cache.layers]

                def fail_layer(*args, **kwargs):
                    raise RuntimeError("injected late draft failure")

                with monkeypatch.context() as patch:
                    patch.setattr(model.layers[-1], "forward", fail_layer)
                    with pytest.raises(RuntimeError, match="late draft failure"):
                        model(noise, target[:, consumed:end], positions, context_cache=cache)
                assert cache.length == consumed
                for layer, (keys, values) in zip(cache.layers, prefix, strict=True):
                    torch.testing.assert_close(layer.keys[:, :consumed], keys, rtol=0, atol=0)
                    torch.testing.assert_close(layer.values[:, :consumed], values, rtol=0, atol=0)
            actual = model(
                noise, target[:, consumed:end], positions,
                context_cache=cache)
            expected = model(noise, target[:, :end], torch.arange(
                position_base, position_base + end + 3, device="cuda")[None])
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            assert cache.length == end
            current = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers]
            if pointers is not None:
                assert current == pointers
            pointers = current
            consumed = end
        old_length = cache.length
        keys = [layer.keys[:, :old_length].clone() for layer in cache.layers]
        with pytest.raises(ValueError, match="capacity exceeded"):
            model(noise, target, torch.arange(10, device="cuda")[None], context_cache=cache)
        assert cache.length == old_length
        for layer, before in zip(cache.layers, keys, strict=True):
            torch.testing.assert_close(layer.keys[:, :old_length], before, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_draft_context_cache_rejects_invalid_capacity(capacity):
    from kestrel.models.qwen35.dflash import DFlashContextCache

    with pytest.raises(ValueError, match="positive integer"):
        DFlashContextCache(capacity)
