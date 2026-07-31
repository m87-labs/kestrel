"""Qwen model numerics and serving-boundary tests."""

from types import SimpleNamespace

import torch

import kestrel.models.qwen35  # noqa: F401
import kestrel.models.qwen35.qwen_model as qwen_model
from kestrel.models import get_spec, known_models
from kestrel.models.qwen35.inference_ops import LinearAttentionLayer
from kestrel.models.qwen35.cache import qwen_kv_layout
from kestrel.models.qwen35.qwen_config import Qwen3_5Config, Qwen3_5TextConfig
from kestrel.models.qwen35.qwen_model import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5SparseMoeBlock,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5TopKRouter,
)
from kestrel.models.qwen35.runtime import Qwen35Runtime
from kestrel.models.qwen35.skills import build_skill_registry
from kestrel.runtime.carried_state import StateRepresentationRequirement


_MODEL_ID = "Qwen/Qwen3.5-2B"


def _text_config_data(**overrides):
    hidden = int(overrides.get("hidden_size", 8))
    heads = int(overrides.get("num_attention_heads", 2))
    data = {
        "vocab_size": 32,
        "hidden_size": hidden,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": 1,
        "hidden_act": "silu",
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        "attention_bias": False,
        "head_dim": hidden // heads,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "layer_types": ["full_attention"],
    }
    data.update(overrides)
    if "num_experts" in overrides:
        data.setdefault("moe_intermediate_size", 8)
        data.setdefault("shared_expert_intermediate_size", 8)
        data.setdefault("num_experts_per_tok", 1)
    return data


def _text_config(**overrides):
    data = _text_config_data(**overrides)
    rope = data.pop("rope_parameters")
    data.pop("hidden_act")
    data.pop("mamba_ssm_dtype")
    data.pop("attention_bias")
    data["tie_word_embeddings"] = False
    data["rope_theta"] = rope["rope_theta"]
    data["partial_rotary_factor"] = rope["partial_rotary_factor"]
    data["mrope_section"] = tuple(rope["mrope_section"])
    return Qwen3_5TextConfig(**data)


def test_supported_variants_register():
    expected = {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-27B-FP8",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    }
    assert expected <= set(known_models())
    assert get_spec(_MODEL_ID).runtime is Qwen35Runtime


def test_hybrid_cache_uses_shared_paged_kv_layout():
    specs, sources = qwen_kv_layout(
        _text_config(
            num_hidden_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
        )
    )

    assert specs[0] is None and specs[2] is None
    assert specs[1].n_heads == 1
    assert sources == (-1, 1, -1)


def test_fused_attention_handles_multitoken_prefill():
    config = _text_config(head_dim=4)
    attention = Qwen3_5Attention(config, layer_idx=0)
    hidden = torch.randn(2, 3, config.hidden_size)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)
    rope = Qwen3_5TextRotaryEmbedding(config)(hidden, positions)

    output, _ = attention(
        hidden,
        position_embeddings=rope,
        attention_mask=None,
        past_key_values=None,
    )

    assert output.shape == hidden.shape


def test_paged_attention_preserves_fused_value_view(monkeypatch):
    config = _text_config(head_dim=4)
    attention = Qwen3_5Attention(config, layer_idx=0)
    hidden = torch.randn(2, 3, config.hidden_size)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)
    rope = Qwen3_5TextRotaryEmbedding(config)(hidden, positions)
    captured = {}

    class _Layer:
        def update(self, **values):
            captured.update(values)

    monkeypatch.setattr(
        qwen_model,
        "paged_attention_forward",
        lambda query, **_kwargs: (query.transpose(1, 2), None),
    )
    output, _ = attention(
        hidden,
        position_embeddings=rope,
        attention_mask=None,
        past_key_values=SimpleNamespace(layers=(_Layer(),)),
        cache_position_ids=torch.arange(3).expand(2, -1),
        slot_mapping=torch.arange(6).reshape(2, 3),
    )

    assert output.shape == hidden.shape
    assert captured["v_val"].shape == (2, 3, 1, 4)
    assert captured["v_val"].stride()[-2:] == (4, 1)


def test_moe_checkpoint_surface_is_fused():
    config = Qwen3_5Config.from_dict({
        "model_type": "qwen3_5_moe",
        "text_config": _text_config_data(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
        ),
        "vision_config": {
            "depth": 1, "hidden_size": 8, "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16, "num_heads": 2, "in_channels": 3,
            "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 8,
            "num_position_embeddings": 16,
        },
        "image_token_id": 30,
        "tie_word_embeddings": False,
    }).text_config
    model = Qwen3_5TextModel(config)

    assert isinstance(model.layers[0].mlp, Qwen3_5SparseMoeBlock)
    keys = set(model.state_dict())
    assert "layers.0.mlp.experts.gate_up_proj" in keys
    assert "layers.0.mlp.experts.down_proj" in keys
    assert "layers.0.mlp.gate_proj.weight" not in keys


def test_router_matches_softmax_topk():
    config = _text_config(
        hidden_size=4, num_experts=5, num_experts_per_tok=2,
        moe_intermediate_size=3)
    torch.manual_seed(1)
    router = Qwen3_5TopKRouter(config)
    hidden = torch.randn(2, 2, 4)

    scores, indices = router(hidden)
    probabilities = torch.softmax(
        torch.nn.functional.linear(hidden.reshape(-1, 4), router.weight),
        dim=-1,
        dtype=torch.float,
    )
    expected_scores, expected_indices = torch.topk(probabilities, 2, dim=-1)
    expected_scores /= expected_scores.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_decode_routes_to_generated_program(monkeypatch):
    runtime = Qwen35Runtime.__new__(Qwen35Runtime)
    calls = []
    slot = SimpleNamespace(
        compute_stream=None,
        meta=SimpleNamespace(
            batch_idx=SimpleNamespace(cpu=torch.tensor([3, 5]))),
    )
    generated = (StateRepresentationRequirement("state", "generated"),)
    native = (StateRepresentationRequirement("state", "native"),)
    runtime._decode_megakernel = SimpleNamespace(
        supports=lambda batch: batch == 1,
        state_requirements_for=lambda _batch: generated,
        run=lambda bound_slot, batch: calls.append(("generated", bound_slot, batch)),
    )
    runtime._decode_state_coordinator = SimpleNamespace(
        prepare=lambda requirements, rows: calls.append((requirements, rows)))
    runtime._native_decode_state_requirements = native
    runtime._decode_graphs = SimpleNamespace(
        run=lambda bound_slot, batch: calls.append(("native", bound_slot, batch)))

    runtime.decode_with_slot(slot, 1)
    runtime.decode_with_slot(slot, 2)

    assert calls == [
        (generated, (3,)),
        ("generated", slot, 1),
        (native, (3, 5)),
        ("native", slot, 2),
    ]


def test_recurrent_checkpoint_reset_uses_cursor():
    config = SimpleNamespace(
        linear_replay_capacity=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    layer = LinearAttentionLayer(config, replay_capacity=4)
    first = torch.randn((2, 2, 3, 3))
    second = torch.randn_like(first)
    layer.update_recurrent_state(first)
    layer.replay_k.fill_(1)
    layer.replay_lengths.fill_(4)
    layer.update_recurrent_state(second)

    torch.testing.assert_close(layer.recurrent_states, second)
    torch.testing.assert_close(
        layer.replay_checkpoint_states, second.transpose(-1, -2))
    assert torch.all(layer.replay_k == 1)
    assert layer.replay_lengths.tolist() == [0, 0]


def test_gdn_ssm_parameters_stay_float32():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        module = Qwen3_5GatedDeltaNet(_text_config(), layer_idx=0)
    finally:
        torch.set_default_dtype(old_dtype)

    assert module.A_log.dtype == torch.float32
    assert module.dt_bias.dtype == torch.float32
    assert module.in_proj.weight.dtype == torch.bfloat16


def test_query_defaults_to_direct_answer():
    skill = build_skill_registry().resolve("query")
    request = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 8},
    )
    assert request.request_context.reasoning is False
    assert request.temperature == 0.0
    assert request.top_p == 1.0
