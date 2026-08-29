from __future__ import annotations

from types import SimpleNamespace

import torch

from kestrel.models.gemma4.config import Gemma4TextConfig, RopeSpec
from kestrel.models.gemma4.model import (
    Gemma4TextDecoderLayer,
    Gemma4TextExperts,
    Gemma4TextRouter,
)
from kestrel.models.registry import get_spec
import kestrel.models.gemma4.model as model_module


def _text_config(*, moe: bool = True) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope={
            "sliding_attention": RopeSpec(kind="default", theta=10_000.0),
            "full_attention": RopeSpec(kind="default", theta=10_000.0),
        },
        sliding_window=16,
        layer_types=("sliding_attention",),
        final_logit_softcapping=30.0,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        num_global_key_value_heads=1,
        global_head_dim=4,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        enable_moe_block=moe,
        num_experts=4 if moe else None,
        top_k_experts=2 if moe else None,
        moe_intermediate_size=6 if moe else None,
    )


def test_gemma4_26b_a4b_variants_are_registered() -> None:
    for repo_id in (
        "google/gemma-4-26B-A4B",
        "google/gemma-4-26B-A4B-it",
    ):
        spec = get_spec(repo_id)
        assert spec.repo_id == repo_id
        assert spec.checkpoint_format == "gemma4"


def test_moe_modules_match_checkpoint_parameter_shapes() -> None:
    config = _text_config()
    layer = Gemma4TextDecoderLayer(
        config,
        0,
        kv_source_layer_idx=0,
        publishes_kv=False,
    )

    assert layer.router.scale.shape == (config.hidden_size,)
    assert layer.router.per_expert_scale.shape == (config.num_experts,)
    assert layer.experts.gate_up_proj.shape == (4, 12, 8)
    assert layer.experts.down_proj.shape == (4, 8, 6)
    assert {
        "router.scale",
        "router.per_expert_scale",
        "router.proj.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "post_feedforward_layernorm_1.weight",
        "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm_2.weight",
    } <= set(layer.state_dict())


def test_router_matches_full_softmax_then_topk_reference() -> None:
    config = _text_config()
    router = Gemma4TextRouter(config)
    with torch.no_grad():
        router.proj.weight.copy_(
            torch.tensor(
                [
                    [0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.4, -0.3],
                    [-0.4, 0.2, 0.1, 0.3, 0.0, -0.1, 0.2, 0.1],
                    [0.1, 0.3, -0.2, 0.4, -0.1, 0.2, 0.0, -0.3],
                    [0.0, -0.2, 0.4, -0.1, 0.3, 0.1, -0.4, 0.2],
                ]
            )
        )
        router.scale.copy_(torch.linspace(0.8, 1.2, config.hidden_size))
        router.per_expert_scale.copy_(torch.tensor([1.0, 0.5, 1.5, 0.75]))
    hidden = torch.tensor(
        [
            [0.5, -0.4, 0.3, -0.2, 0.1, 0.0, 0.7, -0.6],
            [-0.1, 0.2, 0.4, -0.3, 0.6, -0.5, 0.8, 0.1],
        ]
    )

    actual_weights, actual_indices = router(hidden)

    normalized = torch.nn.functional.rms_norm(
        hidden.float(),
        (config.hidden_size,),
        eps=config.rms_norm_eps,
    )
    logits = torch.nn.functional.linear(
        normalized * router.scale * (config.hidden_size**-0.5),
        router.proj.weight,
    )
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_indices = torch.topk(
        probabilities,
        k=config.top_k_experts,
        dim=-1,
    )
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= router.per_expert_scale[expected_indices]

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_indices.to(torch.long), expected_indices)
    assert actual_weights.dtype == torch.float32


def test_experts_use_shared_contiguous_geglu_runtime(monkeypatch) -> None:
    config = _text_config()
    experts = Gemma4TextExperts(config)
    calls: dict[str, object] = {}

    class FakeMoeRuntime:
        def prepare(self, spec, capacity, *, device):
            calls.update(spec=spec, capacity=capacity, device=device)
            return SimpleNamespace(spec=spec)

        def forward(self, handle, **kwargs):
            calls.update(handle=handle, forward=kwargs)
            return kwargs["x"] + 1

    monkeypatch.setattr(model_module, "_moe_runtime", FakeMoeRuntime())
    hidden = torch.zeros((1, 2, config.hidden_size))
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    weights = torch.full((2, 2), 0.5)

    output = experts(hidden, indices, weights)

    spec = calls["spec"]
    forward = calls["forward"]
    assert spec.activation == "gelu"
    assert spec.backend == "auto"
    assert spec.intermediate_size == config.moe_intermediate_size
    assert forward["weights"].weight_scale_layout == "block128"
    assert forward["topk_ids"].dtype == torch.int32
    torch.testing.assert_close(output, hidden + 1)
