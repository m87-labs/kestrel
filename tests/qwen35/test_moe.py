from types import SimpleNamespace

import torch

from kestrel.models.qwen35.qwen_config import Qwen3_5TextConfig
from kestrel.models.qwen35.qwen_model import Qwen3_5Experts
import kestrel.models.qwen35.qwen_model as model_module


def test_moe_capacity_matches_generated_decode_buckets() -> None:
    assert [
        model_module._kestrel_moe_capacity_for_tokens(tokens)
        for tokens in (1, 3, 8, 9, 16)
    ] == [
        (1, "decode"),
        (4, "decode"),
        (8, "decode"),
        (64, "prefill"),
        (64, "prefill"),
    ]


def _moe_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10_000.0,
        partial_rotary_factor=1.0,
        mrope_section=(1, 1, 1),
        head_dim=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        layer_types=("linear_attention",),
        intermediate_size=12,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        expert_weight_format="bf16",
    )


def test_bf16_experts_use_device_selected_moe_backend(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeMoeRuntime:
        def prepare(self, spec, capacity, *, device):
            calls.update(spec=spec, capacity=capacity, device=device)
            return SimpleNamespace(spec=spec)

        def forward(self, handle, **kwargs):
            calls.update(handle=handle, forward=kwargs)
            return kwargs["x"] + 1

    monkeypatch.setattr(model_module, "_kestrel_moe_runtime", FakeMoeRuntime())
    experts = Qwen3_5Experts(_moe_config()).to(dtype=torch.bfloat16)
    hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    weights = torch.full((2, 2), 0.5, dtype=torch.bfloat16)

    output = experts(hidden, indices, weights)

    spec = calls["spec"]
    assert spec.backend == "auto"
    assert spec.weight_format == "bf16"
    assert calls["forward"]["topk_ids"].dtype == torch.int32
    torch.testing.assert_close(output, hidden + 1)
