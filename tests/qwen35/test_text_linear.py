from types import SimpleNamespace

import torch
import torch.nn.functional as F

import kestrel.models.qwen35.qwen_model as model_module
from kestrel.models.qwen35.qwen_model import Qwen3_5Attention, Qwen3_5MLP


def test_attention_uses_runtime_linear_without_changing_semantics(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        head_dim=2,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
    )
    attention = Qwen3_5Attention(config, layer_idx=0).eval()
    inputs = torch.randn(1, 3, config.hidden_size)
    calls = []

    def runtime_linear(x, weight, bias):
        calls.append((x, weight, bias))
        return F.linear(x, weight, bias)

    monkeypatch.setattr(model_module, "_kestrel_linear", runtime_linear)
    monkeypatch.setattr(
        model_module,
        "_kestrel_rmsnorm",
        lambda value, weight, eps: value,
    )
    monkeypatch.setattr(
        model_module,
        "_kestrel_text_mrope_apply",
        lambda query, key, cos, sin: (query, key),
    )
    monkeypatch.setattr(
        model_module,
        "dense_attention",
        lambda query, *args, **kwargs: query.transpose(1, 2),
    )

    actual, weights = attention(
        inputs,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        attention_mask=None,
        cu_seq_lens_q=torch.tensor([0, inputs.shape[1]], dtype=torch.int32),
    )

    assert weights is None
    assert len(calls) == 2
    assert calls[0][1] is attention.qkv_proj.weight
    assert calls[0][2] is attention.qkv_proj.bias
    assert calls[1][1] is attention.o_proj.weight
    assert calls[1][2] is attention.o_proj.bias
    torch.testing.assert_close(
        actual,
        F.linear(calls[1][0], attention.o_proj.weight, attention.o_proj.bias),
    )


def test_mlp_uses_runtime_linear_without_changing_semantics(monkeypatch) -> None:
    config = SimpleNamespace(hidden_size=4)
    intermediate_size = 3
    mlp = Qwen3_5MLP(config, intermediate_size=intermediate_size).eval()
    inputs = torch.randn(2, config.hidden_size)
    calls = []

    def runtime_linear(x, weight, bias):
        calls.append((x, weight, bias))
        return F.linear(x, weight, bias)

    def gated_activation(output, gate_up, **kwargs):
        output.copy_(gate_up[..., :intermediate_size])

    monkeypatch.setattr(model_module, "_kestrel_linear", runtime_linear)
    monkeypatch.setattr(
        model_module,
        "_kestrel_gated_activation_into",
        gated_activation,
    )

    actual = mlp(inputs)

    assert len(calls) == 2
    assert calls[0][1] is mlp.gate_up_proj.weight
    assert calls[0][2] is mlp.gate_up_proj.bias
    assert calls[1][1] is mlp.down_proj.weight
    assert calls[1][2] is mlp.down_proj.bias
    torch.testing.assert_close(
        actual,
        F.linear(calls[1][0], mlp.down_proj.weight, mlp.down_proj.bias),
    )
