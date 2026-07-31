from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kestrel.models.qwen35 import qwen_loader
from kestrel.models.qwen35 import qwen_model
from kestrel.models.qwen35.qwen_config import Qwen3_5Config, Qwen3_5TextConfig
from kestrel.models.qwen35.qwen_model import Qwen3_5RMSNormGated
from kestrel.ops.norm import RMSNorm


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
    moe_fields = {
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
        "num_experts",
    }
    if moe_fields.intersection(overrides):
        data.setdefault("moe_intermediate_size", 8)
        data.setdefault("shared_expert_intermediate_size", 8)
        data.setdefault("num_experts_per_tok", 1)
        data.setdefault("num_experts", 1)
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


def _qwen_config_data(*, text=None, **overrides):
    text_data = _text_config_data(**(text or {}))
    data = {
        "model_type": "qwen3_5",
        "text_config": text_data,
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": int(text_data["hidden_size"]),
            "num_position_embeddings": 16,
        },
        "image_token_id": 30,
        "tie_word_embeddings": False,
    }
    data.update(overrides)
    return data


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        self.b = torch.nn.Parameter(torch.zeros(2))


class _TinyNormModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(2)
        self.gated_norm = Qwen3_5RMSNormGated(2)


class _TinyGatedDeltaNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.dt_bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))


class _TinyQwen(torch.nn.Module):
    def __init__(self, _config, **_kwargs) -> None:
        super().__init__()
        self.config = _config
        self.gdn = _TinyGatedDeltaNet()
        self.dense = torch.nn.Linear(2, 2, bias=False)


class _TinyDenseOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(2, 2, bias=False)


class _TinyLinearAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(2, 7, bias=False)


class _TinyFullAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(2, 8, bias=False)


class _TinyDenseMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(2, 32, bias=False)


class _TinySharedExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(2, 2, bias=False)


class _TinySharedExpertMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_expert = _TinySharedExpert()


class _TinySharedExpertBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinySharedExpertMlp()


class _TinyLinearAttentionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = _TinyLinearAttention()


class _TinyFullAttentionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyFullAttention()


class _TinyFusedGdnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyLinearAttentionBlock()])


class _TinyFusedAttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyFullAttentionBlock()])


class _TinyFusedMlpModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyDenseMlp()])


class _TinySharedExpertModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinySharedExpertBlock()])


class _TinyExpertsMlp(torch.nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.experts = qwen_model.Qwen3_5Experts(config)


class _TinyExpertsBlock(torch.nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.mlp = _TinyExpertsMlp(config)


class _TinyExpertsModel(torch.nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyExpertsBlock(config)])


def test_load_sharded_safetensors_streams_compatible_keys(tmp_path, monkeypatch):
    save_file(
        {
            "a": torch.ones(2),
            "mtp.layers.0.a": torch.full((2,), 9.0),
        },
        tmp_path / "a.safetensors",
    )
    save_file(
        {
            "b": torch.full((2,), 2.0),
            "extra": torch.full((1,), 3.0),
        },
        tmp_path / "b.safetensors",
    )

    calls: list[str] = []

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        calls.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    model = _TinyModel()
    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["a.safetensors", "b.safetensors"],
        device=torch.device("cpu"),
    )

    assert calls == ["a.safetensors", "b.safetensors"]
    assert missing == []
    assert unexpected == ["extra"]
    assert torch.equal(model.a, torch.ones(2))
    assert torch.equal(model.b, torch.full((2,), 2.0))


def test_load_sharded_safetensors_rejects_fp8_non_native_target(
    tmp_path,
    monkeypatch,
):
    weight_fp8 = torch.ones((2, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.ones((1, 1), dtype=torch.float32)
    save_file(
        {
            "dense.weight": weight_fp8,
            "dense.weight_scale_inv": scale_inv,
        },
        tmp_path / "weights.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    with pytest.raises(ValueError, match="reached a non-FP8 target"):
        qwen_loader._load_sharded_safetensors(
            _TinyDenseOnly(),
            "repo",
            ["weights.safetensors"],
            device=torch.device("cpu"),
        )


def test_load_sharded_safetensors_folds_qwen_rms_norm_offsets(tmp_path, monkeypatch):
    save_file(
        {
            "norm.weight": torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
            "gated_norm.weight": torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    model = _TinyNormModel()
    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["model.safetensors"],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    assert model.norm.weight.dtype == torch.float32
    torch.testing.assert_close(
        model.norm.weight,
        torch.tensor([1.25, 0.5], dtype=torch.float32),
    )
    torch.testing.assert_close(
        model.gated_norm.weight,
        torch.tensor([0.25, -0.5], dtype=torch.float32),
    )


def test_qwen_rms_norm_uses_effective_weight_in_float32():
    module = RMSNorm(4, eps=1e-6)
    checkpoint_weight = torch.tensor([0.25, -0.5, 0.0, 0.75], dtype=torch.float32)
    module.weight.data.copy_(checkpoint_weight + 1.0)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual = module(x)
    x_float = x.float()
    expected = x_float * torch.rsqrt(
        x_float.pow(2).mean(-1, keepdim=True) + module.eps
    )
    expected = expected * (checkpoint_weight + 1.0)

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected.to(x.dtype))


def test_load_sharded_safetensors_fuses_gdn_input_projection(tmp_path, monkeypatch):
    qkv = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    z = torch.arange(6, 10, dtype=torch.float32).reshape(2, 2)
    b = torch.arange(10, 12, dtype=torch.float32).reshape(1, 2)
    a = torch.arange(12, 14, dtype=torch.float32).reshape(1, 2)
    save_file(
        {
            "layers.0.linear_attn.in_proj_qkv.weight": qkv,
            "layers.0.linear_attn.in_proj_z.weight": z,
        },
        tmp_path / "a.safetensors",
    )
    save_file(
        {
            "layers.0.linear_attn.in_proj_b.weight": b,
            "layers.0.linear_attn.in_proj_a.weight": a,
        },
        tmp_path / "b.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    model = _TinyFusedGdnModel()
    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["a.safetensors", "b.safetensors"],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.layers[0].linear_attn.in_proj.weight,
        torch.cat([qkv, z, b, a], dim=0),
    )


def test_qwen36_fp8_gdn_dequantizes_into_fused_projection(tmp_path, monkeypatch):
    cfg = _text_config(
        hidden_size=128,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=16,
        expert_weight_format="fp8_e4m3",
    )

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = qwen_model.Qwen3_5GatedDeltaNet(cfg, layer_idx=0)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([_Block()])

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = _Model()
    finally:
        torch.set_default_dtype(old_dtype)

    gdn = model.layers[0].linear_attn
    qkv_shape = (gdn.conv_dim, gdn.hidden_size)
    z_shape = (gdn.value_dim, gdn.hidden_size)
    ba_shape = (gdn.num_v_heads, gdn.hidden_size)
    qkv = torch.randn(qkv_shape, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    z = torch.randn(z_shape, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    out = torch.randn_like(gdn.out_proj.weight, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    b = torch.randn(ba_shape, dtype=torch.bfloat16)
    a = torch.randn(ba_shape, dtype=torch.bfloat16)
    scales = {
        "layers.0.linear_attn.in_proj_qkv.weight_scale_inv": torch.ones(
            ((qkv_shape[0] + 127) // 128, (qkv_shape[1] + 127) // 128),
            dtype=torch.bfloat16,
        ),
        "layers.0.linear_attn.in_proj_z.weight_scale_inv": torch.ones(
            ((z_shape[0] + 127) // 128, (z_shape[1] + 127) // 128),
            dtype=torch.bfloat16,
        ),
        "layers.0.linear_attn.out_proj.weight_scale_inv": torch.ones(
            ((out.shape[0] + 127) // 128, (out.shape[1] + 127) // 128),
            dtype=torch.bfloat16,
        ),
    }
    save_file(
        {
            "layers.0.linear_attn.in_proj_qkv.weight": qkv,
            "layers.0.linear_attn.in_proj_z.weight": z,
            "layers.0.linear_attn.out_proj.weight": out,
            "layers.0.linear_attn.in_proj_b.weight": b,
            "layers.0.linear_attn.in_proj_a.weight": a,
        },
        tmp_path / "weights.safetensors",
    )
    save_file(scales, tmp_path / "scales.safetensors")

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    _missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["weights.safetensors", "scales.safetensors"],
        device=torch.device("cpu"),
    )

    assert unexpected == []
    expected_qkv = qwen_loader._dequantize_fp8_weight(
        qkv,
        scales["layers.0.linear_attn.in_proj_qkv.weight_scale_inv"],
        torch.Size(qkv_shape),
        key="layers.0.linear_attn.in_proj_qkv.weight",
    )
    expected_z = qwen_loader._dequantize_fp8_weight(
        z,
        scales["layers.0.linear_attn.in_proj_z.weight_scale_inv"],
        torch.Size(z_shape),
        key="layers.0.linear_attn.in_proj_z.weight",
    )
    expected_out = qwen_loader._dequantize_fp8_weight(
        out,
        scales["layers.0.linear_attn.out_proj.weight_scale_inv"],
        gdn.out_proj.weight.shape,
        key="layers.0.linear_attn.out_proj.weight",
    )
    torch.testing.assert_close(
        gdn.in_proj.weight,
        torch.cat((expected_qkv, expected_z, b, a), dim=0),
    )
    torch.testing.assert_close(gdn.out_proj.weight, expected_out)
    assert not hasattr(gdn, "in_proj_qkvz")
    assert not hasattr(gdn, "in_proj_b")
    assert not hasattr(gdn, "in_proj_a")


def test_load_sharded_safetensors_fuses_attention_qkv_projection(tmp_path, monkeypatch):
    q = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    k = torch.arange(8, 12, dtype=torch.float32).reshape(2, 2)
    v = torch.arange(12, 16, dtype=torch.float32).reshape(2, 2)
    save_file(
        {
            "layers.0.self_attn.q_proj.weight": q,
            "layers.0.self_attn.k_proj.weight": k,
        },
        tmp_path / "a.safetensors",
    )
    save_file(
        {
            "layers.0.self_attn.v_proj.weight": v,
        },
        tmp_path / "b.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    model = _TinyFusedAttentionModel()
    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["a.safetensors", "b.safetensors"],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.layers[0].self_attn.qkv_proj.weight,
        torch.cat([q, k, v], dim=0),
    )


def test_load_sharded_safetensors_fuses_mlp_gate_up_projection(tmp_path, monkeypatch):
    gate = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    up = torch.arange(32, 64, dtype=torch.float32).reshape(16, 2)
    save_file(
        {
            "layers.0.gate_proj.weight": gate,
        },
        tmp_path / "a.safetensors",
    )
    save_file(
        {
            "layers.0.up_proj.weight": up,
        },
        tmp_path / "b.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    model = _TinyFusedMlpModel()
    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["a.safetensors", "b.safetensors"],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    expected = torch.stack(
        (
            gate.reshape(2, 8, 2),
            up.reshape(2, 8, 2),
        ),
        dim=1,
    ).reshape(32, 2)
    torch.testing.assert_close(
        model.layers[0].gate_up_proj.weight,
        expected,
    )


def test_qwen35_experts_use_interleaved_gate_up_projection():
    config = _text_config(
        hidden_size=2,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
    )
    experts = qwen_model.Qwen3_5Experts(config)
    gate = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    up = torch.arange(16, 32, dtype=torch.float32).reshape(8, 2) / 16
    experts.gate_up_proj.data.copy_(qwen_loader._interleave_gate_up_weight(gate, up).unsqueeze(0))
    experts.down_proj.data.copy_(torch.eye(2, 8).unsqueeze(0))

    hidden_states = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    top_k_index = torch.zeros((1, 1), dtype=torch.long)
    top_k_weights = torch.ones((1, 1), dtype=torch.float32)

    actual = experts(hidden_states, top_k_index, top_k_weights)
    expected_hidden = torch.nn.functional.silu(hidden_states @ gate.T) * (hidden_states @ up.T)
    expected = expected_hidden[:, :2]
    torch.testing.assert_close(actual, expected)


def test_qwen35_fp8_config_uses_quantized_expert_buffers():
    cfg = Qwen3_5Config.from_dict(
        _qwen_config_data(
            model_type="qwen3_5_moe",
            quantization_config={"quant_method": "fp8", "fmt": "e4m3"},
            text={
                "hidden_size": 128,
                "head_dim": 64,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 128,
                "shared_expert_intermediate_size": 128,
            },
        )
    )
    experts = qwen_model.Qwen3_5Experts(cfg.text_config)

    assert experts.gate_up_proj.dtype == torch.uint8
    assert experts.down_proj.dtype == torch.uint8
    assert experts.gate_up_proj_scale.shape == (2, 2, 1, 1)
    assert experts.down_proj_scale.shape == (2, 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_fp8_experts_fallback_when_kestrel_config_missing(monkeypatch):
    config = _text_config(
        hidden_size=128,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=128,
        expert_weight_format="fp8_e4m3",
    )
    experts = qwen_model.Qwen3_5Experts(config).cuda()
    experts.gate_up_proj.data.zero_()
    experts.down_proj.data.zero_()
    experts.gate_up_proj_scale.fill_(1.0)
    experts.down_proj_scale.fill_(1.0)

    def fail_missing_config(*_args):
        raise ValueError(qwen_model.FP8_MOE_REQUIRES_COMPACT_CONFIG)

    monkeypatch.setattr(experts, "_forward_kestrel", fail_missing_config)

    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    top_k_index = torch.zeros((2, 1), device="cuda", dtype=torch.int32)
    top_k_weights = torch.ones((2, 1), device="cuda", dtype=torch.bfloat16)

    actual = experts(hidden_states, top_k_index, top_k_weights)

    assert torch.count_nonzero(actual) == 0
    assert actual.shape == hidden_states.shape


def test_qwen35_fp8_config_keeps_dense_projection_modules_bf16_by_default():
    cfg = Qwen3_5Config.from_dict(
        _qwen_config_data(
            model_type="qwen3_5_moe",
            quantization_config={"quant_method": "fp8", "fmt": "e4m3"},
            text={
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 16,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 128,
                "shared_expert_intermediate_size": 128,
            },
        )
    )

    gdn = qwen_model.Qwen3_5GatedDeltaNet(cfg.text_config, layer_idx=0)
    attn = qwen_model.Qwen3_5Attention(cfg.text_config, layer_idx=0)
    mlp = qwen_model.Qwen3_5MLP(cfg.text_config, intermediate_size=128)

    assert isinstance(gdn.in_proj, torch.nn.Linear)
    assert isinstance(gdn.out_proj, torch.nn.Linear)
    assert not hasattr(gdn, "in_proj_qkv")
    assert not hasattr(gdn, "in_proj_qkvz")

    assert isinstance(attn.qkv_proj, torch.nn.Linear)
    assert isinstance(attn.o_proj, torch.nn.Linear)
    assert not hasattr(attn, "q_proj")

    assert isinstance(mlp.gate_up_proj, torch.nn.Linear)
    assert isinstance(mlp.down_proj, torch.nn.Linear)
    assert not hasattr(mlp, "gate_proj")


def test_qwen35_fp8_dense_bf16_default_keeps_experts_fp8():
    cfg = Qwen3_5Config.from_dict(
        _qwen_config_data(
            model_type="qwen3_5_moe",
            quantization_config={"quant_method": "fp8", "fmt": "e4m3"},
            text={
                "hidden_size": 128,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 16,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 128,
                "shared_expert_intermediate_size": 128,
            },
        )
    )

    gdn = qwen_model.Qwen3_5GatedDeltaNet(cfg.text_config, layer_idx=0)
    attn = qwen_model.Qwen3_5Attention(cfg.text_config, layer_idx=0)
    mlp = qwen_model.Qwen3_5MLP(cfg.text_config, intermediate_size=128)
    experts = qwen_model.Qwen3_5Experts(cfg.text_config)

    assert isinstance(gdn.in_proj, torch.nn.Linear)
    assert isinstance(gdn.out_proj, torch.nn.Linear)
    assert isinstance(attn.qkv_proj, torch.nn.Linear)
    assert isinstance(attn.o_proj, torch.nn.Linear)
    assert isinstance(mlp.gate_up_proj, torch.nn.Linear)
    assert isinstance(mlp.down_proj, torch.nn.Linear)
    assert experts.gate_up_proj.dtype == torch.uint8
    assert experts.down_proj.dtype == torch.uint8


def test_qwen35_fp8_dense_bf16_default_allows_text_dequant():
    key = "layers.0.self_attn.q_proj.weight"
    expert_key = "layers.0.mlp.experts.0.gate_proj.weight"

    assert qwen_loader._allow_fp8_dequant_fallback(key)
    assert not qwen_loader._allow_fp8_dequant_fallback(expert_key)


def test_load_sharded_safetensors_preserves_fp8_expert_weights_and_scales(
    tmp_path,
    monkeypatch,
):
    config = _text_config(
        hidden_size=128,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=128,
        expert_weight_format="fp8_e4m3",
    )
    model = _TinyExpertsModel(config)

    shard: dict[str, torch.Tensor] = {}
    expected_gate_up: list[torch.Tensor] = []
    expected_down: list[torch.Tensor] = []
    expected_gate_up_scale: list[torch.Tensor] = []
    expected_down_scale: list[torch.Tensor] = []
    for expert_idx in range(2):
        prefix = f"layers.0.mlp.experts.{expert_idx}"
        gate = torch.full((128, 128), expert_idx + 1.0, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        up = torch.full((128, 128), expert_idx + 3.0, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        down = torch.full((128, 128), expert_idx + 5.0, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        gate_scale = torch.full((1, 1), 0.25 + expert_idx, dtype=torch.float32)
        up_scale = torch.full((1, 1), 0.5 + expert_idx, dtype=torch.float32)
        down_scale = torch.full((1, 1), 0.75 + expert_idx, dtype=torch.float32)
        shard[f"{prefix}.gate_proj.weight"] = gate
        shard[f"{prefix}.gate_proj.weight_scale_inv"] = gate_scale
        shard[f"{prefix}.up_proj.weight"] = up
        shard[f"{prefix}.up_proj.weight_scale_inv"] = up_scale
        shard[f"{prefix}.down_proj.weight"] = down
        shard[f"{prefix}.down_proj.weight_scale_inv"] = down_scale
        expected_gate_up.append(
            qwen_loader._interleave_gate_up_weight(
                gate.view(torch.uint8),
                up.view(torch.uint8),
            )
        )
        expected_down.append(down.view(torch.uint8))
        expected_gate_up_scale.append(torch.stack((gate_scale, up_scale), dim=0))
        expected_down_scale.append(down_scale)
    save_file(shard, tmp_path / "model.safetensors")

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)

    missing, unexpected = qwen_loader._load_sharded_safetensors(
        model,
        "repo",
        ["model.safetensors"],
        device=torch.device("cpu"),
    )

    experts = model.layers[0].mlp.experts
    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        experts.gate_up_proj,
        torch.stack(expected_gate_up, dim=0),
    )
    torch.testing.assert_close(experts.down_proj, torch.stack(expected_down, dim=0))
    torch.testing.assert_close(
        experts.gate_up_proj_scale,
        torch.stack(expected_gate_up_scale, dim=0),
    )
    torch.testing.assert_close(
        experts.down_proj_scale,
        torch.stack(expected_down_scale, dim=0),
    )


def test_qwen35_kestrel_moe_capacity_buckets_prefill_tokens():
    assert qwen_model._kestrel_moe_capacity_for_tokens(1) == (1, "decode")
    assert qwen_model._kestrel_moe_capacity_for_tokens(16) == (16, "decode")
    assert qwen_model._kestrel_moe_capacity_for_tokens(17) == (64, "prefill")
    assert qwen_model._kestrel_moe_capacity_for_tokens(64) == (64, "prefill")
    assert qwen_model._kestrel_moe_capacity_for_tokens(65) == (128, "prefill")
    assert qwen_model._kestrel_moe_capacity_for_tokens(4095) == (4096, "prefill")
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen_model._kestrel_moe_capacity_for_tokens(0)


def test_load_qwen35_model_preserves_ssm_parameter_dtype(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            _qwen_config_data(
                text={"mamba_ssm_dtype": "float32"},
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        """
        {
          "weight_map": {
            "gdn.A_log": "model.safetensors",
            "gdn.dt_bias": "model.safetensors",
            "dense.weight": "model.safetensors"
          }
        }
        """,
        encoding="utf-8",
    )
    save_file(
        {
            "gdn.A_log": torch.tensor([1.25, 2.5], dtype=torch.float32),
            "gdn.dt_bias": torch.tensor([3.5, 4.75], dtype=torch.float32),
            "dense.weight": torch.ones((2, 2), dtype=torch.float32),
        },
        tmp_path / "model.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    monkeypatch.setattr(qwen_loader, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(qwen_loader, "Qwen3_5ForConditionalGeneration", _TinyQwen)

    model = qwen_loader.load_qwen35_model(
        "repo",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert not hasattr(model.config.text_config, "_attn_implementation")
    assert not hasattr(model.config.vision_config, "_attn_implementation")
    assert model.gdn.A_log.dtype == torch.float32
    assert model.gdn.dt_bias.dtype == torch.float32
    assert model.dense.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        model.gdn.A_log.detach(),
        torch.tensor([1.25, 2.5], dtype=torch.float32),
    )
    torch.testing.assert_close(
        model.gdn.dt_bias.detach(),
        torch.tensor([3.5, 4.75], dtype=torch.float32),
    )

    def reject_hub_download(*_args, **_kwargs):
        pytest.fail("local Qwen checkpoint loading must not access the Hub")

    monkeypatch.setattr(qwen_loader, "hf_hub_download", reject_hub_download)
    local_model = qwen_loader.load_qwen35_model(
        tmp_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert local_model.gdn.A_log.dtype == torch.float32
    assert local_model.gdn.dt_bias.dtype == torch.float32
    assert local_model.dense.weight.dtype == torch.bfloat16


def test_cuda_vision_attention_defaults_to_kestrel_on_sm90():
    assert (
        qwen_loader._vision_attn_implementation(
            torch.device("cuda"),
            "sdpa",
            compute_capability=(9, 0),
        )
        == "kestrel_vision_flash_attention"
    )


def test_cuda_vision_attention_uses_fallback_on_unsupported_sms():
    assert (
        qwen_loader._vision_attn_implementation(
            torch.device("cuda"),
            "sdpa",
            compute_capability=(10, 0),
        )
        == "sdpa"
    )
    assert (
        qwen_loader._vision_attn_implementation(
            torch.device("cuda"),
            "sdpa",
            compute_capability=(11, 0),
        )
        == "sdpa"
    )


def test_cuda_vision_attention_honors_explicit_override_on_sm90():
    assert (
        qwen_loader._vision_attn_implementation(
            torch.device("cuda"),
            "eager",
            compute_capability=(9, 0),
        )
        == "eager"
    )
