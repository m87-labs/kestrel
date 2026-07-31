from __future__ import annotations

import json
import pytest
import torch
from safetensors.torch import save_file

from kestrel.models.gemma4 import loader


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        self.b = torch.nn.Parameter(torch.zeros(2))


class _TinyFusedMlpModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.mlp = torch.nn.Module()
        self.block.mlp.gate_up_proj = torch.nn.Linear(2, 4, bias=False)


class _TinyFusedClippedMlpModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.mlp = torch.nn.Module()
        gate_up = torch.nn.Module()
        gate_up.linear = torch.nn.Linear(2, 4, bias=False)
        for name in ("input_min", "input_max", "output_min", "output_max"):
            gate_up.register_buffer(name, torch.tensor(0.0))
        self.block.mlp.gate_up_proj = gate_up


class _TinyEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(2, 2)


class _TinyInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyEmbeddings()


class _TinyGemma4(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = _TinyInnerModel()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def test_load_weights_accepts_compatible_keys(tmp_path, monkeypatch):
    save_file(
        {
            "a": torch.ones(2),
            "extra": torch.full((1,), 3.0),
        },
        tmp_path / "model.safetensors",
    )

    def fake_snapshot_download(repo_id: str, allow_patterns: list[str]) -> str:
        assert repo_id == "repo"
        assert "*.safetensors" in allow_patterns
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", fake_snapshot_download)

    model = _TinyModel()
    loader.load_weights("repo", model)

    assert torch.equal(model.a, torch.ones(2))
    assert torch.equal(model.b, torch.zeros(2))


def test_load_weights_does_not_materialize_unsupported_towers(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "model.safetensors").touch()

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def keys(self):
            return ("a", "model.audio_tower.weight", "model.embed_audio.weight")

        def get_tensor(self, name):
            assert name == "a"
            return torch.ones(2)

    monkeypatch.setattr(
        loader,
        "snapshot_download",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(loader, "safe_open", lambda *args, **kwargs: FakeSafeOpen())

    model = _TinyModel()
    loader.load_weights("repo", model)

    assert torch.equal(model.a, torch.ones(2))


def test_load_weights_fuses_gate_up_projection(tmp_path, monkeypatch):
    gate = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    up = gate + 10
    save_file(
        {
            "block.mlp.gate_proj.weight": gate,
            "block.mlp.up_proj.weight": up,
        },
        tmp_path / "model.safetensors",
    )
    monkeypatch.setattr(loader, "snapshot_download", lambda *args, **kwargs: str(tmp_path))

    model = _TinyFusedMlpModel()
    loader.load_weights("repo", model)

    torch.testing.assert_close(
        model.block.mlp.gate_up_proj.weight,
        torch.cat((gate, up), dim=0),
    )


def test_load_weights_fuses_clipped_gate_up_projection(tmp_path, monkeypatch):
    gate = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    up = gate + 10
    bounds = {
        "input_min": torch.tensor(-2.0),
        "input_max": torch.tensor(2.0),
        "output_min": torch.tensor(-4.0),
        "output_max": torch.tensor(4.0),
    }
    weights = {
        "block.mlp.gate_proj.linear.weight": gate,
        "block.mlp.up_proj.linear.weight": up,
    }
    for name, value in bounds.items():
        weights[f"block.mlp.gate_proj.{name}"] = value
        weights[f"block.mlp.up_proj.{name}"] = value.clone()
    save_file(weights, tmp_path / "model.safetensors")
    monkeypatch.setattr(loader, "snapshot_download", lambda *args, **kwargs: str(tmp_path))

    model = _TinyFusedClippedMlpModel()
    loader.load_weights("repo", model)

    torch.testing.assert_close(
        model.block.mlp.gate_up_proj.linear.weight,
        torch.cat((gate, up), dim=0),
    )
    for name, value in bounds.items():
        torch.testing.assert_close(
            getattr(model.block.mlp.gate_up_proj, name),
            value,
        )


def test_load_weights_refuses_different_clipping_bounds(tmp_path, monkeypatch):
    weights = {
        "block.mlp.gate_proj.linear.weight": torch.zeros((2, 2)),
        "block.mlp.up_proj.linear.weight": torch.zeros((2, 2)),
    }
    for name in ("input_min", "input_max", "output_min", "output_max"):
        weights[f"block.mlp.gate_proj.{name}"] = torch.tensor(0.0)
        weights[f"block.mlp.up_proj.{name}"] = torch.tensor(
            1.0 if name == "output_max" else 0.0
        )
    save_file(weights, tmp_path / "model.safetensors")
    monkeypatch.setattr(loader, "snapshot_download", lambda *args, **kwargs: str(tmp_path))

    with pytest.raises(ValueError, match="different output_max bounds"):
        loader.load_weights(
            "repo",
            _TinyFusedClippedMlpModel(),
        )


def test_load_model_loads_config_and_ties_embeddings(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "vocab_size": 2,
                    "hidden_size": 2,
                    "intermediate_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "max_position_embeddings": 2048,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "sliding_attention": {
                            "rope_type": "default",
                            "rope_theta": 10_000.0,
                        },
                        "full_attention": {
                            "rope_type": "proportional",
                            "rope_theta": 1_000_000.0,
                            "partial_rotary_factor": 0.25,
                        },
                    },
                    "sliding_window": 512,
                    "layer_types": ["sliding_attention"],
                    "final_logit_softcapping": 30.0,
                    "vocab_size_per_layer_input": 0,
                    "hidden_size_per_layer_input": 0,
                    "global_head_dim": 2,
                    "attention_k_eq_v": False,
                    "num_kv_shared_layers": 0,
                    "use_double_wide_mlp": False,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                },
                "vision_config": {
                    "hidden_size": 2,
                    "intermediate_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 100.0,
                    },
                    "pooling_kernel_size": 3,
                    "patch_size": 14,
                    "position_embedding_size": 16,
                    "use_clipped_linears": False,
                    "standardize": False,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                },
                "audio_config": {},
                "tie_word_embeddings": True,
                "image_token_id": 11,
                "video_token_id": 12,
                "audio_token_id": 13,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.full((2, 2), 7.0),
        },
        tmp_path / "model.safetensors",
    )

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "repo"
        return str(tmp_path / filename)

    def fake_snapshot_download(repo_id: str, allow_patterns: list[str]) -> str:
        assert repo_id == "repo"
        return str(tmp_path)

    monkeypatch.setattr(loader, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(loader, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(loader, "Gemma4ForConditionalGeneration", _TinyGemma4)

    model = loader.load_model(
        "repo",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert not hasattr(model.config, "tie_word_embeddings")
    assert not hasattr(model.config, "audio_config")
    assert not hasattr(model.config, "audio_token_id")
    assert not hasattr(model.config, "video_token_id")
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    torch.testing.assert_close(
        model.model.language_model.embed_tokens.weight.detach(),
        torch.full((2, 2), 7.0, dtype=torch.bfloat16),
    )
