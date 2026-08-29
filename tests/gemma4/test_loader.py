from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kestrel.models.gemma4.config import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    RopeSpec,
)
from kestrel.models.gemma4.loader import (
    _restore_checkpoint_independent_state,
    load_model,
    load_weights,
)
from kestrel.models.gemma4.model import Gemma4InferenceModel, Gemma4TextMLP
from kestrel.runtime.bounded_projection import (
    PackedLinear,
    bind_declared_packed_projections,
)
from kestrel.runtime.generated_decode import materialize_remaining_meta_tensors
from kestrel_kernels.generated_decode import (
    allocate_weight_storage_for_loading,
    finalize_weight_storage_after_loading,
    materialize_weights,
)


class _ToyShardedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = PackedLinear(
            2,
            (2, 2),
            source_names=("gate", "up"),
        )
        self.exact = nn.Linear(2, 2, bias=False)


def _tiny_gemma_config() -> Gemma4Config:
    rope = RopeSpec(kind="default", theta=10_000.0)
    return Gemma4Config(
        text_config=Gemma4TextConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope={
                "sliding_attention": rope,
                "full_attention": rope,
            },
            sliding_window=8,
            layer_types=("sliding_attention", "full_attention"),
            final_logit_softcapping=30.0,
            vocab_size_per_layer_input=0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            global_head_dim=4,
            attention_k_eq_v=True,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
        ),
        vision_config=Gemma4VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            rms_norm_eps=1e-6,
            rope=rope,
            pooling_kernel_size=2,
            patch_size=2,
            position_embedding_size=4,
            use_clipped_linears=False,
            standardize=False,
        ),
        image_token_id=15,
    )


def _rope_config(spec: RopeSpec) -> dict[str, object]:
    return {
        "rope_type": spec.kind,
        "rope_theta": spec.theta,
        "partial_rotary_factor": spec.partial_rotary_factor,
        "factor": spec.factor,
    }


def _tiny_gemma_config_data() -> dict[str, object]:
    config = _tiny_gemma_config()
    text = asdict(config.text_config)
    text.pop("rope")
    text.update(
        rope_parameters={
            name: _rope_config(spec)
            for name, spec in config.text_config.rope.items()
        },
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
    )
    vision = asdict(config.vision_config)
    vision.pop("rope")
    vision.update(
        rope_parameters=_rope_config(config.vision_config.rope),
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
    )
    return {
        "tie_word_embeddings": True,
        "text_config": text,
        "vision_config": vision,
        "image_token_id": config.image_token_id,
    }


def _tiny_generated_weight_descriptor() -> dict[str, object]:
    return {
        "weight_layer_prefix": "model.language_model.layers",
        "weights": [
            {
                "name": "text_embedding_table",
                "source": "model.language_model.embed_tokens.weight",
                "prep": "cast_bf16",
                "shape": [16, 8],
                "dtype": "bf16",
                "per_layer": False,
                "physical_layers": [None],
                "kind": "param",
            },
            {
                "name": "w_qkv_local",
                "sources": [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                "prep": "concat_rows|cast_bf16",
                "shape": [16, 8],
                "dtype": "bf16",
                "per_layer": True,
                "physical_layers": [0],
                "kind": "param",
            },
            {
                "name": "w_gate_up_fresh",
                "source": "mlp.gate_up_proj.weight",
                "prep": "identity",
                "shape": [16, 8],
                "dtype": "bf16",
                "per_layer": True,
                "physical_layers": [0],
                "kind": "param",
            },
        ],
    }


def test_load_weights_streams_packed_projection_parts_across_shards(tmp_path) -> None:
    gate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    exact = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    save_file(
        {"gate.weight": gate, "exact.weight": exact},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"up.weight": up},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "gate.weight": "model-00001-of-00002.safetensors",
                    "exact.weight": "model-00001-of-00002.safetensors",
                    "up.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    model = _ToyShardedModel()

    load_weights(tmp_path, model)

    torch.testing.assert_close(model.packed.weight, torch.cat((gate, up), dim=0))
    torch.testing.assert_close(model.exact.weight, exact)


def test_load_weights_rejects_incomplete_packed_projection(tmp_path) -> None:
    save_file(
        {
            "gate.weight": torch.ones((2, 2)),
            "exact.weight": torch.ones((2, 2)),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(KeyError, match="missing source weights"):
        load_weights(tmp_path, _ToyShardedModel())


def test_load_weights_rejects_unexpected_checkpoint_tensor(tmp_path) -> None:
    save_file(
        {
            "gate.weight": torch.ones((2, 2)),
            "up.weight": torch.ones((2, 2)),
            "exact.weight": torch.ones((2, 2)),
            "unexpected.weight": torch.ones((1,)),
        },
        tmp_path / "model.safetensors",
    )

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        load_weights(tmp_path, _ToyShardedModel())


def test_text_mlp_interleaved_checkpoint_layout_matches_gate_up_math(
    monkeypatch,
) -> None:
    import kestrel.models.gemma4.model as gemma_model

    torch.manual_seed(7)
    config = _tiny_gemma_config().text_config
    mlp = Gemma4TextMLP(config, layer_idx=0)
    gate = torch.randn((config.intermediate_size, config.hidden_size))
    up = torch.randn_like(gate)
    down = torch.randn((config.hidden_size, config.intermediate_size))
    packed = {"gate_proj.weight": gate, "up_proj.weight": up}
    bind_declared_packed_projections(mlp, packed)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(packed["gate_up_proj.weight"])
        mlp.down_proj.weight.copy_(down)

    observed_layouts: list[str] = []

    def gated_activation_into(out, gate_up, *, activation, layout) -> None:
        observed_layouts.append(layout)
        assert activation == "gelu_tanh"
        blocks = gate_up.reshape(*gate_up.shape[:-1], config.intermediate_size // 8, 16)
        gate_out = blocks[..., :8].reshape_as(out)
        up_out = blocks[..., 8:].reshape_as(out)
        out.copy_(torch.nn.functional.gelu(
            gate_out, approximate="tanh"
        ) * up_out)

    monkeypatch.setattr(
        gemma_model,
        "_kestrel_gated_activation_into",
        gated_activation_into,
    )
    hidden = torch.randn((3, config.hidden_size))

    actual = mlp(hidden)
    expected = torch.nn.functional.linear(
        torch.nn.functional.gelu(
            torch.nn.functional.linear(hidden, gate), approximate="tanh"
        ) * torch.nn.functional.linear(hidden, up),
        down,
    )

    assert observed_layouts == ["interleaved_i8"]
    torch.testing.assert_close(actual, expected)


def test_gemma_generated_weight_storage_streams_direct_rows_and_finalizes_retained():
    config = _tiny_gemma_config()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)
    descriptor = _tiny_generated_weight_descriptor()

    storage = allocate_weight_storage_for_loading(
        model,
        descriptor,
        device="cpu",
    )
    final_storages = {
        int(value.untyped_storage()._cdata): int(value.untyped_storage().nbytes())
        for value in storage.buffers.values()
    }
    final_bytes = sum(final_storages.values())
    _restore_checkpoint_independent_state(model, config, device=torch.device("cpu"))
    materialize_remaining_meta_tensors(model, device=torch.device("cpu"))
    layer = model.model.language_model.layers[0]
    layer.self_attn.q_proj.weight.data.fill_(1)
    layer.self_attn.k_proj.weight.data.fill_(2)
    layer.self_attn.v_proj.weight.data.fill_(3)
    layer.mlp.gate_up_proj.weight.data.copy_(
        torch.arange(16, dtype=torch.bfloat16).view(16, 1).expand(16, 8)
    )

    finalize_weight_storage_after_loading(model, descriptor, storage)

    assert final_bytes == 3 * 16 * 8 * 2
    assert storage.aliased_source_bytes == 2 * 16 * 8 * 2
    assert storage.retained_source_bytes == (8 * 8 + 4 * 8 + 4 * 8) * 2
    assert all(
        retained.name != "w_gate_up_fresh"
        for retained in storage.retained_recipes
    )
    assert storage.finalized
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert torch._C._is_alias_of(
        layer.mlp.gate_up_proj.weight,
        storage.buffers["w_gate_up_fresh"][0],
    )
    assert not any(
        tensor.device.type == "meta"
        for tensor in (*model.parameters(), *model.buffers())
    )
    assert not any(
        tensor.device.type == "meta"
        for tensor in model.model.language_model.rotary_emb.inv_freq.values()
    )
    assert materialize_weights(model, descriptor) is storage


def test_load_model_runs_generated_weight_lifecycle_through_tiny_checkpoint(
    tmp_path,
) -> None:
    config = _tiny_gemma_config()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reference = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)
    checkpoint = {
        name: value.detach().clone()
        for name, value in reference.state_dict().items()
        if name != "lm_head.weight"
    }
    expected_embedding = checkpoint[
        "model.language_model.embed_tokens.weight"
    ].clone()
    (tmp_path / "config.json").write_text(
        json.dumps(_tiny_gemma_config_data()),
        encoding="utf-8",
    )
    save_file(checkpoint, tmp_path / "model.safetensors")

    descriptor = _tiny_generated_weight_descriptor()
    lifecycle: list[str] = []
    storage_box = {}

    def prepare_model(model: torch.nn.Module) -> None:
        lifecycle.append("prepare")
        assert all(tensor.device.type == "meta" for tensor in model.parameters())
        storage = allocate_weight_storage_for_loading(
            model,
            descriptor,
            device="cpu",
        )
        assert not storage.finalized
        storage_box["storage"] = storage

    def finalize_model(model: torch.nn.Module) -> None:
        lifecycle.append("finalize")
        storage = storage_box["storage"]
        assert not any(tensor.device.type == "meta" for tensor in model.parameters())
        finalize_weight_storage_after_loading(model, descriptor, storage)
        assert storage.finalized

    model = load_model(
        tmp_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        prepare_model=prepare_model,
        finalize_model=finalize_model,
    )

    storage = storage_box["storage"]
    embedding = model.model.language_model.embed_tokens.weight
    assert lifecycle == ["prepare", "finalize"]
    assert storage.finalized
    assert model.lm_head.weight is embedding
    assert torch._C._is_alias_of(
        embedding,
        storage.buffers["text_embedding_table"],
    )
    assert torch._C._is_alias_of(
        model.model.language_model.layers[0].mlp.gate_up_proj.weight,
        storage.buffers["w_gate_up_fresh"][0],
    )
    torch.testing.assert_close(embedding, expected_embedding)
    assert not any(
        tensor.device.type == "meta"
        for tensor in (*model.parameters(), *model.buffers())
    )
    assert not any(
        tensor.device.type == "meta"
        for tensor in model.model.language_model.rotary_emb.inv_freq.values()
    )
    torch.testing.assert_close(
        model.model.language_model.embed_tokens.embed_scale,
        torch.full_like(
            model.model.language_model.embed_tokens.embed_scale,
            config.text_config.hidden_size**0.5,
        ),
    )
    assert materialize_weights(model, descriptor) is storage
