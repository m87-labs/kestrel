from __future__ import annotations

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
    load_weights,
)
from kestrel.models.gemma4.model import Gemma4InferenceModel
from kestrel.runtime.bounded_projection import PackedLinear
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


def test_gemma_generated_weight_storage_streams_direct_rows_and_finalizes_retained():
    config = _tiny_gemma_config()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Gemma4InferenceModel(config)
    finally:
        torch.set_default_dtype(old_dtype)
    descriptor = {
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
                "prep": "interleave_gate_up_rows8_axis0|cast_bf16",
                "shape": [16, 8],
                "dtype": "bf16",
                "per_layer": True,
                "physical_layers": [0],
                "kind": "param",
            },
        ],
    }

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
    assert storage.aliased_source_bytes == 16 * 8 * 2
    assert storage.retained_source_bytes == (8 * 8 + 4 * 8 + 4 * 8 + 16 * 8) * 2
    assert storage.finalized
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert not any(
        tensor.device.type == "meta"
        for tensor in (*model.parameters(), *model.buffers())
    )
    assert not any(
        tensor.device.type == "meta"
        for tensor in model.model.language_model.rotary_emb.inv_freq.values()
    )
    assert materialize_weights(model, descriptor) is storage
