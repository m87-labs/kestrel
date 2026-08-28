import torch

from kestrel.models.qwen35.qwen_loader import (
    _ATTN_QKV_WEIGHT_PARTS,
    _MLP_GATE_UP_WEIGHT_PARTS,
    _copy_fused_projection_part,
    _dequantize_fp8_weight,
    _interleave_gate_up_weight,
)


def test_fp8_dequantization_chunks_without_changing_bf16_result() -> None:
    torch.manual_seed(0)
    value = torch.randn((1_153, 257), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand((10, 3), dtype=torch.float32)

    actual = _dequantize_fp8_weight(
        value,
        scale,
        value.shape,
        key="model.language_model.layers.0.mlp.gate_proj.weight",
    )
    expanded_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(
        128,
        dim=1,
    )[: value.shape[0], : value.shape[1]]
    expected = (value.float() * expanded_scale).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_projection_parts_copy_directly_into_final_slices() -> None:
    values = {
        "q_proj.weight": torch.full((4, 3), 1.0),
        "k_proj.weight": torch.full((2, 3), 2.0),
        "v_proj.weight": torch.full((2, 3), 3.0),
    }
    target = torch.zeros((8, 3))
    shapes = {f"layer.{key}": value.shape for key, value in values.items()}
    loaded_parts: dict[str, set[str]] = {}
    loaded_keys: set[str] = set()

    for part in ("v_proj.weight", "q_proj.weight", "k_proj.weight"):
        _copy_fused_projection_part(
            {"layer.qkv_proj.weight": target},
            shapes,
            checkpoint_key=f"layer.{part}",
            fused_key="layer.qkv_proj.weight",
            part=part,
            parts=_ATTN_QKV_WEIGHT_PARTS,
            value=values[part],
            loaded_parts=loaded_parts,
            loaded_keys=loaded_keys,
        )

    torch.testing.assert_close(
        target,
        torch.cat([values[part] for part in _ATTN_QKV_WEIGHT_PARTS]),
    )
    assert loaded_keys == {"layer.qkv_proj.weight"}


def test_gate_up_parts_copy_directly_into_interleaved_layout() -> None:
    gate = torch.arange(48, dtype=torch.float32).reshape(16, 3)
    up = gate + 100
    target = torch.zeros((32, 3))
    loaded_parts: dict[str, set[str]] = {}
    loaded_keys: set[str] = set()

    for part, value in (("up_proj.weight", up), ("gate_proj.weight", gate)):
        _copy_fused_projection_part(
            {"layer.gate_up_proj.weight": target},
            {
                "layer.gate_proj.weight": gate.shape,
                "layer.up_proj.weight": up.shape,
            },
            checkpoint_key=f"layer.{part}",
            fused_key="layer.gate_up_proj.weight",
            part=part,
            parts=_MLP_GATE_UP_WEIGHT_PARTS,
            value=value,
            loaded_parts=loaded_parts,
            loaded_keys=loaded_keys,
        )

    torch.testing.assert_close(target, _interleave_gate_up_weight(gate, up))
    assert loaded_keys == {"layer.gate_up_proj.weight"}
