import torch

from kestrel.models.qwen35.qwen_loader import (
    _ATTN_QKV_WEIGHT_PARTS,
    _MLP_GATE_UP_WEIGHT_PARTS,
    _copy_bf16_expert_part,
    _copy_fused_projection_part,
    _dequantize_fp8_weight,
    _interleave_gate_up_weight,
    _materialize_remaining_meta_tensors,
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


def test_bf16_experts_stream_into_final_interleaved_slices() -> None:
    gate = torch.arange(48, dtype=torch.bfloat16).reshape(16, 3)
    up = gate + 100
    down = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    expected_state = {
        "layer.experts.gate_up_proj": torch.empty((2, 32, 3), dtype=torch.bfloat16),
        "layer.experts.down_proj": torch.empty((2, 3, 8), dtype=torch.bfloat16),
    }
    loaded_parts: dict[str, dict[int, set[str]]] = {}
    loaded_keys: set[str] = set()

    for expert_idx in range(2):
        for part, value in (("up_proj.weight", up), ("gate_proj.weight", gate)):
            _copy_bf16_expert_part(
                expected_state,
                checkpoint_key=f"layer.experts.{expert_idx}.{part}",
                target_key="layer.experts.gate_up_proj",
                expert_idx=expert_idx,
                part=part,
                value=value,
                loaded_parts=loaded_parts,
                loaded_keys=loaded_keys,
            )
        _copy_bf16_expert_part(
            expected_state,
            checkpoint_key=f"layer.experts.{expert_idx}.down_proj.weight",
            target_key="layer.experts.down_proj",
            expert_idx=expert_idx,
            part="down_proj.weight",
            value=down,
            loaded_parts=loaded_parts,
            loaded_keys=loaded_keys,
        )

    interleaved = _interleave_gate_up_weight(gate, up)
    torch.testing.assert_close(
        expected_state["layer.experts.gate_up_proj"],
        torch.stack((interleaved, interleaved)),
    )
    torch.testing.assert_close(
        expected_state["layer.experts.down_proj"],
        torch.stack((down, down)),
    )
    assert loaded_keys == {
        "layer.experts.gate_up_proj",
        "layer.experts.down_proj",
    }


def test_remaining_meta_tensors_materialize_without_replacing_bound_storage() -> None:
    module = torch.nn.Module()
    module.bound = torch.nn.Parameter(torch.empty(3))
    with torch.device("meta"):
        shared = torch.nn.Parameter(torch.empty(4))
        module.pending = shared
        module.pending_alias = shared
        module.register_buffer("pending_buffer", torch.empty(2))
    bound = module.bound

    _materialize_remaining_meta_tensors(module, device=torch.device("cpu"))

    assert module.bound is bound
    assert module.pending.device.type == "cpu"
    assert module.pending is module.pending_alias
    assert module.pending_buffer.device.type == "cpu"
