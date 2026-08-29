from contextlib import contextmanager
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from kestrel.models.qwen35.qwen_loader import (
    _ATTN_QKV_WEIGHT_PARTS,
    _MLP_GATE_UP_WEIGHT_PARTS,
    _copy_bf16_expert_part,
    _copy_fused_projection_part,
    _dequantize_fp8_weight,
    _interleave_gate_up_weight,
    _load_sharded_safetensors,
    _materialize_remaining_meta_tensors,
    _restore_rotary_buffers,
)
import kestrel.models.qwen35.qwen_loader as loader_module
from kestrel.ops.rotary import default_inv_freq


class _FusedExpertHolder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.language_model.layers[0]
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(
            torch.empty((2, 32, 3), dtype=torch.bfloat16),
            requires_grad=False,
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


def test_fused_and_separate_expert_checkpoints_load_identical_physical_layout(
    tmp_path,
) -> None:
    gate = torch.arange(2 * 16 * 3, dtype=torch.bfloat16).reshape(2, 16, 3)
    up = gate + 1000
    target_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    fused_dir = tmp_path / "fused"
    separate_dir = tmp_path / "separate"
    fused_dir.mkdir()
    separate_dir.mkdir()
    save_file(
        {target_key: torch.cat((gate, up), dim=1)},
        fused_dir / "model.safetensors",
    )
    save_file(
        {
            f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight": gate[expert]
            for expert in range(2)
        }
        | {
            f"model.language_model.layers.0.mlp.experts.{expert}.up_proj.weight": up[expert]
            for expert in range(2)
        },
        separate_dir / "model.safetensors",
    )
    fused = _FusedExpertHolder()
    separate = _FusedExpertHolder()

    fused_result = _load_sharded_safetensors(
        fused,
        fused_dir,
        ["model.safetensors"],
        device=torch.device("cpu"),
    )
    separate_result = _load_sharded_safetensors(
        separate,
        separate_dir,
        ["model.safetensors"],
        device=torch.device("cpu"),
    )

    assert fused_result == ([], [])
    assert separate_result == ([], [])
    expected = torch.stack(
        tuple(_interleave_gate_up_weight(gate[i], up[i]) for i in range(2))
    )
    torch.testing.assert_close(
        fused.model.language_model.layers[0].mlp.experts.gate_up_proj,
        expected,
    )
    torch.testing.assert_close(
        separate.model.language_model.layers[0].mlp.experts.gate_up_proj,
        expected,
    )


def test_sharded_loader_closes_key_handle_before_per_tensor_read(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fake")
    source = torch.tensor([[1.0, 2.0]])
    events: list[tuple[str, int, str | None]] = []
    next_handle = 0

    class Slice:
        def get_shape(self):
            return source.shape

    @contextmanager
    def fake_safe_open(path, *, framework, device):
        nonlocal next_handle
        assert path == str(checkpoint)
        assert framework == "pt"
        handle_id = next_handle
        next_handle += 1
        tensor_reads = 0
        events.append(("open", handle_id, device))

        class Handle:
            def keys(self):
                events.append(("keys", handle_id, None))
                return ["weight"]

            def get_slice(self, key):
                assert key == "weight"
                events.append(("slice", handle_id, key))
                return Slice()

            def get_tensor(self, key):
                nonlocal tensor_reads
                assert key == "weight"
                tensor_reads += 1
                assert tensor_reads == 1
                events.append(("tensor", handle_id, key))
                return source

        try:
            yield Handle()
        finally:
            events.append(("close", handle_id, device))

    monkeypatch.setattr(loader_module, "safe_open", fake_safe_open)
    model = torch.nn.Linear(2, 1, bias=False)

    missing, unexpected = _load_sharded_safetensors(
        model,
        tmp_path,
        [checkpoint.name],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(model.weight, source)
    tensor_handle = next(
        handle_id for event, handle_id, _ in events if event == "tensor"
    )
    tensor_open_index = events.index(("open", tensor_handle, "cpu"))
    close_event = events[tensor_open_index - 1]
    assert close_event[0] == "close"
    key_handle = close_event[1]
    assert key_handle != tensor_handle
    assert ("keys", key_handle, None) in events


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


def test_remaining_meta_tensors_reject_uninitialized_derived_buffers() -> None:
    module = torch.nn.Module()
    with torch.device("meta"):
        module.register_buffer("derived", torch.empty(2), persistent=False)

    try:
        _materialize_remaining_meta_tensors(module, device=torch.device("cpu"))
    except RuntimeError as exc:
        assert "derived buffer" in str(exc)
    else:
        raise AssertionError("uninitialized derived buffer was accepted")


def test_rotary_buffers_are_recomputed_after_meta_construction() -> None:
    class Rotary(torch.nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            with torch.device("meta"):
                self.register_buffer(
                    "inv_freq", torch.empty(size), persistent=False
                )

    model = SimpleNamespace(
        model=SimpleNamespace(
            visual=SimpleNamespace(rotary_pos_emb=Rotary(4)),
            language_model=SimpleNamespace(rotary_emb=Rotary(3)),
        )
    )
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            head_dim=12,
            rope_theta=1_000_000.0,
            partial_rotary_factor=0.5,
        )
    )

    _restore_rotary_buffers(model, config, device=torch.device("cpu"))

    torch.testing.assert_close(
        model.model.visual.rotary_pos_emb.inv_freq,
        default_inv_freq(8, 10_000.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.model.language_model.rotary_emb.inv_freq,
        default_inv_freq(
            12,
            1_000_000.0,
            partial_rotary_factor=0.5,
        ),
        rtol=0,
        atol=0,
    )
