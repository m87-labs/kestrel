from types import SimpleNamespace

import torch

from kestrel.models.moondream.generated_decode import (
    _engine_weight_buffers,
    _logical_weight_sources,
    _prepare_generated_decode,
    MoondreamDecodeBindings,
)
from kestrel.models.moondream.runtime import (
    _decode_slot_storage_capacity,
    _FP8_KV_SUPPORTED_SMS,
)
from kestrel.runtime.generated_decode import GeneratedDecode


def _cache(value: float) -> SimpleNamespace:
    return SimpleNamespace(
        quantized=True,
        k_scale_tensor=torch.tensor(value, dtype=torch.float32),
        v_scale_tensor=torch.tensor(value + 1, dtype=torch.float32),
        k_cache=torch.full((3, 1, 1, 2), value, dtype=torch.float32),
        v_cache=torch.full((3, 1, 1, 2), value + 1, dtype=torch.float32),
    )


def test_ampere_uses_checkpoint_fp8_kv_for_generated_decode() -> None:
    assert {80, 86}.issubset(_FP8_KV_SUPPORTED_SMS)


def test_moondream_exposes_logical_moe_sources_and_engine_slabs() -> None:
    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 3))
            self.register_buffer("scale", torch.empty(2))

    fused = torch.nn.Module()
    fused.up_experts = Experts()
    fused.down_experts = Experts()
    block = torch.nn.Module()
    block.mlp = torch.nn.ModuleDict({
        "router": torch.nn.Linear(3, 2),
        "mlp": fused,
    })
    text = torch.nn.Module()
    text.blocks = torch.nn.ModuleList([block])
    text.moe_up_w_slab = torch.empty(1, dtype=torch.uint8)
    text.moe_up_scale_slab = torch.empty(1)

    sources = _logical_weight_sources(text)
    buffers = _engine_weight_buffers(text)

    assert sources["blocks.0.mlp.down_experts.weight"] is fused.down_experts.weight
    assert sources["blocks.0.mlp.down_experts.scale"] is fused.down_experts.scale
    assert sources["blocks.0.mlp.up_experts.weight"] is fused.up_experts.weight
    assert sources["blocks.0.mlp.up_experts.scale"] is fused.up_experts.scale
    assert set(buffers) == {"moe_up_w_slab", "moe_up_scale_slab"}
    assert buffers["moe_up_w_slab"] is text.moe_up_w_slab
    assert buffers["moe_up_scale_slab"] is text.moe_up_scale_slab


def test_moondream_bindings_expose_compiler_runtime_resources() -> None:
    first = _cache(1)
    second = _cache(2)
    blocks = [
        SimpleNamespace(
            attn=SimpleNamespace(
                _tau_pos_table=torch.full((5, 2), value, dtype=torch.bfloat16)
            )
        )
        for value in (1, 2)
    ]
    text = SimpleNamespace(
        blocks=blocks,
        cos_sin_cache=torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        ),
    )
    runtime = SimpleNamespace(
        _lora_workspace=None,
        page_size=1,
        model=SimpleNamespace(text=text),
        page_table=SimpleNamespace(page_table=torch.zeros(4, 7)),
    )
    bindings = MoondreamDecodeBindings(
        (SimpleNamespace(cache=first), SimpleNamespace(cache=second))
    )

    assert bindings.is_eligible(runtime)
    inputs = bindings.runtime_inputs(runtime)

    assert set(inputs) == {
        "tau_pos",
        "rope_cos",
        "rope_sin",
        "mK_dequant_scale",
        "mV_dequant_scale",
        "mK",
        "mV",
        "page_table",
        "kv_len",
    }
    assert inputs["kv_len"] == 1
    torch.testing.assert_close(
        inputs["rope_cos"],
        torch.tensor([[1.0, 2.0, 1.0, 2.0], [5.0, 6.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(
        inputs["rope_sin"],
        torch.tensor([[3.0, 4.0, 3.0, 4.0], [7.0, 8.0, 7.0, 8.0]]),
    )
    assert [tensor.shape for tensor in inputs["mK"]] == [
        torch.Size((3, 1, 2)),
        torch.Size((3, 1, 2)),
    ]
    assert inputs["tau_pos"].shape == (2, 5, 2)
    assert inputs["tau_pos"].dtype is torch.float32
    torch.testing.assert_close(inputs["mK_dequant_scale"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(inputs["mV_dequant_scale"], torch.tensor([2.0, 3.0]))


def test_moondream_slot_bindings_use_stable_hidden_and_metadata_buffers() -> None:
    bindings = MoondreamDecodeBindings(())
    slot = SimpleNamespace(
        hidden_last=torch.zeros(8, 16),
        meta=SimpleNamespace(
            batch_idx=SimpleNamespace(gpu=torch.arange(8)),
            input_pos=SimpleNamespace(
                gpu=torch.arange(8, dtype=torch.int32),
                cpu=torch.tensor([3, 6, 4, 1, 0, 0, 0, 0], dtype=torch.int32),
            ),
        ),
    )

    inputs = bindings.slot_inputs(slot, 4)

    assert inputs["x"].data_ptr() == slot.hidden_last.data_ptr()
    assert inputs["batch_idx"].tolist() == [0, 1, 2, 3]
    assert inputs["input_pos"].tolist() == [0, 1, 2, 3]
    assert bindings.launch_extents(slot, 4) == {
        "active_batch": 4,
        "kv_len": 7,
    }


def test_moondream_allocates_selected_generated_program_abi_capacity(
    monkeypatch,
) -> None:
    text = torch.nn.Module()
    text.blocks = torch.nn.ModuleList()
    text.moe_up_w_slab = torch.empty(1, dtype=torch.uint8)
    text.moe_up_scale_slab = torch.empty(1)
    runtime = SimpleNamespace(
        max_batch_size=1,
        max_batch_slots=3,
        model=SimpleNamespace(text=text),
        layer_caches=(),
    )
    program = SimpleNamespace(
        capacity=8,
        static_extent_bindings={},
        runtime_extent_minimums={},
    )
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: (program,)),
    )

    plan = _prepare_generated_decode(runtime)
    storage_capacity = _decode_slot_storage_capacity(
        runtime.max_batch_slots, plan.slot_capacity
    )

    assert plan.slot_capacity == 8
    assert storage_capacity == 8
    assert runtime.max_batch_slots == 3
