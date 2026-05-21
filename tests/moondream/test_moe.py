from __future__ import annotations

from types import SimpleNamespace

import torch

from kestrel.moondream.moe import MoEConfig, MoEModule, _SHARED_MOE_HANDLES
from kestrel.moondream import runtime as runtime_mod
from kestrel.utils import CpuGpuBuffer


class _FakeMoeRuntime:
    def __init__(self) -> None:
        self.capacities = []

    def prepare(self, spec, capacity, *, device):
        self.capacities.append(capacity)
        return SimpleNamespace(
            spec=spec,
            capacity=capacity,
            device=torch.device(device),
            impl=None,
        )


def _make_module(runtime: _FakeMoeRuntime) -> MoEModule:
    module = object.__new__(MoEModule)
    module.top_k = 8
    module.hidden_size = 1024
    module.input_size = 2048
    module.num_experts = 64
    module.config = MoEConfig()
    module._moe_runtime = runtime
    return module


def test_prefill_moe_handles_reuse_power_of_two_capacity_bucket() -> None:
    _SHARED_MOE_HANDLES.clear()
    try:
        runtime = _FakeMoeRuntime()
        module = _make_module(runtime)

        first = module._get_moe_handle(
            hidden_states=torch.empty((129, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=0,
            mode="prefill",
        )
        second = module._get_moe_handle(
            hidden_states=torch.empty((200, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=0,
            mode="prefill",
        )

        assert first is second
        assert [capacity.max_tokens for capacity in runtime.capacities] == [256]
    finally:
        _SHARED_MOE_HANDLES.clear()


def test_decode_moe_handles_remain_exact_capacity_for_cuda_graph_buckets() -> None:
    _SHARED_MOE_HANDLES.clear()
    try:
        runtime = _FakeMoeRuntime()
        module = _make_module(runtime)

        first = module._get_moe_handle(
            hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=0,
            mode="decode",
        )
        second = module._get_moe_handle(
            hidden_states=torch.empty((16, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=0,
            mode="decode",
        )

        assert first is not second
        assert [capacity.max_tokens for capacity in runtime.capacities] == [8, 16]
    finally:
        _SHARED_MOE_HANDLES.clear()


def test_lora_rank_is_part_of_moe_handle_capacity() -> None:
    _SHARED_MOE_HANDLES.clear()
    try:
        runtime = _FakeMoeRuntime()
        module = _make_module(runtime)

        first = module._get_moe_handle(
            hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=1,
            max_lora_rank=4,
            mode="decode",
        )
        second = module._get_moe_handle(
            hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
            weight_format="bf16",
            max_loras=1,
            max_lora_rank=8,
            mode="decode",
        )

        assert first is not second
        assert [capacity.max_lora_rank for capacity in runtime.capacities] == [4, 8]
    finally:
        _SHARED_MOE_HANDLES.clear()


class _FakeLoraWorkspace:
    max_slots = 4

    def __init__(self) -> None:
        self._ranks = [3, 7, 5]

    def moe_lora_rank_for_id(self, lora_id: int) -> int:
        return self._ranks[lora_id]


def _make_decode_meta(max_batch: int) -> SimpleNamespace:
    device = torch.device("cpu")
    return SimpleNamespace(
        lora_slot_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_token_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_lora_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_lora_meta=CpuGpuBuffer(
            max_batch + 3, dtype=torch.int32, device=device, pin_memory=False
        ),
    )


def test_decode_lora_metadata_uses_compact_graph_stable_buffers() -> None:
    runtime = runtime_mod.MoondreamRuntime.__new__(runtime_mod.MoondreamRuntime)
    runtime._lora_workspace = _FakeLoraWorkspace()
    slot = SimpleNamespace(meta=_make_decode_meta(8))

    slot.meta.lora_slot_ids.np[:6] = [0, 3, 1, 3, 2, 0]

    runtime._prepare_decode_lora_metadata(slot, 6)

    torch.testing.assert_close(
        slot.meta.active_lora_meta.gpu[:7],
        torch.tensor([3, 7, 4, 0, 2, 3, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        slot.meta.active_token_ids.gpu[:4],
        torch.tensor([1, 3, 2, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        slot.meta.active_lora_ids.gpu[:3],
        torch.tensor([2, 0, 1], dtype=torch.int32),
    )
    assert slot.meta.moe_lora_metadata.active_token_ids.numel() == 6
    assert slot.meta.moe_lora_metadata.active_lora_ids.numel() == 3
    assert slot.meta.moe_lora_metadata.active_lora_meta.numel() == 7

    slot.meta.lora_slot_ids.np[:6] = 0
    runtime._prepare_decode_lora_metadata(slot, 6)

    torch.testing.assert_close(
        slot.meta.active_lora_meta.gpu[:7],
        torch.zeros((7,), dtype=torch.int32),
    )
