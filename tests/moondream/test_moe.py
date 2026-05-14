from __future__ import annotations

from types import SimpleNamespace

import torch

from kestrel.moondream.moe import MoEConfig, MoEModule, _SHARED_MOE_HANDLES


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
