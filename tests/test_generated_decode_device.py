from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kestrel.runtime.generated_decode import GeneratedDecode


def _program(capacity, *, active_batch=None):
    static = {} if active_batch is None else {"active_batch": active_batch}
    return SimpleNamespace(
        capacity=capacity,
        static_extent_bindings=static,
    )


def test_try_create_binds_physical_sm_count_to_program_resolution():
    runtime = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    spec = SimpleNamespace(
        bindings=SimpleNamespace(is_eligible=lambda value: value is runtime),
        weight_root=Mock(),
        weight_layer_prefix="model.layers",
    )
    properties = SimpleNamespace(major=10, minor=0, multi_processor_count=148)

    with (
        patch("torch.cuda.get_device_properties", return_value=properties),
        patch(
            "kestrel_kernels.generated_decode.resolve_compatible_programs",
            return_value=(),
        ) as resolve,
    ):
        assert GeneratedDecode.try_create(runtime, spec) is None

    resolve.assert_called_once_with(
        spec.weight_root,
        layer_prefix="model.layers",
        arch="sm100",
        device_sms=148,
    )


def test_program_selection_preserves_dynamic_and_exact_same_capacity_variants():
    generated = GeneratedDecode.__new__(GeneratedDecode)
    dynamic_b2 = _program(2)
    dynamic_b4 = _program(4)
    exact_b4 = _program(4, active_batch=4)
    dynamic_b8 = _program(8)
    exact_b8 = _program(8, active_batch=8)
    generated._programs = (
        dynamic_b2,
        dynamic_b4,
        exact_b4,
        dynamic_b8,
        exact_b8,
    )

    assert generated._program_for(1)[1] is dynamic_b2
    assert generated._program_for(2)[1] is dynamic_b2
    assert generated._program_for(3)[1] is dynamic_b4
    assert generated._program_for(4)[1] is exact_b4
    assert generated._program_for(5)[1] is dynamic_b8
    assert generated._program_for(8)[1] is exact_b8
    assert generated._program_for(9) is None
