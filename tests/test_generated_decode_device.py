from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from kestrel.runtime.generated_decode import GeneratedDecode


def _program(capacity, *, active_batch=None, minimum_batch=1):
    static = {} if active_batch is None else {"active_batch": active_batch}
    return SimpleNamespace(
        capacity=capacity,
        static_extent_bindings=static,
        runtime_extent_minimums=(
            {} if minimum_batch == 1 else {"active_batch": minimum_batch}
        ),
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


def test_program_selection_preserves_legacy_static_preference():
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


def test_program_selection_partitions_dynamic_runtime_intervals():
    generated = GeneratedDecode.__new__(GeneratedDecode)
    b1 = _program(1, active_batch=1)
    b2 = _program(2, minimum_batch=2)
    b4 = _program(4, minimum_batch=3)
    b8 = _program(8, minimum_batch=5)
    generated._programs = (b1, b2, b4, b8)

    assert [generated._program_for(batch_size)[1] for batch_size in range(1, 9)] == [
        b1,
        b2,
        b4,
        b4,
        b8,
        b8,
        b8,
        b8,
    ]


def test_program_selection_rejects_invalid_runtime_interval():
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._programs = (_program(4, minimum_batch=5),)

    with pytest.raises(RuntimeError, match="invalid active-batch interval"):
        generated._program_for(4)


def test_slot_capacity_uses_selected_program_physical_capacity(monkeypatch):
    runtime = SimpleNamespace(max_batch_size=1)
    b8 = _program(8)
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: (b8,)),
    )

    assert GeneratedDecode.resolve_slot_capacity(
        runtime,
        object(),
        required_batch_sizes=(1,),
    ) == 8


def test_slot_capacity_refuses_incomplete_required_domain(monkeypatch):
    runtime = SimpleNamespace(max_batch_size=4)
    programs = tuple(
        _program(batch_size, active_batch=batch_size)
        for batch_size in (1, 2, 4)
    )
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: programs),
    )

    assert GeneratedDecode.resolve_slot_capacity(
        runtime,
        object(),
        required_batch_sizes=range(1, 5),
    ) is None
