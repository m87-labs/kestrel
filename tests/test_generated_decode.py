"""Compiler-generated decode runtime contract tests."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    _merge_disjoint_inputs,
    _require_uniform_weight_contract,
)


requires_mkl = pytest.mark.skipif(
    importlib.util.find_spec("mkl") is None,
    reason="requires the optional generated-decode compiler",
)


@requires_mkl
@pytest.mark.parametrize(
    ("batch_size", "expected_capacity"),
    [(3, 8), (5, 8), (7, 8), (11, 16), (16, 16)],
)
def test_run_uses_smallest_capacity_and_logical_extent(
    batch_size,
    expected_capacity,
):
    calls = []
    state = SimpleNamespace(
        invocation=SimpleNamespace(launch=lambda **kwargs: calls.append(kwargs)),
        argument_names={"active_batch", "kv_len"},
        required_launch_extents={"active_batch", "kv_len"},
    )
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._programs = {
        1: (object(),) * 3,
        8: (object(),) * 3,
        16: (object(),) * 3,
    }
    generated._slots = {(7, capacity): state for capacity in (8, 16)}
    generated._input_preparation_plan = ()
    generated._spec = SimpleNamespace(
        label="test",
        launch_extents=lambda slot, extent: {
            "active_batch": extent,
            "kv_len": int(slot.meta.input_pos.cpu[:extent].max()) + 1,
        },
        preparation_callbacks={},
    )
    slot = SimpleNamespace(
        slot_id=7,
        meta=SimpleNamespace(
            input_pos=SimpleNamespace(
                cpu=torch.arange(batch_size, dtype=torch.int32),
            )
        ),
    )

    assert generated.supports(batch_size)
    assert generated._capacity_for(batch_size) == expected_capacity
    generated.run(slot, batch_size)

    assert calls == [{"active_batch": batch_size, "kv_len": batch_size}]


def test_rejects_input_namespace_collisions():
    with pytest.raises(RuntimeError, match="owned by both shared and slot"):
        _merge_disjoint_inputs(
            "test",
            shared={"page_table": object()},
            capacity={"active_rows": object()},
            slot={"page_table": object()},
        )


def test_rejects_cross_capacity_weight_abi_drift():
    programs = {
        1: (
            SimpleNamespace(weight_binding_contract=("weight", "bf16")),
            None,
            None,
        ),
        8: (
            SimpleNamespace(weight_binding_contract=("weight", "fp8")),
            None,
            None,
        ),
    }

    with pytest.raises(RuntimeError, match="weight storage ABI"):
        _require_uniform_weight_contract("test", programs)


def test_accepts_uniform_cross_capacity_weight_abi():
    programs = {
        capacity: (
            SimpleNamespace(weight_binding_contract=("weight", "bf16")),
            None,
            None,
        )
        for capacity in (1, 8)
    }

    _require_uniform_weight_contract("test", programs)


@requires_mkl
def test_run_rejects_missing_dynamic_launch_extent():
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._programs = {1: (object(),) * 3}
    generated._slots = {
        (0, 1): SimpleNamespace(
            invocation=SimpleNamespace(launch=lambda **_kwargs: None),
            argument_names={"active_batch"},
            required_launch_extents={"active_batch"},
        )
    }
    generated._input_preparation_plan = ()
    generated._spec = SimpleNamespace(
        label="test",
        launch_extents=lambda _slot, _batch_size: {},
        preparation_callbacks={},
    )

    with pytest.raises(RuntimeError, match="active_batch"):
        generated.run(SimpleNamespace(slot_id=0), 1)


@requires_mkl
def test_capacity_selection_rejects_uncovered_extents():
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._programs = {
        1: (object(),) * 3,
        8: (object(),) * 3,
        16: (object(),) * 3,
    }

    assert generated._capacity_for(1) == 1
    assert generated._capacity_for(3) == 8
    assert generated._capacity_for(11) == 16
    assert generated._capacity_for(16) == 16
    assert generated._capacity_for(0) is None
    assert generated._capacity_for(17) is None
    assert not generated.supports(17)
