from types import SimpleNamespace

import pytest

from kestrel.runtime.carried_state import (
    CarriedStateCoordinator,
    StateRepresentationRequirement,
)
from kestrel.runtime.generated_decode import GeneratedDecode, _merge_disjoint


def test_generated_decode_uses_smallest_capacity():
    calls = []
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._programs = {1: object(), 8: object(), 16: object()}
    generated._slots = {
        (7, capacity): SimpleNamespace(
            invocation=SimpleNamespace(
                launch=lambda **values: calls.append(values)),
            scalar_names=frozenset({"active_batch"}),
            required_launch_extents=frozenset({"active_batch"}),
        )
        for capacity in (8, 16)
    }
    generated._input_preparation_plan = ()
    generated._spec = SimpleNamespace(
        label="test",
        bindings=SimpleNamespace(
            launch_extents=lambda _slot, batch: {"active_batch": batch}),
        preparation_callbacks={},
    )
    slot = SimpleNamespace(slot_id=7)

    assert generated._capacity_for(3) == 8
    assert generated._capacity_for(11) == 16
    assert not generated.supports(17)
    generated.run(slot, 3)
    assert calls == [{"active_batch": 3}]


def test_generated_inputs_have_one_owner():
    with pytest.raises(RuntimeError, match="shared and slot"):
        _merge_disjoint(
            "test",
            shared={"page_table": object()},
            slot={"page_table": object()},
        )


def test_carried_state_converts_only_after_a_path_switch():
    calls = []
    coordinator = CarriedStateCoordinator(
        buffers=("state",),
        rows=range(4),
        transitions={
            "state": lambda source, target, rows: calls.append(
                (source.representation, target.representation, rows)),
        },
    )
    native = (StateRepresentationRequirement("state", "native"),)
    generated = (StateRepresentationRequirement("state", "generated"),)

    coordinator.prepare(native, (0, 1))
    coordinator.prepare(native, (0, 1))
    coordinator.prepare(generated, (1,))

    assert calls == [("native", "generated", (1,))]
