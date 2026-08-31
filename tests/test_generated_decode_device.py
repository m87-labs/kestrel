from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    prepare_generated_weight_storage_for_loading,
    reserve_generated_binding_storage,
)


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


@pytest.mark.parametrize(
    "missing_capability",
    (
        "allocate_weight_storage_for_loading",
        "finalize_weight_storage_after_loading",
    ),
)
def test_load_time_weight_preparation_fails_soft_only_when_optional(
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    from kestrel_kernels import generated_decode as generated_runtime

    runtime = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        max_batch_size=1,
    )
    program = _program(1, active_batch=1)
    program.descriptor = {"weights": []}
    properties = SimpleNamespace(major=9, minor=0, multi_processor_count=132)
    monkeypatch.setattr(
        generated_runtime,
        "resolve_compatible_programs",
        lambda *_args, **_kwargs: (program,),
    )
    monkeypatch.delattr(generated_runtime, missing_capability)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: properties
    )

    options = dict(
        label="Gemma",
        layer_prefix="model.language_model.layers",
        required_batch_sizes=(1,),
    )
    assert prepare_generated_weight_storage_for_loading(
        runtime, Mock(), required=False, **options
    ) is None
    with pytest.raises(RuntimeError, match="binding and finalization support"):
        prepare_generated_weight_storage_for_loading(
            runtime, Mock(), required=True, **options
        )


def test_binding_reservation_allocates_every_program_slot_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_kernels import generated_decode as generated_runtime

    programs = tuple(
        SimpleNamespace(descriptor={"owned_bytes": owned_bytes})
        for owned_bytes in (10, 20)
    )
    storage = SimpleNamespace(finalized=True, buffers={"weight": torch.empty(1)})
    first = {"slot": 1}
    second = {"slot": 2}
    calls = []
    compute_stream = object()
    stream_contexts = []

    @contextmanager
    def use_stream(stream):
        stream_contexts.append(("enter", stream))
        yield
        stream_contexts.append(("exit", stream))

    def reserve(descriptor, *, weights, runtime_inputs, device):
        calls.append((descriptor, weights, runtime_inputs))
        return (
            torch.empty(
                descriptor["owned_bytes"] + runtime_inputs["slot"],
                dtype=torch.uint8,
                device=device,
            ),
        )

    monkeypatch.setattr(
        generated_runtime,
        "reserve_binding_storage",
        reserve,
        raising=False,
    )
    monkeypatch.setattr(
        "kestrel.runtime.generated_decode.stream_context",
        use_stream,
    )

    reservation = reserve_generated_binding_storage(
        programs,
        weight_storage=storage,
        runtime_inputs_by_slot=(first, second),
        device=torch.device("cpu"),
        stream=compute_stream,
        label="test",
        required=True,
    )

    assert reservation is not None
    assert [tensor.dtype for tensor in reservation] == [torch.uint8] * 4
    assert [tensor.numel() for tensor in reservation] == [11, 21, 12, 22]
    assert calls == [
        (programs[0].descriptor, storage.buffers, first),
        (programs[1].descriptor, storage.buffers, first),
        (programs[0].descriptor, storage.buffers, second),
        (programs[1].descriptor, storage.buffers, second),
    ]
    assert stream_contexts == [
        ("enter", compute_stream),
        ("exit", compute_stream),
    ]


def test_binding_reservation_fails_soft_only_when_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_kernels import generated_decode as generated_runtime

    monkeypatch.delattr(
        generated_runtime,
        "reserve_binding_storage",
        raising=False,
    )
    options = dict(
        programs=(SimpleNamespace(descriptor={}),),
        weight_storage=SimpleNamespace(finalized=True, buffers={}),
        runtime_inputs_by_slot=({},),
        device=torch.device("cpu"),
        stream=None,
        label="test",
    )

    assert reserve_generated_binding_storage(required=False, **options) is None
    with pytest.raises(RuntimeError, match="binding-storage reservation"):
        reserve_generated_binding_storage(required=True, **options)
