from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    PagedDecodeBindings,
    _program_lookup,
    _selectable_programs,
    prepare_generated_weight_storage_for_loading,
    reserve_generated_binding_storage,
)


def test_paged_launch_extents_use_scheduler_position_scalar() -> None:
    slot = SimpleNamespace(meta=SimpleNamespace(max_input_pos=37))

    assert PagedDecodeBindings(layers=()).launch_extents(slot, 3) == {
        "active_batch": 3,
        "kv_len": 38,
    }


def _program(
    capacity,
    *,
    active_batch=None,
    minimum_batch=1,
    minimums=None,
    name=None,
    num_ctas=132,
):
    static = {} if active_batch is None else {"active_batch": active_batch}
    runtime_minimums = (
        {} if minimum_batch == 1 else {"active_batch": minimum_batch}
    )
    runtime_minimums.update(minimums or {})
    return SimpleNamespace(
        capacity=capacity,
        static_extent_bindings=static,
        runtime_extent_minimums=runtime_minimums,
        name=name,
        num_ctas=num_ctas,
    )


def _generated_with_programs(*programs):
    generated = GeneratedDecode.__new__(GeneratedDecode)
    generated._device_sms = 132
    generated._programs = programs
    generated._program_by_batch = _program_lookup(
        programs,
        max(program.capacity for program in programs),
        device_sms=generated._device_sms,
    )
    return generated


def test_try_create_binds_physical_sm_count_to_program_resolution():
    runtime = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    weight_sources = {"logical.weight": torch.empty(1)}
    spec = SimpleNamespace(
        bindings=SimpleNamespace(is_eligible=lambda value: value is runtime),
        weight_root=Mock(),
        weight_layer_prefix="model.layers",
        weight_sources=weight_sources,
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
        weight_sources=weight_sources,
    )


def test_program_selection_preserves_legacy_static_preference():
    dynamic_b2 = _program(2)
    dynamic_b4 = _program(4)
    exact_b4 = _program(4, active_batch=4)
    dynamic_b8 = _program(8)
    exact_b8 = _program(8, active_batch=8)
    generated = _generated_with_programs(
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
    b1 = _program(1, active_batch=1)
    b2 = _program(2, minimum_batch=2)
    b4 = _program(4, minimum_batch=3)
    b8 = _program(8, minimum_batch=5)
    generated = _generated_with_programs(b1, b2, b4, b8)

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


def test_program_selection_reselects_by_live_runtime_extent():
    u2 = _program(8, minimum_batch=5, name="u2")
    u4 = _program(
        8,
        minimum_batch=5,
        minimums={"kv_len": 3201},
        name="u4",
    )
    generated = _generated_with_programs(u2, u4)

    assert generated._program_for(8)[1] is u2
    assert generated._program_for(
        8, {"active_batch": 8, "kv_len": 3200}
    )[1] is u2
    assert generated._program_for(
        8, {"active_batch": 8, "kv_len": 3201}
    )[1] is u4

    with pytest.raises(RuntimeError, match="active_batch disagrees"):
        generated._program_for(8, {"active_batch": 7, "kv_len": 3201})
    with pytest.raises(RuntimeError, match="positive exact integer"):
        generated._program_for(
            8, {"active_batch": 8, "kv_len": "3201"}
        )
    with pytest.raises(RuntimeError, match="positive exact integer"):
        generated._program_for(
            8, {"active_batch": 8, "kv_len": True}
        )
    with pytest.raises(RuntimeError, match="positive exact integer"):
        generated._program_for(True)


def test_selectable_programs_ignore_unrelated_static_runtime_extent():
    fallback = _program(8, minimum_batch=5, name="dynamic")
    static_kv = _program(8, minimum_batch=5, name="static-kv")
    static_kv.static_extent_bindings = {"kv_len": 4096}

    assert _selectable_programs(
        (fallback, static_kv), 8, device_sms=132,
    ) == (fallback,)


def test_program_selection_rejects_invalid_runtime_interval():
    with pytest.raises(RuntimeError, match="invalid active-batch interval"):
        _generated_with_programs(_program(4, minimum_batch=5))


def test_program_selection_does_not_rescan_after_construction(monkeypatch):
    generated = _generated_with_programs(_program(2), _program(4))
    monkeypatch.setattr(
        "kestrel.runtime.generated_decode._select_program",
        lambda *_args, **_kwargs: pytest.fail("program selection rescanned"),
    )

    assert generated.supports(3)
    assert generated._program_for(3)[1].capacity == 4
    assert generated._program_for(5) is None


def test_slot_capacity_uses_selected_program_physical_capacity(monkeypatch):
    runtime = SimpleNamespace(
        max_batch_size=1,
        device=torch.device("cuda", 0),
    )
    b8 = _program(8)
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: (b8,)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=132),
    )

    assert GeneratedDecode.resolve_slot_capacity(
        runtime,
        object(),
        required_batch_sizes=(1,),
    ) == 8


def test_slot_capacity_refuses_incomplete_required_domain(monkeypatch):
    runtime = SimpleNamespace(
        max_batch_size=4,
        device=torch.device("cuda", 0),
    )
    programs = tuple(
        _program(batch_size, active_batch=batch_size)
        for batch_size in (1, 2, 4)
    )
    monkeypatch.setattr(
        GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: programs),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=132),
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

    def reserve(descriptor, *, weights, runtime_inputs, device, stream):
        calls.append((descriptor, weights, runtime_inputs, stream))
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
        (programs[0].descriptor, storage.buffers, first, compute_stream),
        (programs[1].descriptor, storage.buffers, first, compute_stream),
        (programs[0].descriptor, storage.buffers, second, compute_stream),
        (programs[1].descriptor, storage.buffers, second, compute_stream),
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
