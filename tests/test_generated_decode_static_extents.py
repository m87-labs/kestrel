import contextlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from kestrel.runtime import generated_decode as runtime_decode


class _Stream:
    def wait_event(self, _event) -> None:
        pass


class _Event:
    def record(self, _stream) -> None:
        pass


class _Invocation:
    def __init__(self, name: str, launches: list[tuple[str, dict]]) -> None:
        self.name = name
        self.launches = launches

    def launch(self, **extents) -> None:
        self.launches.append((self.name, dict(extents)))

    def prepare_launch(self, **extents):
        return lambda: self.launch(**extents)


@dataclass(frozen=True)
class _Program:
    name: str
    capacity: int
    static_extent_bindings: dict[str, int]
    launches: list[tuple[str, dict]]

    @property
    def descriptor(self):
        scalar_arguments = (
            []
            if "active_batch" in self.static_extent_bindings
            else [
                {
                    "name": "active_batch",
                    "transport": "scalar",
                    "source": "external",
                }
            ]
        )
        return {
            "name": self.name,
            "weights": [],
            "carried_state": [],
            "device_program": {
                "argument_plan": {"arguments": scalar_arguments},
                "physical_abi": {"operands": []},
            },
        }

    def bind(self, _bindings):
        return _Invocation(self.name, self.launches)


class _Bindings:
    def runtime_inputs(self, _runtime):
        return {}

    def slot_inputs(self, _slot, _capacity):
        return {}

    def launch_extents(self, _slot, batch_size):
        return {"active_batch": int(batch_size)}


def _programs(*names: str):
    launches: list[tuple[str, dict]] = []
    definitions = {
        "b1": (1, {"active_batch": 1}),
        "b2": (2, {}),
        "b4": (4, {}),
        "b4_exact": (4, {"active_batch": 4}),
        "b8": (8, {}),
        "b8_exact": (8, {"active_batch": 8}),
        "b4_multi_exact": (4, {"active_batch": 4, "kv_len": 16}),
    }
    return tuple(_Program(name, *definitions[name], launches) for name in names)


def _build(
    monkeypatch,
    programs,
    *,
    bindings=None,
    max_batch_size=8,
    weight_sources=None,
    materialize_calls=None,
):
    launches = programs[0].launches
    kernels = ModuleType("kestrel_kernels")
    generated = ModuleType("kestrel_kernels.generated_decode")
    generated.assemble_bindings = lambda _descriptor, **_kwargs: {}
    generated.derive_runtime_extents = lambda _descriptor, _inputs, *, active_batch: {
        "active_batch": int(active_batch)
    }

    def materialize_weights(*_args, **kwargs):
        if materialize_calls is not None:
            materialize_calls.append(kwargs)
        return SimpleNamespace(buffers={})

    generated.materialize_weights = materialize_weights
    monkeypatch.setitem(sys.modules, "kestrel_kernels", kernels)
    monkeypatch.setitem(sys.modules, "kestrel_kernels.generated_decode", generated)
    monkeypatch.setattr(
        runtime_decode.torch.cuda, "current_stream", lambda _device: _Stream()
    )
    monkeypatch.setattr(runtime_decode.torch.cuda, "Event", _Event)
    monkeypatch.setattr(
        runtime_decode.torch.cuda, "stream", lambda _stream: contextlib.nullcontext()
    )
    runtime = SimpleNamespace(
        max_batch_size=max_batch_size,
        device=SimpleNamespace(type="cuda"),
        compute_stream=_Stream(),
        decode_slots=(SimpleNamespace(slot_id=0, compute_stream=_Stream()),),
    )
    spec = runtime_decode.GeneratedDecodeSpec(
        label="test",
        weight_root=SimpleNamespace(),
        weight_layer_prefix="layers",
        bindings=bindings or _Bindings(),
        weight_sources=weight_sources,
    )
    return runtime_decode.GeneratedDecode(
        runtime, spec=spec, programs=programs
    ), launches


def test_generated_decode_materializes_explicit_weight_sources(monkeypatch):
    sources = {"model.layers.0.weight": object()}
    calls = []

    _build(
        monkeypatch,
        _programs("b1"),
        max_batch_size=1,
        weight_sources=sources,
        materialize_calls=calls,
    )

    assert calls == [{"layer_prefix": "layers", "weight_sources": sources}]


def test_generated_decode_constructs_and_selects_dynamic_exact_siblings(monkeypatch):
    generated, launches = _build(
        monkeypatch,
        _programs("b1", "b2", "b4", "b4_exact", "b8", "b8_exact"),
    )

    for active_batch in (3, 4, 5, 6, 7, 8):
        generated.run(SimpleNamespace(slot_id=0), active_batch)

    assert launches == [
        ("b4", {"active_batch": 3}),
        ("b4_exact", {}),
        ("b8", {"active_batch": 5}),
        ("b8", {"active_batch": 6}),
        ("b8", {"active_batch": 7}),
        ("b8_exact", {}),
    ]


def test_static_launcher_resolves_capacity_once(monkeypatch):
    generated, launches = _build(
        monkeypatch,
        _programs("b1", "b2", "b4", "b8"),
    )

    launch = generated.static_launcher(SimpleNamespace(slot_id=0), 3)
    launch()
    launch()

    assert launches == [
        ("b4", {"active_batch": 3}),
        ("b4", {"active_batch": 3}),
    ]


def test_generated_decode_missing_covering_artifact_fails_before_launch(monkeypatch):
    generated, launches = _build(
        monkeypatch,
        _programs("b1", "b2", "b4_exact", "b8_exact"),
    )

    assert not generated.supports(3)
    with pytest.raises(ValueError, match="no generated decode capacity covers 3"):
        generated.run(SimpleNamespace(slot_id=0), 3)
    assert launches == []


def test_generated_decode_rejects_mismatched_static_construction_extent(monkeypatch):
    class _MismatchedBindings(_Bindings):
        def launch_extents(self, _slot, _batch_size):
            return {"active_batch": 3}

    with pytest.raises(
        RuntimeError,
        match="construction extents disagree with static artifact bindings",
    ):
        _build(
            monkeypatch,
            _programs("b4_exact"),
            bindings=_MismatchedBindings(),
        )


def test_generated_decode_uses_dynamic_fallback_for_other_static_extents(monkeypatch):
    generated, launches = _build(
        monkeypatch,
        _programs("b4", "b4_multi_exact"),
    )

    generated.run(SimpleNamespace(slot_id=0), 4)

    assert [program.name for program in generated._programs] == ["b4"]
    assert launches == [("b4", {"active_batch": 4})]


def test_generated_decode_skips_artifacts_above_runtime_batch_limit(monkeypatch):
    generated, _launches = _build(
        monkeypatch,
        _programs("b1", "b2", "b4", "b4_exact", "b8", "b8_exact"),
        max_batch_size=4,
    )

    assert [program.name for program in generated._programs] == [
        "b1",
        "b2",
        "b4",
        "b4_exact",
    ]


def test_optional_generated_decode_falls_back_when_all_artifacts_are_unreachable(
    monkeypatch,
):
    programs = _programs("b8_exact")
    monkeypatch.setattr(
        runtime_decode.GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: programs),
    )
    runtime = SimpleNamespace(max_batch_size=4)

    assert runtime_decode.GeneratedDecode.try_create(runtime, object()) is None


def test_required_generated_decode_rejects_all_unreachable_artifacts(monkeypatch):
    programs = _programs("b8_exact")
    monkeypatch.setattr(
        runtime_decode.GeneratedDecode,
        "_resolve_programs",
        classmethod(lambda _cls, _runtime, _spec: programs),
    )
    runtime = SimpleNamespace(max_batch_size=4)
    spec = SimpleNamespace(label="test")

    with pytest.raises(
        RuntimeError,
        match=r"does not cover active batch sizes \[1, 2, 3, 4\]",
    ):
        runtime_decode.GeneratedDecode.require(
            runtime, spec, batch_sizes=range(1, 5))
