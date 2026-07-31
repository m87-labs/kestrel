"""Runtime binding for compiler-generated decode bundles."""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from .carried_state import StateRepresentationRequirement


@dataclass(frozen=True)
class DeviceInputPreparation:
    name: str
    produces: tuple[str, ...]
    requires: tuple[str, ...] = ()


class GeneratedDecodeBindings(Protocol):
    def is_eligible(self, runtime: Any) -> bool: ...
    def runtime_inputs(self, runtime: Any) -> Mapping[str, Any]: ...
    def slot_inputs(self, slot: Any, capacity: int) -> Mapping[str, Any]: ...
    def launch_extents(self, slot: Any, batch_size: int) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class PagedDecodeBindings:
    """Common runtime ABI for generated decode over paged KV."""

    layers: Sequence[Any]
    kv_sets: Sequence[tuple[str, str | None]] = (("", None),)
    layer_kinds: Sequence[str] | None = None
    extra_runtime_inputs: Callable[[Any], Mapping[str, Any]] | None = None
    extra_slot_inputs: Callable[[Any, int], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.layer_kinds is not None and len(self.layer_kinds) != len(self.layers):
            raise ValueError("paged KV layers and layer kinds must have equal length")
        if self.layer_kinds is None and any(
            kind is not None for _name, kind in self.kv_sets
        ):
            raise ValueError("filtered paged KV inputs require layer kinds")

    def is_eligible(self, _runtime: Any) -> bool:
        return all(
            layer is None
            or int(layer.k_cache.shape[2]) == int(layer.v_cache.shape[2]) == 1
            for layer in self.layers
        )

    def _paged_tensors(self, field: str, layer_kind: str | None):
        kinds = (
            self.layer_kinds
            if self.layer_kinds is not None
            else (None,) * len(self.layers)
        )
        return [
            None
            if layer is None or (layer_kind is not None and kind != layer_kind)
            else getattr(layer, field)[:, :, 0, :]
            for kind, layer in zip(kinds, self.layers, strict=True)
        ]

    def runtime_inputs(self, runtime: Any) -> Mapping[str, Any]:
        inputs = {"page_table": runtime.page_table.page_table, "kv_len": 1}
        for name, layer_kind in self.kv_sets:
            suffix = f"_{name}" if name else ""
            inputs[f"mK{suffix}"] = self._paged_tensors("k_cache", layer_kind)
            inputs[f"mV{suffix}"] = self._paged_tensors("v_cache", layer_kind)
        return _merge_disjoint(
            "decode",
            paged=inputs,
            extra=(self.extra_runtime_inputs(runtime)
                   if self.extra_runtime_inputs else {}),
        )

    def slot_inputs(self, slot: Any, capacity: int) -> Mapping[str, Any]:
        inputs = {
            "input_ids": slot.decode_token_ids[:capacity],
            "final_norm": slot.hidden_last[:capacity],
            "logits": slot.logits[:capacity],
            "batch_idx": slot.meta.batch_idx.gpu[:capacity],
            "input_pos": slot.meta.input_pos.gpu[:capacity],
        }
        return _merge_disjoint(
            "decode slot",
            standard=inputs,
            extra=(self.extra_slot_inputs(slot, capacity)
                   if self.extra_slot_inputs else {}),
        )

    @staticmethod
    def launch_extents(slot: Any, batch_size: int) -> Mapping[str, int]:
        return {
            "active_batch": int(batch_size),
            "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
        }


@dataclass(frozen=True)
class GeneratedDecodeSpec:
    label: str
    weight_root: torch.nn.Module
    weight_layer_prefix: str
    bindings: GeneratedDecodeBindings
    capacity_inputs: Callable[
        [int, tuple[StateRepresentationRequirement, ...]], Mapping[str, Any]
    ] | None = None
    preparations: Sequence[DeviceInputPreparation] = ()
    not_ready_inputs: frozenset[str] = frozenset()
    preparation_callbacks: Mapping[str, Callable[[Any, int], None]] = field(
        default_factory=dict)


@dataclass(frozen=True)
class _BoundInvocation:
    invocation: Any
    scalar_names: frozenset[str]
    required_launch_extents: frozenset[str]


def _merge_disjoint(label: str, **namespaces: Mapping[str, Any]) -> dict[str, Any]:
    merged = {}
    owners = {}
    for namespace, values in namespaces.items():
        for name, value in values.items():
            if name in merged:
                raise RuntimeError(
                    f"generated {label} input {name!r} is owned by "
                    f"{owners[name]} and {namespace}")
            merged[name] = value
            owners[name] = namespace
    return merged


def _required_engine_inputs(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    arguments = {
        item["name"]
        for item in descriptor["device_program"]["argument_plan"]["arguments"]
        if item["source"] == "external"
    }
    return tuple(dict.fromkeys(
        item["logical_name"]
        for item in descriptor["device_program"]["physical_abi"]["operands"]
        if item["abi_name"] in arguments and item["owner"] == "engine"
    ))


def _preparation_plan(
    descriptor: Mapping[str, Any],
    *,
    ready: set[str],
    preparations: Sequence[DeviceInputPreparation],
) -> tuple[DeviceInputPreparation, ...]:
    producers = {
        output: step for step in preparations for output in step.produces
    }
    selected = []
    visiting = set()

    def require(name: str) -> None:
        if name in ready:
            return
        try:
            step = producers[name]
        except KeyError as exc:
            raise RuntimeError(
                f"generated input {name!r} is neither ready nor prepared") from exc
        if step.name in visiting:
            raise RuntimeError(f"generated input preparation cycle at {step.name!r}")
        if step not in selected:
            visiting.add(step.name)
            for dependency in step.requires:
                require(dependency)
            visiting.remove(step.name)
            selected.append(step)
            ready.update(step.produces)

    for name in _required_engine_inputs(descriptor):
        require(name)
    return tuple(selected)


def _state_requirements(descriptor: Mapping[str, Any]):
    return tuple(
        StateRepresentationRequirement(
            item["buffer"],
            item["representation"],
            tuple(item["storage_axis_order"]),
            item["storage_dtype"],
        )
        for item in descriptor["carried_state"]
    )


class GeneratedDecode:
    """Select bundled capacities and bind them to one serving runtime."""

    @classmethod
    def try_create(
        cls, runtime: Any, spec: GeneratedDecodeSpec,
    ) -> "GeneratedDecode | None":
        if (
            not spec.bindings.is_eligible(runtime)
            or runtime.device.type != "cuda"
            or runtime.dtype is not torch.bfloat16
        ):
            return None
        from kestrel_kernels.generated_decode import resolve_compatible_programs

        properties = torch.cuda.get_device_properties(runtime.device)
        programs = resolve_compatible_programs(
            spec.weight_root,
            layer_prefix=spec.weight_layer_prefix,
            arch=f"sm{properties.major}{properties.minor}",
        )
        if not programs:
            return None
        return cls(runtime, spec=spec, programs=programs)

    def __init__(self, runtime: Any, *, spec: GeneratedDecodeSpec, programs) -> None:
        from kestrel_kernels.generated_decode import (
            assemble_bindings,
            derive_runtime_extents,
            materialize_weights,
        )

        self._programs = {program.capacity: program for program in programs}
        self._spec = spec
        contracts = {repr(program.descriptor["weights"]) for program in programs}
        if len(contracts) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on weight storage")
        self.state_requirements_by_capacity = {
            program.capacity: _state_requirements(program.descriptor)
            for program in programs
        }
        state_sets = {
            tuple(item.buffer for item in requirements)
            for requirements in self.state_requirements_by_capacity.values()
        }
        if len(state_sets) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on carried state")
        self.state_buffers = next(iter(state_sets))

        ambient_stream = torch.cuda.current_stream(runtime.device)
        model_ready = torch.cuda.Event()
        model_ready.record(ambient_stream)
        runtime.compute_stream.wait_event(model_ready)
        weights_ready = torch.cuda.Event()
        with torch.cuda.stream(runtime.compute_stream):
            self.weight_storage = materialize_weights(
                spec.weight_root,
                programs[0].descriptor,
                layer_prefix=spec.weight_layer_prefix,
            )
            shared_inputs = dict(spec.bindings.runtime_inputs(runtime))
            weights_ready.record(runtime.compute_stream)
        ambient_stream.wait_event(weights_ready)

        self._slots = {}
        plans = set()
        for slot in runtime.decode_slots:
            for capacity, program in self._programs.items():
                requirements = self.state_requirements_by_capacity[capacity]
                capacity_inputs = (
                    dict(spec.capacity_inputs(capacity, requirements))
                    if spec.capacity_inputs else {}
                )
                inputs = _merge_disjoint(
                    spec.label,
                    shared=shared_inputs,
                    capacity=capacity_inputs,
                    slot=dict(spec.bindings.slot_inputs(slot, capacity)),
                )
                plan = _preparation_plan(
                    program.descriptor,
                    ready=set(inputs) - spec.not_ready_inputs,
                    preparations=spec.preparations,
                )
                plans.add(tuple(step.name for step in plan))
                construction_batch = min(capacity, int(runtime.max_batch_size))
                extents = derive_runtime_extents(
                    program.descriptor, inputs, active_batch=construction_batch)
                launch_extents = dict(
                    spec.bindings.launch_extents(slot, construction_batch))
                extents.update(launch_extents)
                bindings = assemble_bindings(
                    program.descriptor,
                    weights=self.weight_storage.buffers,
                    runtime_inputs=inputs,
                    runtime_extents=extents,
                    stream=slot.compute_stream,
                    device=runtime.device,
                )
                scalar_names = frozenset(
                    item["name"]
                    for item in program.descriptor["device_program"]
                    ["argument_plan"]["arguments"]
                    if item["transport"] == "scalar"
                )
                unknown = launch_extents.keys() - scalar_names
                if unknown:
                    raise RuntimeError(
                        f"generated {spec.label} has unknown launch extents "
                        f"{sorted(unknown)}")
                self._slots[(int(slot.slot_id), capacity)] = _BoundInvocation(
                    program.bind(bindings), scalar_names,
                    frozenset(launch_extents),
                )
        if len(plans) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on input preparation")
        self._input_preparation_plan = next(iter(plans))
        missing = {
            step.name for step in self._input_preparation_plan
        } - spec.preparation_callbacks.keys()
        if missing:
            raise RuntimeError(
                f"generated {spec.label} has no preparation callbacks for "
                f"{sorted(missing)}")

    def _capacity_for(self, batch_size: int) -> int | None:
        return next(
            (capacity for capacity in sorted(self._programs)
             if capacity >= int(batch_size)),
            None,
        )

    def supports(self, batch_size: int) -> bool:
        return self._capacity_for(batch_size) is not None

    def state_requirements_for(
        self, batch_size: int,
    ) -> tuple[StateRepresentationRequirement, ...]:
        capacity = self._capacity_for(batch_size)
        if capacity is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        return self.state_requirements_by_capacity[capacity]

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        capacity = self._capacity_for(batch_size)
        if capacity is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        for step in self._input_preparation_plan:
            self._spec.preparation_callbacks[step.name](slot, int(batch_size))
        bound = self._slots[(int(slot.slot_id), capacity)]
        extents = dict(self._spec.bindings.launch_extents(slot, int(batch_size)))
        missing = bound.required_launch_extents - extents.keys()
        if missing:
            raise RuntimeError(
                f"generated {self._spec.label} launch misses {sorted(missing)}")
        bound.invocation.launch(**{
            name: value for name, value in extents.items()
            if name in bound.scalar_names
        })


__all__ = [
    "DeviceInputPreparation",
    "GeneratedDecode",
    "GeneratedDecodeBindings",
    "GeneratedDecodeSpec",
    "PagedDecodeBindings",
]
