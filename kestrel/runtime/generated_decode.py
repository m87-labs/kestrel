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
            extra=(
                self.extra_runtime_inputs(runtime) if self.extra_runtime_inputs else {}
            ),
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
            extra=(
                self.extra_slot_inputs(slot, capacity) if self.extra_slot_inputs else {}
            ),
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
    weight_sources: Mapping[str, torch.Tensor] | None = None
    engine_buffers: Mapping[str, torch.Tensor] | None = None
    capacity_inputs: (
        Callable[[int, tuple[StateRepresentationRequirement, ...]], Mapping[str, Any]]
        | None
    ) = None
    preparations: Sequence[DeviceInputPreparation] = ()
    not_ready_inputs: frozenset[str] = frozenset()
    preparation_callbacks: Mapping[str, Callable[[Any, int], None]] = field(
        default_factory=dict
    )


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
                    f"{owners[name]} and {namespace}"
                )
            merged[name] = value
            owners[name] = namespace
    return merged


def _required_engine_inputs(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    arguments = {
        item["name"]
        for item in descriptor["device_program"]["argument_plan"]["arguments"]
        if item["source"] == "external"
    }
    return tuple(
        dict.fromkeys(
            item["logical_name"]
            for item in descriptor["device_program"]["physical_abi"]["operands"]
            if item["abi_name"] in arguments and item["owner"] == "engine"
        )
    )


def _preparation_plan(
    descriptor: Mapping[str, Any],
    *,
    ready: set[str],
    preparations: Sequence[DeviceInputPreparation],
) -> tuple[DeviceInputPreparation, ...]:
    producers = {output: step for step in preparations for output in step.produces}
    selected = []
    visiting = set()

    def require(name: str) -> None:
        if name in ready:
            return
        try:
            step = producers[name]
        except KeyError as exc:
            raise RuntimeError(
                f"generated input {name!r} is neither ready nor prepared"
            ) from exc
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


def _select_program(programs: Sequence[Any], batch_size: int) -> tuple[int, Any] | None:
    batch_size = int(batch_size)
    candidates = []
    for index, program in enumerate(programs):
        static_extents = program.static_extent_bindings
        if static_extents.keys() - {"active_batch"}:
            continue
        static_batch = static_extents.get("active_batch")
        if static_batch is not None and int(static_batch) != batch_size:
            continue
        if int(program.capacity) < batch_size:
            continue
        candidates.append((static_batch is None, int(program.capacity), index, program))
    if not candidates:
        return None
    _dynamic, _capacity, index, program = min(candidates, key=lambda item: item[:3])
    return index, program


def _selectable_programs(
    programs: Sequence[Any], max_batch_size: int
) -> tuple[Any, ...]:
    selected_indexes = set()
    for batch_size in range(1, int(max_batch_size) + 1):
        selected = _select_program(programs, batch_size)
        if selected is not None:
            selected_indexes.add(selected[0])
    return tuple(
        program
        for index, program in enumerate(programs)
        if index in selected_indexes
    )


class GeneratedDecode:
    """Select bundled capacities and bind them to one serving runtime."""

    @classmethod
    def _resolve_programs(
        cls, runtime: Any, spec: GeneratedDecodeSpec,
    ) -> tuple[Any, ...]:
        if (
            not spec.bindings.is_eligible(runtime)
            or runtime.device.type != "cuda"
            or runtime.dtype is not torch.bfloat16
        ):
            return ()
        try:
            from kestrel_kernels.generated_decode import resolve_compatible_programs
        except ModuleNotFoundError as exc:
            if exc.name != "kestrel_kernels.generated_decode":
                raise
            return ()

        properties = torch.cuda.get_device_properties(runtime.device)
        return tuple(resolve_compatible_programs(
            spec.weight_root,
            layer_prefix=spec.weight_layer_prefix,
            arch=f"sm{properties.major}{properties.minor}",
            device_sms=int(properties.multi_processor_count),
            weight_sources=spec.weight_sources,
        ))

    @classmethod
    def require(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
        *,
        capacity: int,
    ) -> "GeneratedDecode":
        programs = cls._resolve_programs(runtime, spec)
        if not programs:
            raise RuntimeError(
                f"{spec.label} requires a compatible bundled generated-decode program"
            )
        return cls(
            runtime,
            spec=spec,
            programs=programs,
            required_capacity=capacity,
        )

    @classmethod
    def try_create(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
    ) -> "GeneratedDecode | None":
        programs = cls._resolve_programs(runtime, spec)
        if not programs:
            return None
        programs = _selectable_programs(programs, runtime.max_batch_size)
        if not programs:
            return None
        return cls(runtime, spec=spec, programs=programs)

    def __init__(
        self,
        runtime: Any,
        *,
        spec: GeneratedDecodeSpec,
        programs,
        required_capacity: int | None = None,
    ) -> None:
        available_programs = tuple(programs)
        self._programs = available_programs
        self._spec = spec
        if required_capacity is not None:
            missing = [
                batch_size
                for batch_size in range(1, int(required_capacity) + 1)
                if not self.supports(batch_size)
            ]
            if missing:
                raise RuntimeError(
                    f"generated {spec.label} decode does not cover active batch "
                    f"sizes {missing}"
                )

        self._programs = _selectable_programs(
            available_programs, runtime.max_batch_size
        )
        if not self._programs:
            raise ValueError("generated decode has no selectable batch artifact")

        from kestrel_kernels.generated_decode import (
            assemble_bindings,
            derive_runtime_extents,
            materialize_weights,
        )

        contracts = {repr(program.descriptor["weights"]) for program in self._programs}
        if len(contracts) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on weight storage"
            )
        self.state_requirements_by_capacity = {}
        for program in self._programs:
            requirements = _state_requirements(program.descriptor)
            existing = self.state_requirements_by_capacity.setdefault(
                program.capacity, requirements
            )
            if existing != requirements:
                raise RuntimeError(
                    f"generated {spec.label} programs at capacity="
                    f"{program.capacity} disagree on carried state"
                )
        state_sets = {
            tuple(item.buffer for item in requirements)
            for requirements in self.state_requirements_by_capacity.values()
        }
        if len(state_sets) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on carried state"
            )
        self.state_buffers = next(iter(state_sets))

        ambient_stream = torch.cuda.current_stream(runtime.device)
        model_ready = torch.cuda.Event()
        model_ready.record(ambient_stream)
        runtime.compute_stream.wait_event(model_ready)
        weights_ready = torch.cuda.Event()
        with torch.cuda.stream(runtime.compute_stream):
            self.weight_storage = materialize_weights(
                spec.weight_root,
                self._programs[0].descriptor,
                layer_prefix=spec.weight_layer_prefix,
                engine_buffers=spec.engine_buffers,
                weight_sources=spec.weight_sources,
            )
            shared_inputs = dict(spec.bindings.runtime_inputs(runtime))
            weights_ready.record(runtime.compute_stream)
        ambient_stream.wait_event(weights_ready)

        self._slots = {}
        plans = {}
        for slot in runtime.decode_slots:
            for program_index, program in enumerate(self._programs):
                capacity = program.capacity
                requirements = self.state_requirements_by_capacity[capacity]
                capacity_inputs = (
                    dict(spec.capacity_inputs(capacity, requirements))
                    if spec.capacity_inputs
                    else {}
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
                plans.setdefault(tuple(step.name for step in plan), plan)
                construction_batch = int(
                    program.static_extent_bindings.get(
                        "active_batch",
                        min(capacity, int(runtime.max_batch_size)),
                    )
                )
                extents = derive_runtime_extents(
                    program.descriptor, inputs, active_batch=construction_batch
                )
                launch_extents = dict(
                    spec.bindings.launch_extents(slot, construction_batch)
                )
                static_extents = program.static_extent_bindings
                mismatched = {
                    name: (static_extents[name], launch_extents[name])
                    for name in static_extents.keys() & launch_extents.keys()
                    if int(static_extents[name]) != int(launch_extents[name])
                }
                if mismatched:
                    raise RuntimeError(
                        f"generated {spec.label} construction extents disagree "
                        f"with static artifact bindings {mismatched}"
                    )
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
                    for item in program.descriptor["device_program"]["argument_plan"][
                        "arguments"
                    ]
                    if item["transport"] == "scalar"
                )
                unknown = launch_extents.keys() - scalar_names - static_extents.keys()
                if unknown:
                    raise RuntimeError(
                        f"generated {spec.label} has unknown launch extents "
                        f"{sorted(unknown)}"
                    )
                self._slots[(int(slot.slot_id), program_index)] = _BoundInvocation(
                    program.bind(bindings),
                    scalar_names,
                    frozenset(launch_extents),
                )
        if len(plans) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on input preparation"
            )
        self._input_preparation_plan = next(iter(plans.values()))
        missing = {
            step.name for step in self._input_preparation_plan
        } - spec.preparation_callbacks.keys()
        if missing:
            raise RuntimeError(
                f"generated {spec.label} has no preparation callbacks for "
                f"{sorted(missing)}"
            )

    def _program_for(self, batch_size: int) -> tuple[int, Any] | None:
        return _select_program(self._programs, batch_size)

    def supports(self, batch_size: int) -> bool:
        return self._program_for(batch_size) is not None

    def state_requirements_for(
        self,
        batch_size: int,
    ) -> tuple[StateRepresentationRequirement, ...]:
        selected = self._program_for(batch_size)
        if selected is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        _index, program = selected
        return self.state_requirements_by_capacity[program.capacity]

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        selected = self._program_for(batch_size)
        if selected is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        program_index, _program = selected
        for step in self._input_preparation_plan:
            self._spec.preparation_callbacks[step.name](slot, int(batch_size))
        bound = self._slots[(int(slot.slot_id), program_index)]
        extents = dict(self._spec.bindings.launch_extents(slot, int(batch_size)))
        missing = bound.required_launch_extents - extents.keys()
        if missing:
            raise RuntimeError(
                f"generated {self._spec.label} launch misses {sorted(missing)}"
            )
        bound.invocation.launch(
            **{
                name: value
                for name, value in extents.items()
                if name in bound.scalar_names
            }
        )


__all__ = [
    "DeviceInputPreparation",
    "GeneratedDecode",
    "GeneratedDecodeBindings",
    "GeneratedDecodeSpec",
    "PagedDecodeBindings",
]
