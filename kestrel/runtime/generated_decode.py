"""Runtime binding for compiler-generated decode bundles."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from kestrel.device import stream_context

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
            "kv_len": int(slot.meta.max_input_pos) + 1,
        }


@dataclass(frozen=True)
class GeneratedDecodeSpec:
    label: str
    weight_root: Any
    weight_layer_prefix: str
    bindings: GeneratedDecodeBindings
    weight_sources: Mapping[str, torch.Tensor] | None = None
    engine_buffers: Mapping[str, torch.Tensor] | None = None
    weight_storage: Any | None = None
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
class _GeneratedDecodePlan:
    """Resolved programs kept stable across slot allocation and binding."""

    runtime: Any
    spec: GeneratedDecodeSpec
    max_batch_size: int
    compatible_programs: tuple[Any, ...]
    selectable_programs: tuple[Any, ...]

    @property
    def slot_capacity(self) -> int:
        """Physical rows required by every program this runtime can select."""

        return max(
            (int(program.capacity) for program in self.selectable_programs),
            default=0,
        )


@dataclass(frozen=True)
class _BoundInvocation:
    invocation: Any
    repeated_dynamic_launch: Callable[..., Any]
    scalar_names: frozenset[str]
    required_launch_extents: frozenset[str]


def _active_batch_interval(program: Any) -> tuple[int, int]:
    """Return the inclusive request-count interval served by one program."""

    capacity = int(program.capacity)
    minimum = int(
        getattr(program, "runtime_extent_minimums", {}).get("active_batch", 1)
    )
    if not 1 <= minimum <= capacity:
        raise RuntimeError(
            "generated decode program has invalid active-batch interval "
            f"[{minimum}, {capacity}]"
        )
    static = program.static_extent_bindings.get("active_batch")
    if static is not None:
        static = int(static)
        if not minimum <= static <= capacity:
            raise RuntimeError(
                "generated decode program has invalid static active batch "
                f"{static} outside [{minimum}, {capacity}]"
            )
        return static, static
    return minimum, capacity


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
        minimum_batch, maximum_batch = _active_batch_interval(program)
        if not minimum_batch <= batch_size <= maximum_batch:
            continue
        static_batch = program.static_extent_bindings.get("active_batch")
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
        program for index, program in enumerate(programs) if index in selected_indexes
    )


def _program_lookup(
    programs: Sequence[Any], max_batch_size: int
) -> tuple[tuple[int, Any] | None, ...]:
    return tuple(
        _select_program(programs, batch_size)
        for batch_size in range(1, int(max_batch_size) + 1)
    )


def _generated_weight_runtime(*, label: str, required: bool) -> Any | None:
    try:
        from kestrel_kernels import generated_decode as generated_runtime
    except ModuleNotFoundError as exc:
        if exc.name not in {"kestrel_kernels", "kestrel_kernels.generated_decode"}:
            raise
        if required:
            raise RuntimeError(
                f"generated {label} decode requires load-time weight support"
            )
        return None
    capabilities = (
        "resolve_compatible_programs",
        "allocate_weight_storage_for_loading",
        "finalize_weight_storage_after_loading",
    )
    if not all(
        callable(getattr(generated_runtime, name, None)) for name in capabilities
    ):
        if required:
            raise RuntimeError(
                f"generated {label} decode requires load-time weight binding "
                "and finalization support"
            )
        return None
    return generated_runtime


def generated_weight_programs_for_loading(
    runtime: Any,
    model: torch.nn.Module,
    *,
    label: str,
    layer_prefix: str,
    required_batch_sizes: Sequence[int],
    required: bool,
) -> tuple[Any, ...]:
    """Resolve the exact load-time program set or fail soft when optional."""

    if runtime.device.type != "cuda" or runtime.dtype is not torch.bfloat16:
        if required:
            raise RuntimeError(
                f"generated {label} decode requires CUDA BF16 model storage"
            )
        return ()
    generated_runtime = _generated_weight_runtime(label=label, required=required)
    if generated_runtime is None:
        return ()

    properties = torch.cuda.get_device_properties(runtime.device)
    programs = tuple(generated_runtime.resolve_compatible_programs(
        model,
        layer_prefix=layer_prefix,
        arch=f"sm{properties.major}{properties.minor}",
        device_sms=int(properties.multi_processor_count),
    ))
    missing = [
        int(batch_size)
        for batch_size in required_batch_sizes
        if _select_program(programs, int(batch_size)) is None
    ]
    if missing:
        if required:
            raise RuntimeError(
                f"generated {label} decode has no load-time artifact coverage "
                f"for batch sizes {missing}"
            )
        return ()
    selected = _selectable_programs(programs, runtime.max_batch_size)
    if not selected:
        if required:
            raise RuntimeError(
                f"generated {label} decode has no selectable load-time artifact"
            )
        return ()
    contracts = {repr(program.descriptor["weights"]) for program in selected}
    if len(contracts) != 1:
        raise RuntimeError(
            f"generated {label} artifacts disagree on weight storage"
        )
    return selected


def prepare_generated_weight_storage_for_loading(
    runtime: Any,
    model: torch.nn.Module,
    *,
    label: str,
    layer_prefix: str,
    required_batch_sizes: Sequence[int],
    required: bool,
) -> Any | None:
    """Bind checkpoint targets into the selected generated program's final slabs."""

    selected = generated_weight_programs_for_loading(
        runtime,
        model,
        label=label,
        layer_prefix=layer_prefix,
        required_batch_sizes=required_batch_sizes,
        required=required,
    )
    if not selected:
        return None
    generated_runtime = _generated_weight_runtime(label=label, required=True)
    assert generated_runtime is not None

    return generated_runtime.allocate_weight_storage_for_loading(
        model,
        selected[0].descriptor,
        device=runtime.device,
        layer_prefix=layer_prefix,
    )


def finalize_generated_weight_storage_after_loading(
    runtime: Any,
    model: torch.nn.Module,
    storage: Any,
    *,
    label: str,
    layer_prefix: str,
    required_batch_sizes: Sequence[int],
) -> Any:
    """Finalize retained recipes against the exact selected weight contract."""

    selected = generated_weight_programs_for_loading(
        runtime,
        model,
        label=label,
        layer_prefix=layer_prefix,
        required_batch_sizes=required_batch_sizes,
        required=True,
    )
    generated_runtime = _generated_weight_runtime(label=label, required=True)
    assert generated_runtime is not None

    return generated_runtime.finalize_weight_storage_after_loading(
        model,
        selected[0].descriptor,
        storage,
        layer_prefix=layer_prefix,
    )


def reserve_generated_binding_storage(
    programs: Sequence[Any],
    *,
    weight_storage: Any,
    runtime_inputs_by_slot: Sequence[Mapping[str, Any]],
    device: torch.device,
    stream: Any,
    label: str,
    required: bool,
) -> tuple[torch.Tensor, ...] | None:
    """Reserve exact allocator-owned storage for later binding assembly.

    Callers keep the shape- and dtype-exact tensors alive while fitting and
    allocating other resident tensors, then release them immediately before
    constructing :class:`GeneratedDecode`. The caching allocator can reuse
    those same size classes for the per-program binding tensors.
    """

    generated_runtime = _generated_weight_runtime(label=label, required=required)
    if generated_runtime is None:
        return None
    reserve = getattr(generated_runtime, "reserve_binding_storage", None)
    if not callable(reserve):
        if required:
            raise RuntimeError(
                f"generated {label} decode requires binding-storage reservation"
            )
        return None
    if getattr(weight_storage, "finalized", None) is not True:
        raise RuntimeError(
            f"generated {label} binding reservation requires finalized weights"
        )
    weights = getattr(weight_storage, "buffers", None)
    if not isinstance(weights, Mapping):
        raise RuntimeError(
            f"generated {label} binding reservation requires weight buffers"
        )
    if not programs or not runtime_inputs_by_slot:
        raise ValueError(
            "generated binding reservation needs programs and decode slots"
        )

    # Binding assembly runs on the runtime compute stream. Reserving on that
    # stream keeps the exact-size blocks reusable by the CUDA caching allocator.
    with stream_context(stream):
        return tuple(
            tensor
            for runtime_inputs in runtime_inputs_by_slot
            for program in programs
            for tensor in reserve(
                program.descriptor,
                weights=weights,
                runtime_inputs=runtime_inputs,
                device=device,
                stream=stream,
            )
        )


def materialize_remaining_meta_tensors(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> None:
    """Materialize checkpoint-backed tensors not claimed by a load-time layout."""

    parameter_replacements: dict[int, torch.nn.Parameter] = {}
    buffer_replacements: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is None or parameter.device.type != "meta":
                continue
            replacement = parameter_replacements.get(id(parameter))
            if replacement is None:
                replacement = torch.nn.Parameter(
                    torch.empty_like(parameter, device=device),
                    requires_grad=parameter.requires_grad,
                )
                parameter_replacements[id(parameter)] = replacement
            module._parameters[name] = replacement
        for name, buffer in tuple(module._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            if name in module._non_persistent_buffers_set:
                raise RuntimeError(
                    "load-time preparation left derived buffer "
                    f"{type(module).__name__}.{name} uninitialized"
                )
            replacement = buffer_replacements.get(id(buffer))
            if replacement is None:
                replacement = torch.empty_like(buffer, device=device)
                buffer_replacements[id(buffer)] = replacement
            module._buffers[name] = replacement


class GeneratedDecode:
    """Select bundled capacities and bind them to one serving runtime."""

    @classmethod
    def _resolve_programs(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
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
        resolution_options = {}
        weight_sources = getattr(spec, "weight_sources", None)
        if weight_sources is not None:
            resolution_options["weight_sources"] = weight_sources
        return tuple(
            resolve_compatible_programs(
                spec.weight_root,
                layer_prefix=spec.weight_layer_prefix,
                arch=f"sm{properties.major}{properties.minor}",
                device_sms=int(properties.multi_processor_count),
                **resolution_options,
            )
        )

    @classmethod
    def resolve_slot_capacity(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
        *,
        required_batch_sizes: Sequence[int] = (),
    ) -> int | None:
        """Return the physical row capacity required by selectable programs."""

        programs = cls._resolve_programs(runtime, spec)
        if any(
            _select_program(programs, int(batch_size)) is None
            for batch_size in required_batch_sizes
        ):
            return None
        selected = _selectable_programs(programs, runtime.max_batch_size)
        if not selected:
            return None
        return max(int(program.capacity) for program in selected)

    @classmethod
    def plan(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
    ) -> _GeneratedDecodePlan:
        """Resolve once so slot storage and later bindings use one program set."""

        max_batch_size = int(runtime.max_batch_size)
        compatible_programs = cls._resolve_programs(runtime, spec)
        return _GeneratedDecodePlan(
            runtime=runtime,
            spec=spec,
            max_batch_size=max_batch_size,
            compatible_programs=compatible_programs,
            selectable_programs=_selectable_programs(
                compatible_programs, max_batch_size
            ),
        )

    @staticmethod
    def _validate_plan(
        runtime: Any,
        spec: GeneratedDecodeSpec,
        plan: _GeneratedDecodePlan,
    ) -> None:
        if plan.runtime is not runtime:
            raise ValueError("generated decode plan belongs to a different runtime")
        if plan.spec is not spec:
            raise ValueError("generated decode plan belongs to a different spec")
        if plan.max_batch_size != int(runtime.max_batch_size):
            raise ValueError(
                "generated decode plan batch limit changed between resolution "
                "and binding"
            )

    @classmethod
    def require(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
        *,
        batch_sizes: Sequence[int],
        plan: _GeneratedDecodePlan | None = None,
    ) -> "GeneratedDecode":
        if plan is None:
            programs = cls._resolve_programs(runtime, spec)
        else:
            cls._validate_plan(runtime, spec, plan)
            programs = plan.compatible_programs
        if not programs:
            raise RuntimeError(
                f"{spec.label} requires a compatible bundled generated-decode program"
            )
        return cls(
            runtime,
            spec=spec,
            programs=programs,
            required_batch_sizes=batch_sizes,
        )

    @classmethod
    def try_create(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
        *,
        required_batch_sizes: Sequence[int] = (),
        plan: _GeneratedDecodePlan | None = None,
    ) -> "GeneratedDecode | None":
        if plan is None:
            compatible_programs = cls._resolve_programs(runtime, spec)
        else:
            cls._validate_plan(runtime, spec, plan)
            compatible_programs = plan.compatible_programs
        if not compatible_programs or any(
            _select_program(compatible_programs, int(batch_size)) is None
            for batch_size in required_batch_sizes
        ):
            return None
        programs = (
            _selectable_programs(compatible_programs, runtime.max_batch_size)
            if plan is None
            else plan.selectable_programs
        )
        if not programs:
            return None
        return cls(
            runtime,
            spec=spec,
            programs=programs,
            required_batch_sizes=required_batch_sizes,
        )

    def __init__(
        self,
        runtime: Any,
        *,
        spec: GeneratedDecodeSpec,
        programs,
        required_batch_sizes: Sequence[int] = (),
    ) -> None:
        available_programs = tuple(programs)
        self._programs = available_programs
        self._spec = spec
        if required_batch_sizes:
            missing = [
                int(batch_size)
                for batch_size in required_batch_sizes
                if _select_program(available_programs, int(batch_size)) is None
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
        self._program_by_batch = _program_lookup(
            self._programs, runtime.max_batch_size
        )

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
            materialization_options = {}
            if spec.weight_sources is not None:
                materialization_options["weight_sources"] = spec.weight_sources
            if spec.engine_buffers is not None:
                materialization_options["engine_buffers"] = spec.engine_buffers
            if spec.weight_storage is None:
                self.weight_storage = materialize_weights(
                    spec.weight_root,
                    self._programs[0].descriptor,
                    layer_prefix=spec.weight_layer_prefix,
                    **materialization_options,
                )
            else:
                expected_contract = repr(self._programs[0].descriptor["weights"])
                actual_contract = getattr(
                    spec.weight_storage, "weight_contract", None
                )
                if actual_contract != expected_contract:
                    raise RuntimeError(
                        "preloaded generated weights do not match the selected program"
                    )
                if getattr(spec.weight_storage, "finalized", None) is not True:
                    raise RuntimeError(
                        "preloaded generated weights were not finalized after loading"
                    )
                self.weight_storage = spec.weight_storage
            shared_inputs = dict(spec.bindings.runtime_inputs(runtime))
            weights_ready.record(runtime.compute_stream)
        ambient_stream.wait_event(weights_ready)

        self._slots = {}
        plans = {}
        for slot in runtime.decode_slots:
            for program_index, program in enumerate(self._programs):
                capacity = program.capacity
                minimum_batch, maximum_batch = _active_batch_interval(program)
                if minimum_batch > int(runtime.max_batch_size):
                    continue
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
                construction_batch = min(maximum_batch, int(runtime.max_batch_size))
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
                invocation = program.bind(bindings)
                self._slots[(int(slot.slot_id), program_index)] = _BoundInvocation(
                    invocation,
                    invocation.prepare_repeated_dynamic_launch(),
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
        batch_size = int(batch_size)
        if batch_size < 1 or batch_size > len(self._program_by_batch):
            return None
        return self._program_by_batch[batch_size - 1]

    def supports(self, batch_size: int) -> bool:
        return self._program_for(batch_size) is not None

    @property
    def artifact_receipts(self) -> tuple[dict[str, Any], ...]:
        """Return exact packed-artifact identities for the selected programs."""

        return tuple(
            json.loads(json.dumps(program.artifact_receipt, default=dict))
            for program in self._programs
            if program.artifact_receipt is not None
        )

    def state_requirements_for(
        self,
        batch_size: int,
    ) -> tuple[StateRepresentationRequirement, ...]:
        selected = self._program_for(batch_size)
        if selected is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        _index, program = selected
        return self.state_requirements_by_capacity[program.capacity]

    def static_launcher(self, slot: Any, batch_size: int) -> Callable[[], None]:
        """Bind a repeated launch whose inputs and extents stay fixed."""

        if self._input_preparation_plan:
            raise ValueError(
                "static generated decode cannot run per-step input preparations"
            )
        selected = self._program_for(batch_size)
        if selected is None:
            raise ValueError(f"no generated decode capacity covers {batch_size}")
        program_index, _program = selected
        bound = self._slots[(int(slot.slot_id), program_index)]
        extents = dict(self._spec.bindings.launch_extents(slot, int(batch_size)))
        missing = bound.required_launch_extents - extents.keys()
        if missing:
            raise RuntimeError(
                f"generated {self._spec.label} launch misses {sorted(missing)}"
            )
        return bound.invocation.prepare_repeated_launch(
            **{
                name: value
                for name, value in extents.items()
                if name in bound.scalar_names
            }
        )

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
        bound.repeated_dynamic_launch(
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
    "finalize_generated_weight_storage_after_loading",
    "generated_weight_programs_for_loading",
    "materialize_remaining_meta_tensors",
    "prepare_generated_weight_storage_for_loading",
    "reserve_generated_binding_storage",
]
