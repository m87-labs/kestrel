"""Descriptor-driven binding for compiler-generated decode programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class GeneratedDecodeSpec:
    label: str
    capacities: Sequence[int]
    compile_program: Callable[[int, Any], Any]
    weight_root: Any
    weight_layer_prefix: str
    runtime_inputs: Callable[[], Mapping[str, Any]]
    slot_inputs: Callable[[Any, int], Mapping[str, Any]]
    runtime_extents: Callable[[int], Mapping[str, int]]
    launch_extents: Callable[[Any, int], Mapping[str, int]]
    unsupported: tuple[type[BaseException], ...] = ()
    capacity_inputs: Callable[[int, tuple[Any, ...]], Mapping[str, Any]] | None = None
    preparations: Sequence[Any] = ()
    not_ready_inputs: frozenset[str] = frozenset()
    preparation_callbacks: Mapping[str, Callable[[Any, int], None]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _BoundInvocation:
    invocation: Any
    argument_names: frozenset[str]


class GeneratedDecode:
    """Own capacity selection, bindings, and invocation lifecycle."""

    @classmethod
    def try_create(
        cls,
        runtime: Any,
        spec: GeneratedDecodeSpec,
    ) -> GeneratedDecode | None:
        if runtime.device.type != "cuda" or runtime.dtype is not torch.bfloat16:
            return None
        try:
            from mkl.compiler.frontend.gpu_model import CalibrationUnavailable
            from mkl.compiler.frontend.validate_program import validate_compiled_tape
            from mkl.megakernel.device_runtime import (
                DeviceRuntimeError,
                resolve_capacity_programs,
                resolve_shipped_aot_bundle,
            )
        except ModuleNotFoundError as exc:
            missing = str(exc.name or "")
            if missing != "mkl" and not missing.startswith("mkl."):
                raise
            return None

        properties = torch.cuda.get_device_properties(runtime.device)
        arch = f"sm{properties.major}{properties.minor}"
        try:
            programs = resolve_capacity_programs(
                spec.capacities,
                max_active_extent=runtime.max_batch_size,
                compile_program=lambda capacity: spec.compile_program(
                    capacity, properties
                ),
                validate_program=validate_compiled_tape,
                resolve_bundle=resolve_shipped_aot_bundle,
                arch=arch,
            )
        except spec.unsupported:
            return None
        except CalibrationUnavailable as exc:
            raise DeviceRuntimeError(
                f"generated {spec.label} decode has no calibration for "
                f"{properties.name!r}: {exc}"
            ) from exc
        if not programs:
            return None
        return cls(runtime, spec=spec, programs=programs)

    def __init__(
        self,
        runtime: Any,
        *,
        spec: GeneratedDecodeSpec,
        programs: Mapping[int, tuple[Any, Any, Any]],
    ) -> None:
        from mkl.compiler.frontend import bind_owned_weight_storage
        from mkl.compiler.frontend import derive_device_carried_state_contracts
        from mkl.megakernel.device_input_preparation import (
            derive_device_input_preparation_plan,
        )
        from mkl.megakernel.device_runtime import (
            assemble_torch_device_bindings,
            bind_aot_device_program,
        )
        from mkl.megakernel.state_runtime import StateRepresentationRequirement

        self._programs = dict(sorted(programs.items()))
        self._spec = spec
        first_compiled = next(iter(self._programs.values()))[0]
        self.state_requirements_by_capacity = {
            int(capacity): tuple(
                StateRepresentationRequirement(
                    contract.buffer,
                    contract.representation,
                    contract.storage_axis_order,
                    contract.storage_dtype,
                )
                for contract in derive_device_carried_state_contracts(
                    compiled.graph,
                    compiled.device_program,
                )
            )
            for capacity, (compiled, _validated, _bundle) in self._programs.items()
        }
        state_buffer_sets = {
            tuple(requirement.buffer for requirement in requirements)
            for requirements in self.state_requirements_by_capacity.values()
        }
        if len(state_buffer_sets) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on carried state"
            )
        self.state_buffers = next(iter(state_buffer_sets))
        ambient_stream = torch.cuda.current_stream(runtime.device)
        model_weights_ready = torch.cuda.Event()
        model_weights_ready.record(ambient_stream)
        runtime.compute_stream.wait_event(model_weights_ready)
        weight_storage_ready = torch.cuda.Event()
        with torch.cuda.stream(runtime.compute_stream):
            self.weight_storage = bind_owned_weight_storage(
                first_compiled,
                spec.weight_root,
                device=runtime.device,
                layer_prefix=spec.weight_layer_prefix,
            )
            shared_inputs = dict(spec.runtime_inputs())
            weight_storage_ready.record(runtime.compute_stream)
        ambient_stream.wait_event(weight_storage_ready)

        self._slots: dict[tuple[int, int], Any] = {}
        plans: list[tuple[str, ...]] = []
        for slot in runtime.decode_slots:
            for batch_capacity, program in self._programs.items():
                compiled, validated, aot_bundle = program
                requirements = self.state_requirements_by_capacity[
                    int(batch_capacity)
                ]
                runtime_inputs = {
                    **shared_inputs,
                    **(
                        spec.capacity_inputs(int(batch_capacity), requirements)
                        if spec.capacity_inputs is not None
                        else {}
                    ),
                    **spec.slot_inputs(slot, int(batch_capacity)),
                }
                plan = derive_device_input_preparation_plan(
                    compiled.device_program,
                    ready_inputs=set(runtime_inputs) - spec.not_ready_inputs,
                    preparations=spec.preparations,
                )
                plans.append(tuple(step.name for step in plan))
                bindings = assemble_torch_device_bindings(
                    compiled,
                    bound_weights=self.weight_storage.buffers,
                    runtime_inputs=runtime_inputs,
                    runtime_extents=spec.runtime_extents(int(batch_capacity)),
                    stream=slot.compute_stream,
                    device=runtime.device,
                )
                self._slots[(int(slot.slot_id), int(batch_capacity))] = _BoundInvocation(
                    invocation=bind_aot_device_program(
                        validated, aot_bundle, bindings.values
                    ),
                    argument_names=frozenset(
                        argument.name
                        for argument in compiled.device_program.argument_plan.arguments
                    ),
                )
        if len(set(plans)) != 1:
            raise RuntimeError(
                f"generated {spec.label} capacities disagree on input preparation"
            )
        self._input_preparation_plan = plans[0]
        missing_callbacks = (
            set(self._input_preparation_plan) - spec.preparation_callbacks.keys()
        )
        if missing_callbacks:
            raise RuntimeError(
                f"generated {spec.label} decode has no callbacks for preparation "
                f"steps {sorted(missing_callbacks)}"
            )

    def _capacity_for(self, batch_size: int) -> int | None:
        from mkl.megakernel.device_runtime import select_compiled_capacity

        return select_compiled_capacity(self._programs, batch_size)

    def supports(self, batch_size: int) -> bool:
        return self._capacity_for(batch_size) is not None

    def state_requirements_for(self, batch_size: int) -> tuple[Any, ...]:
        capacity = self._capacity_for(int(batch_size))
        if capacity is None:
            raise ValueError(
                f"no generated decode capacity covers batch size {batch_size}"
            )
        return self.state_requirements_by_capacity[capacity]

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        batch_size = int(batch_size)
        capacity = self._capacity_for(batch_size)
        if capacity is None:
            raise ValueError(
                f"no generated decode capacity covers batch size {batch_size}"
            )
        for step_name in self._input_preparation_plan:
            self._spec.preparation_callbacks[step_name](slot, batch_size)
        bound = self._slots[(int(slot.slot_id), capacity)]
        candidates = self._spec.launch_extents(slot, batch_size)
        bound.invocation.launch(
            **{
                name: value
                for name, value in candidates.items()
                if name in bound.argument_names
            }
        )


__all__ = ["GeneratedDecode", "GeneratedDecodeSpec"]
