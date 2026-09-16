"""Local rank-team coordination for bundled generated decode programs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .generated_decode import GeneratedDecodeTeamMember


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
}


def generated_decode_team_members(
    devices: Sequence[int], *, topology: str,
) -> tuple[GeneratedDecodeTeamMember, ...]:
    devices = tuple(devices)
    return tuple(
        GeneratedDecodeTeamMember(devices, rank, topology)
        for rank in range(len(devices))
    )


def _collective_pairs(descriptor: dict[str, Any]) -> tuple[tuple[dict, dict], ...]:
    device_program = descriptor["device_program"]
    plan = device_program["distribution_plan"]
    arguments = {
        item["name"]: item
        for item in device_program["argument_plan"]["arguments"]
        if item["source"] == "collective"
    }
    collective_arguments = tuple(
        item for item in device_program["argument_plan"]["arguments"]
        if item["source"] == "collective"
    )
    if len(arguments) != len(collective_arguments):
        raise RuntimeError("generated decode collective argument is duplicated")
    collectives = plan.get("routed_down_collectives", ())
    if not collectives or not arguments:
        raise RuntimeError("generated decode team has no collective ABI")
    pairs = []
    names = set()
    for collective in collectives:
        by_stem: dict[str, dict[str, dict]] = {}
        for buffer in collective["buffers"]:
            name = buffer["name"]
            role = buffer["role"]
            stem, separator, view = role.rpartition("_")
            if (
                not separator or view not in {"local", "multicast"}
                or name in names or name not in arguments
            ):
                raise RuntimeError("generated decode collective roles are malformed")
            argument = arguments[name]
            if (
                argument["transport"] != "raw_pointer"
                or argument["dtype"] != buffer["dtype"]
                or tuple(argument["shape"]) != tuple(buffer["shape"])
                or argument["access"] != buffer["access"]
                or argument["assumed_align"] != buffer["assumed_align"]
            ):
                raise RuntimeError("generated decode collective argument disagrees with plan")
            views = by_stem.setdefault(stem, {})
            if view in views:
                raise RuntimeError("generated decode collective role is duplicated")
            names.add(name)
            views[view] = buffer
        for views in by_stem.values():
            if set(views) != {"local", "multicast"}:
                raise RuntimeError("generated decode collective needs both CUDA views")
            local, multicast = views["local"], views["multicast"]
            if (
                local["dtype"] != multicast["dtype"]
                or local["shape"] != multicast["shape"]
                or local["dtype"] not in _DTYPES
            ):
                raise RuntimeError("generated decode collective pair has inconsistent storage")
            pairs.append((local, multicast))
    if names != set(arguments):
        raise RuntimeError("generated decode collective arguments are incomplete")
    return tuple(pairs)


class GeneratedDecodeRankTeam:
    """Bind one BS1 slot and queue rank launches on a local CUDA team."""

    def __init__(self, runtimes: Sequence[Any]) -> None:
        runtimes = tuple(runtimes)
        if len(runtimes) <= 1:
            raise ValueError("generated decode rank team needs multiple runtimes")
        members = tuple(
            runtime.generated_decode._spec.team_member for runtime in runtimes
        )
        first = members[0]
        if (
            first is None
            or len(runtimes) != len(first.devices)
            or any(
                member != GeneratedDecodeTeamMember(
                    first.devices, rank, first.topology
                )
                or runtime.device != torch.device("cuda", first.devices[rank])
                or int(runtime.max_batch_size) != 1
                for rank, (runtime, member) in enumerate(zip(runtimes, members))
            )
        ):
            raise ValueError("generated decode rank team has inconsistent members")
        programs = tuple(
            runtime.generated_decode.team_program(1) for runtime in runtimes
        )
        identity = programs[0].descriptor["program"]
        device_program = programs[0].descriptor["device_program"]
        contract = (
            device_program["distribution_plan"],
            device_program["argument_plan"],
            programs[0].descriptor["weights"],
        )
        if any(
            program.descriptor["program"] != identity
            or (
                program.descriptor["device_program"]["distribution_plan"],
                program.descriptor["device_program"]["argument_plan"],
                program.descriptor["weights"],
            ) != contract
            for program in programs[1:]
        ):
            raise RuntimeError("generated decode ranks selected different programs")
        plan = device_program["distribution_plan"]
        if (
            plan.get("world_size") != len(first.devices)
            or plan.get("strategy") != "tensor_parallel"
            or plan.get("routed_expert_topology") != first.topology
        ):
            raise RuntimeError("generated decode team disagrees with program topology")

        from kestrel_kernels.multicast import CudaMulticastAllocation

        self.runtimes = runtimes
        self.devices = first.devices
        self._allocations = []
        self._bound_runtimes = []
        self._closed = False
        self._bound_team = None
        self._bound_members = ()
        self._bound_slots = ()
        self._step = 0
        rank_inputs = [dict() for _ in runtimes]
        try:
            for local, multicast in _collective_pairs(programs[0].descriptor):
                allocation = CudaMulticastAllocation(
                    self.devices, local["shape"], dtype=_DTYPES[local["dtype"]]
                )
                self._allocations.append(allocation)
                for rank in range(len(runtimes)):
                    rank_inputs[rank][local["name"]] = allocation.local[rank]
                    rank_inputs[rank][multicast["name"]] = allocation.multicast[rank]
            for runtime, inputs in zip(runtimes, rank_inputs):
                runtime.generated_decode.bind_team(inputs)
                self._bound_runtimes.append(runtime)
            from kestrel_kernels.generated_decode import BoundGeneratedDecodeRankTeam

            slots = tuple(runtime.decode_slots[0] for runtime in runtimes)
            members = tuple(
                runtime.generated_decode.team_slot_launch(slot, 1)[0]
                for runtime, slot in zip(runtimes, slots)
            )

            def stage(rank: int, _step: int) -> None:
                runtimes[rank].generated_decode.prepare_team_inputs(slots[rank], 1)

            self._bound_team = BoundGeneratedDecodeRankTeam(
                self.devices, members, stage_step=stage,
            )
            self._bound_members = members
            self._bound_slots = slots
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def queue_one(self, *, slot_id: int = 0) -> Any:
        """Queue slot0 on every rank; call wait before reading its outputs."""

        if self._closed:
            raise RuntimeError("generated decode rank team is closed")
        if type(slot_id) is not int or slot_id != 0:
            raise ValueError("generated decode rank team is bound to slot0")
        slots = tuple(runtime.decode_slots[0] for runtime in self.runtimes)
        if any(
            slot is not bound_slot
            for slot, bound_slot in zip(
                slots, self._bound_slots, strict=True,
            )
        ):
            raise RuntimeError("generated decode team slot object changed after binding")
        launches = tuple(
            runtime.generated_decode.team_slot_launch(slot, 1)
            for runtime, slot in zip(self.runtimes, slots)
        )
        if (
            any(
                bound is not member
                for (bound, _), member in zip(
                    launches, self._bound_members, strict=True,
                )
            )
            or any(scalars != launches[0][1] for _, scalars in launches[1:])
        ):
            raise RuntimeError("generated decode ranks changed slot program or extents")
        self._bound_team.launch_step(self._step, **launches[0][1])
        self._step += 1
        return slots[0]

    def wait(self) -> None:
        if self._closed:
            raise RuntimeError("generated decode rank team is closed")
        self._bound_team.wait()

    def close(self) -> None:
        if self._closed:
            return
        if self._bound_team is not None:
            self._bound_team.close()
        for runtime in reversed(self._bound_runtimes):
            runtime.generated_decode.unbind_team()
        self._bound_runtimes.clear()
        self._bound_members = ()
        self._bound_slots = ()
        self._bound_team = None
        first_error = None
        for allocation in reversed(self._allocations):
            try:
                allocation.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "GeneratedDecodeRankTeam":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = ["GeneratedDecodeRankTeam", "generated_decode_team_members"]
