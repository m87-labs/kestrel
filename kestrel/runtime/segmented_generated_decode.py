"""Explicit segmented execution of compiler-generated decode fragments."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from kestrel_kernels.generated_decode import GeneratedDecodeError
from kestrel_kernels.generated_decode_segment_attention import FixedPlanPageOneAttention
from kestrel_kernels.generated_decode_segments import (
    BoundSegmentedDecode,
    FragmentInvocation,
    bind_fragment,
    validate_fragment_plan,
)


def _merge_inputs(*inputs: Mapping[str, Any]) -> dict[str, Any]:
    merged = {}
    for source in inputs:
        collision = merged.keys() & source.keys()
        if collision:
            raise GeneratedDecodeError(
                f"segmented decode input ownership collides on {sorted(collision)}"
            )
        merged.update(source)
    return merged


class SegmentedGeneratedDecode:
    """One explicitly selected capacity, retaining the whole-model owner."""

    @classmethod
    def from_shipped_inventory(
        cls,
        runtime: Any,
        whole: Any,
        *,
        registration: str,
        parent_program: str,
        capacity: int,
        max_kv_len: int,
    ) -> "SegmentedGeneratedDecode":
        from kestrel_kernels.generated_decode import (
            resolve_attention_fragment_inventory,
        )

        properties = torch.cuda.get_device_properties(runtime.device)
        inventory = resolve_attention_fragment_inventory(
            registration=registration,
            parent_program=parent_program,
            arch=f"sm{properties.major}{properties.minor}",
            device_sms=int(properties.multi_processor_count),
        )
        programs = inventory["programs"]
        return cls(
            runtime, whole,
            capacity=capacity,
            programs=programs,
            plan=inventory["plan"],
            boundary=inventory["boundary"],
            max_kv_len=max_kv_len,
        )

    def __init__(
        self,
        runtime: Any,
        whole: Any,
        *,
        capacity: int,
        programs: Mapping[str, Any],
        plan: tuple[Mapping[str, Any], ...],
        boundary: Mapping[str, Any],
        max_kv_len: int,
    ) -> None:
        selected = whole._program_for(capacity)
        if selected is None or selected[1].capacity != capacity:
            raise GeneratedDecodeError("segmented decode has no exact parent capacity")
        parent = selected[1]
        parent_id = parent.descriptor["program"]
        if (
            parent_id.get("registration"), parent_id.get("program"), capacity,
            boundary.get("input_positions"),
        ) != ("qwen35_dense", "qwen35_27b_fp8_b8", 8, "input_pos"):
            raise GeneratedDecodeError(
                "segmented pre-capture is not validated for this parent program"
            )
        steps = validate_fragment_plan(
            tuple(FragmentInvocation(str(step["role"]),
                                     tuple(step["physical_layers"]))
                  for step in plan),
            num_layers=int(parent.descriptor["num_layers"]),
        )
        required_roles = {step.role for step in steps} - {"attention"}
        if set(programs) != required_roles:
            raise GeneratedDecodeError("segmented decode fragment inventory is incomplete")
        if any(program.capacity != capacity for program in programs.values()):
            raise GeneratedDecodeError("segmented decode fragment capacity differs")
        terminal_abi_names = {}
        for role, program in programs.items():
            metadata = getattr(program, "attention_fragment", None)
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("role") != role
                or metadata.get("capacity") != capacity
                or metadata.get("parent_registration") != parent_id["registration"]
                or metadata.get("parent_program") != parent_id["program"]
            ):
                raise GeneratedDecodeError(
                    "segmented decode fragment belongs to a different parent"
                )
            arguments = program.descriptor["device_program"]["argument_plan"]["arguments"]
            scalar_names = {
                argument["name"] for argument in arguments
                if argument["transport"] == "scalar"
            }
            if ("kv_len" in scalar_names) != (role == "prefix"):
                raise GeneratedDecodeError(
                    "segmented pre-capture has an unsupported KV scalar owner"
                )
            terminal_abi_names[role] = metadata.get("terminal_abi_name")
        self._whole = whole
        self._capacity = capacity
        self._slots = {}
        spec = whole._spec
        parent_weights = whole.weight_storage.buffers
        shared_inputs = dict(spec.bindings.runtime_inputs(runtime))
        state_inputs = (
            dict(spec.capacity_inputs(
                capacity, whole.state_requirements_by_capacity[capacity]
            )) if spec.capacity_inputs else {}
        )
        for slot in runtime.decode_slots:
            inputs = _merge_inputs(
                shared_inputs,
                state_inputs,
                dict(spec.bindings.slot_inputs(slot, capacity)),
            )
            attention = FixedPlanPageOneAttention(
                boundary=boundary,
                page_table=inputs[boundary["page_table"]],
                capacity=capacity,
                max_kv_len=max_kv_len,
                stream=slot.compute_stream,
            )
            bound = []
            residual = None
            attn_result = None
            for step in steps:
                if step.role == "attention":
                    continue
                program = programs[step.role]
                overlay = {}
                terminal = terminal_abi_names[step.role]
                if step.role != "first_body":
                    if residual is None:
                        raise GeneratedDecodeError(
                            "segmented decode has no preceding residual"
                        )
                    overlay["x"] = residual
                if step.role in {"first_body", "suffix"}:
                    if not isinstance(terminal, str) or not terminal:
                        raise GeneratedDecodeError(
                            "segmented decode has no residual terminal binding"
                        )
                    residual = torch.empty_like(slot.hidden_last[:capacity])
                    overlay[terminal] = residual
                elif step.role == "body" and terminal != "x":
                    raise GeneratedDecodeError(
                        "segmented decode body must carry its residual in place"
                    )
                if step.role == "prefix":
                    width = int(boundary["query_heads"]) * int(boundary["head_dim"])
                    overlay[boundary["query"]] = torch.empty(
                        (capacity, width), dtype=torch.bfloat16,
                        device=runtime.device,
                    )
                    if boundary["gate"] is not None:
                        overlay[boundary["gate"]] = torch.empty(
                            (capacity, width), dtype=torch.bfloat16,
                            device=runtime.device,
                        )
                    attn_result = torch.empty(
                        (capacity, width), dtype=torch.bfloat16,
                        device=runtime.device,
                    )
                elif step.role in {"suffix", "final_suffix"}:
                    if attn_result is None:
                        raise GeneratedDecodeError(
                            "attention suffix has no external output storage"
                        )
                    overlay[boundary["output"]] = attn_result
                    attn_result = None
                bound.append(bind_fragment(
                    step, program,
                    parent_descriptor=parent.descriptor,
                    parent_weights=parent_weights,
                    parent_runtime_inputs=inputs,
                    fragment_runtime_inputs=overlay,
                    terminal_output=terminal or "",
                    active_batch=capacity,
                    stream=slot.compute_stream,
                    device=runtime.device,
                ))
            composite = BoundSegmentedDecode(
                bound, plan=steps, num_layers=parent.descriptor["num_layers"],
                attention=attention,
            )
            # Live positions are positive by admission, so the QKV fallback
            # scalar is not used; the captured graph reads mutable GPU metadata.
            with torch.cuda.stream(slot.compute_stream):
                graph = composite.capture(
                    stream=slot.compute_stream,
                    active_batch=capacity,
                    kv_len=1,
                )
            self._slots[int(slot.slot_id)] = (composite, attention, graph)

    def supports(self, batch_size: int) -> bool:
        return batch_size == self._capacity

    @property
    def artifact_receipts(self):
        return self._whole.artifact_receipts

    def state_requirements_for(self, batch_size: int):
        if not self.supports(batch_size):
            raise GeneratedDecodeError("segmented decode lacks requested capacity")
        return self._whole.state_requirements_for(batch_size)

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        if not self.supports(batch_size):
            raise GeneratedDecodeError("segmented decode lacks requested capacity")
        for step in self._whole._input_preparation_plan:
            self._whole._spec.preparation_callbacks[step.name](slot, batch_size)
        _composite, attention, graph = self._slots[int(slot.slot_id)]
        attention.update_metadata(slot)
        with torch.cuda.stream(slot.compute_stream):
            graph.replay()


__all__ = ["SegmentedGeneratedDecode"]
