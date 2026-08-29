"""Qwen runtime tensors for bundled generated decode."""

from typing import Any

import torch

from kestrel.runtime.generated_decode import (
    DeviceInputPreparation,
    GeneratedDecode,
    GeneratedDecodeSpec,
    PagedDecodeBindings,
)


def _state_tensors(state_pool: Any, field: str) -> list[torch.Tensor | None]:
    return [
        None if storage is None else getattr(storage, field)
        for storage in state_pool.layers
    ]


def prepare_generated_weight_storage(
    runtime: Any,
    model: torch.nn.Module,
    *,
    required: bool,
) -> Any | None:
    """Bind checkpoint load targets directly into the final generated layout."""

    if runtime.device.type != "cuda" or runtime.dtype is not torch.bfloat16:
        if required:
            raise RuntimeError(
                "generated Qwen decode requires CUDA BF16 model storage"
            )
        return None
    try:
        from kestrel_kernels import generated_decode as generated_runtime
    except ModuleNotFoundError as exc:
        if exc.name not in {"kestrel_kernels", "kestrel_kernels.generated_decode"}:
            raise
        if required:
            raise RuntimeError(
                "generated Qwen decode requires load-time weight binding support"
            )
        return None
    allocate_weight_storage_for_loading = getattr(
        generated_runtime, "allocate_weight_storage_for_loading", None
    )
    resolve_compatible_programs = getattr(
        generated_runtime, "resolve_compatible_programs", None
    )
    if not callable(allocate_weight_storage_for_loading) or not callable(
        resolve_compatible_programs
    ):
        if required:
            raise RuntimeError(
                "generated Qwen decode requires load-time weight binding support"
            )
        return None

    properties = torch.cuda.get_device_properties(runtime.device)
    programs = tuple(resolve_compatible_programs(
        model,
        layer_prefix="model.language_model.layers",
        arch=f"sm{properties.major}{properties.minor}",
        device_sms=int(properties.multi_processor_count),
    ))
    missing_batches = []
    for batch_size in range(1, runtime.max_batch_size + 1):
        covered = False
        for program in programs:
            static = program.static_extent_bindings.get("active_batch")
            minimum = int(
                program.runtime_extent_minimums.get("active_batch", 1)
            )
            covered = covered or (
                (static is None and minimum <= batch_size <= program.capacity)
                or static == batch_size
            )
        if not covered:
            missing_batches.append(batch_size)
    if missing_batches:
        if required:
            raise RuntimeError(
                "generated Qwen decode has no load-time artifact coverage for "
                f"batch sizes {missing_batches}"
            )
        return None
    contracts = {repr(program.descriptor["weights"]) for program in programs}
    if len(contracts) != 1:
        raise RuntimeError("generated Qwen artifacts disagree on weight storage")
    return allocate_weight_storage_for_loading(
        model,
        programs[0].descriptor,
        device=runtime.device,
        layer_prefix="model.language_model.layers",
    )


def _generated_decode_spec(runtime: Any) -> GeneratedDecodeSpec:
    bindings = PagedDecodeBindings(
        runtime._paged_kv,
        extra_runtime_inputs=lambda bound_runtime: {
            "rope_delta_table": bound_runtime._decode_rope_deltas,
            "rope_inv_freq": (
                bound_runtime.model.model.language_model.rotary_emb.inv_freq),
        },
        extra_slot_inputs=lambda slot, capacity: {
            "position_ids": slot.position_ids[1:4, :capacity, 0],
        },
    )

    state_cache = {}
    recurrent_by_form = {}

    def state_inputs(_capacity, requirements):
        if not state_cache:
            state_cache["conv"] = _state_tensors(
                runtime._linear_state_pool, "conv_states")
            state_cache["recurrent"] = _state_tensors(
                runtime._linear_state_pool, "recurrent_states")
        recurrent = state_cache["recurrent"]
        requirement = next(
            (item for item in requirements
             if item.buffer == "gdn_recurrent_state"),
            None,
        )
        if requirement is not None:
            form = requirement.physical_form
            recurrent = recurrent_by_form.get(form)
            if recurrent is None:
                recurrent = runtime._linear_state_pool.recurrent_tensors_for_form(form)
                recurrent_by_form[form] = recurrent
        return {
            "gdn_conv_state": state_cache["conv"],
            "gdn_recurrent_state": recurrent,
        }

    return GeneratedDecodeSpec(
        label="Qwen",
        weight_root=runtime.model,
        weight_layer_prefix="model.language_model.layers",
        bindings=bindings,
        weight_storage=getattr(runtime, "_generated_weight_storage", None),
        capacity_inputs=state_inputs,
        preparations=(
            DeviceInputPreparation(
                "gather_rope_deltas", ("rope_deltas",), ("batch_idx",)),
            DeviceInputPreparation(
                "prepare_position_ids",
                ("position_ids",),
                ("input_pos", "rope_deltas"),
            ),
        ),
        not_ready_inputs=frozenset({"position_ids"}),
        preparation_callbacks={
            "gather_rope_deltas": runtime._gather_decode_rope_deltas,
            "prepare_position_ids": runtime._prepare_decode_position_ids,
        },
    )


def generated_decode_slot_capacity(
    runtime: Any, *, required: bool = False,
) -> int | None:
    required_batch_sizes = range(1, runtime.max_batch_size + 1)
    capacity = GeneratedDecode.resolve_slot_capacity(
        runtime,
        _generated_decode_spec(runtime),
        required_batch_sizes=required_batch_sizes,
    )
    if required and capacity is None:
        raise RuntimeError(
            "Qwen requires a compatible bundled generated-decode program"
        )
    return capacity


def create_generated_decode(
    runtime: Any, *, required: bool = False,
) -> GeneratedDecode | None:
    spec = _generated_decode_spec(runtime)
    if required:
        return GeneratedDecode.require(
            runtime, spec, batch_sizes=range(1, runtime.max_batch_size + 1)
        )
    return GeneratedDecode.try_create(
        runtime,
        spec,
        required_batch_sizes=range(1, runtime.max_batch_size + 1),
    )


__all__ = [
    "create_generated_decode",
    "generated_decode_slot_capacity",
    "prepare_generated_weight_storage",
]
