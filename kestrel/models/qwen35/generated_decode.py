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


def create_generated_decode(
    runtime: Any, *, required: bool = False,
) -> GeneratedDecode | None:
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

    spec = GeneratedDecodeSpec(
        label="Qwen",
        weight_root=runtime.model,
        weight_layer_prefix="model.language_model.layers",
        bindings=bindings,
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
    if required:
        return GeneratedDecode.require(
            runtime, spec, batch_sizes=range(1, runtime.max_batch_size + 1)
        )
    return GeneratedDecode.try_create(
        runtime,
        spec,
        required_batch_sizes=range(1, runtime.max_batch_size + 1),
    )


__all__ = ["create_generated_decode"]
