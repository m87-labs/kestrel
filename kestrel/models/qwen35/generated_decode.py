"""Qwen runtime tensors for bundled generated decode."""

from typing import Any

import torch

from kestrel.runtime.generated_decode import (
    DeviceInputPreparation,
    GeneratedDecode,
    GeneratedDecodeSpec,
)


def _state_tensors(state_pool: Any, field: str) -> list[torch.Tensor | None]:
    return [
        None if storage is None else getattr(storage, field)
        for storage in state_pool.layers
    ]


def _paged_tensors(paged_layers: Any, field: str) -> list[torch.Tensor | None]:
    result = []
    for layer in paged_layers:
        if layer is None:
            result.append(None)
            continue
        tensor = getattr(layer, field)
        if int(tensor.shape[2]) != 1:
            raise ValueError(
                f"generated decode requires unit KV pages, got {tuple(tensor.shape)}")
        result.append(tensor[:, :, 0, :])
    return result


def create_generated_decode(runtime: Any) -> GeneratedDecode | None:
    layers = runtime._paged_kv.layers
    if any(
        layer is not None
        and (int(layer.k_cache.shape[2]) != 1 or int(layer.v_cache.shape[2]) != 1)
        for layer in layers
    ):
        return None

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

    page_table = runtime.page_table.page_table
    return GeneratedDecode.try_create(
        runtime,
        GeneratedDecodeSpec(
            label="Qwen",
            weight_root=runtime.model,
            weight_layer_prefix="model.language_model.layers",
            runtime_inputs=lambda: {
                "page_table": page_table,
                "rope_inv_freq": (
                    runtime.model.model.language_model.rotary_emb.inv_freq),
                "mK": _paged_tensors(layers, "k_cache"),
                "mV": _paged_tensors(layers, "v_cache"),
                "kv_len": 1,
            },
            capacity_inputs=state_inputs,
            slot_inputs=lambda slot, capacity: {
                "input_ids": slot.decode_token_ids[:capacity],
                "final_norm": slot.hidden_last[:capacity],
                "logits": slot.logits[:capacity],
                "batch_idx": slot.meta.batch_idx.gpu[:capacity],
                "input_pos": slot.meta.input_pos.gpu[:capacity],
                "position_ids": slot.position_ids[1:4, :capacity, 0],
            },
            launch_extents=lambda slot, batch_size: {
                "active_batch": batch_size,
                "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
            },
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
        ),
    )


__all__ = ["create_generated_decode"]
