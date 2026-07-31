"""Qwen descriptors for compiler-generated decode."""

from __future__ import annotations

from typing import Any

import torch

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec


class _UnsupportedDecodeConfig(ValueError):
    pass


def _layer_kinds(config: Any) -> list[int]:
    mapping = {"linear_attention": 0, "full_attention": 1}
    try:
        return [mapping[str(kind)] for kind in config.layer_types]
    except KeyError as exc:
        raise _UnsupportedDecodeConfig(
            f"unsupported Qwen decode layer type {exc.args[0]!r}"
        ) from exc


def _rope_config(config: Any) -> tuple[float, list[int]]:
    partial = float(config.partial_rotary_factor)
    sections = [int(value) for value in config.mrope_section]
    if len(sections) != 3:
        raise _UnsupportedDecodeConfig(
            f"Qwen decode needs three M-RoPE sections, got {sections}"
        )
    return partial, sections


def _compile_from_config(
    config: Any,
    *,
    batch_capacity: int = 1,
    num_ctas: int,
    gpu: str,
):
    from mkl.compiler.frontend.models.qwen import compile_qwen35

    if config.is_moe:
        raise _UnsupportedDecodeConfig(
            "the current Qwen decode trace covers dense MLP layers"
        )
    partial_rotary, rope_sections = _rope_config(config)
    return compile_qwen35(
        n_layers=int(config.num_hidden_layers),
        hidden=int(config.hidden_size),
        inter=int(config.intermediate_size),
        nh=int(config.num_attention_heads),
        nkv=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        num_k_heads=int(config.linear_num_key_heads),
        num_v_heads=int(config.linear_num_value_heads),
        key_head_dim=int(config.linear_key_head_dim),
        value_head_dim=int(config.linear_value_head_dim),
        conv_kernel=int(config.linear_conv_kernel_dim),
        partial_rotary=partial_rotary,
        rope_sections=rope_sections,
        vocab_size=int(config.vocab_size),
        rms_norm_eps=float(config.rms_norm_eps),
        tie_word_embeddings=bool(config.tie_word_embeddings),
        num_ctas=int(num_ctas),
        num_splits=None,
        max_kv_len=int(config.max_position_embeddings),
        batch_tile=int(batch_capacity),
        static_extent_bindings={},
        gpu=str(gpu),
        layer_types=_layer_kinds(config),
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
        if tensor.shape[2] != 1:
            raise ValueError(
                f"generated decode requires unit KV pages, got {tuple(tensor.shape)}"
            )
        result.append(tensor[:, :, 0, :])
    return result


def create_generated_decode(runtime: Any) -> GeneratedDecode | None:
    """Describe Qwen inputs; shared runtime code owns binding and launch."""

    layers = runtime._paged_kv.layers
    if any(
        layer is not None
        and (int(layer.k_cache.shape[2]) != 1 or int(layer.v_cache.shape[2]) != 1)
        for layer in layers
    ):
        return None
    if runtime.device.type != "cuda" or runtime.dtype is not torch.bfloat16:
        return None
    try:
        from mkl.compiler.frontend.models.aot import DECODE_BATCH_CAPACITIES
        from mkl.megakernel.device_input_preparation import DeviceInputPreparation
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        if missing != "mkl" and not missing.startswith("mkl."):
            raise
        return None

    config = runtime.model.model.language_model.config
    state_cache: dict[str, list[torch.Tensor | None]] = {}
    recurrent_states_by_form: dict[Any, list[torch.Tensor | None]] = {}

    def state_inputs(_capacity: int, requirements: tuple[Any, ...]) -> dict[str, Any]:
        if not state_cache:
            state_cache["conv"] = _state_tensors(
                runtime._linear_state_pool, "conv_states"
            )
            state_cache["recurrent"] = _state_tensors(
                runtime._linear_state_pool, "recurrent_states"
            )
        recurrent_requirements = tuple(
            requirement
            for requirement in requirements
            if requirement.buffer == "gdn_recurrent_state"
        )
        if len(recurrent_requirements) > 1:
            raise RuntimeError(
                "generated decode declares recurrent state more than once"
            )
        recurrent_states = state_cache["recurrent"]
        if recurrent_requirements:
            form = recurrent_requirements[0].physical_form
            recurrent_states = recurrent_states_by_form.get(form)
            if recurrent_states is None:
                recurrent_states = (
                    runtime._linear_state_pool.recurrent_tensors_for_form(form)
                )
                recurrent_states_by_form[form] = recurrent_states
        return {
            "gdn_conv_state": state_cache["conv"],
            "gdn_recurrent_state": recurrent_states,
        }

    page_table = runtime.page_table.page_table
    return GeneratedDecode.try_create(
        runtime,
        GeneratedDecodeSpec(
            label="Qwen",
            capacities=DECODE_BATCH_CAPACITIES,
            compile_program=lambda capacity, properties: _compile_from_config(
                config,
                batch_capacity=capacity,
                num_ctas=properties.multi_processor_count,
                gpu=properties.name,
            ),
            weight_root=runtime.model,
            weight_layer_prefix="model.language_model.layers",
            runtime_inputs=lambda: {
                "page_table": page_table,
                "rope_inv_freq": (
                    runtime.model.model.language_model.rotary_emb.inv_freq
                ),
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
            runtime_extents=lambda _capacity: {
                "n_pages": int(runtime.page_table.n_pages),
                "page_table_capacity": int(page_table.shape[1]),
                "state_rows": int(page_table.shape[0]),
            },
            launch_extents=lambda slot, batch_size: {
                "active_batch": batch_size,
                "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
            },
            preparations=(
                DeviceInputPreparation(
                    "gather_rope_deltas",
                    produces=("rope_deltas",),
                    requires=("batch_idx",),
                ),
                DeviceInputPreparation(
                    "prepare_position_ids",
                    produces=("position_ids",),
                    requires=("input_pos", "rope_deltas"),
                ),
            ),
            not_ready_inputs=frozenset({"position_ids"}),
            preparation_callbacks={
                "gather_rope_deltas": (
                    lambda slot, batch_size: runtime._gather_decode_rope_deltas(
                        slot, batch_size
                    )
                ),
                "prepare_position_ids": (
                    lambda slot, batch_size: runtime._prepare_decode_position_ids(
                        slot, batch_size
                    )
                ),
            },
            unsupported=(_UnsupportedDecodeConfig,),
        ),
    )


__all__ = ["create_generated_decode"]
