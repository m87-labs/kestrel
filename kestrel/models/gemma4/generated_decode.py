"""Gemma descriptors for compiler-generated decode."""

from __future__ import annotations

from typing import Any

import torch

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec


class _UnsupportedDecodeConfig(ValueError):
    pass


def _compile_from_config(
    config: Any,
    *,
    batch_capacity: int = 1,
    max_kv_len: int,
    num_ctas: int,
    gpu: str,
):
    from mkl.compiler.frontend.models.gemma import compile_gemma4_decode

    if config.attention_k_eq_v:
        raise _UnsupportedDecodeConfig(
            "generated Gemma decode requires an explicit V projection"
        )
    ple_hidden = int(config.hidden_size_per_layer_input)
    if ple_hidden <= 0:
        raise _UnsupportedDecodeConfig(
            "generated Gemma decode requires traced PLE inputs"
        )
    layer_types = tuple(config.layer_types)
    supported = {"sliding_attention", "full_attention"}
    if not layer_types or set(layer_types) - supported:
        raise _UnsupportedDecodeConfig(
            f"unsupported Gemma layer types {sorted(set(layer_types) - supported)}"
        )
    first_shared = len(layer_types) - config.num_kv_shared_layers
    branches = {
        (
            "global" if layer_type == "full_attention" else "local",
            "shared" if layer >= first_shared else "fresh",
        )
        for layer, layer_type in enumerate(layer_types)
    }
    required = {
        ("local", "fresh"),
        ("global", "fresh"),
        ("local", "shared"),
        ("global", "shared"),
    }
    if branches != required:
        raise _UnsupportedDecodeConfig(
            "generated Gemma decode requires fresh/shared local/global coverage"
        )

    return compile_gemma4_decode(
        layer_types=list(layer_types),
        num_kv_shared_layers=config.num_kv_shared_layers,
        hidden=config.hidden_size,
        inter=config.intermediate_size,
        nh=config.num_attention_heads,
        nkv=config.num_key_value_heads,
        global_nkv=config.num_global_key_value_heads,
        local_head_dim=config.head_dim,
        global_head_dim=config.global_head_dim,
        window=config.sliding_window,
        max_kv_len=max_kv_len,
        ple_hidden=ple_hidden,
        ple_vocab=config.vocab_size_per_layer_input,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        final_logit_softcapping=config.final_logit_softcapping,
        tie_word_embeddings=True,
        double_wide_mlp=config.use_double_wide_mlp,
        num_ctas=num_ctas,
        num_splits=None,
        batch_tile=batch_capacity,
        static_extent_bindings={},
        gpu=gpu,
    )


def _paged_tensors(
    layers: Any,
    layer_types: tuple[str, ...],
    *,
    kind: str,
    field: str,
) -> list[torch.Tensor | None]:
    values = []
    for layer_type, layer in zip(layer_types, layers, strict=True):
        if layer is None or layer_type != kind:
            values.append(None)
            continue
        tensor = getattr(layer, field)
        if tensor.shape[2] != 1:
            raise ValueError(
                f"generated decode requires unit KV pages, got {tuple(tensor.shape)}"
            )
        values.append(tensor[:, :, 0, :])
    return values


def _rope_tables(runtime: Any) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    positions = torch.arange(
        runtime.max_seq_length,
        dtype=torch.int64,
        device=runtime.device,
    ).view(1, -1)
    probe = torch.empty((1, 1, 1), dtype=runtime.dtype, device=runtime.device)
    rotary = runtime.model.model.language_model.rotary_emb
    return {
        kind: tuple(
            table[0].float().contiguous()
            for table in rotary(probe, positions, kind)
        )
        for kind in ("sliding_attention", "full_attention")
    }


def _runtime_extents(runtime: Any, capacity: int) -> dict[str, int]:
    page_table = runtime.page_table.page_table
    return {
        "active_batch": capacity,
        "n_pages": runtime.page_table.n_pages,
        "page_table_capacity": page_table.shape[1],
        "position_capacity": runtime.max_seq_length,
        "state_rows": page_table.shape[0],
    }


def create_generated_decode(runtime: Any) -> GeneratedDecode | None:
    """Describe Gemma inputs; shared runtime code owns binding and launch."""

    layers = runtime._kv_cache.layers
    if not layers or any(
        layer is not None
        and (layer.k_cache.shape[2] != 1 or layer.v_cache.shape[2] != 1)
        for layer in layers
    ):
        return None
    try:
        from mkl.compiler.frontend.models.aot import DECODE_BATCH_CAPACITIES
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        if missing != "mkl" and not missing.startswith("mkl."):
            raise
        return None

    config = runtime.model.model.language_model.config

    def runtime_inputs() -> dict[str, Any]:
        layer_types = tuple(config.layer_types)
        page_table = runtime.page_table.page_table
        ropes = _rope_tables(runtime)
        local_cos, local_sin = ropes["sliding_attention"]
        global_cos, global_sin = ropes["full_attention"]
        return {
            "page_table": page_table,
            "kv_len": 1,
            "rope_cos_local": local_cos,
            "rope_sin_local": local_sin,
            "rope_cos_global": global_cos,
            "rope_sin_global": global_sin,
            "mK_local": _paged_tensors(
                layers, layer_types, kind="sliding_attention", field="k_cache"
            ),
            "mV_local": _paged_tensors(
                layers, layer_types, kind="sliding_attention", field="v_cache"
            ),
            "mK_global": _paged_tensors(
                layers, layer_types, kind="full_attention", field="k_cache"
            ),
            "mV_global": _paged_tensors(
                layers, layer_types, kind="full_attention", field="v_cache"
            ),
        }

    return GeneratedDecode.try_create(
        runtime,
        GeneratedDecodeSpec(
            label="Gemma",
            capacities=DECODE_BATCH_CAPACITIES,
            compile_program=lambda capacity, properties: _compile_from_config(
                config,
                batch_capacity=capacity,
                max_kv_len=runtime.max_seq_length,
                num_ctas=properties.multi_processor_count,
                gpu=properties.name,
            ),
            weight_root=runtime.model,
            weight_layer_prefix="model.language_model.layers",
            runtime_inputs=runtime_inputs,
            slot_inputs=lambda slot, capacity: {
                "input_ids": slot.decode_token_ids[:capacity],
                "final_norm": slot.hidden_last[:capacity],
                "logits": slot.logits[:capacity],
                "batch_idx": slot.meta.batch_idx.gpu[:capacity],
                "input_pos": slot.meta.input_pos.gpu[:capacity],
            },
            runtime_extents=lambda capacity: _runtime_extents(runtime, capacity),
            launch_extents=lambda slot, batch_size: {
                "active_batch": batch_size,
                "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
            },
            unsupported=(_UnsupportedDecodeConfig,),
        ),
    )


__all__ = ["create_generated_decode"]
