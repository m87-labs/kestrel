"""Gemma descriptors for compiler-generated decode."""

from __future__ import annotations

from typing import Any

import torch

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec


def _compile_from_config(
    config: Any,
    *,
    batch_capacity: int = 1,
    max_kv_len: int,
    num_ctas: int,
    gpu: str,
):
    from mkl.compiler.frontend import DecodeCompileTarget
    from mkl.compiler.frontend.models.gemma import (
        Gemma4DecodeTraceConfig,
        compile_gemma4_decode,
    )

    trace = Gemma4DecodeTraceConfig.from_model_config(
        config,
        max_kv_len=max_kv_len,
    )
    return compile_gemma4_decode(
        trace,
        target=DecodeCompileTarget(
            batch_capacity=batch_capacity,
            num_ctas=num_ctas,
            gpu=gpu,
        ),
    )


def _paged_kv(
    layers: Any,
    layer_types: tuple[str, ...],
    *,
    kind: str,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    keys = []
    values = []
    for layer_type, layer in zip(layer_types, layers, strict=True):
        if layer is None or layer_type != kind:
            keys.append(None)
            values.append(None)
            continue
        for tensor, output in (
            (layer.k_cache, keys),
            (layer.v_cache, values),
        ):
            if tensor.shape[2] != 1:
                raise ValueError(
                    "generated decode requires unit KV pages, "
                    f"got {tuple(tensor.shape)}"
                )
            output.append(tensor[:, :, 0, :])
    return keys, values


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
        from mkl.compiler.frontend.models.gemma import (
            UnsupportedGemma4DecodeConfig,
        )
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
        local_k, local_v = _paged_kv(
            layers, layer_types, kind="sliding_attention"
        )
        global_k, global_v = _paged_kv(
            layers, layer_types, kind="full_attention"
        )
        return {
            "page_table": page_table,
            "kv_len": 1,
            "rope_cos_local": local_cos,
            "rope_sin_local": local_sin,
            "rope_cos_global": global_cos,
            "rope_sin_global": global_sin,
            "mK_local": local_k,
            "mV_local": local_v,
            "mK_global": global_k,
            "mV_global": global_v,
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
            runtime_extents=lambda _capacity: {
                "n_pages": runtime.page_table.n_pages,
                "page_table_capacity": runtime.page_table.page_table.shape[1],
                "position_capacity": runtime.max_seq_length,
                "state_rows": runtime.page_table.page_table.shape[0],
            },
            launch_extents=lambda slot, batch_size: {
                "active_batch": batch_size,
                "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
            },
            unsupported=(UnsupportedGemma4DecodeConfig,),
        ),
    )


__all__ = ["create_generated_decode"]
