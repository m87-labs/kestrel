"""Gemma descriptors for compiler-generated decode."""

from __future__ import annotations

from typing import Any

import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeSpec,
    PagedDecodeBindings,
)


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
            table[0].float().contiguous() for table in rotary(probe, positions, kind)
        )
        for kind in ("sliding_attention", "full_attention")
    }


def create_generated_decode(runtime: Any) -> GeneratedDecode | None:
    """Describe Gemma inputs; shared runtime code owns binding and launch."""

    layers = runtime._kv_cache.layers
    if not layers or not PagedDecodeBindings(layers).is_eligible(runtime):
        return None
    config = runtime.model.model.language_model.config

    def rope_inputs(bound_runtime: Any) -> dict[str, torch.Tensor]:
        ropes = _rope_tables(bound_runtime)
        local_cos, local_sin = ropes["sliding_attention"]
        global_cos, global_sin = ropes["full_attention"]
        return {
            "rope_cos_local": local_cos,
            "rope_sin_local": local_sin,
            "rope_cos_global": global_cos,
            "rope_sin_global": global_sin,
        }

    bindings = PagedDecodeBindings(
        layers,
        kv_sets=(
            ("local", "sliding_attention"),
            ("global", "full_attention"),
        ),
        layer_kinds=tuple(config.layer_types),
        position_capacity=runtime.max_seq_length,
        extra_runtime_inputs=rope_inputs,
    )
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
            bindings=bindings,
            unsupported=(UnsupportedGemma4DecodeConfig,),
        ),
    )


__all__ = ["create_generated_decode"]
