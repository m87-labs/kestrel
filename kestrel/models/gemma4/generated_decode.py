"""Gemma runtime tensors for bundled generated decode."""

from typing import Any

import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeSpec,
    PagedDecodeBindings,
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


def create_generated_decode(runtime: Any) -> GeneratedDecode:
    """Describe Gemma inputs; shared runtime code owns binding and launch."""

    layers = runtime._kv_cache
    if not layers:
        raise RuntimeError("Gemma generated decode requires paged K/V storage")
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
        extra_runtime_inputs=rope_inputs,
    )
    return GeneratedDecode.require(
        runtime,
        GeneratedDecodeSpec(
            label="Gemma",
            weight_root=runtime.model,
            weight_layer_prefix="model.language_model.layers",
            bindings=bindings,
        ),
        capacity=runtime.max_batch_size,
    )


__all__ = ["create_generated_decode"]
