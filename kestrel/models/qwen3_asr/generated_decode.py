"""Qwen3-ASR bindings for Kestrel's shared generated-decode runtime."""

from __future__ import annotations

from typing import Any

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeSpec,
    PagedDecodeBindings,
)


def create_generated_decode(runtime: Any) -> GeneratedDecode:
    """Bind bundled programs covering every admitted batch size."""

    spec = GeneratedDecodeSpec(
        label="Qwen3-ASR",
        weight_root=runtime.model,
        weight_layer_prefix="model.language_model.layers",
        bindings=PagedDecodeBindings(
            runtime._paged_kv,
            extra_runtime_inputs=lambda bound: {
                "rope_cosine": bound._rope_cosine,
                "rope_sine": bound._rope_sine,
            },
        ),
    )
    return GeneratedDecode.require(
        runtime,
        spec,
        batch_sizes=range(1, runtime.max_batch_size + 1),
    )


__all__ = ["create_generated_decode"]
