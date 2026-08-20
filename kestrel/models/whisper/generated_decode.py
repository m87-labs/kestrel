"""Whisper bindings for Kestrel's shared generated-decode runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec

from .weights import WhisperModelWeights, named_whisper_tensors


@dataclass(frozen=True, slots=True)
class WhisperDecodeBindings:
    """Map resident Whisper state onto the compiler-emitted decode ABI."""

    def is_eligible(self, runtime: Any) -> bool:
        return len(runtime.self_kv.keys) == len(runtime.self_kv.values) == 4

    def runtime_inputs(self, runtime: Any) -> Mapping[str, Any]:
        return {
            "page_table": runtime.page_table.page_table,
            "self_key": runtime.self_kv.keys,
            "self_value": runtime.self_kv.values,
            "cross_key": runtime.cross_kv.keys,
            "cross_value": runtime.cross_kv.values,
        }

    def slot_inputs(self, slot: Any, capacity: int) -> Mapping[str, Any]:
        return {
            "token_ids": slot.decode_token_ids[:capacity],
            "input_pos": slot.meta.input_pos.gpu[:capacity],
            "batch_idx": slot.meta.batch_idx.gpu[:capacity],
            "logits": slot.logits[:capacity],
        }

    @staticmethod
    def launch_extents(_slot: Any, batch_size: int) -> Mapping[str, int]:
        return {"active_batch": int(batch_size)}


def create_generated_decode(
    runtime: Any,
    weights: WhisperModelWeights,
) -> GeneratedDecode:
    """Require the packed programs for every admitted Whisper batch size."""

    return GeneratedDecode.require(
        runtime,
        GeneratedDecodeSpec(
            label="Whisper",
            weight_root=runtime,
            weight_layer_prefix="model.decoder.layers",
            weight_sources=named_whisper_tensors(weights),
            bindings=WhisperDecodeBindings(),
        ),
        batch_sizes=runtime._decode_batch_capacities,
    )


__all__ = ["WhisperDecodeBindings", "create_generated_decode"]
