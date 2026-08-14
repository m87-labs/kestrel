"""Moondream tensors for bundled compiler-generated decode."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from kestrel.runtime.generated_decode import GeneratedDecode, GeneratedDecodeSpec


def _rope_tables(text: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cache = text.cos_sin_cache.float()
    if int(cache.shape[-1]) % 2:
        raise RuntimeError("Moondream rotary cache width must be even")
    half = int(cache.shape[-1]) // 2
    cos = torch.cat((cache[:, :half], cache[:, :half]), dim=1)
    sin = torch.cat((cache[:, half:], cache[:, half:]), dim=1)
    return cos.contiguous(), sin.contiguous()


@dataclass(frozen=True)
class MoondreamDecodeBindings:
    layers: Sequence[Any]

    def is_eligible(self, runtime: Any) -> bool:
        return (
            runtime._lora_workspace is None
            and runtime.page_size == 1
            and len(self.layers) == len(runtime.model.text.blocks)
            and all(layer.cache.quantized for layer in self.layers)
            and all(
                int(layer.cache.k_cache.shape[2])
                == int(layer.cache.v_cache.shape[2])
                == 1
                for layer in self.layers
            )
            and all(
                hasattr(block.attn, "_tau_pos_table")
                for block in runtime.model.text.blocks
            )
        )

    def runtime_inputs(self, runtime: Any) -> Mapping[str, Any]:
        rope_cos, rope_sin = _rope_tables(runtime.model.text)
        caches = [layer.cache for layer in self.layers]
        return {
            "tau_pos": [
                block.attn._tau_pos_table.detach().float().contiguous()
                for block in runtime.model.text.blocks
            ],
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "mK_dequant_scale": [cache.k_scale_tensor for cache in caches],
            "mV_dequant_scale": [cache.v_scale_tensor for cache in caches],
            "mK": [cache.k_cache[:, :, 0, :] for cache in caches],
            "mV": [cache.v_cache[:, :, 0, :] for cache in caches],
            "page_table": runtime.page_table.page_table,
        }

    def slot_inputs(self, slot: Any, capacity: int) -> Mapping[str, Any]:
        return {
            "x": slot.hidden_last[:capacity],
            "batch_idx": slot.meta.batch_idx.gpu[:capacity],
            "input_pos": slot.meta.input_pos.gpu[:capacity],
        }

    @staticmethod
    def launch_extents(slot: Any, batch_size: int) -> Mapping[str, int]:
        return {
            "active_batch": int(batch_size),
            "kv_len": int(slot.meta.input_pos.cpu[:batch_size].max()) + 1,
        }


def create_generated_decode(
    runtime: Any,
    *,
    required: bool = False,
) -> GeneratedDecode | None:
    spec = GeneratedDecodeSpec(
        label="Moondream",
        weight_root=runtime.model.text,
        weight_layer_prefix="blocks",
        bindings=MoondreamDecodeBindings(runtime.layer_caches),
    )
    if required:
        return GeneratedDecode.require(
            runtime,
            spec,
            capacity=runtime.max_batch_size,
        )
    return GeneratedDecode.try_create(runtime, spec)


__all__ = ["MoondreamDecodeBindings", "create_generated_decode"]
