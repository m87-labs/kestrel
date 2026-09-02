"""Moondream tensors for bundled compiler-generated decode."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeSpec,
    _GeneratedDecodePlan,
)


def _logical_weight_sources(text: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Expose Moondream's compiler namespace without changing module ownership."""

    sources = dict(text.named_parameters(remove_duplicate=False))
    sources.update(dict(text.named_buffers(remove_duplicate=False)))
    for layer, block in enumerate(text.blocks):
        if not hasattr(block.mlp, "router"):
            continue
        fused = block.mlp["mlp"]
        prefix = f"blocks.{layer}.mlp."
        sources[prefix + "up_experts.weight"] = fused.up_experts.weight
        sources[prefix + "up_experts.scale"] = fused.up_experts.scale
        sources[prefix + "down_experts.weight"] = fused.down_experts.weight
        sources[prefix + "down_experts.scale"] = fused.down_experts.scale
    return sources


def _engine_weight_buffers(text: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = ("moe_up_w_slab", "moe_up_scale_slab")
    return {
        name: buffer
        for name in names
        if (buffer := getattr(text, name, None)) is not None
    }


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
            "tau_pos": torch.stack([
                block.attn._tau_pos_table.detach().float()
                for block in runtime.model.text.blocks
            ]).contiguous(),
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "mK_dequant_scale": torch.stack([
                cache.k_scale_tensor for cache in caches
            ]).contiguous(),
            "mV_dequant_scale": torch.stack([
                cache.v_scale_tensor for cache in caches
            ]).contiguous(),
            "mK": [cache.k_cache[:, :, 0, :] for cache in caches],
            "mV": [cache.v_cache[:, :, 0, :] for cache in caches],
            "page_table": runtime.page_table.page_table,
            # Scalar launch arguments need a construction-time placeholder;
            # ``launch_extents`` supplies the live maximum position for every run.
            "kv_len": 1,
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


def _prepare_generated_decode(runtime: Any) -> _GeneratedDecodePlan:
    spec = GeneratedDecodeSpec(
        label="Moondream",
        weight_root=runtime.model.text,
        weight_layer_prefix="blocks",
        weight_sources=_logical_weight_sources(runtime.model.text),
        engine_buffers=_engine_weight_buffers(runtime.model.text),
        bindings=MoondreamDecodeBindings(runtime.layer_caches),
    )
    return GeneratedDecode.plan(runtime, spec)


def create_generated_decode(
    runtime: Any,
    *,
    required: bool = False,
    plan: _GeneratedDecodePlan | None = None,
) -> GeneratedDecode | None:
    if plan is None:
        plan = _prepare_generated_decode(runtime)
    spec = plan.spec
    if required:
        return GeneratedDecode.require(
            runtime,
            spec,
            batch_sizes=range(1, runtime.max_batch_size + 1),
            plan=plan,
        )
    return GeneratedDecode.try_create(runtime, spec, plan=plan)


__all__ = [
    "MoondreamDecodeBindings",
    "create_generated_decode",
]
