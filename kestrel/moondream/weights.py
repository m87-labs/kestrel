"""Utilities to load Moondream text weights into the self-contained model.

Adapted from the Moondream project (Apache-2.0).
"""


from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import safetensors
import torch
import torch.nn as nn

from ..ops import precompute_freqs_cis

@contextmanager
def safetensors_open(path: str):
    with safetensors.safe_open(path, framework="pt") as f:
        def get_tensor(name: str) -> torch.Tensor:
            return f.get_tensor(name)

        def get_keys() -> List[str]:
            return f.keys()

        get_tensor.keys = get_keys  # type: ignore[attr-defined]
        yield get_tensor


def _assign_text_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
    text = model.text

    weight_map: Dict[str, torch.Tensor] = {
        "text_model.transformer.embd.wte.weight": text.wte,
        "text_model.lm_head.ln.weight": text["post_ln"].weight,
        "text_model.lm_head.ln.bias": text["post_ln"].bias,
        "text_model.lm_head.linear.weight": text["lm_head"].weight,
        "text_model.lm_head.linear.bias": text["lm_head"].bias,
    }

    for i, block in enumerate(text["blocks"]):
        prefix = f"text_model.transformer.h.{i}"
        is_moe = hasattr(block.mlp, "router")
        weight_map.update(
            {
                f"{prefix}.ln.weight": block["ln"].weight,
                f"{prefix}.ln.bias": block["ln"].bias,
                f"{prefix}.mixer.Wqkv.weight": block["attn"]["qkv"].weight,
                f"{prefix}.mixer.Wqkv.bias": block["attn"]["qkv"].bias,
                f"{prefix}.mixer.out_proj.weight": block["attn"]["proj"].weight,
                f"{prefix}.mixer.out_proj.bias": block["attn"]["proj"].bias,
                f"{prefix}.tau_wq": block["attn"]["tau"]["wq"],
                f"{prefix}.tau_wv": block["attn"]["tau"]["wv"],
                f"{prefix}.tau_alpha": block["attn"]["tau"]["alpha"],
            }
        )
        if is_moe:
            weight_map.update(
                {
                    f"{prefix}.gate.weight": block["mlp"]["router"].weight,
                    f"{prefix}.gate.bias": block["mlp"]["router"].bias,
                    f"{prefix}.mlp.experts.weight": block["mlp"]["mlp"].up_experts.weight,
                    f"{prefix}.mlp.output_experts.weight": block["mlp"]["mlp"].down_experts.weight,
                }
            )
        else:
            weight_map.update(
                {
                    f"{prefix}.mlp.fc1.weight": block["mlp"]["fc1"].weight,
                    f"{prefix}.mlp.fc1.bias": block["mlp"]["fc1"].bias,
                    f"{prefix}.mlp.fc2.weight": block["mlp"]["fc2"].weight,
                    f"{prefix}.mlp.fc2.bias": block["mlp"]["fc2"].bias,
                }
            )

    for key, tensor in weight_map.items():
        tensor.data.copy_(get_tensor(key))

    for param in text.parameters():
        param.data = param.data.contiguous()


def _assign_vision_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
    if not hasattr(model, "vision"):
        return

    vision = model.vision
    weight_map: Dict[str, torch.Tensor] = {
        "vision_encoder.encoder.model.visual.patch_embed.linear.weight": vision[
            "patch_emb"
        ].weight,
        "vision_encoder.encoder.model.visual.patch_embed.linear.bias": vision[
            "patch_emb"
        ].bias,
        "vision_encoder.encoder.model.visual.pos_embed": vision.pos_emb,
        "vision_encoder.encoder.model.visual.norm.weight": vision["post_ln"].weight,
        "vision_encoder.encoder.model.visual.norm.bias": vision["post_ln"].bias,
        "vision_encoder.projection.mlp.fc1.weight": vision["proj_mlp"]["fc1"].weight,
        "vision_encoder.projection.mlp.fc1.bias": vision["proj_mlp"]["fc1"].bias,
        "vision_encoder.projection.mlp.fc2.weight": vision["proj_mlp"]["fc2"].weight,
        "vision_encoder.projection.mlp.fc2.bias": vision["proj_mlp"]["fc2"].bias,
    }

    for i, block in enumerate(vision["blocks"]):
        prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
        weight_map.update(
            {
                f"{prefix}.norm1.weight": block["ln1"].weight,
                f"{prefix}.norm1.bias": block["ln1"].bias,
                f"{prefix}.norm2.weight": block["ln2"].weight,
                f"{prefix}.norm2.bias": block["ln2"].bias,
                f"{prefix}.attn.qkv.weight": block["attn"]["qkv"].weight,
                f"{prefix}.attn.qkv.bias": block["attn"]["qkv"].bias,
                f"{prefix}.attn.proj.weight": block["attn"]["proj"].weight,
                f"{prefix}.attn.proj.bias": block["attn"]["proj"].bias,
                f"{prefix}.mlp.fc1.weight": block["mlp"]["fc1"].weight,
                f"{prefix}.mlp.fc1.bias": block["mlp"]["fc1"].bias,
                f"{prefix}.mlp.fc2.weight": block["mlp"]["fc2"].weight,
                f"{prefix}.mlp.fc2.bias": block["mlp"]["fc2"].bias,
            }
        )

    for key, tensor in weight_map.items():
        tensor.data.copy_(get_tensor(key))

    for param in vision.parameters():
        param.data = param.data.contiguous()


def _assign_region_weights(
    get_tensor: Callable[[str], torch.Tensor],
    region: nn.Module,
    *,
    convert: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    if not isinstance(region, nn.Module):
        raise TypeError("region must be an nn.Module with encoder/decoder attributes")

    # Linear layers
    region["coord_encoder"].weight.data.copy_(convert(get_tensor("region_model.coordinate_encoder.weight")))
    region["coord_encoder"].bias.data.copy_(convert(get_tensor("region_model.coordinate_encoder.bias")))
    region["coord_decoder"].weight.data.copy_(convert(get_tensor("region_model.coordinate_head.weight")))
    region["coord_decoder"].bias.data.copy_(convert(get_tensor("region_model.coordinate_head.bias")))
    region["size_encoder"].weight.data.copy_(convert(get_tensor("region_model.size_encoder.weight")))
    region["size_encoder"].bias.data.copy_(convert(get_tensor("region_model.size_encoder.bias")))
    region["size_decoder"].weight.data.copy_(convert(get_tensor("region_model.size_head.weight")))
    region["size_decoder"].bias.data.copy_(convert(get_tensor("region_model.size_head.bias")))

    # Fourier feature parameters are stored transposed in checkpoints.
    region.coord_features.data.copy_(
        convert(get_tensor("region_model.coordinate_features.weight")).T
    )
    region.size_features.data.copy_(
        convert(get_tensor("region_model.size_features.weight")).T
    )


def _refresh_rotary_tables(model: nn.Module) -> None:
    if not hasattr(model, "text") or not hasattr(model, "config"):
        return
    text_cfg = model.config.text
    cache: torch.Tensor = model.text.cos_sin_cache
    cos_sin_cache = precompute_freqs_cis(
        text_cfg.dim // (2 * text_cfg.n_heads),
        text_cfg.max_context,
        dtype=cache.dtype,
        device=cache.device,
    )
    cache.data.copy_(cos_sin_cache)


def load_text_weights(
    path: str,
    model: nn.Module,
    *,
    tensor_hook: Callable[[str, torch.Tensor], None] | None = None,
) -> None:
    load_moondream_weights(
        path,
        model,
        load_vision=False,
        tensor_hook=tensor_hook,
        region=None,
    )


def load_moondream_weights(
    path: str,
    model: nn.Module,
    *,
    load_vision: bool = True,
    tensor_hook: Callable[[str, torch.Tensor], None] | None = None,
    region: Optional[nn.Module] = None,
) -> None:
    target_param = next(model.text.parameters())
    target_dtype = target_param.dtype
    target_device = target_param.device

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(target_dtype)

    if path.endswith(".safetensors"):
        with safetensors_open(path) as get_tensor:
            name_map = {k.replace("._orig_mod", ""): k for k in get_tensor.keys()}

            def getter(name: str) -> torch.Tensor:
                raw = get_tensor(name_map[name])
                if tensor_hook is not None:
                    tensor_hook(name, raw)
                return convert(raw)

            _assign_text_weights(getter, model)
            if load_vision:
                _assign_vision_weights(getter, model)
            if region is not None:
                _assign_region_weights(getter, region, convert=convert)
    else:
        tensors_raw = torch.load(path, map_location=target_device, weights_only=True)
        tensors: dict[str, torch.Tensor] = {}
        for key, value in tensors_raw.items():
            name = key.replace("._orig_mod", "")
            if tensor_hook is not None:
                tensor_hook(name, value)
            tensors[name] = convert(value)

        def getter(name: str) -> torch.Tensor:
            return tensors[name]

        _assign_text_weights(getter, model)
        if load_vision:
            _assign_vision_weights(getter, model)
        if region is not None:
            _assign_region_weights(getter, region, convert=convert)

    _refresh_rotary_tables(model)


__all__ = ["load_moondream_weights", "load_text_weights"]
