"""Utilities to load Moondream text weights into the self-contained model.

Adapted from the Moondream project (Apache-2.0).
"""


from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import safetensors
import torch
import torch.nn as nn

from ..ops import precompute_freqs_cis
from .text import build_tau_pos_tables


@dataclass
class MoEScales:
    """Per-layer FP8 MoE scales captured during weight loading."""

    up_scales: List[Optional[torch.Tensor]]  # [n_layers] of [E, N] or None
    down_scales: List[Optional[torch.Tensor]]  # [n_layers] of [E, N] or None

    @classmethod
    def empty(cls, n_layers: int) -> "MoEScales":
        return cls(
            up_scales=[None] * n_layers,
            down_scales=[None] * n_layers,
        )

    def has_scales_for_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer has both up and down scales."""
        return (
            self.up_scales[layer_idx] is not None
            and self.down_scales[layer_idx] is not None
        )

    def has_any_scales(self) -> bool:
        """Check if any MoE scales were captured (indicates FP8 checkpoint)."""
        return any(s is not None for s in self.up_scales)

@contextmanager
def safetensors_open(path: str):
    with safetensors.safe_open(path, framework="pt") as f:
        def get_tensor(name: str) -> torch.Tensor:
            return f.get_tensor(name)

        def get_keys() -> List[str]:
            return f.keys()

        get_tensor.keys = get_keys  # type: ignore[attr-defined]
        yield get_tensor


def _assign_text_weights(
    get_tensor: Callable[[str], torch.Tensor],
    model: nn.Module,
    *,
    moe_scales: Optional[MoEScales] = None,
    get_raw_tensor: Optional[Callable[[str], torch.Tensor]] = None,
) -> None:
    text = model.text
    use_fp8_moe = moe_scales is not None and moe_scales.has_any_scales()

    weight_map: Dict[str, torch.Tensor] = {
        "text_model.transformer.embd.wte.weight": text.wte,
        "text_model.lm_head.ln.weight": text["post_ln"].weight,
        "text_model.lm_head.ln.bias": text["post_ln"].bias,
        "text_model.lm_head.linear.weight": text["lm_head"].weight,
        "text_model.lm_head.linear.bias": text["lm_head"].bias,
    }

    # Track MoE layers for FP8 handling
    moe_layers: List[tuple[int, nn.Module]] = []

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
            }
        )
        if is_moe:
            weight_map.update(
                {
                    f"{prefix}.gate.weight": block["mlp"]["router"].weight,
                    f"{prefix}.gate.bias": block["mlp"]["router"].bias,
                }
            )
            # Check if this specific layer has FP8 scales
            layer_has_fp8 = use_fp8_moe and moe_scales.has_scales_for_layer(i)
            if layer_has_fp8:
                # FP8 MoE weights handled separately below
                moe_layers.append((i, block))
            else:
                weight_map.update(
                    {
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

    # Handle FP8 MoE weights: load raw FP8 tensors and attach scales
    if use_fp8_moe and get_raw_tensor is not None:
        assert moe_scales is not None
        for layer_idx, block in moe_layers:
            prefix = f"text_model.transformer.h.{layer_idx}"
            fused_mlp = block["mlp"]["mlp"]

            # Load FP8 weights (stored as float8_e4m3fn, viewed as uint8)
            up_weight_fp8 = get_raw_tensor(f"{prefix}.mlp.experts.weight")
            down_weight_fp8 = get_raw_tensor(f"{prefix}.mlp.output_experts.weight")

            # Convert to uint8 view for storage
            up_weight_uint8 = up_weight_fp8.view(torch.uint8)
            down_weight_uint8 = down_weight_fp8.view(torch.uint8)

            # Replace weight data with uint8 FP8 bits
            # First resize the weight tensor to match uint8 storage
            up_experts = fused_mlp.up_experts
            down_experts = fused_mlp.down_experts

            # Create new uint8 weight parameter and copy FP8 bits
            device = up_experts.weight.device
            up_experts.weight = nn.Parameter(
                up_weight_uint8.to(device), requires_grad=False
            )
            down_experts.weight = nn.Parameter(
                down_weight_uint8.to(device), requires_grad=False
            )

            # Register scale buffers
            up_scale = moe_scales.up_scales[layer_idx]
            down_scale = moe_scales.down_scales[layer_idx]
            assert up_scale is not None and down_scale is not None
            up_experts.register_buffer("scale", up_scale.to(device))
            down_experts.register_buffer("scale", down_scale.to(device))

    # Tau weights (q/v scaling). Kestrel stores these fused as a single wqwv matrix
    # for performance, but older checkpoints store tau_wq and tau_wv separately.
    for i, block in enumerate(text["blocks"]):
        prefix = f"text_model.transformer.h.{i}"
        try:
            tau_wqwv = get_tensor(f"{prefix}.tau_wqwv")
        except KeyError:
            tau_wq = get_tensor(f"{prefix}.tau_wq")
            tau_wv = get_tensor(f"{prefix}.tau_wv")
            tau_wqwv = torch.cat([tau_wq, tau_wv], dim=0)
        block["attn"]["tau"]["wqwv"].data.copy_(tau_wqwv)
        block["attn"]["tau"]["alpha"].data.copy_(get_tensor(f"{prefix}.tau_alpha"))

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

    region["ln"].weight.data.copy_(convert(get_tensor("region_model.ln.weight")))
    region["ln"].bias.data.copy_(convert(get_tensor("region_model.ln.bias")))

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


def _refresh_tau_pos_tables(model: nn.Module) -> None:
    if not hasattr(model, "text") or not hasattr(model, "config"):
        return
    text_cfg = model.config.text
    target_param = next(model.text.parameters())
    build_tau_pos_tables(
        model.text,
        text_cfg.n_heads,
        text_cfg.max_context,
        dtype=target_param.dtype,
        device=target_param.device,
    )


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


def _capture_moe_scales(
    tensors_raw: Dict[str, torch.Tensor],
    n_layers: int,
) -> MoEScales:
    """Extract MoE FP8 scales from raw checkpoint tensors."""
    moe_scales = MoEScales.empty(n_layers)

    for name, tensor in tensors_raw.items():
        if not name.startswith("text_model.transformer.h."):
            continue
        parts = name.split(".")
        if len(parts) < 6:
            continue
        try:
            layer_idx = int(parts[3])
        except ValueError:
            continue
        if not (0 <= layer_idx < n_layers):
            continue
        if parts[4] != "moe_quant":
            continue
        target = parts[5]
        if target == "up_scale":
            moe_scales.up_scales[layer_idx] = tensor.detach().clone()
        elif target == "down_scale":
            moe_scales.down_scales[layer_idx] = tensor.detach().clone()

    return moe_scales


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

    # Determine number of layers for MoE scale capture
    n_layers = len(model.text.blocks)

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(target_dtype)

    if path.endswith(".safetensors"):
        with safetensors_open(path) as get_tensor:
            name_map = {k.replace("._orig_mod", ""): k for k in get_tensor.keys()}

            # First pass: collect all tensors to check for MoE scales
            all_tensors = {}
            for orig_name in get_tensor.keys():
                name = orig_name.replace("._orig_mod", "")
                all_tensors[name] = get_tensor(orig_name)

            # Capture MoE scales
            moe_scales = _capture_moe_scales(all_tensors, n_layers)

            # Call tensor_hook for all tensors
            if tensor_hook is not None:
                for name, tensor in all_tensors.items():
                    tensor_hook(name, tensor)

            def getter(name: str) -> torch.Tensor:
                return convert(all_tensors[name])

            def raw_getter(name: str) -> torch.Tensor:
                return all_tensors[name]

            _assign_text_weights(
                getter, model, moe_scales=moe_scales, get_raw_tensor=raw_getter
            )
            if load_vision:
                _assign_vision_weights(getter, model)
            if region is not None:
                _assign_region_weights(getter, region, convert=convert)
    else:
        tensors_raw = torch.load(path, map_location=target_device, weights_only=True)

        # Normalize names and capture MoE scales
        tensors_normalized: dict[str, torch.Tensor] = {}
        for key, value in tensors_raw.items():
            name = key.replace("._orig_mod", "")
            tensors_normalized[name] = value

        moe_scales = _capture_moe_scales(tensors_normalized, n_layers)

        # Call tensor_hook and prepare converted tensors
        tensors: dict[str, torch.Tensor] = {}
        for name, value in tensors_normalized.items():
            if tensor_hook is not None:
                tensor_hook(name, value)
            tensors[name] = convert(value)

        def getter(name: str) -> torch.Tensor:
            return tensors[name]

        def raw_getter(name: str) -> torch.Tensor:
            return tensors_normalized[name]

        _assign_text_weights(
            getter, model, moe_scales=moe_scales, get_raw_tensor=raw_getter
        )
        if load_vision:
            _assign_vision_weights(getter, model)
        if region is not None:
            _assign_region_weights(getter, region, convert=convert)

    _refresh_rotary_tables(model)
    _refresh_tau_pos_tables(model)


__all__ = ["load_moondream_weights", "load_text_weights", "MoEScales"]
