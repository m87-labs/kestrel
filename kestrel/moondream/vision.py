"""Vision encoder components ported from the Moondream reference implementation."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvips

from .config import VisionConfig
from .image_crops import reconstruct_from_crops, select_tiling
from kestrel.utils import log_gpu_memory
from kestrel.utils.image import ImageArray, as_uint8_image_array


def prepare_crops(
    image: Union[pyvips.Image, ImageArray],
    config: VisionConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if device.type != "cuda":
        raise RuntimeError("Vision preprocessing expects a CUDA device")

    np_image = as_uint8_image_array(image)
    margin = config.overlap_margin * config.enc_patch_size
    window = max(config.crop_size - 2 * margin, 1)

    height, width = np_image.shape[:2]
    h_tiles, w_tiles = select_tiling(
        max(1, height - 2 * margin),
        max(1, width - 2 * margin),
        window,
        config.max_crops,
    )

    cpu_tensor = torch.from_numpy(np_image).to(torch.uint8).pin_memory()
    img = cpu_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True) / 255.0

    crops = _bilinear_overlap_crops(
        img,
        crop_size=config.crop_size,
        window=window,
        margin=margin,
        tiling=(h_tiles, w_tiles),
    )
    crops = crops.sub_(0.5).div_(0.5)
    crops = crops.to(dtype=dtype)
    return crops, (h_tiles, w_tiles)


def _bilinear_overlap_crops(
    img: torch.Tensor,
    *,
    crop_size: int,
    window: int,
    margin: int,
    tiling: Tuple[int, int],
) -> torch.Tensor:
    """Generate overlapping crops using bilinear GPU resizing."""

    if img.ndim != 4:
        raise ValueError(f"Expected image tensor of shape (1, C, H, W); received {img.shape}")

    device = img.device
    _, channels, height, width = img.shape

    h_tiles, w_tiles = tiling
    num_crops = h_tiles * w_tiles + 1
    crops = torch.zeros(
        (num_crops, channels, crop_size, crop_size),
        device=device,
        dtype=img.dtype,
    )

    global_crop = torch.nn.functional.interpolate(
        img, size=(crop_size, crop_size), mode="bilinear", align_corners=False
    )
    crops[0] = global_crop[0]

    target_h = h_tiles * window + 2 * margin
    target_w = w_tiles * window + 2 * margin
    target_h = max(target_h, crop_size)
    target_w = max(target_w, crop_size)

    resized = torch.nn.functional.interpolate(
        img,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    idx = 1
    for tile_y in range(h_tiles):
        for tile_x in range(w_tiles):
            y0 = tile_y * window
            x0 = tile_x * window
            y1 = min(y0 + crop_size, target_h)
            x1 = min(x0 + crop_size, target_w)

            crop = crops[idx]
            crop_slice = resized[0, :, y0:y1, x0:x1]
            crop[:, : y1 - y0, : x1 - x0].copy_(crop_slice)
            idx += 1

    return crops


def create_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    bsz, channels, height, width = x.shape
    p1 = p2 = patch_size
    x = x.reshape(bsz, channels, height // p1, p1, width // p2, p2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(bsz, (height // p1) * (width // p2), channels * p1 * p2)
    return x


def vision_encoder(crops: torch.Tensor, module: nn.Module, config: VisionConfig) -> torch.Tensor:
    x = create_patches(crops, config.enc_patch_size)
    x = module.patch_emb(x)
    x = x + module.pos_emb
    for block in module.blocks:
        x_norm = F.layer_norm(x, block.ln1.normalized_shape, block.ln1.weight, block.ln1.bias)
        x = x + _vision_attn(x_norm, block.attn, config.enc_n_heads)
        x_norm = F.layer_norm(x, block.ln2.normalized_shape, block.ln2.weight, block.ln2.bias)
        x = x + _vision_mlp(x_norm, block.mlp)
    x = F.layer_norm(x, module.post_ln.normalized_shape, module.post_ln.weight, module.post_ln.bias)
    return x


def _vision_attn(x: torch.Tensor, attn: nn.ModuleDict, n_heads: int) -> torch.Tensor:
    qkv = attn["qkv"](x)
    dim = x.shape[-1]
    head_dim = dim // n_heads
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(x.size(0), -1, n_heads, head_dim).transpose(1, 2)
    k = k.view(x.size(0), -1, n_heads, head_dim).transpose(1, 2)
    v = v.view(x.size(0), -1, n_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).contiguous().view(x.size(0), -1, dim)
    return attn["proj"](out)


def _vision_mlp(x: torch.Tensor, mlp: nn.ModuleDict) -> torch.Tensor:
    x = F.gelu(mlp["fc1"](x), approximate="tanh")
    return mlp["fc2"](x)


def vision_projection(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    module: nn.Module,
    config: VisionConfig,
) -> torch.Tensor:
    dtype = global_features.dtype
    reconstructed = local_features.to(dtype=dtype).permute(2, 0, 1)
    reconstructed = F.adaptive_avg_pool2d(
        reconstructed.to(dtype=torch.float32),
        output_size=(config.enc_n_layers, config.enc_n_layers),
    ).to(dtype)
    reconstructed = reconstructed.permute(1, 2, 0).reshape(-1, config.enc_dim)
    features = torch.cat([global_features, reconstructed], dim=-1)
    return _vision_mlp(features, module.proj_mlp)


def build_vision_model(config: VisionConfig, dtype: torch.dtype) -> nn.Module:
    patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
    grid_size = config.crop_size // config.enc_patch_size
    num_patches = grid_size * grid_size

    model = nn.ModuleDict(
        {
            "patch_emb": nn.Linear(patch_dim, config.enc_dim, dtype=dtype),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(config.enc_dim, 3 * config.enc_dim, dtype=dtype),
                                    "proj": nn.Linear(config.enc_dim, config.enc_dim, dtype=dtype),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(config.enc_dim, config.enc_ff_dim, dtype=dtype),
                                    "fc2": nn.Linear(config.enc_ff_dim, config.enc_dim, dtype=dtype),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype),
            "proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(config.enc_dim * 2, config.proj_inner_dim, dtype=dtype),
                    "fc2": nn.Linear(config.proj_inner_dim, config.proj_out_dim, dtype=dtype),
                }
            ),
        }
    )
    model.pos_emb = nn.Parameter(torch.zeros(1, num_patches, config.enc_dim, dtype=dtype))
    return model


def encode_image(
    image: Union[pyvips.Image, ImageArray],
    module: nn.Module,
    config: VisionConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.inference_mode():
        crops, tiling = prepare_crops(image, config, device, dtype)
        torch._dynamo.mark_dynamic(crops, 0)
        log_gpu_memory("vision:after_prepare_crops", device)
        outputs = vision_encoder(crops, module, config)
        log_gpu_memory("vision:after_encoder", device)
        global_features = outputs[0]
        local = outputs[1:].reshape(
            -1,
            config.enc_n_layers,
            config.enc_n_layers,
            config.enc_dim,
        )
        reconstructed = reconstruct_from_crops(
            local.to(dtype=torch.float32),
            tiling,
            overlap_margin=config.overlap_margin,
            patch_size=1,
        )
        reconstructed = reconstructed.to(device=device, dtype=outputs.dtype)
        projected = vision_projection(global_features, reconstructed, module, config)
        log_gpu_memory("vision:after_projection", device)
    return projected


__all__ = [
    "prepare_crops",
    "create_patches",
    "vision_encoder",
    "vision_projection",
    "build_vision_model",
    "encode_image",
]
