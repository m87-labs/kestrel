"""Local Qwen 3.5 image preprocessing."""

import math
from dataclasses import dataclass
from typing import Any

import kestrel_native
import numpy as np
import torch
from kestrel.utils.image import decode_to_srgb


@dataclass(frozen=True)
class QwenImageProcessorConfig:
    shortest_edge: int = 65536
    longest_edge: int = 16777216
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(
    image: Any,
    config: QwenImageProcessorConfig = QwenImageProcessorConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = np.ascontiguousarray(decode_to_srgb(image))
    height, width = rgb.shape[:2]
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=config.patch_size * config.merge_size,
        min_pixels=config.shortest_edge,
        max_pixels=config.longest_edge,
    )
    resized = kestrel_native.resize_bicubic(rgb, resized_h, resized_w)
    array = resized.astype(np.float32)
    array *= 1.0 / 255.0
    np.subtract(
        array,
        np.asarray(config.image_mean, dtype=np.float32),
        out=array,
    )
    np.divide(
        array,
        np.asarray(config.image_std, dtype=np.float32),
        out=array,
    )
    channels = int(array.shape[2])
    grid_h = resized_h // config.patch_size
    grid_w = resized_w // config.patch_size
    patches = torch.from_numpy(array).reshape(
        grid_h // config.merge_size,
        config.merge_size,
        config.patch_size,
        grid_w // config.merge_size,
        config.merge_size,
        config.patch_size,
        channels,
    ).permute(0, 3, 1, 4, 6, 2, 5)
    pixel_values = (
        patches.unsqueeze(5)
        .expand(
            -1,
            -1,
            -1,
            -1,
            -1,
            config.temporal_patch_size,
            -1,
            -1,
        )
        .reshape(
            grid_h * grid_w,
            channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size,
        )
    )
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return pixel_values, image_grid_thw


__all__ = [
    "QwenImageProcessorConfig",
    "preprocess_image",
    "smart_resize",
]
