"""Local Qwen 3.5 image preprocessing."""

import math
from typing import Any

import kestrel_native
import numpy as np
import torch
from kestrel.utils.image import decode_to_srgb


_MIN_PIXELS = 256 * 256
_MAX_PIXELS = 4096 * 4096
_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2
_RESIZE_FACTOR = _PATCH_SIZE * _MERGE_SIZE


def _smart_resize(height: int, width: int) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / _RESIZE_FACTOR) * _RESIZE_FACTOR
    w_bar = round(width / _RESIZE_FACTOR) * _RESIZE_FACTOR
    if h_bar * w_bar > _MAX_PIXELS:
        beta = math.sqrt((height * width) / _MAX_PIXELS)
        h_bar = max(
            _RESIZE_FACTOR,
            math.floor(height / beta / _RESIZE_FACTOR) * _RESIZE_FACTOR,
        )
        w_bar = max(
            _RESIZE_FACTOR,
            math.floor(width / beta / _RESIZE_FACTOR) * _RESIZE_FACTOR,
        )
    elif h_bar * w_bar < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / _RESIZE_FACTOR) * _RESIZE_FACTOR
        w_bar = math.ceil(width * beta / _RESIZE_FACTOR) * _RESIZE_FACTOR
    return h_bar, w_bar


def preprocess_image(image: Any) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = np.ascontiguousarray(decode_to_srgb(image))
    height, width = rgb.shape[:2]
    resized_h, resized_w = _smart_resize(height, width)
    resized = kestrel_native.resize_bicubic(rgb, resized_h, resized_w)
    array = resized.astype(np.float32)
    array *= 2.0 / 255.0
    array -= 1.0
    channels = int(array.shape[2])
    grid_h = resized_h // _PATCH_SIZE
    grid_w = resized_w // _PATCH_SIZE
    patches = torch.from_numpy(array).reshape(
        grid_h // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
        grid_w // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
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
            _TEMPORAL_PATCH_SIZE,
            -1,
            -1,
        )
        .reshape(
            grid_h * grid_w,
            channels
            * _TEMPORAL_PATCH_SIZE
            * _PATCH_SIZE
            * _PATCH_SIZE,
        )
    )
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return pixel_values, image_grid_thw


__all__ = ["preprocess_image"]
