"""Image tiling helpers for the Moondream vision encoder."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def select_tiling(height: int, width: int, crop_size: int, max_crops: int) -> Tuple[int, int]:
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))

def reconstruct_from_crops(
    crops: torch.Tensor | np.ndarray,
    tiling: Tuple[int, int],
    *,
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor | np.ndarray:
    """Stitch tiled crops back into a full feature map."""

    tiling_h, tiling_w = tiling
    is_numpy = isinstance(crops, np.ndarray)
    if is_numpy:
        crop_tensor = torch.from_numpy(crops)
    else:
        crop_tensor = crops

    crop_height, crop_width = crop_tensor.shape[1:3]
    margin_pixels = overlap_margin * patch_size
    window_h = crop_height - 2 * margin_pixels
    window_w = crop_width - 2 * margin_pixels

    output_h = window_h * tiling_h + 2 * margin_pixels
    output_w = window_w * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crop_tensor.shape[-1]),
        device=crop_tensor.device,
        dtype=crop_tensor.dtype,
    )

    for tile_y in range(tiling_h):
        for tile_x in range(tiling_w):
            idx = tile_y * tiling_w + tile_x
            crop = crop_tensor[idx]

            y_start = 0 if tile_y == 0 else margin_pixels
            y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels
            x_start = 0 if tile_x == 0 else margin_pixels
            x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels

            dest_y0 = tile_y * window_h + (0 if tile_y == 0 else margin_pixels)
            dest_y1 = dest_y0 + (y_end - y_start)
            dest_x0 = tile_x * window_w + (0 if tile_x == 0 else margin_pixels)
            dest_x1 = dest_x0 + (x_end - x_start)

            reconstructed[dest_y0:dest_y1, dest_x0:dest_x1] = crop[ y_start:y_end, x_start:x_end ]

    if is_numpy:
        return reconstructed.cpu().numpy()
    return reconstructed


__all__ = [
    "select_tiling",
    "reconstruct_from_crops",
]
