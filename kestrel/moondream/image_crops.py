"""Image tiling helpers for the Moondream vision encoder."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


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


def overlap_crop_image(
    image: np.ndarray,
    *,
    overlap_margin: int,
    max_crops: int,
    base_size: Tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    original_h, original_w = image.shape[:2]

    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2

    crop_patches = base_size[0] // patch_size
    crop_window_patches = crop_patches - (2 * overlap_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    n_crops = tiling[0] * tiling[1] + 1
    crops = np.zeros((n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    image = _resize_image(image, (int(target_size[0]), int(target_size[1])))

    crops[0] = _resize_image(image, (int(base_size[0]), int(base_size[1])))

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            y0 = i * crop_window_size
            x0 = j * crop_window_size
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])
            crop_region = image[y0:y_end, x0:x_end]
            crops[1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]] = crop_region

    return crops, tiling


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a uint8 image to ``size`` (height, width) using bilinear filtering."""

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C); received {image.shape}")

    target_h, target_w = size
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image.copy()

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
    )
    resized = F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(dtype=torch.uint8)
    return resized.cpu().numpy()


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
    "overlap_crop_image",
    "reconstruct_from_crops",
]
