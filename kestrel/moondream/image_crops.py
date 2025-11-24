"""Image tiling helpers for the Moondream vision encoder."""


import math
from typing import Tuple, TypedDict

import numpy as np
import torch
import pyvips
from PIL import Image

from kestrel.utils.image import _vips_to_uint8_numpy


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


class OverlapCropOutput(TypedDict):
    crops: np.ndarray
    tiling: Tuple[int, int]


def overlap_crop_image(
    image: pyvips.Image | np.ndarray,
    *,
    overlap_margin: int,
    max_crops: int,
    base_size: Tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> OverlapCropOutput:
    """Create a global crop plus overlapping local crops with consistent margins."""

    if isinstance(image, np.ndarray):
        original_h, original_w = image.shape[:2]
        num_bands = image.shape[2] if image.ndim == 3 else 1
    else:
        original_h, original_w = image.height, image.width
        num_bands = image.bands

    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2

    crop_patches = base_size[0] // patch_size
    crop_window_patches = crop_patches - (2 * overlap_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling = select_tiling(
        max(1, original_h - total_margin_pixels),
        max(1, original_w - total_margin_pixels),
        crop_window_size,
        max_crops,
    )

    num_crops = tiling[0] * tiling[1] + 1
    crops = np.zeros((num_crops, base_size[0], base_size[1], num_bands), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    if isinstance(image, np.ndarray):
        original_numpy = image
    else:
        original_numpy = _vips_to_uint8_numpy(image)

    pil_image = Image.fromarray(original_numpy)

    scale_x = target_size[1] / original_w
    scale_y = target_size[0] / original_h
    target_width = int(round(original_w * scale_x))
    target_height = int(round(original_h * scale_y))
    resized_pil = pil_image.resize((target_width, target_height), Image.LANCZOS)
    resized_numpy = np.array(resized_pil, dtype=np.uint8)

    scale_x_global = base_size[1] / original_w
    scale_y_global = base_size[0] / original_h
    global_width = int(round(original_w * scale_x_global))
    global_height = int(round(original_h * scale_y_global))
    global_pil = pil_image.resize((global_width, global_height), Image.LANCZOS)
    crops[0] = np.array(global_pil, dtype=np.uint8)

    for tile_y in range(tiling[0]):
        for tile_x in range(tiling[1]):
            y0 = tile_y * crop_window_size
            x0 = tile_x * crop_window_size

            y1 = min(y0 + base_size[0], resized_numpy.shape[0])
            x1 = min(x0 + base_size[1], resized_numpy.shape[1])

            idx = 1 + tile_y * tiling[1] + tile_x
            width = max(0, x1 - x0)
            height = max(0, y1 - y0)
            if width == 0 or height == 0:
                continue
            crop_region = resized_numpy[y0:y1, x0:x1]
            crops[idx, :height, :width] = crop_region

    return {"crops": crops, "tiling": tiling}


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

            reconstructed[dest_y0:dest_y1, dest_x0:dest_x1] = crop[y_start:y_end, x_start:x_end]

    if is_numpy:
        return reconstructed.cpu().numpy()
    return reconstructed


__all__ = [
    "select_tiling",
    "overlap_crop_image",
    "reconstruct_from_crops",
]
