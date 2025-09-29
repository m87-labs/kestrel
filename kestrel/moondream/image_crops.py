"""Image tiling helpers for the Moondream vision encoder."""

from __future__ import annotations

import math
from typing import Tuple, TypedDict

import numpy as np
import torch

try:
    import pyvips  # type: ignore[attr-defined]

    HAS_VIPS = True
except Exception:  # pragma: no cover - optional dependency
    from PIL import Image

    HAS_VIPS = False


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
    image: np.ndarray,
    *,
    overlap_margin: int,
    max_crops: int,
    base_size: Tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> OverlapCropOutput:
    """Create a global crop plus overlapping local crops with consistent margins."""

    original_h, original_w = image.shape[:2]

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
    crops = np.zeros((num_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    if HAS_VIPS:
        vips_image = pyvips.Image.new_from_array(image)
        scale_x = target_size[1] / original_w
        scale_y = target_size[0] / original_h
        resized = vips_image.resize(scale_x, vscale=scale_y)
        image = resized.numpy()

        scale_x = base_size[1] / original_w
        scale_y = base_size[0] / original_h
        global_vips = vips_image.resize(scale_x, vscale=scale_y)
        crops[0] = global_vips.numpy()
    else:
        pil_img = Image.fromarray(image)
        resized = pil_img.resize(
            (int(target_size[1]), int(target_size[0])),
            resample=Image.Resampling.LANCZOS,
        )
        image = np.asarray(resized)

        global_pil = pil_img.resize(
            (int(base_size[1]), int(base_size[0])),
            resample=Image.Resampling.LANCZOS,
        )
        crops[0] = np.asarray(global_pil)

    for tile_y in range(tiling[0]):
        for tile_x in range(tiling[1]):
            y0 = tile_y * crop_window_size
            x0 = tile_x * crop_window_size

            y1 = min(y0 + base_size[0], image.shape[0])
            x1 = min(x0 + base_size[1], image.shape[1])

            idx = 1 + tile_y * tiling[1] + tile_x
            crop_region = image[y0:y1, x0:x1]
            crops[idx, : crop_region.shape[0], : crop_region.shape[1]] = crop_region

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
