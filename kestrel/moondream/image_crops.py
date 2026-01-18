"""Image tiling helpers for the Moondream vision encoder."""

import ctypes
import ctypes.util
import math
from typing import Tuple, TypedDict

import numpy as np
import pyvips
import torch

# Set VIPS concurrency to a low value to avoid internal thread contention
# when using multiple Python threads. With low concurrency, each vips operation
# uses fewer internal threads, allowing Python-level parallelism to scale better.
# Must use C API since env var is only read at vips initialization.
_vips_lib = ctypes.CDLL(ctypes.util.find_library("vips"))
_vips_lib.vips_concurrency_set.argtypes = [ctypes.c_int]
_vips_lib.vips_concurrency_set(2)

from kestrel.utils.image import _vips_to_uint8_numpy


def _ensure_vips(image: pyvips.Image | np.ndarray) -> pyvips.Image:
    """Convert numpy array to pyvips Image if needed.

    MEMORY SAFETY: This function uses memoryview to avoid copying data. The returned
    VipsImage holds a reference to the numpy array's memory. Callers must ensure the
    input array remains alive and unmodified until all vips operations (including
    lazy operations like resize) are fully evaluated (e.g., via write_to_memory).
    """
    if isinstance(image, pyvips.Image):
        return image
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    bands = image.shape[2] if image.ndim == 3 else 1
    return pyvips.Image.new_from_memory(memoryview(image), w, h, bands, "uchar")


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
    # Use empty instead of zeros - all regions are fully written by the code below.
    crops = np.empty((num_crops, base_size[0], base_size[1], num_bands), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    vips_image = _ensure_vips(image)

    # Resize for tiled crops
    scale_x = target_size[1] / original_w
    scale_y = target_size[0] / original_h
    resized_vips = vips_image.resize(scale_x, vscale=scale_y, kernel="lanczos3")
    resized_numpy = _vips_to_uint8_numpy(resized_vips)

    # Resize for global crop
    scale_x_global = base_size[1] / original_w
    scale_y_global = base_size[0] / original_h
    global_vips = vips_image.resize(scale_x_global, vscale=scale_y_global, kernel="lanczos3")
    crops[0] = _vips_to_uint8_numpy(global_vips)

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
