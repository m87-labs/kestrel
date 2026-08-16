"""Gemma 4 image preprocessing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kestrel.utils.image import decode_to_srgb


@dataclass(frozen=True)
class GemmaImageInputs:
    pixel_values: torch.Tensor
    image_position_ids: torch.Tensor
    num_image_tokens: int


MAX_PATCHES = 2520
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MAX_IMAGE_TOKENS = MAX_PATCHES // POOLING_KERNEL_SIZE**2


def _pick_grid(
    height: int,
    width: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    aspect = width / height
    ideal_w = math.sqrt(MAX_PATCHES * aspect)
    ideal_h = math.sqrt(MAX_PATCHES / aspect)
    grid_w = int(ideal_w // POOLING_KERNEL_SIZE) * POOLING_KERNEL_SIZE
    grid_h = int(ideal_h // POOLING_KERNEL_SIZE) * POOLING_KERNEL_SIZE
    max_side = MAX_IMAGE_TOKENS * POOLING_KERNEL_SIZE
    if grid_h == 0 and grid_w == 0:
        raise ValueError("image aspect ratio cannot fit the patch budget")
    if grid_h == 0:
        grid_h = POOLING_KERNEL_SIZE
        grid_w = min((width // height) * POOLING_KERNEL_SIZE, max_side)
    elif grid_w == 0:
        grid_w = POOLING_KERNEL_SIZE
        grid_h = min((height // width) * POOLING_KERNEL_SIZE, max_side)
    if grid_h * grid_w > MAX_PATCHES:
        raise ValueError("image grid exceeds the patch budget")
    return grid_h, grid_w


def preprocess_image(
    image: Any,
) -> GemmaImageInputs:
    array = np.asarray(image) if not isinstance(image, (bytes, bytearray)) else image
    if (
        not isinstance(array, (bytes, bytearray))
        and np.issubdtype(array.dtype, np.floating)
        and array.size
    ):
        finite = array[np.isfinite(array)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            array = array * 255.0
    array = np.ascontiguousarray(decode_to_srgb(array))
    grid_h, grid_w = _pick_grid(*array.shape[:2])
    resized_h = grid_h * PATCH_SIZE
    resized_w = grid_w * PATCH_SIZE

    num_valid = grid_h * grid_w
    pixels = torch.from_numpy(array).permute(2, 0, 1)
    if array.shape[:2] != (resized_h, resized_w):
        # Match Gemma4ImageProcessor's uint8 antialiased bicubic boundary.
        # The generic native resize has different edge and antialias semantics,
        # which changes the vision tokens before model execution.
        pixels = F.interpolate(
            pixels.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    pixels = pixels.permute(1, 2, 0).contiguous()
    patches = (
        pixels.reshape(
            grid_h,
            PATCH_SIZE,
            grid_w,
            PATCH_SIZE,
            3,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(
            num_valid,
            3 * PATCH_SIZE * PATCH_SIZE,
        )
    )

    pixel_values = torch.zeros(
        (
            MAX_PATCHES,
            3 * PATCH_SIZE * PATCH_SIZE,
        ),
        dtype=torch.bfloat16,
    )
    valid_pixels = patches.to(torch.float32).mul_(1.0 / 255.0)
    # Gemma 4's processor config rescales to [0, 1] with do_normalize=false.
    pixel_values[:num_valid].copy_(valid_pixels)

    image_position_ids = torch.full(
        (MAX_PATCHES, 2),
        -1,
        dtype=torch.long,
    )
    image_position_ids[:num_valid, 0] = torch.arange(grid_w).repeat(grid_h)
    image_position_ids[:num_valid, 1] = torch.arange(grid_h).repeat_interleave(grid_w)

    return GemmaImageInputs(
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        num_image_tokens=num_valid // POOLING_KERNEL_SIZE**2,
    )


__all__ = [
    "GemmaImageInputs",
    "MAX_IMAGE_TOKENS",
    "MAX_PATCHES",
    "preprocess_image",
]
