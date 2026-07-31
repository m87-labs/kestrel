"""Gemma 4 image preprocessing."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from kestrel.utils.image import decode_to_srgb
from PIL import Image


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
    aspect = width / height
    ideal_w = math.sqrt(MAX_PATCHES * aspect)
    ideal_h = math.sqrt(MAX_PATCHES / aspect)
    grid_w = max(
        POOLING_KERNEL_SIZE,
        int(ideal_w // POOLING_KERNEL_SIZE) * POOLING_KERNEL_SIZE,
    )
    grid_h = max(
        POOLING_KERNEL_SIZE,
        int(ideal_h // POOLING_KERNEL_SIZE) * POOLING_KERNEL_SIZE,
    )
    return grid_h, grid_w


def preprocess_image(
    image: Any,
) -> GemmaImageInputs:
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if np.issubdtype(image.dtype, np.floating) and image.size:
        finite = image[np.isfinite(image)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            image = image * 255.0
    array = decode_to_srgb(image)
    grid_h, grid_w = _pick_grid(*array.shape[:2])
    resized_h = grid_h * PATCH_SIZE
    resized_w = grid_w * PATCH_SIZE

    num_valid = grid_h * grid_w
    if not array.flags.writeable:
        array = array.copy()
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if tuple(pixels.shape[-2:]) != (resized_h, resized_w):
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    pixels = pixels[0].permute(1, 2, 0)
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

    output_patches = max(MAX_PATCHES, num_valid)
    pixel_values = torch.full(
        (
            output_patches,
            3 * PATCH_SIZE * PATCH_SIZE,
        ),
        0,
        dtype=torch.bfloat16,
    )
    valid_pixels = patches.to(torch.float32).mul_(1.0 / 255.0)
    # Gemma 4's processor config rescales to [0, 1] with do_normalize=false.
    pixel_values[:num_valid].copy_(valid_pixels)

    image_position_ids = torch.full(
        (output_patches, 2),
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
