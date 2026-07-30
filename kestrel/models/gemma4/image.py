"""Gemma 4 image preprocessing."""

from __future__ import annotations

import io
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Gemma4ImageProcessorConfig:
    max_patches: int = 2520
    patch_size: int = 16
    pooling_kernel_size: int = 3


@dataclass(frozen=True)
class GemmaImageInputs:
    pixel_values: torch.Tensor
    image_position_ids: torch.Tensor
    num_image_tokens: int


DEFAULT_IMAGE_CONFIG = Gemma4ImageProcessorConfig()
MAX_PATCHES = DEFAULT_IMAGE_CONFIG.max_patches
MAX_IMAGE_TOKENS = MAX_PATCHES // (DEFAULT_IMAGE_CONFIG.pooling_kernel_size**2)
IMAGE_SEQ_LENGTH = MAX_IMAGE_TOKENS


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                finite = array[np.isfinite(array)]
                if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported Gemma 4 image input type: {type(image)!r}")


def _pick_grid(
    height: int,
    width: int,
    config: Gemma4ImageProcessorConfig,
) -> tuple[int, int]:
    aspect = width / height
    ideal_w = math.sqrt(config.max_patches * aspect)
    ideal_h = math.sqrt(config.max_patches / aspect)
    grid_w = max(
        config.pooling_kernel_size,
        int(ideal_w // config.pooling_kernel_size) * config.pooling_kernel_size,
    )
    grid_h = max(
        config.pooling_kernel_size,
        int(ideal_h // config.pooling_kernel_size) * config.pooling_kernel_size,
    )
    return grid_h, grid_w


def preprocess_image(
    image: Any,
    config: Gemma4ImageProcessorConfig = DEFAULT_IMAGE_CONFIG,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaImageInputs:
    pil = _to_pil_rgb(image)
    grid_h, grid_w = _pick_grid(pil.height, pil.width, config)
    resized_h = grid_h * config.patch_size
    resized_w = grid_w * config.patch_size

    num_valid = grid_h * grid_w
    array = np.asarray(pil, dtype=np.uint8)
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
            config.patch_size,
            grid_w,
            config.patch_size,
            3,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(
            num_valid,
            3 * config.patch_size * config.patch_size,
        )
    )

    output_patches = max(config.max_patches, num_valid)
    pixel_values = torch.full(
        (
            output_patches,
            3 * config.patch_size * config.patch_size,
        ),
        0,
        dtype=dtype,
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
        num_image_tokens=num_valid // (config.pooling_kernel_size**2),
    )


def preprocess(image: Any) -> GemmaImageInputs:
    return preprocess_image(image)


class Gemma4ImagePreprocessor:
    def __init__(
        self,
        *,
        num_workers: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._dtype = dtype
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="kestrel-gemma4-img",
        )

    def submit(self, image: Any) -> Future[GemmaImageInputs]:
        return self._executor.submit(
            preprocess_image,
            image,
            dtype=self._dtype,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


__all__ = [
    "Gemma4ImageProcessorConfig",
    "Gemma4ImagePreprocessor",
    "GemmaImageInputs",
    "IMAGE_SEQ_LENGTH",
    "MAX_IMAGE_TOKENS",
    "MAX_PATCHES",
    "preprocess",
    "preprocess_image",
]
