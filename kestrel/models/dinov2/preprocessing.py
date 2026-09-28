"""DINOv2 image preprocessing without a Transformers runtime dependency.

Resize/crop and numeric behavior follow the PIL backend of Hugging Face's
``BitImageProcessor`` (Copyright 2022 The HuggingFace Inc. team, Apache-2.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch

from .config import Dinov2ProcessorConfig


def _numpy_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("NumPy images must have rank 2 or 3")

    first_is_channel = image.shape[0] in (1, 3, 4)
    last_is_channel = image.shape[-1] in (1, 3, 4)
    if first_is_channel and not last_is_channel:
        image = np.moveaxis(image, 0, -1)
    elif not last_is_channel:
        raise ValueError("NumPy images must be HWC or CHW with 1, 3, or 4 channels")
    elif first_is_channel and image.shape[1] in (1, 3, 4):
        raise ValueError("ambiguous NumPy channel dimension")

    channels = image.shape[-1]
    if channels == 1:
        image = np.repeat(image, 3, axis=-1)
    elif channels == 4:
        image = image[..., :3]
    elif channels != 3:
        raise ValueError(f"NumPy image has {channels} channels; expected 1, 3, or 4")
    return np.ascontiguousarray(image)


def _to_pil_rgb(image: Image.Image | np.ndarray) -> tuple[Image.Image, bool]:
    """Return an RGB PIL image and whether float input was internally scaled."""

    if isinstance(image, Image.Image):
        return image.convert("RGB"), False
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a PIL image or NumPy array")

    image = _numpy_rgb(image)
    if image.dtype == np.bool_:
        raise TypeError("boolean images are not supported")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"unsupported NumPy image dtype {image.dtype}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image values must be finite")

    internally_scaled = False
    if image.dtype == np.uint8:
        converted = image
    elif np.allclose(image, image.astype(np.int64)):
        if image.min() < 0 or image.max() > 255:
            raise ValueError("integer-valued image data must lie in [0, 255]")
        converted = image.astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating) and image.min() >= 0 and image.max() <= 1:
        converted = (image.astype(np.float64) * 255.0).astype(np.uint8)
        internally_scaled = True
    else:
        raise ValueError(
            "fractional image data must lie in [0, 1]; integer-valued data must lie "
            "in [0, 255]"
        )
    return Image.fromarray(converted, mode="RGB"), internally_scaled


def _resize_shortest_edge(image: Image.Image, shortest_edge: int, resample: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width <= height:
        new_width = shortest_edge
        new_height = int(shortest_edge * height / width)
    else:
        new_height = shortest_edge
        new_width = int(shortest_edge * width / height)
    return image.resize((new_width, new_height), resample=Image.Resampling(resample))


def _center_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    pad_height = max(height - image_height, 0)
    pad_width = max(width - image_width, 0)
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = (pad_height + 1) // 2
        left = pad_width // 2
        right = (pad_width + 1) // 2
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)))
        image_height, image_width = image.shape[:2]
    top = int((image_height - height) / 2.0)
    left = int((image_width - width) / 2.0)
    return image[top : top + height, left : left + width]


class Dinov2ImageProcessor:
    """Batch-one processor producing contiguous FP32 ``[1, 3, H, W]`` tensors."""

    def __init__(self, config: Dinov2ProcessorConfig) -> None:
        config.validate_v1()
        self.config = config

    def __call__(self, image: Any) -> torch.Tensor:
        pil_image, internally_scaled = _to_pil_rgb(image)
        resized = _resize_shortest_edge(
            pil_image,
            self.config.shortest_edge,
            self.config.resample,
        )
        pixels = _center_crop(
            np.asarray(resized),
            self.config.crop_height,
            self.config.crop_width,
        )

        if internally_scaled:
            pixels = (pixels.astype(np.float64) / 255.0).astype(np.float32)
        pixels = (pixels.astype(np.float64) * self.config.rescale_factor).astype(np.float32)
        mean = np.asarray(self.config.image_mean, dtype=np.float32)
        std = np.asarray(self.config.image_std, dtype=np.float32)
        pixels = ((pixels - mean) / std).astype(np.float32)
        chw = np.ascontiguousarray(np.moveaxis(pixels, -1, 0))
        return torch.from_numpy(chw).unsqueeze(0)


__all__ = ["Dinov2ImageProcessor"]
