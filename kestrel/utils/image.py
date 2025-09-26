"""Utilities for working with pyvips images within Kestrel."""

from __future__ import annotations

import base64
import binascii
from typing import Optional, Union

import numpy as np
import pyvips

ImageArray = np.ndarray


def load_vips_from_base64(data: str) -> pyvips.Image:
    """Decode a base64 string (raw or ``data:image/...``) into a pyvips image."""

    if data.startswith("data:image"):
        header, _, payload = data.partition(",")
        if not payload:
            raise ValueError("Invalid data URL: missing payload")
        raw = _b64decode(payload)
    else:
        raw = _b64decode(data)

    try:
        image = pyvips.Image.new_from_buffer(raw, "", access="sequential")
    except pyvips.Error as exc:
        raise ValueError("Invalid image payload") from exc

    return ensure_srgb(image)


def ensure_srgb(image: pyvips.Image) -> pyvips.Image:
    """Return an image in 8-bit sRGB without alpha."""

    if image.format != "uchar":
        image = image.cast("uchar")

    if image.hasalpha():
        # Remove alpha channel by premultiplying over an opaque black background
        image = image.flatten(background=[0, 0, 0])

    if image.bands == 1:
        image = image.colourspace("srgb")
    elif image.bands > 3:
        image = image.extract_band(0, n=3)

    try:
        if image.interpretation not in ("srgb", "rgb", "rgb16"):
            image = image.colourspace("srgb")
    except pyvips.Error:
        # Some formats (e.g. already linear) may not support conversion; fall back.
        pass

    if image.bands == 1:
        image = image.bandjoin([image, image])
    elif image.bands > 3:
        image = image.extract_band(0, n=3)
    return image


def vips_to_uint8_numpy(image: pyvips.Image) -> ImageArray:
    """Convert a pyvips image into a HxWxC uint8 NumPy array."""

    memory = image.write_to_memory()
    array = np.frombuffer(memory, dtype=np.uint8)
    height, width, bands = image.height, image.width, image.bands
    array = array.reshape(height, width, bands)
    # The buffer returned by pyvips is read-only; take a contiguous copy so we can
    # hand it to consumers that expect writable memory (e.g., torch.from_numpy).
    return np.ascontiguousarray(array)


def as_uint8_image_array(image: Union[pyvips.Image, ImageArray]) -> ImageArray:
    """Normalize image inputs to a contiguous uint8 HxWxC array."""

    if isinstance(image, pyvips.Image):
        return vips_to_uint8_numpy(image)

    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError(
                f"Vision image must have shape (H, W, C); received ndim={image.ndim}"
            )
        if image.shape[2] not in (3, 4):
            raise ValueError(
                f"Vision image must have 3 or 4 channels; received {image.shape[2]}"
            )
        array = image[..., :3]
        if array.dtype != np.uint8:
            raise TypeError(
                f"Vision image array must have dtype=uint8; received {array.dtype}"
            )
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array

    raise TypeError(
        "image must be a pyvips.Image or an HxWxC uint8 numpy array"
    )


def _b64decode(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc


__all__ = [
    "ensure_srgb",
    "load_vips_from_base64",
    "vips_to_uint8_numpy",
    "as_uint8_image_array",
    "ImageArray",
]
