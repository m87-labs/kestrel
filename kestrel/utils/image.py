"""Utilities for working with pyvips images within Kestrel."""

from __future__ import annotations

import base64
import binascii
from typing import Optional

import numpy as np
import pyvips


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
    return image.copy_memory()


def vips_to_uint8_numpy(image: pyvips.Image) -> np.ndarray:
    """Convert a pyvips image into a HxWxC uint8 NumPy array."""

    memory = image.write_to_memory()
    array = np.frombuffer(memory, dtype=np.uint8)
    height, width, bands = image.height, image.width, image.bands
    array = array.reshape(height, width, bands)
    # The buffer returned by pyvips is read-only; take a contiguous copy so we can
    # hand it to consumers that expect writable memory (e.g., torch.from_numpy).
    return np.ascontiguousarray(array)


def _b64decode(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc


__all__ = [
    "ensure_srgb",
    "load_vips_from_base64",
    "vips_to_uint8_numpy",
]
