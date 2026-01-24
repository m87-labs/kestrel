"""Utilities for working with images within Kestrel."""


import base64
import binascii

import numpy as np


def load_image_bytes_from_base64(data: str) -> bytes:
    """Decode a base64 string (raw or ``data:image/...``) into raw bytes."""

    if data.startswith("data:image"):
        header, _, payload = data.partition(",")
        if not payload:
            raise ValueError("Invalid data URL: missing payload")
        return _b64decode(payload)
    else:
        return _b64decode(data)


def ensure_srgb(image: np.ndarray) -> np.ndarray:
    """Return an image in 8-bit sRGB without alpha.

    Handles grayscale, RGBA, and other channel counts by converting to RGB.
    """

    np_image = np.clip(image, 0, 255).astype(np.uint8)

    if np_image.ndim == 2:
        np_image = np.stack([np_image] * 3, axis=-1)
    elif np_image.ndim == 3:
        if np_image.shape[2] == 1:
            np_image = np.repeat(np_image, 3, axis=2)
        elif np_image.shape[2] == 4:
            np_image = np_image[:, :, :3]
        elif np_image.shape[2] > 3:
            np_image = np_image[:, :, :3]

    return np_image


def _b64decode(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc


__all__ = [
    "ensure_srgb",
    "load_image_bytes_from_base64",
]
