"""Utilities for working with pyvips images within Kestrel."""


import base64
import binascii

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


def ensure_srgb(image: pyvips.Image | np.ndarray) -> pyvips.Image | np.ndarray:
    """Return an image in 8-bit sRGB without alpha."""

    if isinstance(image, np.ndarray):
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

    if image.format != "uchar":
        image = image.cast("uchar")

    if image.hasalpha():
        image = image.flatten(background=[0, 0, 0])

    if image.bands == 1:
        image = image.colourspace("srgb")
    elif image.bands > 3:
        image = image.extract_band(0, n=3)

    try:
        if image.interpretation not in ("srgb", "rgb", "rgb16"):
            image = image.colourspace("srgb")
    except pyvips.Error:
        pass

    if image.bands == 1:
        image = image.bandjoin([image, image])
    elif image.bands > 3:
        image = image.extract_band(0, n=3)
    return image.copy_memory()


def _vips_to_uint8_numpy(image: pyvips.Image) -> np.ndarray:
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
]
