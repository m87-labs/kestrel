"""Tests for segmentation mask bitmap encoding helpers."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from kestrel import seg_refiner as seg_refiner_mod


@pytest.mark.skipif(
    not seg_refiner_mod._HAS_SEG_DEPS,
    reason="Segmentation optional dependencies are not installed",
)
def test_encode_mask_png_base64_roundtrip() -> None:
    from PIL import Image

    mask = np.zeros((11, 13), dtype=np.uint8)
    mask[2:6, 3:9] = 1

    b64 = seg_refiner_mod._encode_mask_png_base64(mask)
    raw = base64.b64decode(b64)

    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)

    assert arr.shape == mask.shape
    assert arr.dtype == np.uint8
    assert arr.min() == 0
    assert arr.max() == 255
    assert (arr[2:6, 3:9] == 255).all()
    assert (arr[:2, :] == 0).all()

