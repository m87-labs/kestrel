"""Qwen image preprocessing tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kestrel.models.qwen35 import qwen_image


def test_preprocess_image_uses_native_resize_for_numpy(monkeypatch):
    calls = []

    def resize_bicubic(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        calls.append((image.shape, image.flags.c_contiguous, target_h, target_w))
        return np.full((target_h, target_w, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(
        qwen_image,
        "kestrel_native",
        SimpleNamespace(resize_bicubic=resize_bicubic),
    )

    image = np.zeros((32, 32, 4), dtype=np.uint8)
    pixel_values, image_grid_thw = qwen_image.preprocess_image(image)

    assert calls == [((32, 32, 3), True, 256, 256)]
    assert pixel_values.shape == (256, 1536)
    assert image_grid_thw.tolist() == [[1, 16, 16]]
