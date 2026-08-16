from __future__ import annotations

import hashlib

import numpy as np
import torch

from kestrel.models.gemma4.image import preprocess_image


def _sha256_tensor(tensor) -> str:
    raw = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def test_preprocess_image_matches_reference_antialiased_bicubic() -> None:
    image = np.arange(37 * 61 * 3, dtype=np.uint8).reshape(37, 61, 3)

    result = preprocess_image(image)

    # Generated from the pinned Gemma4ImageProcessor reference implementation:
    # uint8 Torchvision bicubic with antialiasing, then BF16 conversion.
    assert result.num_image_tokens == 273
    assert _sha256_tensor(result.pixel_values) == (
        "34484611b0697c9f40f6f3698b20272e1f1f798daa611067e61e1ede48ce94ee"
    )
    assert _sha256_tensor(result.image_position_ids) == (
        "6a8ec8b7ec82189bbf0b31e1a6e0294c0b69ef164d56eb5e2b07433f0508ca48"
    )
