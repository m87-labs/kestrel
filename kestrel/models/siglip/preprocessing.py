"""A single RGB resize for independent 378-pixel SigLIP images."""
from io import BytesIO

import numpy as np
from PIL import Image
import torch


def preprocess_image(image) -> torch.Tensor:
    """Return raw uint8 [1,3,378,378]; normalization stays in the megakernel."""
    if isinstance(image, (bytes, bytearray)):
        with Image.open(BytesIO(image)) as decoded:
            image = decoded.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("SigLIP NumPy images must be uint8 HWC RGB or RGBA")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("SigLIP image must be a PIL image, encoded bytes, or uint8 NumPy image")
    image = image.convert("RGB").resize((378, 378), Image.Resampling.BICUBIC)
    pixels = np.array(image, dtype=np.uint8, copy=True).transpose(2, 0, 1).copy()
    return torch.from_numpy(pixels).unsqueeze(0)
