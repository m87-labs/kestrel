"""CPU RGB resize and ImageNet normalization for RF-DETR inference."""
import io

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def preprocess_image(image, resolution):
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as decoded:
            image = decoded.convert('RGB')
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('NumPy images must be HWC RGB')
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    if not isinstance(image, torch.Tensor) or image.device.type != 'cpu':
        raise TypeError('image must be a PIL image, encoded bytes, HWC RGB array or CPU CHW tensor')
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError('tensor images must have shape [3, height, width]')
    if image.dtype == torch.uint8:
        image = image.float() / 255
    elif image.is_floating_point():
        image = image.float()
    else:
        raise TypeError('image pixels must be uint8 or floating point in [0, 1]')
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('image pixels must be finite and in [0, 1]')
    # Upstream predict uses tensor bilinear resize, antialias=False, then normalize.
    image = F.interpolate(image.unsqueeze(0), size=(resolution, resolution),
                          mode='bilinear', align_corners=False, antialias=False)
    mean = image.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = image.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    return ((image - mean) / std).contiguous()
