"""Read only the qualified MD3 vision tensors, without constructing a model."""
from pathlib import Path

import torch

from kestrel.models.moondream.weights import md3_vision_parameter_sources


def load_siglip_weights(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Load a local MD3 .pt or safetensors checkpoint's BF16 vision parameters.

    This is the Moondream-trained 378-pixel encoder, not an arbitrary upstream
    SigLIP checkpoint. Safetensors reads only requested vision tensors; torch
    checkpoints are memory mapped and never move text weights onto the GPU.
    """
    if not isinstance(checkpoint, (str, Path)):
        raise TypeError("SigLIP model_path must name a local MD3 checkpoint")
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SigLIP checkpoint does not exist: {path}")
    sources = md3_vision_parameter_sources(27, include_projection=False)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            tensors = {name: reader.get_tensor(source) for name, source in sources.items()}
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(raw, dict):
            raise ValueError("MD3 checkpoint must contain a tensor state dictionary")
        # Compiled source checkpoints may retain module wrapper components.
        raw = {key.replace("._orig_mod", ""): value for key, value in raw.items()}
        tensors = {name: raw[source] for name, source in sources.items()}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"SigLIP weight {name!r} must be floating point")
        value = value.to(torch.bfloat16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"SigLIP weight {name!r} must be finite in BF16")
        tensors[name] = value
    return tensors
