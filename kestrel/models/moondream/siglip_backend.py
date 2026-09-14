"""Select the shipped Hopper SigLIP encoder when its contract matches."""
from typing import Any, Protocol

import torch

from kestrel.device import get_device_capability


_CROP_COUNTS = tuple(range(2, 14))
_SIGLIP_GEOMETRY = (27, 1152, 4304, 16, 14, 378, 12, 3)
_create_hopper_encoder = None


class SiglipEncoderBackend(Protocol):
    """Runtime contract implemented by a complete vision encoder backend."""

    crop_counts: tuple[int, ...]
    crop_dtype: torch.dtype

    def encode_crops(
        self, crops: torch.Tensor, tiling: tuple[int, int],
    ) -> torch.Tensor: ...

    def close(self) -> None: ...


def _is_hopper_siglip(
    vision: Any,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or dtype != torch.bfloat16:
        return False
    if tuple(get_device_capability(device)) != (9, 0):
        return False
    geometry = (
        config.enc_n_layers,
        config.enc_dim,
        config.enc_ff_dim,
        config.enc_n_heads,
        config.enc_patch_size,
        config.crop_size,
        config.max_crops,
        config.in_channels,
    )
    return geometry == _SIGLIP_GEOMETRY and len(vision.blocks) == 27


def create_siglip_backend(
    *,
    model_name: str,
    vision: Any,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> SiglipEncoderBackend | None:
    """Create the required Hopper backend, or select the native tower elsewhere."""
    if not _is_hopper_siglip(vision, config, device, dtype):
        return None

    factory = _create_hopper_encoder
    if factory is None:
        from kestrel_kernels.megakernel.siglip import create_encoder

        factory = create_encoder
    backend = factory(
        model_name=model_name,
        vision=vision,
        config=config,
        device=device,
        dtype=dtype,
    )
    if tuple(getattr(backend, "crop_counts", ())) != _CROP_COUNTS:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        raise RuntimeError("Hopper SigLIP backend does not cover every image crop count 2..13")
    if getattr(backend, "crop_dtype", None) is not torch.uint8:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        raise RuntimeError("Hopper SigLIP backend must consume raw uint8 crops")
    if not callable(getattr(backend, "encode_crops", None)) or not callable(
        getattr(backend, "close", None)
    ):
        raise TypeError("Hopper SigLIP backend does not implement the encoder runtime contract")
    return backend


__all__ = ["SiglipEncoderBackend", "create_siglip_backend"]
