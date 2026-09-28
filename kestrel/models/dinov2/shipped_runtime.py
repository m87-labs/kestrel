"""Compiler-free DINOv2 adapter backed by the shipped H100 megakernel."""

from collections.abc import Mapping

import torch

from .model import Dinov2Output
from .runtime import Dinov2ExecutableCapability


class Dinov2ShippedExecutable:
    """Serve one complete DINOv2 forward from the kestrel-kernels AOT archive."""

    def __init__(
        self,
        config,
        state_dict: Mapping[str, torch.Tensor],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Dinov2ShippedExecutable requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if dtype is not torch.bfloat16:
            raise ValueError("Dinov2ShippedExecutable requires bfloat16")
        from kestrel_kernels.megakernel.dinov2 import Dinov2MegakernelEncoder

        self._encoder = Dinov2MegakernelEncoder(
            state_dict=state_dict,
            config=config,
            device=device,
            dtype=dtype,
        )
        self.capability = Dinov2ExecutableCapability(
            device=device,
            dtype=dtype,
            image_size=224,
            batch_size=1,
            task="embed",
        )
        self._shutdown = False

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> Dinov2Output:
        if self._shutdown:
            raise RuntimeError("Dinov2ShippedExecutable is shut down")
        hidden = self._encoder.forward(pixel_values)
        return Dinov2Output(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0, :],
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._encoder.close()
        self._shutdown = True


__all__ = ["Dinov2ShippedExecutable"]
