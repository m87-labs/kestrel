"""Kestrel single-pass runtime for DINOv2 image embeddings."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from typing import Any

import torch
from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape


class Dinov2Runtime:
    """Batch-one runtime serving the canonical ``embed`` task."""

    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = 1
    image_size = 224

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        processor: Callable[[Any], torch.Tensor],
        executable: Any,
    ) -> None:
        self._model_name = cfg.model
        self.device = resolve_device(cfg.resolved_device())
        self.dtype = cfg.resolved_dtype()
        self.primary_stream = compute_stream
        self.compute_stream = compute_stream
        if not callable(processor):
            raise TypeError("DINOv2 processor must be callable")
        self.executable = executable
        self.processor = processor
        self._shutdown = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def tasks(self) -> tuple[str, ...]:
        return ("embed",)

    def preprocess_image_async(self, image: Any) -> Future[torch.Tensor]:
        future: Future[torch.Tensor] = Future()
        try:
            future.set_result(self.processor(image))
        # Any processor failure belongs on the returned Future, regardless of type.
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def _pixel_values(self, inputs: Any) -> torch.Tensor:
        if not isinstance(inputs, Mapping):
            raise TypeError("embed inputs must be a mapping")
        unexpected = set(inputs) - {"image", "pixel_values"}
        if unexpected:
            raise ValueError(f"unsupported embed inputs: {sorted(unexpected)}")
        forms = [name for name in ("image", "pixel_values") if name in inputs]
        if len(forms) != 1:
            raise ValueError("embed requires exactly one of image or pixel_values")

        if forms[0] == "image":
            if inputs["image"] is None:
                raise ValueError("image must not be None")
            pixels = self.processor(inputs["image"])
        else:
            pixels = inputs["pixel_values"]
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor")
        expected = (1, 3, self.image_size, self.image_size)
        if tuple(pixels.shape) != expected:
            raise ValueError(
                f"pixel_values must have shape {expected}, got {tuple(pixels.shape)}"
            )
        if not pixels.is_floating_point():
            raise TypeError("pixel_values must be floating point")
        if pixels.device.type == "cpu":
            # Validate after conversion as finite FP32/FP64 can overflow BF16.
            pixels = pixels.to(dtype=self.dtype).contiguous()
            if not torch.isfinite(pixels).all():
                raise ValueError("pixel_values must be finite in the model dtype")
            return pixels.to(self.device)
        if (pixels.device != self.device or pixels.dtype != self.dtype
                or not pixels.is_contiguous()):
            raise ValueError(
                "pixel_values must be contiguous and match the model device/dtype"
            )
        return pixels

    @torch.inference_mode()
    def forward(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[dict[str, torch.Tensor], ...]:
        if self._shutdown:
            raise RuntimeError("Dinov2Runtime is shut down")
        if task != "embed":
            raise ValueError(f"Dinov2Runtime does not support task {task!r}")
        if len(inputs) != 1:
            raise ValueError("DINOv2 serves one embed request per forward")
        pixel_values = self._pixel_values(inputs[0])
        last_hidden_state = self.executable.forward(pixel_values)
        if not isinstance(last_hidden_state, torch.Tensor):
            raise TypeError("DINOv2 executable must return a torch.Tensor")
        if tuple(last_hidden_state.shape) != (1, 257, 384):
            raise RuntimeError("DINOv2 executable must return shape [1, 257, 384]")
        if last_hidden_state.dtype is not torch.float32 or last_hidden_state.device != self.device:
            raise RuntimeError("DINOv2 executable must return FP32 on the model device")
        return (
            {
                "last_hidden_state": last_hidden_state,
                "pooler_output": last_hidden_state[:, 0, :],
            },
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self.executable.close()
        finally:
            self.executable = None
            empty_cache(self.device)


__all__ = [
    "Dinov2Runtime",
]
