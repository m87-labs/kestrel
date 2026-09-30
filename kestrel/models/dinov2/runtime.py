"""Kestrel single-pass runtime for DINOv2 image embeddings."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from typing import Any, Protocol

import torch
from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape
from .model import Dinov2Model, Dinov2Output
from .metadata import DEFAULT_DINOV2_MODEL


class Dinov2CompiledExecutable(Protocol):
    """Prepared backend selected once by the model factory."""

    def forward(self, pixel_values: torch.Tensor) -> Dinov2Output: ...

    def shutdown(self) -> None: ...


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
        kv_pool: Any = None,
        max_lora_rank: int | None = None,
        model: Dinov2Model | None = None,
        processor: Callable[[Any], torch.Tensor],
        compiled_executable: Dinov2CompiledExecutable | None = None,
    ) -> None:
        # Kestrel supplies these shared runtime resources to every registered model.
        # Single-pass image embeddings use neither paged KV storage nor adapters.
        del kv_pool, max_lora_rank
        self._model_name = getattr(cfg, "model", DEFAULT_DINOV2_MODEL)
        self.device = resolve_device(
            cfg.resolved_device()
            if hasattr(cfg, "resolved_device")
            else getattr(cfg, "device", "cpu")
        )
        self.dtype = (
            cfg.resolved_dtype()
            if hasattr(cfg, "resolved_dtype")
            else getattr(cfg, "dtype", torch.float32)
        )
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("runtime dtype must be a torch.dtype")
        self.primary_stream = compute_stream
        self.compute_stream = compute_stream
        self._compiled_executable = compiled_executable
        if (model is None) == (compiled_executable is None):
            raise ValueError("DINOv2 requires exactly one eager or compiled backend")
        self._use_compiled = compiled_executable is not None

        if not callable(processor):
            raise TypeError("DINOv2 processor must be callable")
        if model is not None:
            model.eval()

        self.model = model
        self.processor = processor
        self._shutdown = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def execution_backend(self) -> str:
        return "compiled" if self._use_compiled else "eager"

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
        if self._use_compiled:
            if (pixels.device != self.device or pixels.dtype != self.dtype
                    or not pixels.is_contiguous()):
                raise ValueError(
                    "compiled pixel_values must be contiguous and match the model device/dtype"
                )
            return pixels
        return pixels.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def forward(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[dict[str, torch.Tensor], ...]:
        if self._shutdown:
            raise RuntimeError("Dinov2Runtime is shut down")
        if task != "embed":
            raise ValueError(f"Dinov2Runtime does not support task {task!r}")
        (request,) = inputs
        pixel_values = self._pixel_values(request)
        if self._use_compiled:
            assert self._compiled_executable is not None
            output = self._compiled_executable.forward(pixel_values)
        else:
            assert self.model is not None
            output = self.model(pixel_values)
        if not isinstance(output, Dinov2Output):
            raise TypeError(
                f"DINOv2 backend must return Dinov2Output, got {type(output).__name__}"
            )
        expected_hidden = (1, 257, 384)
        expected_pooler = (1, 384)
        if tuple(output.last_hidden_state.shape) != expected_hidden:
            raise RuntimeError(
                "DINOv2 backend returned last_hidden_state shape "
                f"{tuple(output.last_hidden_state.shape)}, expected {expected_hidden}"
            )
        if tuple(output.pooler_output.shape) != expected_pooler:
            raise RuntimeError(
                f"DINOv2 backend returned pooler_output shape "
                f"{tuple(output.pooler_output.shape)}, expected {expected_pooler}"
            )
        # Keep FP32 output and a CLS view on every backend.
        last_hidden_state = output.last_hidden_state.to(dtype=torch.float32)
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
            if self._compiled_executable is not None:
                self._compiled_executable.shutdown()
        finally:
            self._compiled_executable = None
            self.model = None
            empty_cache(self.device)


__all__ = [
    "Dinov2CompiledExecutable",
    "Dinov2Runtime",
]
