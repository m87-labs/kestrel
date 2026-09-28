"""Kestrel single-pass runtime for DINOv2 image embeddings."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape
from .model import Dinov2Model, Dinov2Output
from .weights import DEFAULT_DINOV2_MODEL


@dataclass(frozen=True)
class Dinov2ExecutableCapability:
    """The exact request domain one injected compiled executable claims to serve.

    Capability resolution happens before constructing the runtime. The runtime compares
    this declaration verbatim and never probes GPU architecture, imports a backend to see
    whether it happens to exist, or infers support from executable attributes.
    """

    device: torch.device
    dtype: torch.dtype
    image_size: int = 224
    batch_size: int = 1
    task: str = "embed"

    def __post_init__(self) -> None:
        # Canonicalized so an index-less "cuda" and the executable's fully-qualified
        # "cuda:N" compare equal instead of silently routing to the eager fallback.
        object.__setattr__(self, "device", resolve_device(self.device))
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("compiled executable dtype must be a torch.dtype")
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError(
                "compiled executable image_size and batch_size must be positive"
            )
        if self.task != "embed":
            raise ValueError("DINOv2 compiled executables must declare task='embed'")

    def matches(
        self,
        *,
        task: str,
        device: torch.device,
        dtype: torch.dtype,
        image_size: int,
        batch_size: int,
    ) -> bool:
        return (
            self.task == task
            and self.device == device
            and self.dtype == dtype
            and self.image_size == image_size
            and self.batch_size == batch_size
        )


class Dinov2CompiledExecutable(Protocol):
    """Declared compiled-backend seam consumed by :class:`Dinov2Runtime`."""

    capability: Dinov2ExecutableCapability

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
        self._use_compiled = bool(
            compiled_executable is not None
            and compiled_executable.capability.matches(
                task="embed",
                device=self.device,
                dtype=self.dtype,
                image_size=self.image_size,
                batch_size=1,
            )
        )
        if compiled_executable is not None and not self._use_compiled:
            # The eager fallback on a declared-capability mismatch is deliberate, but an
            # explicitly injected executable that never serves is a configuration error
            # the caller should hear about, not discover from a slow checksum.
            warnings.warn(
                "Dinov2Runtime: injected compiled executable declares "
                f"{compiled_executable.capability} but the runtime domain is "
                f"(task='embed', device={self.device}, dtype={self.dtype}, "
                f"image_size={self.image_size}, batch_size=1); serving the eager "
                "fallback instead",
                RuntimeWarning,
                stacklevel=2,
            )

        if not callable(processor):
            raise TypeError("DINOv2 processor must be callable")
        if not self._use_compiled and model is None:
            raise RuntimeError("DINOv2 eager fallback has no model")
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
        # One public dtype across eager and compiled execution. The compiled final-norm
        # seam is FP32 for checkpoint fidelity, while an eager BF16 model naturally
        # returns BF16; exposing those backend details would make the same runtime request
        # change type when a compiled capability is injected. Normalize once at the public
        # boundary and derive the pooler from that tensor so its CLS-view semantics are
        # identical on every backend.
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
        if self._compiled_executable is not None:
            self._compiled_executable.shutdown()
        empty_cache(self.device)


__all__ = [
    "Dinov2CompiledExecutable",
    "Dinov2ExecutableCapability",
    "Dinov2Runtime",
]
