"""Production runtime construction for the registered DINOv2 models."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from kestrel.device import get_device_capability, resolve_device

from .preprocessing import Dinov2ImageProcessor
from .runtime import Dinov2Runtime
from .shipped_runtime import Dinov2ShippedExecutable
from .weights import (
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV2_REPO_ID,
    DEFAULT_DINOV2_REVISION,
    load_dinov2,
)

_MODEL_REPOS = {
    DEFAULT_DINOV2_MODEL: DEFAULT_DINOV2_REPO_ID,
    DEFAULT_DINOV2_REPO_ID: DEFAULT_DINOV2_REPO_ID,
}
def repo_id_for_model(model_name: str) -> str:
    return _MODEL_REPOS.get(model_name, model_name)


def _use_shipped_encoder(device: torch.device, dtype: torch.dtype) -> bool:
    """Select the shipped family only for its exact Hopper BF16 domain."""
    if device.type != "cuda" or dtype is not torch.bfloat16:
        return False
    return tuple(get_device_capability(device)) == (9, 0)


def create_dinov2_runtime(
    cfg: Any,
    *,
    compute_stream: Any = None,
    kv_pool: Any = None,
    max_lora_rank: int | None = None,
) -> Dinov2Runtime:
    """Load one checkpoint and build the best runtime supported by this request.

    The registered batch-one, 224px BF16 runtime loads its complete encoder from
    ``kestrel-kernels`` on Hopper. Every other device/dtype combination receives
    the eager model. Missing or malformed Hopper artifacts are startup failures.
    """

    device = resolve_device(
        cfg.resolved_device()
        if hasattr(cfg, "resolved_device")
        else getattr(cfg, "device", "cpu")
    )
    dtype = (
        cfg.resolved_dtype()
        if hasattr(cfg, "resolved_dtype")
        else getattr(cfg, "dtype", torch.float32)
    )
    if not isinstance(dtype, torch.dtype):
        raise TypeError("runtime dtype must be a torch.dtype")

    checkpoint = getattr(cfg, "model_path", None)
    if checkpoint is not None and not isinstance(checkpoint, (str, Path)):
        raise TypeError("DINOv2 checkpoint must be a string or pathlib.Path")
    source = checkpoint or repo_id_for_model(
        getattr(cfg, "model", DEFAULT_DINOV2_MODEL)
    )
    use_shipped = _use_shipped_encoder(device, dtype)
    loaded = load_dinov2(
        source,
        revision=DEFAULT_DINOV2_REVISION,
        device=torch.device("cpu") if use_shipped else device,
        dtype=torch.float32 if use_shipped else dtype,
    )
    processor = Dinov2ImageProcessor(loaded.processor_config)

    if not use_shipped:
        return Dinov2Runtime(
            cfg,
            compute_stream=compute_stream,
            kv_pool=kv_pool,
            max_lora_rank=max_lora_rank,
            model=loaded.model,
            processor=processor,
        )

    with torch.cuda.device(device):
        stream_context = (
            nullcontext()
            if compute_stream is None
            else torch.cuda.stream(compute_stream)
        )
        with stream_context:
            executable = Dinov2ShippedExecutable(
                loaded.model_config,
                loaded.model.state_dict(),
                device=device,
                dtype=dtype,
            )
    return Dinov2Runtime(
        cfg,
        compute_stream=compute_stream,
        kv_pool=kv_pool,
        max_lora_rank=max_lora_rank,
        processor=processor,
        compiled_executable=executable,
    )


__all__ = ["create_dinov2_runtime", "repo_id_for_model"]
