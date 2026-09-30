"""Construct the shipped DINOv2 runtime for supported Hopper devices."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from kestrel.device import get_device_capability, resolve_device

from .metadata import DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID
from .preprocessing import Dinov2ImageProcessor
from .runtime import Dinov2Runtime
from .weights import load_dinov2


def create_dinov2_runtime(
    cfg: Any,
    *,
    compute_stream: Any = None,
    kv_pool: Any = None,
    max_lora_rank: int | None = None,
) -> Dinov2Runtime:
    """Load the complete encoder from kestrel-kernels; unsupported targets fail."""
    del kv_pool, max_lora_rank
    device = resolve_device(cfg.resolved_device())
    dtype = cfg.resolved_dtype()
    if (
        device.type != "cuda"
        or dtype is not torch.bfloat16
        or tuple(get_device_capability(device)) != (9, 0)
    ):
        raise ValueError("DINOv2 currently requires Hopper CUDA and BF16")

    checkpoint = cfg.model_path
    if checkpoint is not None and not isinstance(checkpoint, (str, Path)):
        raise TypeError("DINOv2 checkpoint must be a string or pathlib.Path")
    source = checkpoint or (
        DEFAULT_DINOV2_REPO_ID if cfg.model == DEFAULT_DINOV2_MODEL else cfg.model
    )
    loaded = load_dinov2(source)
    processor = Dinov2ImageProcessor(loaded.processor_config)

    from kestrel_kernels.megakernel.dinov2 import Dinov2MegakernelEncoder

    with torch.cuda.device(device), (
        torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
    ):
        executable = Dinov2MegakernelEncoder(
            state_dict=loaded.state_dict,
            config=loaded.model_config,
            device=device,
            dtype=dtype,
        )
    return Dinov2Runtime(
        cfg, processor=processor, executable=executable, compute_stream=compute_stream
    )
