"""Runtime configuration objects for the Kestrel inference engine."""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RuntimeConfig:
    """Knobs controlling the text-only inference prototype."""

    model_path: str | Path | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Effective batch size (excluding reserved batch_idx 0).
    max_batch_size: int = 4
    page_size: int = 1
    kv_cache_pages: int = 65536
    enable_cuda_graphs: bool = True
    enable_prefix_cache: bool = True
    # Model: "moondream2" or "moondream3-preview"
    model: str = "moondream3-preview"

    def __post_init__(self):
        if self.model_path is None:
            from kestrel.model_download import ensure_model_weights

            self.model_path = ensure_model_weights(self.model)

    def resolved_dtype(self) -> torch.dtype:
        """Return the torch dtype to use for the runtime."""

        return self.dtype

    def resolved_device(self) -> torch.device:
        """Return the torch device requested for inference."""

        device = torch.device(self.device)
        # Ensure CUDA devices have an explicit index for torch.cuda.set_device()
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device


__all__ = ["RuntimeConfig"]
