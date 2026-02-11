"""Runtime configuration objects for the Kestrel inference engine."""


from dataclasses import dataclass
from pathlib import Path

import torch


_SMALL_VRAM_THRESHOLD_BYTES = 24 * 1024**3
_KV_CACHE_PAGES_SMALL_VRAM = 16384
_KV_CACHE_PAGES_LARGE_VRAM = 65536


def _default_kv_cache_pages_for_device(device: str) -> int:
    """Choose a KV-cache default from the device's total VRAM size."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return _KV_CACHE_PAGES_LARGE_VRAM

    try:
        device_index = (
            torch.cuda.current_device()
            if torch_device.index is None
            else int(torch_device.index)
        )
        total_memory = int(torch.cuda.get_device_properties(device_index).total_memory)
    except Exception:
        return _KV_CACHE_PAGES_LARGE_VRAM

    if total_memory <= _SMALL_VRAM_THRESHOLD_BYTES:
        return _KV_CACHE_PAGES_SMALL_VRAM
    return _KV_CACHE_PAGES_LARGE_VRAM


@dataclass
class RuntimeConfig:
    """Knobs controlling the text-only inference prototype."""

    model_path: str | Path | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Effective batch size (excluding reserved batch_idx 0).
    max_batch_size: int = 4
    page_size: int = 1
    # Auto-select based on VRAM when not provided:
    # <=24 GiB -> 16384 pages, otherwise 65536 pages.
    kv_cache_pages: int | None = None
    enable_cuda_graphs: bool = True
    enable_prefix_cache: bool = True
    # Model: "moondream2" or "moondream3-preview"
    model: str = "moondream3-preview"

    def __post_init__(self):
        if self.kv_cache_pages is None:
            self.kv_cache_pages = _default_kv_cache_pages_for_device(self.device)
        elif self.kv_cache_pages <= 0:
            raise ValueError("kv_cache_pages must be a positive integer")

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
