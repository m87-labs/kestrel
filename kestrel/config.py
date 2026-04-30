"""Runtime configuration objects for the Kestrel inference engine."""


from dataclasses import dataclass
from pathlib import Path
import re

import torch


_SMALL_VRAM_THRESHOLD_BYTES = 24 * 1024**3
_KV_CACHE_PAGES_SMALL_VRAM = 16384
_KV_CACHE_PAGES_LARGE_VRAM = 65536

# Apple Silicon unified-memory tiers. The KV cache is statically allocated
# upfront; on a 16 GB Mac, 65536 pages is ~12.6 GB at the BF16 MD2 page
# size — leaves no room for the model + activations and forces every
# inference into swap. Tier by total system memory (we don't have a
# separate VRAM number on unified-memory Macs).
_MPS_PAGES_BY_TOTAL_GIB = (
    # (total_gib_ceiling, pages)
    (18, 4096),    # base 16 GB Macs (M1/M2/M3/M4)
    (36, 16384),   # 24 / 32 GB tier (Pro chips)
    (72, 32768),   # 36 / 48 / 64 GB tier
    (None, 65536), # 96+ GB Ultra
)
_SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _mps_total_memory_bytes() -> int | None:
    """Best-effort total unified memory on Apple Silicon.

    Falls through to ``hw.memsize`` via sysctl if torch's MPS API
    doesn't expose it. Returns ``None`` if neither is available.
    """
    try:
        # Available on PyTorch 2.x on macOS arm64. Returns the
        # recommended working-set size for the MPS device, which is
        # tied to total system memory.
        return int(torch.mps.recommended_max_memory())
    except (AttributeError, RuntimeError):
        pass
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2)
        return int(out.strip())
    except Exception:
        return None


def _default_kv_cache_pages_for_device(device: str) -> int:
    """Choose a KV-cache default from the device's total memory."""
    torch_device = torch.device(device)
    if torch_device.type == "mps":
        total = _mps_total_memory_bytes()
        if total is None:
            # Conservative: any unknown MPS device is treated as the
            # smallest tier — better to under-allocate than swap-thrash.
            return _MPS_PAGES_BY_TOTAL_GIB[0][1]
        total_gib = total / (1024**3)
        for ceiling, pages in _MPS_PAGES_BY_TOTAL_GIB:
            if ceiling is None or total_gib <= ceiling:
                return pages
        return _MPS_PAGES_BY_TOTAL_GIB[-1][1]

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
    # Auto-select based on device memory when not provided.
    # CUDA: <=24 GiB -> 16384 pages, otherwise 65536 pages.
    # MPS:  tiered by total unified memory (4096 / 16384 / 32768 / 65536
    # for 16 / 32 / 64 / 96+ GiB Macs).
    kv_cache_pages: int | None = None
    enable_cuda_graphs: bool = True
    enable_prefix_cache: bool = True
    # Model: "moondream2" or "moondream3-preview"
    model: str = "moondream3-preview"
    service_name: str = "local"

    def __post_init__(self):
        normalized_service_name = self.service_name.strip()
        self.service_name = normalized_service_name or "local"
        if not _SERVICE_NAME_PATTERN.fullmatch(self.service_name):
            raise ValueError("service_name must match [A-Za-z0-9_-]+")

        if self.kv_cache_pages is None:
            self.kv_cache_pages = _default_kv_cache_pages_for_device(self.device)
        elif self.kv_cache_pages <= 0:
            raise ValueError("kv_cache_pages must be a positive integer")

        if self.model_path is None:
            from kestrel.model_download import ensure_model_weights

            self.model_path = ensure_model_weights(self.model)

    def resolved_dtype(self) -> torch.dtype:
        """Return the torch dtype to use for the runtime.

        On MPS we override the default ``bfloat16`` to ``float16``:
        Apple Silicon's matmul + SDPA throughput is consistently
        higher in fp16, fp16's 10-bit mantissa is a precision
        *upgrade* over bf16's 7-bit on the activation range Moondream
        occupies, and Moondream's bf16-trained weights round to fp16
        without saturating fp16's 65504 dynamic-range cap. Users who
        explicitly want fp32 on MPS still get fp32 — only the bf16
        default is overridden.
        """

        if self.dtype == torch.bfloat16 and self.resolved_device().type == "mps":
            return torch.float16
        return self.dtype

    def resolved_device(self) -> torch.device:
        """Return the torch device requested for inference."""

        device = torch.device(self.device)
        # Ensure CUDA devices have an explicit index for torch.cuda.set_device()
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device


__all__ = ["RuntimeConfig"]
