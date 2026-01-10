"""Runtime utilities for loading precompiled CuTe DSL kernels.

This module provides shared utilities for loading precompiled kernels at runtime.
Each kernel family (topk, cute_moe, moe_align, flash_attn) uses these utilities
to load their respective precompiled .so files.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import cutlass.cute as cute


# Path to precompiled kernels directory
PRECOMPILED_DIR = Path(__file__).parent / "precompiled"


@lru_cache(maxsize=1)
def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper).

    Returns architecture from TORCH_CUDA_ARCH_LIST if set, otherwise
    detects from the current CUDA device.
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list:
        # Parse first arch from env var (e.g., "9.0" -> "sm90")
        first_arch = arch_list.split(";")[0].strip()
        if "." in first_arch:
            major, minor = first_arch.split(".")
            return f"sm{major}{minor}"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def load_precompiled_module(filename: str) -> Any | None:
    """Load a precompiled kernel module (.so file) if it exists.

    Args:
        filename: The .so filename (not full path)

    Returns:
        The loaded module, or None if the file doesn't exist.
    """
    so_path = PRECOMPILED_DIR / filename
    if not so_path.exists():
        return None
    return cute.runtime.load_module(str(so_path))
