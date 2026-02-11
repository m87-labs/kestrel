#!/usr/bin/env python3
"""Exhaustive grid search for CuTe MoE kernel configurations.

Sweeps over all relevant tile sizes, warp counts, and pipeline stages to find
optimal configurations for each token count. Results are saved to JSON files
and the best configs should be added to configs/cute_moe_configs.json.

IMPORTANT: This script tests BOTH up and down kernels together with shared
routing (same block_m for moe_align_block_size). This ensures the selected
configs are compatible - both kernels must use the same block_m in practice.

RUNNING THE GRID SEARCH
=======================

1. Sync code to p1:
   ./sync.sh p1

2. Run grid search for specific token counts:
   ssh p1 'cd ~/code/kestrel && KESTREL_CUTE_MOE_JIT=1 ~/.local/bin/uv run python \\
     scripts/grid_search_cute_moe.py --num-tokens 8 16 32 --output results.json'

3. Run full grid search (all token counts):
   ssh p1 'cd ~/code/kestrel && KESTREL_CUTE_MOE_JIT=1 nohup ~/.local/bin/uv run python \\
     scripts/grid_search_cute_moe.py --output /tmp/cute_moe_grid_search.json \\
     > /tmp/cute_moe_grid_search.log 2>&1 &'

4. Monitor progress:
   ssh p1 'tail -f /tmp/cute_moe_grid_search.log'

SELECTING THE BEST CONFIG
=========================

For each block_m, we find the best (up_config, down_config) pair, then pick
the block_m that minimizes total time (up_time + down_time).

When multiple configs have similar performance (within ~5% or 2*std_dev), prefer
the "simpler" config using this priority:

1. Fewer pipeline stages (num_stages) - reduces register pressure
2. Fewer warps (num_warps) - simpler scheduling
3. Smaller tiles - less shared memory pressure

Save selected configs to: kestrel-kernels/python/kestrel_kernels/configs/

OPTIONS
=======

  --num-tokens N...  Token counts to test (default: all standard counts)
  --output FILE      Write results to JSON file
  --quick            Use smaller search space for faster iteration
  --iters N          Benchmark iterations (default: 200)
  --warmup N         Warmup iterations (default: 5)
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Enable JIT compilation for autotuning (covers both cute_moe and moe_align kernels)
os.environ["KESTREL_CUTE_MOE_JIT"] = "1"

import torch

# Allow `uv run python scripts/grid_search_cute_moe.py ...` without needing
# PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import kestrel_kernels as _kk  # noqa: E402
import kestrel_kernels.cute_moe as _kk_cute_moe  # noqa: E402
import kestrel_kernels.flash_attn.cute as _kk_fa_cute  # noqa: E402

_KK_SRC_CANDIDATES = (
    _REPO_ROOT / "kestrel-kernels" / "python" / "kestrel_kernels",
    _REPO_ROOT.parent / "kestrel-proprietary" / "kestrel-kernels" / "python" / "kestrel_kernels",
)
_KK_SRC = next((p for p in _KK_SRC_CANDIDATES if p.exists()), None)
if _KK_SRC is not None:
    kk_path = str(_KK_SRC)
    if kk_path not in _kk.__path__:
        # Allow source package discovery for submodules while keeping compiled extensions.
        _kk.__path__.append(kk_path)
    cute_moe_src = _KK_SRC / "cute_moe"
    if cute_moe_src.exists():
        cute_moe_path = str(cute_moe_src)
        if cute_moe_path not in _kk_cute_moe.__path__:
            # Make JIT template modules visible to the already-imported cute_moe package.
            _kk_cute_moe.__path__.append(cute_moe_path)
    fa_cute_src = _KK_SRC / "flash_attn" / "cute"
    if fa_cute_src.exists():
        fa_cute_path = str(fa_cute_src)
        if fa_cute_path not in _kk_fa_cute.__path__:
            # Ensure flash_attn cute helpers are importable for CuTe templates.
            _kk_fa_cute.__path__.append(fa_cute_path)

from kestrel.fused_moe.routing import moe_align_block_size
from kestrel_kernels.arch import normalize_gpu_name_key, sm_arch_at_least
from kestrel_kernels.cute_moe import (
    CuteMoeConfig,
    invoke_cute_moe_up,
    invoke_cute_moe_down,
    invoke_cute_moe_up_fp8,
    invoke_cute_moe_down_fp8,
)


# Moondream-ish MoE config
NUM_EXPERTS = 64
TOP_K = 8
D_MODEL = 2048
D_EXPERT = 1024

# Token counts to optimize for (decode + prefill range)
TOKEN_COUNTS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]


def _config_filename_for_current_gpu(*, dtype: str) -> str:
    gpu_key = normalize_gpu_name_key(torch.cuda.get_device_name())
    return f"cute_moe_E{NUM_EXPERTS}_H{D_MODEL}_I{D_EXPERT}_{dtype}_{gpu_key}.json"


def _config_description_for_current_gpu() -> str:
    return f"CuTe MoE kernel configs optimized for {torch.cuda.get_device_name()}"


def _current_cuda_arch() -> str:
    """Return current CUDA arch in `smXY` form."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"

# Full search space for BF16 (sweeps both warp and wgmma kernel types)
# - warp kernel: block_m can be any multiple of 16 up to num_threads
# - wgmma kernel: block_m must be multiple of 64, num_warps derived
FULL_SEARCH_SPACE = {
    "kernel_type": ["warp", "wgmma"],
    "block_m": {
        "warp": [16, 32, 48, 64, 80, 96, 112, 128],  # multiples of 16
        "wgmma": [64, 128, 192, 256],                 # multiples of 64
    },
    "block_n": [32, 64, 128, 256],
    "block_k": [64, 128, 256],
    "num_warps": [2, 4, 8],  # for warp kernel; wgmma derives from block_m
    "num_stages": [1, 2, 3, 4, 5],
}

# Quick search space (for faster iteration)
QUICK_SEARCH_SPACE = {
    "kernel_type": ["warp", "wgmma"],
    "block_m": {
        "warp": [16, 32, 64],
        "wgmma": [64, 128],
    },
    "block_n": [64, 128, 256],
    "block_k": [128, 256],
    "num_warps": [2, 4],
    "num_stages": [1, 2, 3],
}

# FP8 SM90+ search space - principled pruning based on MoE problem dimensions:
# - UP kernel: M=tokens*8 distributed across 64 experts, N=1024, K=2048
# - DOWN kernel: M=tokens*8 distributed across 64 experts, N=2048, K=1024
#
# Token-based kernel selection (based on BF16 optimal configs):
# - Tokens 1-48: warp only (small M tiles, WGMMA overhead not worth it)
# - Tokens 64-512: both warp and wgmma (transition region)
# - Tokens 1024+: wgmma only (large M, WGMMA wins at scale)
#
# Block_m pruning: only power-of-2 values (16, 32, 64, 128, 192 for WGMMA)
# Non-power-of-2 (48, 80, 96, 112) rarely optimal in BF16 benchmarks.

def get_fp8_search_space_sm90(num_tokens: int) -> dict:
    """Get FP8 SM90+ search space pruned by token count (warp+wgmma)."""
    # Common parameters across all token counts
    block_n = [64, 128, 256]
    block_k_warp = [16, 32, 64, 128]
    block_k_wgmma = [32, 64, 128, 256]
    num_warps = [2, 4, 8]
    num_stages = [1, 2, 3, 4, 5, 6, 7, 8]

    if num_tokens <= 48:
        # Small tokens: warp kernel only, small block_m
        return {
            "kernel_type": ["warp"],
            "block_m": {"warp": [16, 32]},
            "block_n": block_n,
            "block_k": {"warp": block_k_warp},
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif num_tokens <= 512:
        # Medium tokens: both warp and wgmma
        return {
            "kernel_type": ["warp", "wgmma"],
            "block_m": {
                "warp": [16, 32, 64],
                "wgmma": [64, 128],
            },
            "block_n": block_n,
            "block_k": {
                "warp": block_k_warp,
                "wgmma": block_k_wgmma,
            },
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    else:
        # Large tokens (1024+): wgmma only
        # Crossover from warp to wgmma happens at ~256 tokens
        return {
            "kernel_type": ["wgmma"],
            "block_m": {"wgmma": [64, 128, 192]},
            "block_n": block_n,
            "block_k": {"wgmma": block_k_wgmma},
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

# Full FP8 SM90+ search space (for reference, not used directly - use get_fp8_search_space_sm90)
FP8_FULL_SEARCH_SPACE = {
    "kernel_type": ["warp", "wgmma"],
    "block_m": {
        "warp": [16, 32, 64],       # pruned: removed 48, 80, 96, 112, 128
        "wgmma": [64, 128, 192],    # pruned: removed 256 (register pressure)
    },
    "block_n": [64, 128, 256],
    "block_k": {
        "warp": [16, 32, 64, 128],   # must be divisible by 16
        "wgmma": [32, 64, 128, 256], # must be divisible by 32
    },
    "num_warps": [2, 4, 8],  # for warp kernel; wgmma derives from block_m
    "num_stages": [1, 2, 3, 4, 5, 6, 7, 8],
}

FP8_QUICK_SEARCH_SPACE = {
    "kernel_type": ["warp", "wgmma"],
    "block_m": {
        "warp": [16, 32],
        "wgmma": [64, 128],
    },
    "block_n": [64, 128, 256],
    "block_k": {
        "warp": [64, 128],
        "wgmma": [64, 128],
    },
    "num_warps": [2, 4],
    "num_stages": [2, 3, 4],
}

# FP8 SM89 search space (warp + qmma).
# qmma keeps block_n=128 and num_warps=4, while sweeping block_m/block_k/num_stages.
FP8_SM89_FULL_SEARCH_SPACE = {
    "kernel_type": ["warp", "qmma"],
    "block_m": {
        "warp": [16, 32, 64, 96, 128],
        "qmma": [16, 32, 48, 64, 80, 96, 112, 128],
    },
    "block_n": {
        "warp": [64, 128, 256],
        "qmma": [128],
    },
    "block_k": {
        "warp": [16, 32, 64, 128],
        "qmma": [64, 128],
    },
    "num_warps": {
        "warp": [2, 4, 8],
        "qmma": [4],
    },
    "num_stages": {
        "warp": [1, 2, 3, 4, 5, 6, 7, 8],
        "qmma": [2, 3, 4, 5],
    },
}

FP8_SM89_QUICK_SEARCH_SPACE = {
    "kernel_type": ["warp", "qmma"],
    "block_m": {
        "warp": [16, 32, 64],
        "qmma": [16, 32, 64, 96, 128],
    },
    "block_n": {
        "warp": [64, 128, 256],
        "qmma": [128],
    },
    "block_k": {
        "warp": [64, 128],
        "qmma": [64, 128],
    },
    "num_warps": {
        "warp": [2, 4],
        "qmma": [4],
    },
    "num_stages": {
        "warp": [2, 3, 4],
        "qmma": [2, 3, 4, 5],
    },
}


def _filter_kernel_types(search_space: dict, kernel_types: list[str]) -> dict:
    """Keep only selected kernel types from a search-space dict."""
    out = {"kernel_type": list(kernel_types)}
    for key, values in search_space.items():
        if key == "kernel_type":
            continue
        if isinstance(values, dict):
            out[key] = {k: v for k, v in values.items() if k in kernel_types}
        else:
            out[key] = values
    return out


def get_fp8_search_space(num_tokens: int, *, arch: str, quick: bool) -> dict:
    """Architecture-aware FP8 search-space selector.

    - SM90+: preserve existing warp/wgmma behavior.
    - SM89: use warp/qmma search space.
    - <SM89: warp-only fallback.
    """
    if sm_arch_at_least(arch, 90):
        if quick:
            return FP8_QUICK_SEARCH_SPACE
        return get_fp8_search_space_sm90(num_tokens)

    if arch == "sm89":
        # Minor search-time optimization from manual checks:
        # - <=512 tokens: keep both warp and qmma in the search.
        # - >512 tokens: search qmma only.
        if num_tokens <= 512:
            return FP8_SM89_QUICK_SEARCH_SPACE if quick else FP8_SM89_FULL_SEARCH_SPACE
        if quick:
            return {
                "kernel_type": ["qmma"],
                "block_m": {"qmma": [64, 96, 128]},
                "block_n": {"qmma": [128]},
                "block_k": {"qmma": [128]},
                "num_warps": {"qmma": [4]},
                "num_stages": {"qmma": [4]},
            }
        return _filter_kernel_types(FP8_SM89_FULL_SEARCH_SPACE, ["qmma"])

    # Legacy fallback path: warp-only FP8.
    return {
        "kernel_type": ["warp"],
        "block_m": {"warp": [16, 32, 64]},
        "block_n": {"warp": [64, 128, 256]},
        "block_k": {"warp": [64, 128]},
        "num_warps": {"warp": [2, 4]},
        "num_stages": {"warp": [2, 3, 4]},
    }


@dataclass
class BenchResult:
    """Result of benchmarking a single config."""
    config: CuteMoeConfig
    kind: Literal["up", "down"]
    num_tokens: int
    time_us: float
    std_us: float = 0.0  # Standard deviation in microseconds
    error: str | None = None


@dataclass
class CombinedBenchResult:
    """Result of benchmarking both up and down with shared routing."""
    block_m: int
    num_tokens: int
    up_config: CuteMoeConfig
    up_time_us: float
    up_std_us: float
    down_config: CuteMoeConfig
    down_time_us: float
    down_std_us: float
    total_time_us: float  # up_time_us + down_time_us


def quantize_fp8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor row-wise to FP8 E4M3."""
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / 448.0  # FP8 E4M3 max
    x_fp8 = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    return x_fp8, scale.squeeze(-1).to(torch.float32)


def quantize_fp8_colwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor column-wise (per output channel) to FP8 E4M3."""
    # w: [E, N, K] -> scale per [E, N]
    abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / 448.0
    w_fp8 = (w / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    return w_fp8, scale.squeeze(-1).to(torch.float32)


def is_valid_config(config: CuteMoeConfig, kind: str, num_tokens: int) -> bool:
    """Check if config is valid for the given parameters."""
    num_threads = config.num_warps * 32

    # CRITICAL: block_m must not exceed num_threads!
    # The kernel's metadata loading loop uses `if tx < block_m` to load per-row
    # metadata into shared memory. With only num_threads threads, only rows 0 to
    # num_threads-1 get their metadata loaded. Rows beyond that read uninitialized
    # shared memory, causing illegal memory access.
    if config.block_m > num_threads:
        return False

    # block_m should be appropriate for token count
    # For very small batches, larger block_m wastes work
    assignments = num_tokens * TOP_K if kind == "up" else num_tokens * TOP_K
    if config.block_m > assignments * 2:
        return False  # Too much padding overhead

    # block_n must be divisible by 8 * num_warps (MMA layout constraint)
    # The MMA atom is (16, 8, 16) and warps are tiled (1, num_warps, 1) in N
    mma_n_coverage = 8 * config.num_warps
    if config.block_n % mma_n_coverage != 0:
        return False

    # Warp constraints: need enough threads for the tile
    # Each warp is 32 threads; tiles need sufficient parallelism
    min_threads_needed = max(config.block_n // 8, config.block_k // 8)
    if num_threads < min_threads_needed:
        return False

    # Avoid configs known to cause IR verification failures or resource exhaustion
    # block_n=32 with num_warps=8 fails LDSM alignment verification
    if config.block_n == 32 and config.num_warps == 8:
        return False

    # High stages (>=4) with large tiles can exceed shared memory or register budget
    if config.num_stages >= 4:
        if config.block_k >= 256 or config.block_n >= 256:
            return False

    return True


# Global lock file for serializing GPU benchmarks across parallel processes
_LOCK_FILE = Path("/tmp/cute_moe_grid_search.lock")


class GPUBenchmarkLock:
    """Context manager for exclusive GPU access during benchmarking.

    JIT compilation is CPU-bound and can run in parallel, but actual GPU
    benchmarking must be serialized to get accurate timing measurements.
    """

    def __init__(self):
        self._lock_file = None

    def __enter__(self):
        _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(_LOCK_FILE, "w")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self._lock_file:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()


@torch.no_grad()
def jit_compile_config(
    config: CuteMoeConfig,
    kind: Literal["up", "down"],
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> str | None:
    """JIT compile a config without timing. Returns error string or None."""
    is_fp8 = A_scale is not None
    dtype_str = "fp8" if is_fp8 else "bf16"
    print(f"        JIT {kind} ({dtype_str}): m={config.block_m} n={config.block_n} k={config.block_k} "
          f"w={config.num_warps} s={config.num_stages} t={config.kernel_type}...", end="", flush=True)
    try:
        if is_fp8:
            if kind == "up":
                invoke_cute_moe_up_fp8(
                    A, A_scale, B, B_scale, C,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    config=config,
                )
            else:
                invoke_cute_moe_down_fp8(
                    A, A_scale, B, B_scale, C,
                    topk_weights=topk_weights,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    config=config,
                )
        else:
            if kind == "up":
                invoke_cute_moe_up(
                    A, B, C,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    config=config,
                )
            else:
                invoke_cute_moe_down(
                    A, B, C,
                    topk_weights=topk_weights,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    config=config,
                )
        torch.cuda.synchronize()
        print(" OK", flush=True)
        return None
    except Exception as e:
        print(f" FAILED: {e}", flush=True)
        return str(e)


def _invoke_kernel(
    kind: Literal["up", "down"],
    config: CuteMoeConfig,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> None:
    """Helper to invoke the correct kernel based on dtype."""
    is_fp8 = A_scale is not None
    if is_fp8:
        if kind == "up":
            invoke_cute_moe_up_fp8(
                A, A_scale, B, B_scale, C,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                config=config,
            )
        else:
            invoke_cute_moe_down_fp8(
                A, A_scale, B, B_scale, C,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                config=config,
            )
    else:
        if kind == "up":
            invoke_cute_moe_up(
                A, B, C,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                config=config,
            )
        else:
            invoke_cute_moe_down(
                A, B, C,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                config=config,
            )


@torch.no_grad()
def benchmark_compiled_config(
    config: CuteMoeConfig,
    kind: Literal["up", "down"],
    num_tokens: int,
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    warmup: int = 5,
    iters: int = 200,
) -> BenchResult:
    """Benchmark an already-compiled config. Must hold GPU lock."""
    try:
        # Brief warmup (already compiled, just warming up GPU caches)
        for _ in range(warmup):
            _invoke_kernel(kind, config, A, B, C, A_scale, B_scale, topk_weights,
                          sorted_token_ids, expert_ids, num_tokens_post_padded)
        torch.cuda.synchronize()

        # Benchmark
        times_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _invoke_kernel(kind, config, A, B, C, A_scale, B_scale, topk_weights,
                          sorted_token_ids, expert_ids, num_tokens_post_padded)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        mean_ms = sum(times_ms) / len(times_ms)
        std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
        time_us = mean_ms * 1000
        std_us = std_ms * 1000

        return BenchResult(config=config, kind=kind, num_tokens=num_tokens, time_us=time_us, std_us=std_us)

    except Exception as e:
        return BenchResult(
            config=config, kind=kind, num_tokens=num_tokens, time_us=float("inf"), error=str(e)
        )


def run_combined_grid_search(
    num_tokens: int,
    search_space: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    kernel_dtype: str = "bf16",  # "bf16" or "fp8"
    warmup: int,
    iters: int,
    use_lock: bool = False,
) -> list[CombinedBenchResult]:
    """Run grid search testing both up and down kernels with shared routing.

    For each block_m, we:
    1. Generate routing with that block_m (shared between up and down)
    2. For each config: JIT compile (parallel), acquire lock, benchmark (serialized), release lock
    3. Find the best up and down configs for this block_m
    4. Record total time = up_time + down_time

    Args:
        kernel_dtype: "bf16" for BF16 kernels, "fp8" for FP8 kernels
        use_lock: If True, acquire GPU lock during benchmarking (for parallel runs)

    Returns list of CombinedBenchResult, one per block_m tested.
    """
    torch.manual_seed(0)

    assignments = num_tokens * TOP_K
    is_fp8 = kernel_dtype == "fp8"

    # Allocate BF16 tensors (used directly for BF16, quantized for FP8)
    # Up: [M, d_model] x [E, 2*d_expert, d_model] -> [M, top_k, 2*d_expert]
    A_up_bf16 = torch.randn(num_tokens, D_MODEL, device=device, dtype=dtype)
    B_up_bf16 = torch.randn(NUM_EXPERTS, D_EXPERT * 2, D_MODEL, device=device, dtype=dtype)
    C_up = torch.empty(num_tokens, TOP_K, D_EXPERT * 2, device=device, dtype=dtype)

    # Down: [M*top_k, d_expert] x [E, d_model, d_expert] -> [M, top_k, d_model]
    A_down_bf16 = torch.randn(assignments, D_EXPERT, device=device, dtype=dtype)
    B_down_bf16 = torch.randn(NUM_EXPERTS, D_MODEL, D_EXPERT, device=device, dtype=dtype)
    C_down = torch.empty(num_tokens, TOP_K, D_MODEL, device=device, dtype=dtype)
    topk_weights = torch.randn(num_tokens, TOP_K, device=device, dtype=dtype).view(-1)

    # For FP8, quantize tensors
    if is_fp8:
        A_up_fp8, A_up_scale = quantize_fp8_rowwise(A_up_bf16)
        B_up_fp8, B_up_scale = quantize_fp8_colwise(B_up_bf16)
        A_down_fp8, A_down_scale = quantize_fp8_rowwise(A_down_bf16)
        B_down_fp8, B_down_scale = quantize_fp8_colwise(B_down_bf16)

    # Generate topk_ids once (same routing pattern for all block_m)
    topk_ids = torch.randint(0, NUM_EXPERTS, (num_tokens, TOP_K), device=device, dtype=torch.int32)

    combined_results: list[CombinedBenchResult] = []

    def compile_and_benchmark(config: CuteMoeConfig, kind: Literal["up", "down"],
                               sorted_token_ids, expert_ids, num_tokens_post_padded) -> BenchResult | None:
        """JIT compile a config, then acquire lock and benchmark it."""
        if is_fp8:
            A = A_up_fp8 if kind == "up" else A_down_fp8
            A_scale = A_up_scale if kind == "up" else A_down_scale
            B = B_up_fp8 if kind == "up" else B_down_fp8
            B_scale = B_up_scale if kind == "up" else B_down_scale
        else:
            A = A_up_bf16 if kind == "up" else A_down_bf16
            B = B_up_bf16 if kind == "up" else B_down_bf16
            A_scale = B_scale = None
        C = C_up if kind == "up" else C_down
        tw = None if kind == "up" else topk_weights

        # JIT compile (no lock needed, can run in parallel)
        error = jit_compile_config(
            config, kind,
            A=A, B=B, C=C,
            A_scale=A_scale, B_scale=B_scale,
            topk_weights=tw,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
        )
        if error is not None:
            return None

        # Benchmark (acquire lock if needed)
        if use_lock:
            with GPUBenchmarkLock():
                result = benchmark_compiled_config(
                    config, kind, num_tokens,
                    A=A, B=B, C=C,
                    A_scale=A_scale, B_scale=B_scale,
                    topk_weights=tw,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    warmup=warmup,
                    iters=iters,
                )
        else:
            result = benchmark_compiled_config(
                config, kind, num_tokens,
                A=A, B=B, C=C,
                A_scale=A_scale, B_scale=B_scale,
                topk_weights=tw,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                warmup=warmup,
                iters=iters,
            )

        if result.error is not None:
            return None
        print(f"          -> {result.time_us:.2f} us", flush=True)
        return result

    # Helper to resolve per-kernel-type search params (supports scalar list or dict form).
    def get_param_values(param_name: str, kernel_type: str) -> list[int]:
        values = search_space[param_name]
        if isinstance(values, dict):
            return values[kernel_type]
        return values

    # Test each kernel_type and block_m combination
    for kernel_type in search_space["kernel_type"]:
        block_m_values = search_space["block_m"][kernel_type]
        for block_m in block_m_values:
            print(f"\n  kernel_type={kernel_type}, block_m={block_m}", flush=True)

            # Generate shared routing for this block_m
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, block_m, NUM_EXPERTS
            )

            # Generate all configs for this block_m and kernel_type
            up_configs = []
            down_configs = []
            block_n_values = get_param_values("block_n", kernel_type)
            block_k_values = get_param_values("block_k", kernel_type)
            num_stages_values = get_param_values("num_stages", kernel_type)
            dtype_str = "fp8" if is_fp8 else "bf16"

            if kernel_type == "wgmma":
                # WGMMA variants: num_warps is derived from block_m
                num_warps = (block_m // 64) * 4
                for block_n, block_k, num_stages in itertools.product(
                    block_n_values,
                    block_k_values,
                    num_stages_values,
                ):
                    try:
                        config = CuteMoeConfig(
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            dtype=dtype_str,
                            kernel_type=kernel_type,
                        )
                    except ValueError:
                        continue
                    up_configs.append(config)
                    down_configs.append(config)
            else:
                # Warp/QMMA kernels: num_warps is taken from search-space.
                num_warps_values = get_param_values("num_warps", kernel_type)
                for block_n, block_k, num_warps, num_stages in itertools.product(
                    block_n_values,
                    block_k_values,
                    num_warps_values,
                    num_stages_values,
                ):
                    try:
                        config = CuteMoeConfig(
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            dtype=dtype_str,
                            kernel_type=kernel_type,
                        )
                    except ValueError:
                        continue
                    if is_valid_config(config, "up", num_tokens):
                        up_configs.append(config)
                    if is_valid_config(config, "down", num_tokens):
                        down_configs.append(config)

            # Remove duplicates (include kernel_type in key)
            up_configs = list({(c.block_n, c.block_k, c.num_warps, c.num_stages, c.kernel_type): c for c in up_configs}.values())
            down_configs = list({(c.block_n, c.block_k, c.num_warps, c.num_stages, c.kernel_type): c for c in down_configs}.values())

            print(f"    UP: {len(up_configs)} configs, DOWN: {len(down_configs)} configs", flush=True)

            if not up_configs or not down_configs:
                print(f"    Skipping: no valid configs", flush=True)
                continue

            # Compile and benchmark UP configs
            print(f"    Testing UP configs...", flush=True)
            up_results = []
            for config in up_configs:
                result = compile_and_benchmark(config, "up", sorted_token_ids, expert_ids, num_tokens_post_padded)
                if result is not None:
                    up_results.append(result)

            if not up_results:
                print(f"    No valid UP results", flush=True)
                continue
            best_up = min(up_results, key=lambda r: r.time_us)
            print(f"    Best UP: {best_up.time_us:.2f} us (n={best_up.config.block_n} k={best_up.config.block_k} "
                  f"w={best_up.config.num_warps} s={best_up.config.num_stages} t={best_up.config.kernel_type})", flush=True)

            # Compile and benchmark DOWN configs
            print(f"    Testing DOWN configs...", flush=True)
            down_results = []
            for config in down_configs:
                result = compile_and_benchmark(config, "down", sorted_token_ids, expert_ids, num_tokens_post_padded)
                if result is not None:
                    down_results.append(result)

            if not down_results:
                print(f"    No valid DOWN results", flush=True)
                continue
            best_down = min(down_results, key=lambda r: r.time_us)
            print(f"    Best DOWN: {best_down.time_us:.2f} us (n={best_down.config.block_n} k={best_down.config.block_k} "
                  f"w={best_down.config.num_warps} s={best_down.config.num_stages} t={best_down.config.kernel_type})", flush=True)

            total_time = best_up.time_us + best_down.time_us
            print(f"    TOTAL: {total_time:.2f} us", flush=True)

            combined_results.append(CombinedBenchResult(
                block_m=block_m,
                num_tokens=num_tokens,
                up_config=best_up.config,
                up_time_us=best_up.time_us,
                up_std_us=best_up.std_us,
                down_config=best_down.config,
                down_time_us=best_down.time_us,
                down_std_us=best_down.std_us,
                total_time_us=total_time,
            ))

    return combined_results


def format_config_json(config: CuteMoeConfig) -> dict:
    """Format config as JSON-ready dict."""
    return {
        "block_m": config.block_m,
        "block_n": config.block_n,
        "block_k": config.block_k,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "kernel_type": config.kernel_type,
    }


def run_single_token(args) -> None:
    """Run grid search for a single token count (used by parallel launcher)."""
    import traceback
    print(f"[PID {os.getpid()}] Starting run_single_token", flush=True)

    if not torch.cuda.is_available():
        print(f"[PID {os.getpid()}] CUDA not available!", flush=True)
        raise SystemExit("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    num_tokens = args.token_counts[0]
    kernel_dtype = getattr(args, 'kernel_dtype', 'bf16')
    arch = _current_cuda_arch()

    # Select search space based on kernel dtype and token count
    if kernel_dtype == "fp8":
        search_space = get_fp8_search_space(num_tokens, arch=arch, quick=args.quick)
    else:
        search_space = QUICK_SEARCH_SPACE if args.quick else FULL_SEARCH_SPACE

    print(f"[PID {os.getpid()}] CuTe MoE Grid Search: tokens={num_tokens}", flush=True)
    print(f"  Device: {torch.cuda.get_device_name()}", flush=True)
    print(f"  Arch: {arch}", flush=True)
    print(f"  Data dtype: {dtype}", flush=True)
    print(f"  Kernel dtype: {kernel_dtype}", flush=True)
    print(f"  Search space: {'quick' if args.quick else 'full'}", flush=True)
    print(f"  Warmup: {args.warmup}, Iters: {args.iters}", flush=True)
    print(flush=True)

    total_start = time.time()

    try:
        print(f"[PID {os.getpid()}] Calling run_combined_grid_search...", flush=True)
        results = run_combined_grid_search(
            num_tokens, search_space,
            device=device, dtype=dtype,
            kernel_dtype=kernel_dtype,
            warmup=args.warmup, iters=args.iters,
            use_lock=True,  # Use lock for parallel runs
        )
        print(f"[PID {os.getpid()}] run_combined_grid_search returned {len(results)} results", flush=True)
    except Exception as e:
        print(f"[PID {os.getpid()}] EXCEPTION in run_combined_grid_search:", flush=True)
        traceback.print_exc()
        raise

    total_time = time.time() - total_start

    if results:
        best = min(results, key=lambda r: r.total_time_us)
        print(f"\nBest for tokens={num_tokens}:")
        print(f"  block_m={best.block_m}")
        print(f"  UP:   {best.up_time_us:.2f} us - m={best.up_config.block_m} n={best.up_config.block_n} k={best.up_config.block_k} "
              f"w={best.up_config.num_warps} s={best.up_config.num_stages} t={best.up_config.kernel_type}")
        print(f"  DOWN: {best.down_time_us:.2f} us - m={best.down_config.block_m} n={best.down_config.block_n} k={best.down_config.block_k} "
              f"w={best.down_config.num_warps} s={best.down_config.num_stages} t={best.down_config.kernel_type}")
        print(f"  TOTAL: {best.total_time_us:.2f} us")

        # Machine-readable summary line for easy extraction
        print(f"\n### RESULT tokens={num_tokens} ###")
        print(f"UP_CONFIG: m={best.up_config.block_m} n={best.up_config.block_n} k={best.up_config.block_k} "
              f"w={best.up_config.num_warps} s={best.up_config.num_stages} t={best.up_config.kernel_type} time={best.up_time_us:.2f}")
        print(f"DOWN_CONFIG: m={best.down_config.block_m} n={best.down_config.block_n} k={best.down_config.block_k} "
              f"w={best.down_config.num_warps} s={best.down_config.num_stages} t={best.down_config.kernel_type} time={best.down_time_us:.2f}")

        # Save results
        if args.output:
            output_data = {
                "num_tokens": num_tokens,
                "device": torch.cuda.get_device_name(),
                "dtype": str(dtype),
                "total_time_s": total_time,
                "best": {
                    "block_m": best.block_m,
                    "up_config": format_config_json(best.up_config),
                    "up_time_us": best.up_time_us,
                    "down_config": format_config_json(best.down_config),
                    "down_time_us": best.down_time_us,
                    "total_time_us": best.total_time_us,
                },
            }
            Path(args.output).write_text(json.dumps(output_data, indent=2))
            print(f"\nResults saved to: {args.output}")
    else:
        print(f"\nNo valid configs found!")

    print(f"\nTotal time: {total_time:.1f}s")


def launch_parallel(args) -> None:
    """Launch token counts as parallel processes with limited concurrency."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old lock file
    if _LOCK_FILE.exists():
        _LOCK_FILE.unlink()

    token_counts = args.token_counts or TOKEN_COUNTS
    max_parallel = args.max_parallel

    print(f"Launching {len(token_counts)} processes ({max_parallel} at a time)...")
    print(f"  Token counts: {token_counts}")
    print(f"  Output dir: {output_dir}")
    print()

    # Track running and pending processes
    pending = list(token_counts)
    running = []  # (num_tokens, proc, log_file)
    completed_count = 0

    def launch_one(num_tokens: int):
        """Launch a single process for a token count."""
        output_file = output_dir / f"tokens{num_tokens}.json"
        log_file = output_dir / f"tokens{num_tokens}.log"

        cmd = [
            sys.executable, __file__,
            "--num-tokens", str(num_tokens),
            "--output", str(output_file),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--dtype", args.dtype,
            "--kernel-dtype", getattr(args, 'kernel_dtype', 'bf16'),
        ]
        if args.quick:
            cmd.append("--quick")

        log = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"  Started: tokens={num_tokens} (PID {proc.pid})")
        return (num_tokens, proc, log_file, log)

    # Initial batch
    while pending and len(running) < max_parallel:
        num_tokens = pending.pop(0)
        running.append(launch_one(num_tokens))

    print(f"\nMonitor progress: tail -f {output_dir}/*.log")
    print(f"Or check completed: ls -la {output_dir}/*.json")
    print()

    # Process completions and launch new ones
    while running:
        for i, (num_tokens, proc, log_file, log_handle) in enumerate(running):
            ret = proc.poll()
            if ret is not None:
                completed_count += 1
                log_handle.close()
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                print(f"  [{completed_count}/{len(token_counts)}] tokens={num_tokens}: {status}")
                running.pop(i)

                # Launch next one if any pending
                if pending:
                    next_tokens = pending.pop(0)
                    running.append(launch_one(next_tokens))
                break
        else:
            time.sleep(1)

    print(f"\nAll processes completed. Aggregating results...")

    # Aggregate results
    aggregate_results(output_dir)


def aggregate_results(output_dir: Path) -> None:
    """Aggregate results from all parallel processes."""
    best_configs: dict[int, dict] = {}

    for json_file in sorted(output_dir.glob("tokens*.json")):
        try:
            data = json.loads(json_file.read_text())
            num_tokens = data.get("num_tokens")
            if num_tokens and "best" in data:
                best_configs[num_tokens] = data["best"]
        except Exception as e:
            print(f"  Warning: Failed to read {json_file}: {e}")

    if not best_configs:
        print("No results found!")
        return

    # Print JSON-ready summary
    print("\n# Best configs for JSON file:")
    print(f"# File: {_config_filename_for_current_gpu(dtype='bf16')}")
    print("{")
    print(f'  "description": "{_config_description_for_current_gpu()}",')

    # Print up configs
    print('  "up": {')
    for i, num_tokens in enumerate(sorted(best_configs.keys())):
        best = best_configs[num_tokens]
        cfg = best["up_config"]
        time_us = best["up_time_us"]
        comma = "," if i < len(best_configs) - 1 else ""
        print(f'    "{num_tokens}": {{"block_m": {cfg["block_m"]}, "block_n": {cfg["block_n"]}, '
              f'"block_k": {cfg["block_k"]}, "num_warps": {cfg["num_warps"]}, '
              f'"num_stages": {cfg["num_stages"]}, "kernel_type": "{cfg["kernel_type"]}", "time_us": {time_us:.1f}}}{comma}')
    print('  },')

    # Print down configs
    print('  "down": {')
    for i, num_tokens in enumerate(sorted(best_configs.keys())):
        best = best_configs[num_tokens]
        cfg = best["down_config"]
        time_us = best["down_time_us"]
        comma = "," if i < len(best_configs) - 1 else ""
        print(f'    "{num_tokens}": {{"block_m": {cfg["block_m"]}, "block_n": {cfg["block_n"]}, '
              f'"block_k": {cfg["block_k"]}, "num_warps": {cfg["num_warps"]}, '
              f'"num_stages": {cfg["num_stages"]}, "kernel_type": "{cfg["kernel_type"]}", "time_us": {time_us:.1f}}}{comma}')
    print('  }')
    print("}")

    # Write aggregated results
    aggregate_file = output_dir / "aggregate.json"
    aggregate_data = {
        "description": _config_description_for_current_gpu(),
        "up": {
            str(t): {**best_configs[t]["up_config"], "time_us": best_configs[t]["up_time_us"]}
            for t in sorted(best_configs.keys())
        },
        "down": {
            str(t): {**best_configs[t]["down_config"], "time_us": best_configs[t]["down_time_us"]}
            for t in sorted(best_configs.keys())
        },
    }
    aggregate_file.write_text(json.dumps(aggregate_data, indent=2))
    print(f"\nAggregated results saved to: {aggregate_file}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Tokens':>6} | {'UP Time':>8} | {'UP Config':^40} | {'DOWN Time':>9} | {'DOWN Config':^40}")
    print("-" * 100)
    for num_tokens in sorted(best_configs.keys()):
        best = best_configs[num_tokens]
        up = best["up_config"]
        down = best["down_config"]
        up_str = f"m={up['block_m']} n={up['block_n']} k={up['block_k']} w={up['num_warps']} s={up['num_stages']} {up['kernel_type']}"
        down_str = f"m={down['block_m']} n={down['block_n']} k={down['block_k']} w={down['num_warps']} s={down['num_stages']} {down['kernel_type']}"
        print(f"{num_tokens:>6} | {best['up_time_us']:>7.1f}us | {up_str:<40} | {best['down_time_us']:>8.1f}us | {down_str:<40}")
    print("=" * 100)


def run_sequential(args) -> None:
    """Run grid search sequentially for all token counts."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    kernel_dtype = getattr(args, 'kernel_dtype', 'bf16')
    arch = _current_cuda_arch()
    token_counts = args.token_counts or TOKEN_COUNTS

    # For BF16, select search space once (not token-dependent)
    # For FP8, we select inside the loop based on token count
    bf16_search_space = QUICK_SEARCH_SPACE if args.quick else FULL_SEARCH_SPACE

    print(f"CuTe MoE Combined Grid Search")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Arch: {arch}")
    print(f"  Data dtype: {dtype}")
    print(f"  Kernel dtype: {kernel_dtype}")
    print(f"  Token counts: {token_counts}")
    print(f"  Search space: {'quick' if args.quick else 'full (token-aware for FP8)'}")
    print(f"  Model: E={NUM_EXPERTS}, H={D_MODEL}, I={D_EXPERT}, top_k={TOP_K}")
    print()

    # Results indexed by token count
    best_results: dict[int, CombinedBenchResult] = {}

    total_start = time.time()

    for num_tokens in token_counts:
        print(f"\n{'='*60}")
        print(f"Token count: {num_tokens}")
        print(f"{'='*60}")

        # Select search space (FP8 uses token-count-aware pruning)
        if kernel_dtype == "fp8":
            search_space = get_fp8_search_space(num_tokens, arch=arch, quick=args.quick)
        else:
            search_space = bf16_search_space

        results = run_combined_grid_search(
            num_tokens, search_space,
            device=device, dtype=dtype,
            kernel_dtype=kernel_dtype,
            warmup=args.warmup, iters=args.iters,
        )

        if results:
            # Find best combined result (minimum total time)
            best = min(results, key=lambda r: r.total_time_us)
            best_results[num_tokens] = best

            print(f"\n  Best for tokens={num_tokens}:")
            print(f"    block_m={best.block_m}")
            print(f"    UP:   {best.up_time_us:.2f} us - n={best.up_config.block_n} k={best.up_config.block_k} "
                  f"w={best.up_config.num_warps} s={best.up_config.num_stages}")
            print(f"    DOWN: {best.down_time_us:.2f} us - n={best.down_config.block_n} k={best.down_config.block_k} "
                  f"w={best.down_config.num_warps} s={best.down_config.num_stages}")
            print(f"    TOTAL: {best.total_time_us:.2f} us")
        else:
            print(f"  No valid configs found!")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Print JSON-ready summary
    print("\n\n# Best configs for JSON file:")
    print(f"# File: {_config_filename_for_current_gpu(dtype=kernel_dtype)}")
    print("{")
    print(f'  "description": "{_config_description_for_current_gpu()}",')

    # Print up configs
    print('  "up": {')
    for i, num_tokens in enumerate(sorted(best_results.keys())):
        best = best_results[num_tokens]
        cfg = best.up_config
        comma = "," if i < len(best_results) - 1 else ""
        print(f'    "{num_tokens}": {{"block_m": {cfg.block_m}, "block_n": {cfg.block_n}, '
              f'"block_k": {cfg.block_k}, "num_warps": {cfg.num_warps}, '
              f'"num_stages": {cfg.num_stages}, "kernel_type": "{cfg.kernel_type}", "time_us": {best.up_time_us:.1f}}}{comma}')
    print('  },')

    # Print down configs
    print('  "down": {')
    for i, num_tokens in enumerate(sorted(best_results.keys())):
        best = best_results[num_tokens]
        cfg = best.down_config
        comma = "," if i < len(best_results) - 1 else ""
        print(f'    "{num_tokens}": {{"block_m": {cfg.block_m}, "block_n": {cfg.block_n}, '
              f'"block_k": {cfg.block_k}, "num_warps": {cfg.num_warps}, '
              f'"num_stages": {cfg.num_stages}, "kernel_type": "{cfg.kernel_type}", "time_us": {best.down_time_us:.1f}}}{comma}')
    print('  }')
    print("}")

    # Save to JSON if requested
    if args.output:
        output_data = {
            "description": _config_description_for_current_gpu(),
            "device": torch.cuda.get_device_name(),
            "dtype": str(dtype),
            "model": {
                "num_experts": NUM_EXPERTS,
                "hidden_size": D_MODEL,
                "intermediate_size": D_EXPERT,
                "top_k": TOP_K,
            },
            "search_space": search_space,
            "total_time_s": total_time,
            "up": {
                str(t): {
                    **format_config_json(best_results[t].up_config),
                    "time_us": best_results[t].up_time_us,
                }
                for t in sorted(best_results.keys())
            },
            "down": {
                str(t): {
                    **format_config_json(best_results[t].down_config),
                    "time_us": best_results[t].down_time_us,
                }
                for t in sorted(best_results.keys())
            },
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CuTe MoE grid search")
    parser.add_argument("--launch-parallel", action="store_true",
                        help="Launch all token counts as parallel processes")
    parser.add_argument("--max-parallel", type=int, default=5,
                        help="Max concurrent processes for --launch-parallel (default: 5)")
    parser.add_argument("--num-tokens", type=int, action="append", dest="token_counts",
                        help="Token counts to test (can repeat)")
    parser.add_argument("--output", type=str, help="Write results to JSON file")
    parser.add_argument("--output-dir", type=str, default="/tmp/cute_moe_grid_search",
                        help="Directory for parallel output files")
    parser.add_argument("--quick", action="store_true", help="Use smaller search space")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                        help="Data dtype for tensors")
    parser.add_argument("--kernel-dtype", choices=["bf16", "fp8"], default="bf16",
                        help="Kernel dtype: bf16 for BF16 WGMMA, fp8 for FP8 WGMMA")
    args = parser.parse_args()

    if args.launch_parallel:
        launch_parallel(args)
    elif args.token_counts and len(args.token_counts) == 1:
        # Single token mode (used by parallel launcher)
        run_single_token(args)
    else:
        # Sequential mode
        run_sequential(args)


if __name__ == "__main__":
    main()
