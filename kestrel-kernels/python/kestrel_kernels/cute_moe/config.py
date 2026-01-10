"""CuTe MoE kernel configuration and config loading."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class CuteMoeConfig:
    # Tile sizes (tuned for Moondream MoE shapes; adjust via benchmarking).
    block_m: int = 16
    block_n: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2
    dtype: str = "bf16"  # "bf16" or "fp8"
    kernel_type: str = "warp"  # "warp" or "wgmma"

    def __post_init__(self) -> None:
        """Validate config constraints at construction time."""
        if self.dtype not in ("bf16", "fp8"):
            raise ValueError(f"dtype must be 'bf16' or 'fp8', got '{self.dtype}'")
        if self.kernel_type not in ("warp", "wgmma"):
            raise ValueError(f"kernel_type must be 'warp' or 'wgmma', got '{self.kernel_type}'")

        # CRITICAL: block_m must not exceed num_threads!
        # The kernel's metadata loading loop uses `if tx < block_m` to load per-row
        # metadata into shared memory. With only num_threads threads, rows beyond
        # num_threads-1 read uninitialized shared memory -> illegal memory access.
        if self.block_m > self.num_threads:
            raise ValueError(
                f"block_m ({self.block_m}) must not exceed num_threads ({self.num_threads}). "
                f"With {self.num_warps} warps, max block_m is {self.num_threads}."
            )

        if self.dtype == "fp8":
            self._validate_fp8()
        else:
            self._validate_bf16()

    def _validate_bf16(self) -> None:
        """Validate constraints for BF16 kernel based on kernel_type."""
        # block_k must be divisible by 16 (MMA K dimension for BF16).
        if self.block_k % 16 != 0:
            raise ValueError(
                f"block_k ({self.block_k}) must be divisible by 16 (BF16 MMA K dimension)."
            )

        if self.kernel_type == "wgmma":
            # WGMMA constraints
            if self.block_m % 64 != 0:
                raise ValueError(
                    f"BF16 WGMMA requires block_m ({self.block_m}) to be divisible by 64."
                )
            if self.block_k % 32 != 0:
                raise ValueError(
                    f"BF16 WGMMA requires block_k ({self.block_k}) to be divisible by 32."
                )
            # WGMMA requires (block_m // 64) warpgroups, each warpgroup = 4 warps
            required_warpgroups = self.block_m // 64
            required_warps = required_warpgroups * 4
            if self.num_warps != required_warps:
                raise ValueError(
                    f"BF16 WGMMA requires num_warps={required_warps} for block_m={self.block_m} "
                    f"(got {self.num_warps}). Each warpgroup (4 warps) handles 64 rows."
                )
        else:  # "warp"
            # Warp kernel constraints
            # block_m must be multiple of 16 (MMA atom M dimension)
            if self.block_m % 16 != 0:
                raise ValueError(
                    f"BF16 warp kernel requires block_m ({self.block_m}) to be divisible by 16."
                )
            # block_n must be divisible by 8 * num_warps (MMA layout constraint)
            mma_n_coverage = 8 * self.num_warps
            if self.block_n % mma_n_coverage != 0:
                raise ValueError(
                    f"block_n ({self.block_n}) must be divisible by 8 * num_warps ({mma_n_coverage})."
                )
            # block_n=32 with num_warps=8 fails LDSM alignment verification
            if self.block_n == 32 and self.num_warps == 8:
                raise ValueError(
                    "block_n=32 with num_warps=8 causes LDSM alignment verification failure."
                )

        # Shared memory constraint: sA + sB + metadata must fit in H100's 228KB.
        # BF16 = 2 bytes per element.
        bytes_per_elem = 2
        sA_bytes = self.num_stages * self.block_m * self.block_k * bytes_per_elem
        sB_bytes = self.num_stages * self.block_n * self.block_k * bytes_per_elem
        total_bytes = sA_bytes + sB_bytes
        max_bytes = 200 * 1024
        if total_bytes > max_bytes:
            raise ValueError(
                f"Shared memory exceeded: sA={sA_bytes // 1024}KB + sB={sB_bytes // 1024}KB "
                f"= {total_bytes // 1024}KB "
                f"(stages={self.num_stages}, m={self.block_m}, n={self.block_n}, k={self.block_k}). "
                f"Max ~{max_bytes // 1024}KB."
            )

    def _validate_fp8(self) -> None:
        """Validate constraints for FP8 kernels based on kernel_type."""
        if self.kernel_type == "wgmma":
            # FP8 WGMMA requires block_m to be a multiple of 64 (one warpgroup per 64 rows).
            if self.block_m % 64 != 0:
                raise ValueError(
                    f"FP8 WGMMA requires block_m ({self.block_m}) to be divisible by 64."
                )

            # FP8 WGMMA requires block_k divisible by 32 (K dimension alignment for FP8 MMA).
            if self.block_k % 32 != 0:
                raise ValueError(
                    f"FP8 WGMMA requires block_k ({self.block_k}) to be divisible by 32."
                )

            # FP8 WGMMA: num_warps must match warpgroups = block_m // 64, each warpgroup = 4 warps.
            required_warpgroups = self.block_m // 64
            required_warps = required_warpgroups * 4
            if self.num_warps != required_warps:
                raise ValueError(
                    f"FP8 WGMMA constraint: block_m={self.block_m} requires {required_warpgroups} "
                    f"warpgroup(s) = {required_warps} warps, but num_warps={self.num_warps}."
                )

            # Register pressure constraint: m=256 n=256 exceeds register limit.
            if self.block_m >= 256 and self.block_n >= 256:
                raise ValueError(
                    f"FP8 WGMMA: block_m={self.block_m} with block_n={self.block_n} "
                    f"exceeds register limit. Use smaller tiles."
                )
        else:  # "warp"
            # FP8 warp-level MMA kernel: block_m must be multiple of 16 (MMA atom M dimension)
            if self.block_m % 16 != 0:
                raise ValueError(
                    f"FP8 warp kernel requires block_m ({self.block_m}) to be divisible by 16."
                )
            # block_n must be divisible by 8 * num_warps (MMA layout constraint)
            # MMA atom N=8, warps tile (1, num_warps, 1) in N dimension
            mma_n_coverage = 8 * self.num_warps
            if self.block_n % mma_n_coverage != 0:
                raise ValueError(
                    f"FP8 warp kernel: block_n ({self.block_n}) must be divisible by "
                    f"8 * num_warps ({mma_n_coverage})."
                )
            # block_k should be divisible by 16 for warp-level MMA
            if self.block_k % 16 != 0:
                raise ValueError(
                    f"FP8 warp kernel requires block_k ({self.block_k}) to be divisible by 16."
                )

        # Shared memory constraint: sA + sB + metadata must fit in H100's 228KB.
        # FP8 = 1 byte per element (half of BF16).
        bytes_per_elem = 1
        sA_bytes = self.num_stages * self.block_m * self.block_k * bytes_per_elem
        sB_bytes = self.num_stages * self.block_n * self.block_k * bytes_per_elem
        total_bytes = sA_bytes + sB_bytes
        max_bytes = 200 * 1024
        if total_bytes > max_bytes:
            raise ValueError(
                f"Shared memory exceeded: sA={sA_bytes // 1024}KB + sB={sB_bytes // 1024}KB "
                f"= {total_bytes // 1024}KB "
                f"(stages={self.num_stages}, m={self.block_m}, n={self.block_n}, k={self.block_k}). "
                f"Max ~{max_bytes // 1024}KB."
            )

    @property
    def num_threads(self) -> int:
        return 32 * self.num_warps


# Config auto-loading from JSON files
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Cache for CUDA architecture string
_cuda_arch: str | None = None


def _get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper, 'sm100' for Blackwell)."""
    global _cuda_arch
    if _cuda_arch is None:
        major, minor = torch.cuda.get_device_capability()
        _cuda_arch = f"sm{major}{minor}"
    return _cuda_arch


@lru_cache(maxsize=None)
def _load_cute_moe_configs(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str,
    arch: str,
) -> dict | None:
    """Load configs for given model shape and hardware. Return None if not found."""
    filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
    config_file = _CONFIGS_DIR / filename
    if not config_file.exists():
        return None
    with open(config_file) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_cute_moe_block_m(
    num_tokens: int,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "bf16",
) -> int:
    """Get the block_m value for routing alignment.

    Both UP and DOWN kernels share the same block_m for a given token count,
    so this returns the common value used by moe_align_block_size.

    Args:
        num_tokens: Number of tokens (batch size)
        num_experts: Number of experts (E)
        hidden_size: Hidden/model dimension (H)
        intermediate_size: Expert intermediate dimension (I)
        dtype: Data type ("bf16" or "fp8")

    Returns:
        block_m value for routing alignment.

    Raises:
        ValueError: If no config file exists for the model shape + GPU arch.
    """
    arch = _get_cuda_arch()
    configs = _load_cute_moe_configs(num_experts, hidden_size, intermediate_size, dtype, arch)

    if configs is None:
        filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
        raise ValueError(
            f"No CuTe MoE configs for this model shape. "
            f"Expected file: {_CONFIGS_DIR / filename}"
        )

    # Use "up" config to get block_m (UP and DOWN have matching block_m)
    up_configs = configs.get("up", {})
    if not up_configs:
        raise ValueError("No 'up' configs in config file")

    # Find nearest token count
    token_keys = [int(k) for k in up_configs.keys()]
    nearest = min(token_keys, key=lambda t: abs(t - num_tokens))
    cfg = up_configs[str(nearest)]

    return cfg["block_m"]


@lru_cache(maxsize=None)
def get_cute_moe_config(
    kind: Literal["up", "down"],
    num_tokens: int,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "bf16",
) -> CuteMoeConfig:
    """Get optimal config for given parameters. Raises if not available.

    Args:
        kind: "up" or "down" kernel type
        num_tokens: Number of tokens (batch size)
        num_experts: Number of experts (E)
        hidden_size: Hidden/model dimension (H)
        intermediate_size: Expert intermediate dimension (I)
        dtype: Data type ("bf16" or "fp8")

    Returns:
        CuteMoeConfig with optimal tile sizes for the given parameters.

    Raises:
        ValueError: If no config file exists for the model shape + GPU arch.
    """
    arch = _get_cuda_arch()
    configs = _load_cute_moe_configs(num_experts, hidden_size, intermediate_size, dtype, arch)

    if configs is None:
        filename = f"cute_moe_E{num_experts}_H{hidden_size}_I{intermediate_size}_{dtype}_{arch}.json"
        raise ValueError(
            f"No CuTe MoE configs for this model shape. "
            f"Expected file: {_CONFIGS_DIR / filename}"
        )

    kind_configs = configs.get(kind, {})
    if not kind_configs:
        raise ValueError(f"No '{kind}' configs in config file")

    # Find nearest token count
    token_keys = [int(k) for k in kind_configs.keys()]
    nearest = min(token_keys, key=lambda t: abs(t - num_tokens))
    cfg = kind_configs[str(nearest)]

    return CuteMoeConfig(
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        block_k=cfg["block_k"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        dtype=dtype,
        kernel_type=cfg["kernel_type"],
    )
