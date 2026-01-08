"""CuTe MoE kernel implementations for Hopper (SM90).

This package provides fused Mixture-of-Experts GEMM kernels using NVIDIA's
CuTe DSL with optimized memory access patterns for both BF16 and FP8 precision.
"""

from kestrel_kernels.cute_moe.config import (
    CuteMoeConfig,
    get_cute_moe_config,
    get_cute_moe_block_m,
)
from kestrel_kernels.cute_moe.dispatch import (
    invoke_cute_moe_up,
    invoke_cute_moe_down,
    invoke_cute_moe_up_fp8,
    invoke_cute_moe_down_fp8,
)
from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_warp import _FusedMoeMatmulCuTe
from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_wgmma import (
    _FusedMoeMatmulCuTeWgmmaBf16,
    _should_use_wgmma_bf16,
)
from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_wgmma import _FusedMoeMatmulCuTeFp8
from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_warp import _FusedMoeMatmulCuTeWarpFp8

__all__ = [
    "CuteMoeConfig",
    "get_cute_moe_config",
    "get_cute_moe_block_m",
    "invoke_cute_moe_up",
    "invoke_cute_moe_down",
    "invoke_cute_moe_up_fp8",
    "invoke_cute_moe_down_fp8",
    "_FusedMoeMatmulCuTe",
    "_FusedMoeMatmulCuTeWgmmaBf16",
    "_FusedMoeMatmulCuTeFp8",
    "_FusedMoeMatmulCuTeWarpFp8",
    "_should_use_wgmma_bf16",
]
