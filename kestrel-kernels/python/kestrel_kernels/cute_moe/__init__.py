"""CuTe MoE kernel implementations for Hopper (SM90).

This package provides fused Mixture-of-Experts GEMM kernels using NVIDIA's
CuTe DSL with optimized memory access patterns for both BF16 and FP8 precision.

Note: Kernel template source files (cute_moe_*.py) are not included in the
distributed wheel. They are only needed for JIT compilation during development.
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

__all__ = [
    "CuteMoeConfig",
    "get_cute_moe_config",
    "get_cute_moe_block_m",
    "invoke_cute_moe_up",
    "invoke_cute_moe_down",
    "invoke_cute_moe_up_fp8",
    "invoke_cute_moe_down_fp8",
]
