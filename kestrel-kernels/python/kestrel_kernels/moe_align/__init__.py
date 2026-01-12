"""MoE align block size kernel.

This package provides GPU-accelerated MoE token alignment operations.

Note: Kernel template source files (kernel.py) are not included in the
distributed wheel. They are only needed for JIT compilation during development.
"""

from kestrel_kernels.moe_align.dispatch import (
    MoeAlignCuTeConfig,
    moe_align_block_size,
    moe_lora_align_block_size,
)

__all__ = [
    "MoeAlignCuTeConfig",
    "moe_align_block_size",
    "moe_lora_align_block_size",
]
