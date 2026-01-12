"""TopK kernel with bitonic sort.

This package provides a GPU-accelerated top-k operation using bitonic sort.

Note: Kernel template source files (kernel.py) are not included in the
distributed wheel. They are only needed for JIT compilation during development.
"""

from kestrel_kernels.topk.dispatch import topk_fwd

__all__ = ["topk_fwd"]
