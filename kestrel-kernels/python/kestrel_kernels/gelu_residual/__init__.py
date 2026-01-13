"""GELU residual activation kernel implementations."""

from kestrel_kernels.gelu_residual.dispatch import gelu_residual_cute

__all__ = [
    "gelu_residual_cute",
]
