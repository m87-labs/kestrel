"""Fused MoE kernels adapted from vLLM for single-GPU decode."""

from .module import FusedMoEModule, FusedMoEConfig

__all__ = ["FusedMoEModule", "FusedMoEConfig"]
