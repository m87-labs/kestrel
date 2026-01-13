"""FP8 row-wise quantization kernel implementations (CuTe DSL)."""

from kestrel_kernels.fp8_quant_cute.dispatch import fp8_quant_cute

__all__ = [
    "fp8_quant_cute",
]
