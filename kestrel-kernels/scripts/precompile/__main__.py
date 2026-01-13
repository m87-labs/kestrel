#!/usr/bin/env python3
"""Entry point for precompilation.

Usage:
    python -m scripts.precompile                    # Compile all kernels
    python -m scripts.precompile --kernels topk     # Compile only topk
    python -m scripts.precompile --kernels topk cute_moe  # Compile multiple
"""

import argparse
import sys

# Ensure torch is imported first (for libtorch.so)
import torch

from . import topk, cute_moe, moe_align, flash_attn, gelu_residual, fp8_quant
from .utils import get_cuda_arch


KERNEL_MODULES = {
    "topk": topk,
    "cute_moe": cute_moe,
    "moe_align": moe_align,
    "flash_attn": flash_attn,
    "gelu_residual": gelu_residual,
    "fp8_quant": fp8_quant,
}


def main():
    parser = argparse.ArgumentParser(
        description="Precompile CuTe DSL kernels for AOT deployment"
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=list(KERNEL_MODULES.keys()) + ["all"],
        default=["all"],
        help="Which kernel families to precompile (default: all)",
    )
    args = parser.parse_args()

    arch = get_cuda_arch()
    print(f"CUDA architecture: {arch}")
    print()

    # Determine which kernels to compile
    if "all" in args.kernels:
        kernels_to_compile = list(KERNEL_MODULES.keys())
    else:
        kernels_to_compile = args.kernels

    total_failed = 0

    for kernel_name in kernels_to_compile:
        print(f"{'=' * 60}")
        print(f"Precompiling: {kernel_name}")
        print(f"{'=' * 60}")

        module = KERNEL_MODULES[kernel_name]
        try:
            module.main()
        except SystemExit as e:
            if e.code != 0:
                total_failed += 1
        except Exception as e:
            print(f"Error precompiling {kernel_name}: {e}")
            total_failed += 1

        print()

    if total_failed > 0:
        print(f"\n{total_failed} kernel family(ies) had failures")
        sys.exit(1)

    print("All precompilation completed successfully!")


if __name__ == "__main__":
    main()
