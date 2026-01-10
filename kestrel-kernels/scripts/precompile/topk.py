#!/usr/bin/env python3
"""Precompile topk kernel variants for AOT deployment."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import BFloat16, Int32

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from kestrel_kernels.topk import TopK, make_fake_tensor

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link


@dataclass(frozen=True)
class TopKVariant:
    """A single TopK kernel variant to precompile."""

    dtype: type
    N: int
    k: int
    softmax: bool

    def _dtype_name(self) -> str:
        return self.dtype.__name__.lower()

    def filename(self, arch: str) -> str:
        softmax_str = "softmax" if self.softmax else "nosoftmax"
        return f"topk_{self._dtype_name()}_n{self.N}_k{self.k}_{softmax_str}_{arch}.so"

    def function_name(self, arch: str) -> str:
        softmax_str = "softmax" if self.softmax else "nosoftmax"
        return f"topk_{self._dtype_name()}_n{self.N}_k{self.k}_{softmax_str}_{arch}"


# Target kernel variants to precompile
PRECOMPILE_VARIANTS = [
    TopKVariant(BFloat16, 64, 8, True),
]


def compile_variant(
    variant: TopKVariant, arch: str, output_dir: Path
) -> tuple[TopKVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    try:
        print(
            f"Compiling: dtype={variant.dtype.__name__}, N={variant.N}, "
            f"k={variant.k}, softmax={variant.softmax}, arch={arch}"
        )

        # Create fake tensors for compilation
        batch_sym = cute.sym_int()
        div = math.gcd(128 // variant.dtype.width, variant.N)

        x_cute = make_fake_tensor(variant.dtype, (batch_sym, variant.N), div)
        values_cute = make_fake_tensor(variant.dtype, (batch_sym, variant.k), div)
        indices_cute = make_fake_tensor(Int32, (batch_sym, variant.k), div)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Create and compile the kernel
        topk_op = TopK(variant.dtype, variant.N, variant.k, softmax=variant.softmax)
        compiled = cute.compile(
            topk_op,
            x_cute,
            values_cute,
            indices_cute,
            stream_fake,
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "topk_tmp")

        print(f"Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        return (variant, None, str(e))


def main():
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    output_dir = get_precompiled_dir()

    failed = []
    succeeded = []

    for variant in PRECOMPILE_VARIANTS:
        result_variant, so_path, error = compile_variant(variant, arch, output_dir)
        if error:
            failed.append((result_variant, error))
        else:
            succeeded.append((result_variant, so_path))

    print(f"\nTopK precompilation complete:")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed variants:")
        for variant, error in failed:
            print(f"  {variant}: {error}")
        sys.exit(1)

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
