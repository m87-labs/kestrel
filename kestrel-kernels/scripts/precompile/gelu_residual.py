#!/usr/bin/env python3
"""Precompile GELU residual kernel variants for AOT deployment."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import BFloat16

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from kestrel_kernels.gelu_residual.kernel import GeluResidualHighOccupancy, _to_cute_2d

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link


@dataclass(frozen=True)
class GeluResidualVariant:
    """A single GELU residual kernel variant to precompile."""

    dtype: type
    hidden: int

    def _dtype_name(self) -> str:
        return self.dtype.__name__.lower()

    def filename(self, arch: str) -> str:
        return f"gelu_residual_{self._dtype_name()}_h{self.hidden}_{arch}.so"

    def function_name(self, arch: str) -> str:
        return f"gelu_residual_{self._dtype_name()}_h{self.hidden}_{arch}"


# Target kernel variants to precompile
PRECOMPILE_VARIANTS = [
    GeluResidualVariant(BFloat16, 1024),  # Moondream MoE hidden dimension
]


def compile_variant(
    variant: GeluResidualVariant, arch: str, output_dir: Path
) -> tuple[GeluResidualVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    try:
        print(
            f"Compiling: dtype={variant.dtype.__name__}, hidden={variant.hidden}, arch={arch}"
        )

        # Create fake tensors for compilation
        num_tokens = 1  # Dynamic, but need a concrete value for fake tensor
        inp_fake = torch.empty((num_tokens, 2 * variant.hidden), dtype=torch.bfloat16, device="cuda")
        out_fake = torch.empty((num_tokens, variant.hidden), dtype=torch.bfloat16, device="cuda")

        mIn = _to_cute_2d(inp_fake)
        mOut = _to_cute_2d(out_fake)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Create and compile the kernel
        kernel = GeluResidualHighOccupancy(hidden=variant.hidden, dtype=variant.dtype)
        compiled = cute.compile(
            kernel, mOut, mIn, stream_fake,
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "gelu_residual_tmp")

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

    print(f"\nGELU residual precompilation complete:")
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
