#!/usr/bin/env python3
"""Precompile FP8 quantization kernel variants for AOT deployment."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import BFloat16

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from kestrel_kernels.fp8_quant_cute.kernel import (
    Fp8QuantSinglePass,
    Fp8QuantWarpPerRow,
    _to_cute_2d_bf16,
    _to_cute_2d_u8,
    _to_cute_1d_f32,
)

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link


@dataclass(frozen=True)
class Fp8QuantVariant:
    """A single FP8 quant kernel variant to precompile."""

    dtype: type
    hidden: int
    warps_per_block: int  # 1 for small batches, 8 for large batches

    def _dtype_name(self) -> str:
        return self.dtype.__name__.lower()

    def filename(self, arch: str) -> str:
        return f"fp8_quant_{self._dtype_name()}_h{self.hidden}_w{self.warps_per_block}_{arch}.so"

    def function_name(self, arch: str) -> str:
        return f"fp8_quant_{self._dtype_name()}_h{self.hidden}_w{self.warps_per_block}_{arch}"


# Target kernel variants to precompile
# Benchmarked crossover points:
# - hidden=1024: w=1 better up to ~512 rows, w=8 better above
# - hidden=2048: w=1 always better (more register pressure favors fewer warps)
PRECOMPILE_VARIANTS = [
    # hidden=2048 (MoE input) - always use w=1
    Fp8QuantVariant(BFloat16, 2048, 1),
    # hidden=1024 (after up projection) - w=1 for <=512, w=8 for >512
    Fp8QuantVariant(BFloat16, 1024, 1),
    Fp8QuantVariant(BFloat16, 1024, 8),
]


def _can_use_single_pass(hidden: int) -> bool:
    """Check if single-pass kernel can be used.

    Single-pass kernel supports:
    - hidden=1024: 4 vectors per lane (16 i32 registers)
    - hidden=2048: 8 vectors per lane (32 i32 registers)
    """
    if hidden % 8 != 0:
        return False
    num_vecs = hidden // 8
    vecs_per_lane = num_vecs // 32
    return vecs_per_lane in (4, 8)  # hidden=1024 or hidden=2048


def compile_variant(
    variant: Fp8QuantVariant, arch: str, output_dir: Path
) -> tuple[Fp8QuantVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    try:
        use_single_pass = _can_use_single_pass(variant.hidden)
        kernel_type = "single-pass" if use_single_pass else "two-pass"
        print(
            f"Compiling: dtype={variant.dtype.__name__}, hidden={variant.hidden}, "
            f"warps_per_block={variant.warps_per_block}, kernel={kernel_type}, arch={arch}"
        )

        # Create fake tensors for compilation
        num_rows = 1  # Dynamic, but need a concrete value for fake tensor
        inp_fake = torch.empty((num_rows, variant.hidden), dtype=torch.bfloat16, device="cuda")
        out_bits_fake = torch.empty((num_rows, variant.hidden), dtype=torch.uint8, device="cuda")
        out_scale_fake = torch.empty((num_rows,), dtype=torch.float32, device="cuda")

        mIn = _to_cute_2d_bf16(inp_fake)
        mOut = _to_cute_2d_u8(out_bits_fake)
        mScale = _to_cute_1d_f32(out_scale_fake)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Create and compile the kernel
        # Use single-pass for hidden=1024/2048, two-pass fallback for others
        if use_single_pass:
            kernel = Fp8QuantSinglePass(
                hidden=variant.hidden,
                dtype=variant.dtype,
                warps_per_block=variant.warps_per_block,
            )
        else:
            kernel = Fp8QuantWarpPerRow(
                hidden=variant.hidden,
                dtype=variant.dtype,
            )
        compiled = cute.compile(
            kernel, mOut, mScale, mIn, num_rows, stream_fake,
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "fp8_quant_tmp")

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

    print(f"\nFP8 quant precompilation complete:")
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
