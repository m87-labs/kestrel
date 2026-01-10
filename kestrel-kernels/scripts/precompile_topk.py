#!/usr/bin/env python3
"""Precompile topk kernel variants for AOT deployment."""

import math
import subprocess
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import BFloat16, Int32

# Import from the local package
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from kestrel_kernels.topk import TopK, make_fake_tensor


def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper, 'sm100' for Blackwell)."""
    import os
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list:
        # Parse first arch from env var (e.g., "9.0" -> "sm90", "9.0;8.0" -> "sm90")
        first_arch = arch_list.split(";")[0].strip()
        if "." in first_arch:
            major, minor = first_arch.split(".")
            return f"sm{major}{minor}"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


# Target kernel variants to precompile
# (dtype, N, k, softmax) - architecture is determined at compile time
PRECOMPILE_VARIANTS = [
    (BFloat16, 64, 8, True),
]


def get_kernel_filename(dtype, N: int, k: int, softmax: bool, arch: str) -> str:
    """Generate a canonical filename for a kernel variant."""
    dtype_name = dtype.__name__.lower()  # e.g., "bfloat16"
    softmax_str = "softmax" if softmax else "nosoftmax"
    return f"topk_{dtype_name}_n{N}_k{k}_{softmax_str}_{arch}.so"


def get_function_name(dtype, N: int, k: int, softmax: bool, arch: str) -> str:
    """Generate a canonical function name for a kernel variant."""
    dtype_name = dtype.__name__.lower()
    softmax_str = "softmax" if softmax else "nosoftmax"
    return f"topk_{dtype_name}_n{N}_k{k}_{softmax_str}_{arch}"


def precompile_variant(dtype, N: int, k: int, softmax: bool, arch: str, output_dir: Path):
    """Compile and export a single kernel variant."""
    print(f"Compiling: dtype={dtype.__name__}, N={N}, k={k}, softmax={softmax}, arch={arch}")

    # Create fake tensors for compilation (same as in topk.py)
    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, N)

    x_cute = make_fake_tensor(dtype, (batch_sym, N), div)
    values_cute = make_fake_tensor(dtype, (batch_sym, k), div)
    indices_cute = make_fake_tensor(Int32, (batch_sym, k), div)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Create and compile the kernel
    topk_op = TopK(dtype, N, k, softmax=softmax)
    compiled = cute.compile(
        topk_op,
        x_cute,
        values_cute,
        indices_cute,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    # Export to object file
    obj_path = output_dir / "topk_tmp.o"
    function_name = get_function_name(dtype, N, k, softmax, arch)
    compiled.export_to_c(str(obj_path), function_name=function_name)

    # Get runtime libraries and link to shared object
    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)

    so_filename = get_kernel_filename(dtype, N, k, softmax, arch)
    so_path = output_dir / so_filename

    # Link command
    cmd = [
        "gcc",
        "-shared",
        "-o",
        str(so_path),
        str(obj_path),
        *runtime_libs,
    ]
    print(f"Linking: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Clean up object file
    obj_path.unlink()

    print(f"Created: {so_path}")
    return so_path


def main():
    # Detect architecture
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    # Output directory for precompiled kernels
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "python" / "kestrel_kernels" / "precompiled"
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py in precompiled directory
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Precompiled CuTe DSL kernels."""\n')

    # Compile all variants for the current architecture
    for dtype, N, k, softmax in PRECOMPILE_VARIANTS:
        precompile_variant(dtype, N, k, softmax, arch, output_dir)

    print(f"\nPrecompilation complete. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
