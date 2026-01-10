"""Shared utilities for precompilation scripts."""

import os
import subprocess
from pathlib import Path

import torch
import cutlass.cute as cute


def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper).

    Reads from TORCH_CUDA_ARCH_LIST environment variable if set,
    otherwise detects from the current CUDA device.
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list:
        # Parse first arch from env var (e.g., "9.0" -> "sm90", "9.0;8.0" -> "sm90")
        first_arch = arch_list.split(";")[0].strip()
        if "." in first_arch:
            major, minor = first_arch.split(".")
            return f"sm{major}{minor}"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def get_precompiled_dir() -> Path:
    """Return the precompiled kernels directory, creating it if needed."""
    output_dir = Path(__file__).parent.parent.parent / "python" / "kestrel_kernels" / "precompiled"
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py if it doesn't exist
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Precompiled CuTe DSL kernels."""\n')

    return output_dir


def compile_and_link(
    compiled,
    function_name: str,
    output_path: Path,
    temp_prefix: str = "kernel_tmp",
) -> Path:
    """Export compiled CuTe kernel to object file and link to shared object.

    Args:
        compiled: The compiled CuTe kernel object.
        function_name: The exported function name.
        output_path: Path for the output .so file.
        temp_prefix: Prefix for temporary object file.

    Returns:
        Path to the created .so file.
    """
    obj_path = output_path.parent / f"{temp_prefix}_{os.getpid()}.o"

    try:
        # Export to object file
        compiled.export_to_c(str(obj_path), function_name=function_name)

        # Get runtime libraries
        runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)

        # Link to shared object
        cmd = [
            "gcc",
            "-shared",
            "-o",
            str(output_path),
            str(obj_path),
            *runtime_libs,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return output_path
    finally:
        # Clean up temp object file
        if obj_path.exists():
            obj_path.unlink()
