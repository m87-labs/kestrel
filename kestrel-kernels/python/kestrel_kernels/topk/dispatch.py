"""TopK kernel dispatch and loading.

This module provides the public API for top-k operations and handles
loading precompiled kernels at runtime.
"""

import torch

from cutlass import BFloat16, Float16, Float32, Int32, Int64

from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}

_cute_dtype_to_name = {
    BFloat16: "bfloat16",
    Float16: "float16",
    Float32: "float32",
}

# Precompiled kernel registry
_precompiled_cache: dict = {}


def _load_precompiled_kernel(dtype, N: int, k: int, softmax: bool):
    """Load a precompiled kernel if available, return None otherwise."""
    compile_key = (dtype, N, k, softmax)

    # Check if already loaded
    if compile_key in _precompiled_cache:
        return _precompiled_cache[compile_key]

    # Build filename
    dtype_name = _cute_dtype_to_name.get(dtype)
    if dtype_name is None:
        return None

    arch = get_cuda_arch()
    softmax_str = "softmax" if softmax else "nosoftmax"
    filename = f"topk_{dtype_name}_n{N}_k{k}_{softmax_str}_{arch}.so"
    function_name = f"topk_{dtype_name}_n{N}_k{k}_{softmax_str}_{arch}"

    # Load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[compile_key] = kernel_fn
    return kernel_fn


_compile_cache: dict = {}


@torch.library.custom_op("kestrel::topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(
    x: torch.Tensor, k: int, softmax: bool, values: torch.Tensor, indices: torch.Tensor
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert k > 0 and k <= x.shape[1], "k must be positive and <= N"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    compile_key = (dtype, N, k, softmax)
    if compile_key not in _compile_cache:
        precompiled = _load_precompiled_kernel(dtype, N, k, softmax)
        if precompiled is None:
            dtype_name = _cute_dtype_to_name.get(dtype, dtype.__name__)
            arch = get_cuda_arch()
            raise RuntimeError(
                f"No precompiled kernel for topk(dtype={dtype_name}, N={N}, k={k}, "
                f"softmax={softmax}, arch={arch}). "
                f"Run precompile_topk.py on this architecture to generate it."
            )
        _compile_cache[compile_key] = precompiled
    _compile_cache[compile_key](x, values, indices)


def topk_fwd(x: torch.Tensor, k: int, softmax: bool = False):
    """Top-k with fused softmax using bitonic sort.

    Args:
        x: Input tensor of shape (M, N), N must be power of 2 and <= 4096
        k: Number of top elements, must be power of 2 and <= 128
        softmax: Whether to apply softmax to the top-k values

    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    M = x.size(0)
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    _topk_fwd(x, k, softmax, values, indices)
    return values, indices


__all__ = ["topk_fwd"]
