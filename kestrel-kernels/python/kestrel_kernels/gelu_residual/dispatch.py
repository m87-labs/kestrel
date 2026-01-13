"""GELU residual kernel dispatch and loading.

This module provides the public API for GELU residual activation and handles
loading precompiled kernels at runtime.
"""

import torch

from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module


# Precompiled kernel registry
_precompiled_cache: dict = {}


def _load_precompiled_kernel(hidden: int):
    """Load a precompiled kernel if available, return None otherwise."""
    cache_key = hidden

    # Check if already loaded
    if cache_key in _precompiled_cache:
        return _precompiled_cache[cache_key]

    # Build filename
    arch = get_cuda_arch()
    filename = f"gelu_residual_bfloat16_h{hidden}_{arch}.so"
    function_name = f"gelu_residual_bfloat16_h{hidden}_{arch}"

    # Load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[cache_key] = kernel_fn
    return kernel_fn


_compile_cache: dict = {}


@torch.library.custom_op("kestrel::gelu_residual_cute", mutates_args={"out"})
def _gelu_residual_cute(out: torch.Tensor, inp: torch.Tensor) -> None:
    assert inp.is_cuda and out.is_cuda
    assert inp.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert inp.is_contiguous() and out.is_contiguous()
    assert inp.dim() >= 2 and out.dim() == inp.dim()

    hidden = out.shape[-1]
    assert inp.shape[-1] == 2 * hidden

    num_tokens = inp.numel() // inp.shape[-1]

    cache_key = hidden
    if cache_key not in _compile_cache:
        precompiled = _load_precompiled_kernel(hidden)
        if precompiled is None:
            arch = get_cuda_arch()
            raise RuntimeError(
                f"No precompiled kernel for gelu_residual(hidden={hidden}, arch={arch}). "
                f"Run precompile on this architecture to generate it."
            )
        _compile_cache[cache_key] = precompiled

    _compile_cache[cache_key](
        out.view(num_tokens, hidden),
        inp.view(num_tokens, 2 * hidden),
    )


def gelu_residual_cute(out: torch.Tensor, inp: torch.Tensor) -> None:
    """GELU residual kernel for standard input layout.

    Computes: out = GELU(x) * (y + 1)
    where inp = [x0..x_{d-1}, y0..y_{d-1}] (x and y concatenated)

    Args:
        out: Output tensor of shape [num_tokens, hidden], BF16
        inp: Input tensor of shape [num_tokens, 2*hidden], BF16
    """
    _gelu_residual_cute(out, inp)


__all__ = ["gelu_residual_cute"]
