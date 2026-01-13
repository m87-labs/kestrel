"""FP8 quantization kernel dispatch and loading.

This module provides the public API for FP8 row-wise quantization and handles
loading precompiled kernels at runtime.
"""

import torch

from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module


# Precompiled kernel registry: (hidden, warps_per_block, use_pdl) -> kernel_fn
_precompiled_cache: dict = {}


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


def _get_warps_per_block(hidden: int, num_rows: int) -> int:
    """Choose warps per block based on hidden size and batch size.

    Benchmarked crossover points:
    - hidden=1024: w=1 better up to ~512 rows, w=8 better above
    - hidden=2048: w=1 always better (more register pressure favors fewer warps)
    """
    if hidden >= 2048:
        # More registers per thread -> always use 1 warp/block
        return 1
    elif num_rows <= 512:
        return 1
    else:
        return 8


def _load_precompiled_kernel(hidden: int, warps_per_block: int, use_pdl: bool = False):
    """Load a precompiled kernel if available, return None otherwise."""
    cache_key = (hidden, warps_per_block, use_pdl)

    # Check if already loaded
    if cache_key in _precompiled_cache:
        return _precompiled_cache[cache_key]

    # Build filename
    arch = get_cuda_arch()
    pdl_suffix = "_pdl" if use_pdl else ""
    filename = f"fp8_quant_bfloat16_h{hidden}_w{warps_per_block}{pdl_suffix}_{arch}.so"
    function_name = f"fp8_quant_bfloat16_h{hidden}_w{warps_per_block}{pdl_suffix}_{arch}"

    # Load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    kernel_fn = getattr(mod, function_name)
    _precompiled_cache[cache_key] = kernel_fn
    return kernel_fn


def _fp8_quant_cute_impl(
    out_bits: torch.Tensor,
    out_scale: torch.Tensor,
    inp: torch.Tensor,
    use_pdl: bool = False,
) -> None:
    """Internal implementation that supports PDL parameter."""
    assert inp.is_cuda and out_bits.is_cuda and out_scale.is_cuda
    assert inp.dtype == torch.bfloat16
    assert out_bits.dtype == torch.uint8
    assert out_scale.dtype == torch.float32
    assert inp.is_contiguous() and out_bits.is_contiguous() and out_scale.is_contiguous()
    assert inp.dim() == 2 and out_bits.dim() == 2 and out_scale.dim() == 1
    assert inp.shape == out_bits.shape
    assert out_scale.shape[0] == inp.shape[0]

    num_rows, hidden = inp.shape

    if num_rows == 0:
        return

    warps_per_block = _get_warps_per_block(hidden, num_rows)

    precompiled = _load_precompiled_kernel(hidden, warps_per_block, use_pdl)
    if precompiled is None:
        arch = get_cuda_arch()
        pdl_str = ", use_pdl=True" if use_pdl else ""
        raise RuntimeError(
            f"No precompiled kernel for fp8_quant(hidden={hidden}, warps={warps_per_block}{pdl_str}, arch={arch}). "
            f"Run precompile on this architecture to generate it."
        )

    precompiled(out_bits, out_scale, inp, num_rows)


@torch.library.custom_op("kestrel::fp8_quant_cute", mutates_args={"out_bits", "out_scale"})
def _fp8_quant_cute(
    out_bits: torch.Tensor,
    out_scale: torch.Tensor,
    inp: torch.Tensor,
) -> None:
    """Custom op wrapper (no PDL support for torch.compile compatibility)."""
    _fp8_quant_cute_impl(out_bits, out_scale, inp, use_pdl=False)


def fp8_quant_cute(
    out_bits: torch.Tensor,
    out_scale: torch.Tensor,
    inp: torch.Tensor,
    *,
    use_pdl: bool = False,
) -> None:
    """FP8 row-wise quantization kernel.

    Converts BF16 tensor to FP8 (e4m3fn) with per-row dynamic scaling.

    Args:
        out_bits: Output FP8 tensor of shape [M, K], dtype uint8
        out_scale: Output scale tensor of shape [M], dtype float32
        inp: Input tensor of shape [M, K], dtype bfloat16
        use_pdl: Enable Programmatic Dependent Launch for overlapping with
            subsequent kernels (e.g., MoE matmul). Default False.
    """
    _fp8_quant_cute_impl(out_bits, out_scale, inp, use_pdl=use_pdl)


__all__ = ["fp8_quant_cute"]
