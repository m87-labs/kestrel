import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from ..utils import fmin
from .utils import compare_and_swap
from .sorting_networks import optimal_sort


@cute.jit
def bitonic_merge(
    arr: cute.Tensor,
    n: Optional[cutlass.Constexpr[int]] = None,
    start: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = True,
) -> None:
    if const_expr(n is None):
        n = cute.size(arr.shape)
    if const_expr(n > 1):
        num_levels = int(math.log2(n))
        assert n == 2**num_levels, "n must be a power of 2"
        for level in cutlass.range_constexpr(num_levels):
            length = n >> level
            step = length // 2
            for i in cutlass.range(n // length, unroll_full=True):
                start_i = start + i * length
                for j in cutlass.range(step, unroll_full=True):
                    compare_and_swap(arr, start_i + j, start_i + j + step, ascending)


@cute.jit
def bitonic_sort(
    arr: cute.Tensor,
    n: Optional[cutlass.Constexpr[int]] = None,
    start: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = True,
) -> None:
    if const_expr(n is None):
        n = cute.size(arr.shape)
    assert n <= 128
    if const_expr(n > 1):
        if const_expr(n in [2, 4, 8, 16, 32, 64]):
            optimal_sort(arr, n, start, ascending)
        else:
            assert n % 2 == 0
            bitonic_sort(arr, n // 2, start, True)
            bitonic_sort(arr, n // 2, start + n // 2, False)
            bitonic_merge(arr, n, start, ascending)


@cute.jit
def bitonic_topk_merge(
    arr0: cute.Tensor,
    arr1: cute.Tensor,
    k: Optional[cutlass.Constexpr[int]] = None,
    start0: cutlass.Constexpr[int] = 0,
    start1: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = False,
) -> None:
    if const_expr(k is None):
        k = cute.size(arr0.shape)
    if const_expr(arr0.element_type == Float32):
        minmax_fn = fmin if ascending else cute.arch.fmax
    else:
        minmax_fn = min if ascending else max
    for i in cutlass.range(k, unroll_full=True):
        arr0[start0 + i] = minmax_fn(arr0[start0 + i], arr1[start1 + k - 1 - i])
    bitonic_merge(arr0, k, start0, ascending)


@cute.jit
def bitonic_topk(
    arr: cute.Tensor,
    k: cutlass.Constexpr[int],
    ascending: cutlass.Constexpr[bool] = False,
    warp_width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Tensor:
    assert arr.element_type in [Float32, Int32]
    n = cute.size(arr.shape)
    assert k == 1 << int(math.log2(k)), "k must be a power of 2"
    assert n % k == 0, "n must be divisible by k"
    topk_vals = cute.make_fragment(k, arr.element_type)
    for v in cutlass.range(k, unroll_full=True):
        topk_vals[v] = arr[v]
    bitonic_sort(topk_vals, ascending=ascending)
    for i in cutlass.range(1, n // k, unroll_full=True):
        other_vals = cute.make_fragment(k, arr.element_type)
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = arr[i * k + v]
        bitonic_sort(other_vals, ascending=ascending)
        bitonic_topk_merge(topk_vals, other_vals, ascending=ascending)
    for i in cutlass.range(int(math.log2(warp_width)), unroll_full=True):
        other_vals = cute.make_fragment(k, arr.element_type)
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = cute.arch.shuffle_sync_bfly(topk_vals[v], offset=1 << i)
        bitonic_topk_merge(topk_vals, other_vals, ascending=ascending)
    return topk_vals
