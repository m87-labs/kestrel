"""Bitonic top-k kernel and helpers."""

import math
from functools import partial
from typing import Optional, Type

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, BFloat16, Float16, Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def make_fake_tensor(dtype, shape, divisibility=1) -> Optional[cute.Tensor]:
    if dtype is None:
        return None
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(*[cute.sym_int64(divisibility=divisibility)] * (len(shape) - 1), 1),
        assumed_align=divisibility * dtype.width // 8,
    )


@dsl_user_op
def fmin(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        nvvm.fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    num_copy_bits = min(128, num_copy_elems * src.element_type.width)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=num_copy_bits
    )
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_copy_bits
    )
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@cute.jit
def compare_and_swap(
    arr: cute.Tensor, i: int, j: int, ascending: bool = True, use_selection: bool = False
) -> None:
    if const_expr(use_selection):
        a, b = arr[i], arr[j]
        if (a > b) ^ (not ascending):
            arr[i] = b
            arr[j] = a
    else:
        min_fn = min if const_expr(arr.element_type != Float32) else fmin
        max_fn = max if const_expr(arr.element_type != Float32) else cute.arch.fmax
        if const_expr(ascending):
            arr[i], arr[j] = min_fn(arr[i], arr[j]), max_fn(arr[i], arr[j])
        else:
            arr[i], arr[j] = max_fn(arr[i], arr[j]), min_fn(arr[i], arr[j])


networks = {
    2: [[(0, 1)]],
    4: [
        [(0, 2), (1, 3)],
        [(0, 1), (2, 3)],
        [(1, 2)],
    ],
    8: [
        [(0, 2), (1, 3), (4, 6), (5, 7)],
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        [(2, 4), (3, 5)],
        [(1, 4), (3, 6)],
        [(1, 2), (3, 4), (5, 6)],
    ],
    16: [
        [(0, 13), (1, 12), (2, 15), (3, 14), (4, 8), (5, 6), (7, 11), (9, 10)],
        [(0, 5), (1, 7), (2, 9), (3, 4), (6, 13), (8, 14), (10, 15), (11, 12)],
        [(0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15)],
        [(0, 2), (1, 3), (4, 10), (5, 11), (6, 7), (8, 9), (12, 14), (13, 15)],
        [(1, 2), (3, 12), (4, 6), (5, 7), (8, 10), (9, 11), (13, 14)],
        [(1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14)],
        [(2, 4), (3, 6), (9, 12), (11, 13)],
        [(3, 5), (6, 8), (7, 9), (10, 12)],
        [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)],
        [(6, 7), (8, 9)],
    ],
    32: [
        [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31)],
        [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15), (16, 18), (17, 19), (20, 22), (21, 23), (24, 26), (25, 27), (28, 30), (29, 31)],
        [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 20), (17, 21), (18, 22), (19, 23), (24, 28), (25, 29), (26, 30), (27, 31)],
        [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (16, 24), (17, 25), (18, 26), (19, 27), (20, 28), (21, 29), (22, 30), (23, 31)],
        [(0, 16), (1, 8), (2, 4), (3, 12), (5, 10), (6, 9), (7, 14), (11, 13), (15, 31), (17, 24), (18, 20), (19, 28), (21, 26), (22, 25), (23, 30), (27, 29)],
        [(1, 2), (3, 5), (4, 8), (6, 22), (7, 11), (9, 25), (10, 12), (13, 14), (17, 18), (19, 21), (20, 24), (23, 27), (26, 28), (29, 30)],
        [(1, 17), (2, 18), (3, 19), (4, 20), (5, 10), (7, 23), (8, 24), (11, 27), (12, 28), (13, 29), (14, 30), (21, 26)],
        [(3, 17), (4, 16), (5, 21), (6, 18), (7, 9), (8, 20), (10, 26), (11, 23), (13, 25), (14, 28), (15, 27), (22, 24)],
        [(1, 4), (3, 8), (5, 16), (7, 17), (9, 21), (10, 22), (11, 19), (12, 20), (14, 24), (15, 26), (23, 28), (27, 30)],
        [(2, 5), (7, 8), (9, 18), (11, 17), (12, 16), (13, 22), (14, 20), (15, 19), (23, 24), (26, 29)],
        [(2, 4), (6, 12), (9, 16), (10, 11), (13, 17), (14, 18), (15, 22), (19, 25), (20, 21), (27, 29)],
        [(5, 6), (8, 12), (9, 10), (11, 13), (14, 16), (15, 17), (18, 20), (19, 23), (21, 22), (25, 26)],
        [(3, 5), (6, 7), (8, 9), (10, 12), (11, 14), (13, 16), (15, 18), (17, 20), (19, 21), (22, 23), (24, 25), (26, 28)],
        [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28)],
    ],
    64: [
        [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15), (16, 18), (17, 19), (20, 22), (21, 23), (24, 26), (25, 27), (28, 30), (29, 31), (32, 34), (33, 35), (36, 38), (37, 39), (40, 42), (41, 43), (44, 46), (45, 47), (48, 50), (49, 51), (52, 54), (53, 55), (56, 58), (57, 59), (60, 62), (61, 63)],
        [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39), (40, 41), (42, 43), (44, 45), (46, 47), (48, 49), (50, 51), (52, 53), (54, 55), (56, 57), (58, 59), (60, 61), (62, 63)],
        [(0, 52), (1, 2), (3, 55), (4, 48), (5, 6), (7, 51), (8, 60), (9, 10), (11, 63), (12, 56), (13, 14), (15, 59), (16, 32), (17, 18), (19, 35), (20, 24), (21, 22), (23, 27), (25, 26), (28, 44), (29, 30), (31, 47), (33, 34), (36, 40), (37, 38), (39, 43), (41, 42), (45, 46), (49, 50), (53, 54), (57, 58), (61, 62)],
        [(0, 20), (1, 53), (2, 54), (3, 23), (4, 28), (5, 49), (6, 50), (7, 31), (8, 36), (9, 61), (10, 62), (11, 39), (12, 16), (13, 57), (14, 58), (15, 19), (17, 33), (18, 34), (21, 25), (22, 26), (24, 52), (27, 55), (29, 45), (30, 46), (32, 56), (35, 59), (37, 41), (38, 42), (40, 60), (43, 63), (44, 48), (47, 51)],
        [(0, 4), (1, 21), (2, 22), (3, 7), (5, 29), (6, 30), (8, 12), (9, 37), (10, 38), (11, 15), (13, 17), (14, 18), (16, 20), (19, 23), (24, 32), (25, 53), (26, 54), (27, 35), (28, 36), (31, 39), (33, 57), (34, 58), (40, 44), (41, 61), (42, 62), (43, 47), (45, 49), (46, 50), (48, 52), (51, 55), (56, 60), (59, 63)],
        [(0, 8), (1, 5), (2, 6), (3, 11), (4, 12), (7, 15), (9, 13), (10, 14), (16, 40), (17, 21), (18, 22), (19, 43), (20, 44), (23, 47), (24, 28), (25, 33), (26, 34), (27, 31), (29, 37), (30, 38), (32, 36), (35, 39), (41, 45), (42, 46), (48, 56), (49, 53), (50, 54), (51, 59), (52, 60), (55, 63), (57, 61), (58, 62)],
        [(1, 9), (2, 10), (4, 8), (5, 13), (6, 14), (7, 11), (12, 48), (15, 51), (16, 24), (17, 41), (18, 42), (19, 27), (20, 28), (21, 45), (22, 46), (23, 31), (25, 29), (26, 30), (32, 40), (33, 37), (34, 38), (35, 43), (36, 44), (39, 47), (49, 57), (50, 58), (52, 56), (53, 61), (54, 62), (55, 59)],
        [(4, 16), (5, 9), (6, 10), (7, 19), (8, 24), (11, 27), (13, 49), (14, 50), (17, 25), (18, 26), (20, 32), (21, 29), (22, 30), (23, 35), (28, 40), (31, 43), (33, 41), (34, 42), (36, 52), (37, 45), (38, 46), (39, 55), (44, 56), (47, 59), (53, 57), (54, 58)],
        [(1, 4), (5, 17), (6, 18), (8, 16), (9, 25), (10, 26), (11, 19), (12, 24), (15, 27), (21, 33), (22, 34), (29, 41), (30, 42), (36, 48), (37, 53), (38, 54), (39, 51), (44, 52), (45, 57), (46, 58), (47, 55), (59, 62)],
        [(2, 8), (9, 17), (10, 18), (12, 20), (13, 25), (14, 26), (15, 23), (24, 32), (27, 35), (28, 36), (31, 39), (37, 49), (38, 50), (40, 48), (43, 51), (45, 53), (46, 54), (55, 61)],
        [(2, 4), (12, 16), (13, 21), (14, 22), (15, 19), (20, 24), (23, 27), (25, 33), (26, 34), (28, 32), (29, 37), (30, 38), (31, 35), (36, 40), (39, 43), (41, 49), (42, 50), (44, 48), (47, 51), (59, 61)],
        [(4, 16), (5, 20), (10, 40), (13, 17), (14, 18), (21, 25), (22, 26), (23, 53), (24, 28), (27, 31), (29, 33), (30, 34), (32, 36), (35, 39), (37, 41), (38, 42), (43, 58), (45, 49), (46, 50), (47, 59)],
        [(3, 17), (6, 36), (7, 21), (8, 32), (9, 24), (11, 41), (13, 28), (14, 44), (15, 45), (18, 48), (19, 49), (22, 52), (25, 29), (26, 30), (27, 57), (31, 55), (33, 37), (34, 38), (35, 50), (39, 54), (42, 56), (46, 60)],
        [(6, 20), (8, 16), (10, 24), (11, 25), (14, 28), (15, 29), (17, 33), (18, 32), (21, 37), (22, 36), (26, 42), (27, 41), (30, 46), (31, 45), (34, 48), (35, 49), (38, 52), (39, 53), (43, 57), (47, 55)],
        [(3, 18), (5, 8), (6, 12), (7, 22), (15, 21), (17, 32), (19, 33), (23, 37), (26, 40), (30, 44), (31, 46), (41, 56), (42, 48), (45, 60), (51, 57), (55, 58)],
        [(3, 16), (7, 20), (11, 26), (18, 24), (19, 25), (22, 28), (23, 29), (27, 33), (30, 36), (34, 40), (35, 41), (37, 52), (38, 44), (39, 45), (43, 56), (47, 60)],
        [(3, 9), (7, 13), (10, 16), (11, 17), (14, 20), (15, 30), (19, 34), (21, 36), (23, 38), (25, 40), (26, 32), (27, 42), (29, 44), (31, 37), (33, 48), (43, 49), (46, 52), (47, 53), (50, 56), (54, 60)],
        [(3, 8), (7, 10), (9, 12), (11, 18), (13, 14), (15, 24), (17, 22), (19, 28), (21, 26), (23, 25), (27, 34), (29, 36), (30, 32), (31, 33), (35, 44), (37, 42), (38, 40), (39, 48), (41, 46), (45, 52), (49, 50), (51, 54), (53, 56), (55, 60)],
        [(3, 6), (7, 12), (11, 16), (15, 17), (18, 20), (19, 24), (21, 22), (23, 30), (25, 32), (26, 28), (27, 29), (31, 38), (33, 40), (34, 36), (35, 37), (39, 44), (41, 42), (43, 45), (46, 48), (47, 52), (51, 56), (57, 60)],
        [(3, 5), (6, 8), (7, 9), (10, 12), (11, 13), (14, 16), (15, 18), (17, 20), (19, 21), (22, 24), (23, 26), (25, 28), (27, 30), (29, 32), (31, 34), (33, 36), (35, 38), (37, 40), (39, 41), (42, 44), (43, 46), (45, 48), (47, 49), (50, 52), (51, 53), (54, 56), (55, 57), (58, 60)],
        [(3, 4), (7, 8), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 36), (37, 38), (39, 40), (41, 42), (43, 44), (45, 46), (47, 48), (49, 50), (51, 52), (55, 56), (59, 60)],
    ],
}


@cute.jit
def optimal_sort(
    arr: cute.Tensor,
    n: cutlass.Constexpr[int],
    start: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = True,
) -> None:
    assert n in networks
    for level in networks[n]:
        for i, j in level:
            compare_and_swap(arr, start + i, start + j, ascending)


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


class TopK:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, k: int, softmax: bool = False):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width
        self.k = k
        self.softmax = softmax
        assert N == 2 ** int(math.log2(N)), "N must be a power of 2"
        assert k == 2 ** int(math.log2(k)), "k must be a power of 2"
        assert k <= 128
        assert N <= 4096

    def _threads_per_row(self):
        N = self.N
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tiled_copy(self):
        N = self.N
        vecsize = self.vecsize
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = tiled_copy_2d(self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mValues.element_type == self.dtype
        assert mIndices.element_type == Int32
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy()
        num_threads = tiled_copy.size
        self.kernel(mX, mValues, mIndices, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = None if is_even_N else predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy_fn = partial(copy, pred=tXpX)

        if tXcX[0][0] < shape[0]:
            copy_fn(tXgX, tXrX)
        tXrX_f32 = cute.make_fragment(tXrX.shape, Float32)
        tXrX_f32.store(tXrX.load().to(Float32))

        log_N = int(math.log2(self.N))
        idx_mask = (1 << log_N) - 1
        vecsize = const_expr(cute.size(tv_layout.shape[1]))
        tXrX_i32 = cute.recast_tensor(tXrX_f32, Int32)
        for i in cutlass.range(cute.size(tXrX_i32), unroll_full=True):
            col_idx = Int32(tXcX[i // vecsize][1] + i % vecsize)
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            encoded_idx = encoded_idx & idx_mask
            tXrX_i32[i] = (tXrX_i32[i] & ~idx_mask) | encoded_idx

        if const_expr(not is_even_N):
            fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        topk_vals = bitonic_topk(tXrX_f32, self.k, warp_width=threads_per_row)

        vecsize_out = const_expr(min(self.k, vecsize, 128 // mIndices.element_type.width))
        assert self.k % vecsize_out == 0
        nvec_per_thread = const_expr(cute.ceil_div(self.k, vecsize_out * threads_per_row))
        mask = cute.arch.WARP_SIZE - threads_per_row
        mask_and_clamp = mask << 8 | (cute.arch.WARP_SIZE - 1)
        topk_vals_split = cute.make_fragment((vecsize_out, nvec_per_thread), Float32)
        for i in cutlass.range(cute.ceil_div(self.k, vecsize_out), unroll_full=True):
            should_receive = tidx % threads_per_row == i % threads_per_row
            for v in cutlass.range(vecsize_out, unroll_full=True):
                if const_expr(threads_per_row > 1):
                    if i * vecsize_out + v < self.k:
                        val = cute.arch.shuffle_sync(
                            topk_vals[i * vecsize_out + v], offset=0, mask_and_clamp=mask_and_clamp
                        )
                        if should_receive:
                            topk_vals_split[v, i // threads_per_row] = val
                else:
                    topk_vals_split[v, i // threads_per_row] = topk_vals[i * vecsize_out + v]

        topk_vals_i32 = cute.recast_tensor(topk_vals_split, Int32)
        topk_indices = cute.make_fragment(topk_vals_i32.shape, Int32)
        for i in cutlass.range(cute.size(topk_vals_i32), unroll_full=True):
            encoded_idx = topk_vals_i32[i] & idx_mask
            topk_vals_i32[i] = topk_vals_i32[i] & ~idx_mask
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = Int32(col_idx & idx_mask)

        if const_expr(self.softmax):
            for i in cutlass.range(cute.size(topk_vals_split, mode=[1]), unroll_full=True):
                col = i * threads_per_row + tidx % threads_per_row
                if col >= self.k // vecsize_out:
                    for v in cutlass.range(vecsize_out, unroll_full=True):
                        topk_vals_split[v, i] = -Float32.inf
            max_val = cute.arch.shuffle_sync(topk_vals[0], offset=0, mask_and_clamp=mask_and_clamp)
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(
                topk_vals_split.load() * log2_e - (max_val * log2_e), fastmath=True
            )
            denom = cute.arch.warp_reduction_sum(
                exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
                threads_in_group=threads_per_row,
            )
            topk_vals_split.store(exp_x * cute.arch.rcp_approx(denom))

        topk_vals_out = cute.make_fragment_like(topk_vals_split, mValues.element_type)
        topk_vals_out.store(topk_vals_split.load().to(mValues.element_type))

        row = tXcX[0][0]
        if tiler_mn[0] == 0 or row < shape[0]:
            mValues_store = cute.tiled_divide(mValues[row, None], (vecsize_out,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (vecsize_out,))
            for i in cutlass.range(cute.size(topk_vals_out.shape, [1]), unroll_full=True):
                col = i * threads_per_row + tidx % threads_per_row
                if col < self.k // vecsize_out:
                    cute.autovec_copy(topk_vals_out[None, i], mValues_store[None, col])
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])


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
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = make_fake_tensor(dtype, (batch_sym, N), div)
        values_cute = make_fake_tensor(dtype, (batch_sym, k), div)
        indices_cute = make_fake_tensor(Int32, (batch_sym, k), div)
        topk_op = TopK(dtype, N, k, softmax=softmax)
        _compile_cache[compile_key] = cute.compile(
            topk_op,
            x_cute,
            values_cute,
            indices_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
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


__all__ = ["topk_fwd", "TopK"]
