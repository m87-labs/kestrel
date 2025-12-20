import math
from functools import partial
from typing import Type

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from .utils import predicate_k, fill_oob
from .copy_utils import tiled_copy_2d, copy
from .compile_utils import make_fake_tensor
from .cute_dsl_utils import torch2cute_dtype_map
from .sort.bitonic_sort import bitonic_topk


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
