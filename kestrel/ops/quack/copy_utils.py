from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op


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
