"""GELU residual activation kernel in CuTe DSL.

Computes: out = GELU(x) * (y + 1)
where input is [num_tokens, 2*d] split into x=[..., :d] and y=[..., d:]
and output is [num_tokens, d].

GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import math as mlir_math, llvm
from cutlass.cute.runtime import from_dlpack


@dsl_user_op
def cvt_bf16x2_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """Convert two Float32 values to packed bf16x2 (as Int32)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_streaming_b128(
    v0: Int32, v1: Int32, v2: Int32, v3: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    """Store 128 bits with cache streaming hint (st.global.cs.v4.b32).

    Bypasses L1 and marks for early L2 eviction - useful for write-only stores.
    """
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_b128(gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> tuple:
    """Load 128 bits with default caching (ld.global.v4.b32).

    Uses default L1+L2 caching behavior.
    Returns tuple of 4 Int32 values.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    v0 = Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    v1 = Int32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip))
    v2 = Int32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip))
    v3 = Int32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip))
    return (v0, v1, v2, v3)


@dsl_user_op
def cvt_f32x2_bf16x2(packed: Int32, *, loc=None, ip=None) -> tuple:
    """Convert packed bf16x2 to two Float32 values.

    Input: i32 containing two packed bf16 values [lo, hi]
    Output: (f32 lo, f32 hi)
    """
    # Use braces to create local scope, avoiding variable name conflicts
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 bf16_0, bf16_1;
            mov.b32 {bf16_0, bf16_1}, $2;
            cvt.f32.bf16 $0, bf16_0;
            cvt.f32.bf16 $1, bf16_1;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    lo = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    hi = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    return (lo, hi)


@dsl_user_op
def erf_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute erf(x) for Float32."""
    return Float32(
        mlir_math.erf(
            Float32(x).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """Compute a*b+c as fused multiply-add."""
    return Float32(
        mlir_math.fma(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )



@cute.jit
def gelu_f32(x: Float32) -> Float32:
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))

    Optimized to use FMA: 0.5*x + 0.5*x*erf(x*sqrt_half) = fma(0.5*erf_val, x, 0.5*x)
    """
    sqrt_half = Float32(0.7071067811865476)  # 1/sqrt(2)
    half = Float32(0.5)
    erf_val = erf_f32(x * sqrt_half)
    half_x = half * x
    # gelu = 0.5*x + 0.5*x*erf_val = fma(0.5*erf_val, x, 0.5*x)
    return fma_f32(half * erf_val, x, half_x)


class GeluResidualHighOccupancy:
    """GELU residual kernel optimized for high occupancy (prefill).

    Uses explicit vectorized loads, loop unrolling, and streaming stores.
    Processes 8 outputs per iteration with two 128-bit loads (one for x, one for y).
    """

    def __init__(
        self,
        hidden: int,
        dtype: Type[cutlass.Numeric] = BFloat16,
        num_threads: int = 128,
    ):
        self.hidden = hidden
        self.dtype = dtype
        self.num_threads = num_threads
        # 128-bit load = 8 bf16 values = 8 outputs per iteration
        self.out_per_vec = 8
        assert hidden % self.out_per_vec == 0
        self.num_vecs = hidden // self.out_per_vec

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mIn: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_tokens = mIn.shape[0]
        self.kernel(mOut, mIn).launch(
            grid=[num_tokens, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mOut: cute.Tensor, mIn: cute.Tensor):
        """High occupancy kernel with explicit vectorized loads and streaming stores."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        hidden = const_expr(self.hidden)
        num_threads = const_expr(self.num_threads)
        num_vecs = const_expr(self.num_vecs)

        in_row = mIn[bidx, None]
        out_row = mOut[bidx, None]
        in_ptr = in_row.iterator
        out_ptr = out_row.iterator

        # x region starts at offset 0, y region starts at offset hidden
        x_base = in_ptr.toint()
        y_base = in_ptr.toint() + hidden * 2  # 2 bytes per bf16

        for vec_off in cutlass.range_constexpr(num_vecs // num_threads):
            vec_idx = tidx + vec_off * num_threads

            # Compute pointers for this vector (8 bf16 = 16 bytes per vector)
            vec_offset = vec_idx * 8 * 2  # 8 elements * 2 bytes
            x_ptr = cute.make_ptr(self.dtype, x_base + vec_offset, cute.AddressSpace.gmem, assumed_align=16)
            y_ptr = cute.make_ptr(self.dtype, y_base + vec_offset, cute.AddressSpace.gmem, assumed_align=16)

            # Load 8 x values and 8 y values (each as 4 x i32 containing packed bf16x2)
            x0_packed, x1_packed, x2_packed, x3_packed = load_b128(x_ptr)
            y0_packed, y1_packed, y2_packed, y3_packed = load_b128(y_ptr)

            # Unpack bf16x2 to f32 pairs and compute GELU residual
            x0, x1 = cvt_f32x2_bf16x2(x0_packed); y0, y1 = cvt_f32x2_bf16x2(y0_packed)
            g0 = gelu_f32(x0); r0 = fma_f32(g0, y0, g0)
            g1 = gelu_f32(x1); r1 = fma_f32(g1, y1, g1)

            x2, x3 = cvt_f32x2_bf16x2(x1_packed); y2, y3 = cvt_f32x2_bf16x2(y1_packed)
            g2 = gelu_f32(x2); r2 = fma_f32(g2, y2, g2)
            g3 = gelu_f32(x3); r3 = fma_f32(g3, y3, g3)

            x4, x5 = cvt_f32x2_bf16x2(x2_packed); y4, y5 = cvt_f32x2_bf16x2(y2_packed)
            g4 = gelu_f32(x4); r4 = fma_f32(g4, y4, g4)
            g5 = gelu_f32(x5); r5 = fma_f32(g5, y5, g5)

            x6, x7 = cvt_f32x2_bf16x2(x3_packed); y6, y7 = cvt_f32x2_bf16x2(y3_packed)
            g6 = gelu_f32(x6); r6 = fma_f32(g6, y6, g6)
            g7 = gelu_f32(x7); r7 = fma_f32(g7, y7, g7)

            # Pack results to bf16x2 and store with streaming hint
            packed0 = cvt_bf16x2_f32(r0, r1)
            packed1 = cvt_bf16x2_f32(r2, r3)
            packed2 = cvt_bf16x2_f32(r4, r5)
            packed3 = cvt_bf16x2_f32(r6, r7)

            out_ptr_vec = cute.make_ptr(
                self.dtype,
                out_ptr.toint() + vec_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            store_streaming_b128(packed0, packed1, packed2, packed3, out_ptr_vec)


# Cache for compiled kernels
_compile_cache: dict = {}


def _to_cute_2d(t: "torch.Tensor") -> cute.Tensor:
    """Convert 2D tensor to CuTe tensor with dynamic first dimension."""
    return (
        from_dlpack(t.detach(), assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)  # stride[1] == 1 for row-major
        .mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order(), divisibility=8)
    )


def _load_precompiled_kernel(hidden: int):
    """Load a precompiled kernel if available, return None otherwise."""
    from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module

    arch = get_cuda_arch()
    filename = f"gelu_residual_bfloat16_h{hidden}_{arch}.so"
    function_name = f"gelu_residual_bfloat16_h{hidden}_{arch}"

    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    return getattr(mod, function_name)


def gelu_residual_cute(out: "torch.Tensor", inp: "torch.Tensor") -> None:
    """GELU residual kernel for standard input layout.

    Computes: out = GELU(x) * (y + 1)
    where inp = [x0..x_{d-1}, y0..y_{d-1}] (x and y concatenated)

    Args:
        out: Output tensor of shape [num_tokens, hidden], BF16
        inp: Input tensor of shape [num_tokens, 2*hidden], BF16
    """
    import torch

    assert inp.is_cuda and out.is_cuda
    assert inp.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert inp.is_contiguous() and out.is_contiguous()
    assert inp.dim() >= 2 and out.dim() == inp.dim()

    hidden = out.shape[-1]
    assert inp.shape[-1] == 2 * hidden

    num_tokens = inp.numel() // inp.shape[-1]

    cache_key = hidden

    if cache_key not in _compile_cache:
        # Try precompiled first
        precompiled = _load_precompiled_kernel(hidden)
        if precompiled is not None:
            _compile_cache[cache_key] = precompiled
        else:
            # Fall back to JIT compilation
            inp_view = inp.view(num_tokens, 2 * hidden)
            out_view = out.view(num_tokens, hidden)

            mIn = _to_cute_2d(inp_view)
            mOut = _to_cute_2d(out_view)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            kernel = GeluResidualHighOccupancy(hidden=hidden, dtype=BFloat16)
            _compile_cache[cache_key] = cute.compile(
                kernel, mOut, mIn, stream_fake,
                options="--enable-tvm-ffi"
            )

    _compile_cache[cache_key](
        out.view(num_tokens, hidden),
        inp.view(num_tokens, 2 * hidden),
    )


__all__ = [
    "GeluResidualHighOccupancy",
    "gelu_residual_cute",
]
