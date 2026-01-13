"""FP8 row-wise quantization kernel in CuTe DSL.

Converts BF16 tensors to FP8 (e4m3fn) with per-row dynamic scale computation.
Used for quantizing MoE activations before FP8 GEMM.

Algorithm (two passes):
1. Compute per-row absmax via warp reduction
2. scale = max(absmax / 448.0, 1e-6)
3. Quantize: out = clamp(in * inv_scale, -448, 448) -> FP8

Kernel variants:
- Warp-per-row: Better SM utilization for large batches
- Block-per-row: Better for small batches (1-8 rows)
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack


# Constants matching CUDA kernel
FP8_E4M3_MAX = 448.0
MIN_SCALE = 1e-6


@dsl_user_op
def load_b128(gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> tuple:
    """Load 128 bits with default caching (ld.global.v4.b32)."""
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
def store_b128(
    v0: Int32, v1: Int32, v2: Int32, v3: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    """Store 128 bits (st.global.v4.b32)."""
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvt_f32x2_bf16x2(packed: Int32, *, loc=None, ip=None) -> tuple:
    """Convert packed bf16x2 to two Float32 values."""
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
def abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute abs(x) using PTX."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max(a, b) using PTX."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def min_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min(a, b) using PTX."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_xor_f32(val: Float32, lane_mask: int, *, loc=None, ip=None) -> Float32:
    """Warp shuffle XOR for float32 values."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {lane_mask}, 0x1f, 0xffffffff;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute 1/x using PTX reciprocal (rcp.approx.f32 followed by Newton-Raphson)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_f32x2_e4m3x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """Convert two Float32 values to packed e4m3x2 (as Int32).

    Returns u16 packed into lower 16 bits of i32.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8x2;
                cvt.rn.satfinite.e4m3x2.f32 fp8x2, $2, $1;
                cvt.u32.u16 $0, fp8x2;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_u16x2(lo: Int32, hi: Int32, *, loc=None, ip=None) -> Int32:
    """Pack two u16 values (in lower 16 bits of each i32) into one u32.

    Result: hi[15:0] << 16 | lo[15:0]
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(lo).ir_value(loc=loc, ip=ip),
                Int32(hi).ir_value(loc=loc, ip=ip),
            ],
            "bfi.b32 $0, $2, $1, 16, 16;",
            "=r,r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def warp_reduce_max(val: Float32) -> Float32:
    """Butterfly reduction for max within a warp (32 threads)."""
    # Unrolled butterfly reduction
    other = shfl_xor_f32(val, 16)
    val = max_f32(val, other)
    other = shfl_xor_f32(val, 8)
    val = max_f32(val, other)
    other = shfl_xor_f32(val, 4)
    val = max_f32(val, other)
    other = shfl_xor_f32(val, 2)
    val = max_f32(val, other)
    other = shfl_xor_f32(val, 1)
    val = max_f32(val, other)
    return val


@cute.jit
def absmax_from_packed_bf16x2(p0: Int32, p1: Int32, p2: Int32, p3: Int32) -> Float32:
    """Compute absmax of 8 bf16 values from 4 packed bf16x2 words."""
    v0, v1 = cvt_f32x2_bf16x2(p0)
    v2, v3 = cvt_f32x2_bf16x2(p1)
    v4, v5 = cvt_f32x2_bf16x2(p2)
    v6, v7 = cvt_f32x2_bf16x2(p3)

    m = abs_f32(v0)
    m = max_f32(m, abs_f32(v1))
    m = max_f32(m, abs_f32(v2))
    m = max_f32(m, abs_f32(v3))
    m = max_f32(m, abs_f32(v4))
    m = max_f32(m, abs_f32(v5))
    m = max_f32(m, abs_f32(v6))
    m = max_f32(m, abs_f32(v7))
    return m


@cute.jit
def quantize_8_values(
    p0: Int32, p1: Int32, p2: Int32, p3: Int32, inv_scale: Float32
) -> tuple:
    """Quantize 8 bf16 values to FP8 e4m3, returns 2 packed u32s (8 bytes total)."""
    fp8_max = Float32(FP8_E4M3_MAX)
    neg_fp8_max = Float32(-FP8_E4M3_MAX)

    # Unpack, scale, clamp, and convert to FP8
    v0, v1 = cvt_f32x2_bf16x2(p0)
    s0 = min_f32(max_f32(v0 * inv_scale, neg_fp8_max), fp8_max)
    s1 = min_f32(max_f32(v1 * inv_scale, neg_fp8_max), fp8_max)
    q01 = cvt_f32x2_e4m3x2(s0, s1)  # 2 bytes in lower 16 bits

    v2, v3 = cvt_f32x2_bf16x2(p1)
    s2 = min_f32(max_f32(v2 * inv_scale, neg_fp8_max), fp8_max)
    s3 = min_f32(max_f32(v3 * inv_scale, neg_fp8_max), fp8_max)
    q23 = cvt_f32x2_e4m3x2(s2, s3)

    v4, v5 = cvt_f32x2_bf16x2(p2)
    s4 = min_f32(max_f32(v4 * inv_scale, neg_fp8_max), fp8_max)
    s5 = min_f32(max_f32(v5 * inv_scale, neg_fp8_max), fp8_max)
    q45 = cvt_f32x2_e4m3x2(s4, s5)

    v6, v7 = cvt_f32x2_bf16x2(p3)
    s6 = min_f32(max_f32(v6 * inv_scale, neg_fp8_max), fp8_max)
    s7 = min_f32(max_f32(v7 * inv_scale, neg_fp8_max), fp8_max)
    q67 = cvt_f32x2_e4m3x2(s6, s7)

    # Pack into two u32s: q01|q23 and q45|q67
    # Each qXY has 2 FP8 bytes in lower 16 bits
    out0 = pack_u16x2(q01, q23)  # bytes 0,1,2,3
    out1 = pack_u16x2(q45, q67)  # bytes 4,5,6,7

    return (out0, out1)


@dsl_user_op
def shfl_sync_f32(val: Float32, src_lane: int, *, loc=None, ip=None) -> Float32:
    """Broadcast float32 value from src_lane to all lanes in warp."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {src_lane}, 0x1f, 0xffffffff;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class Fp8QuantWarpPerRow:
    """FP8 quantization kernel with one warp per row.

    Better for large batches where we have many rows to process.
    Each warp handles one row, with 8 warps (rows) per block.
    """

    WARPS_PER_BLOCK = 8
    WARP_SIZE = 32

    def __init__(
        self,
        hidden: int,
        dtype: Type[cutlass.Numeric] = BFloat16,
    ):
        self.hidden = hidden
        self.dtype = dtype
        # 8 bf16 values per 128-bit vector
        self.vec_size = 8
        assert hidden % self.vec_size == 0
        self.num_vecs = hidden // self.vec_size

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,      # [M, K] uint8
        mScale: cute.Tensor,    # [M] float32
        mIn: cute.Tensor,       # [M, K] bfloat16
        num_rows: Int32,
        stream: cuda.CUstream,
    ):
        blocks = (num_rows + self.WARPS_PER_BLOCK - 1) // self.WARPS_PER_BLOCK
        self.kernel(mOut, mScale, mIn, num_rows).launch(
            grid=[blocks, 1, 1],
            block=[self.WARPS_PER_BLOCK * self.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mScale: cute.Tensor,
        mIn: cute.Tensor,
        num_rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        hidden = const_expr(self.hidden)
        num_vecs = const_expr(self.num_vecs)
        warp_size = const_expr(self.WARP_SIZE)
        warps_per_block = const_expr(self.WARPS_PER_BLOCK)

        warp_id = tidx // warp_size
        lane_id = tidx % warp_size
        row = bidx * warps_per_block + warp_id

        # Guard all work with bounds check (no early return allowed in CuTe DSL)
        if row < num_rows:
            # Get row pointers
            in_row = mIn[row, None]
            out_row = mOut[row, None]
            in_ptr = in_row.iterator.toint()
            out_ptr = out_row.iterator.toint()

            # Pass 1: Compute per-row absmax
            thread_max = Float32(0.0)

            for vec_idx in range(lane_id, num_vecs, warp_size):
                vec_offset = vec_idx * 8 * 2  # 8 bf16 values * 2 bytes
                ptr = cute.make_ptr(
                    self.dtype, in_ptr + vec_offset, cute.AddressSpace.gmem, assumed_align=16
                )
                p0, p1, p2, p3 = load_b128(ptr)
                local_max = absmax_from_packed_bf16x2(p0, p1, p2, p3)
                thread_max = max_f32(thread_max, local_max)

            # Warp reduction
            row_max = warp_reduce_max(thread_max)

            # Compute scale
            fp8_max = Float32(FP8_E4M3_MAX)
            min_scale = Float32(MIN_SCALE)
            scale = row_max / fp8_max
            scale = max_f32(scale, min_scale)
            inv_scale = rcp_f32(scale)

            # Lane 0 writes scale
            if lane_id == 0:
                scale_ptr = cute.make_ptr(
                    Float32, mScale.iterator.toint() + row * 4, cute.AddressSpace.gmem, assumed_align=4
                )
                store_f32(scale, scale_ptr)

            # Pass 2: Quantize and store
            for vec_idx in range(lane_id, num_vecs, warp_size):
                vec_offset = vec_idx * 8 * 2  # 8 bf16 * 2 bytes
                in_vec_ptr = cute.make_ptr(
                    self.dtype, in_ptr + vec_offset, cute.AddressSpace.gmem, assumed_align=16
                )
                p0, p1, p2, p3 = load_b128(in_vec_ptr)

                # Quantize 8 values -> 8 bytes
                q0, q1 = quantize_8_values(p0, p1, p2, p3, inv_scale)

                # Store 8 bytes (as two 4-byte stores, or use 64-bit store)
                out_vec_ptr = cute.make_ptr(
                    Uint8, out_ptr + vec_idx * 8, cute.AddressSpace.gmem, assumed_align=8
                )
                # Store as u64 (8 bytes)
                store_u64_from_u32x2(q0, q1, out_vec_ptr)


class Fp8QuantSinglePass:
    """Single-pass FP8 quantization kernel that stores data in registers.

    For hidden=1024: 4 vectors per lane (16 i32 registers)
    For hidden=2048: 8 vectors per lane (32 i32 registers)

    Load data into registers, compute absmax, broadcast scale via shfl, quantize.
    This avoids the second global memory read at the cost of register pressure.

    For small batches (< 256 rows), uses 1 warp per block to maximize block count.
    For large batches, uses 8 warps per block for better cache utilization.
    """

    WARP_SIZE = 32

    def __init__(
        self,
        hidden: int,
        dtype: Type[cutlass.Numeric] = BFloat16,
        warps_per_block: int = 8,
    ):
        self.hidden = hidden
        self.dtype = dtype
        self.vec_size = 8
        self.warps_per_block = warps_per_block
        assert hidden % self.vec_size == 0
        self.num_vecs = hidden // self.vec_size
        assert self.num_vecs % self.WARP_SIZE == 0
        self.vecs_per_lane = self.num_vecs // self.WARP_SIZE
        # Support hidden=1024 (4 vecs/lane) and hidden=2048 (8 vecs/lane)
        assert self.vecs_per_lane in (4, 8), f"Unsupported vecs_per_lane={self.vecs_per_lane}"

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mScale: cute.Tensor,
        mIn: cute.Tensor,
        num_rows: Int32,
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr[bool] = False,
    ):
        blocks = (num_rows + self.warps_per_block - 1) // self.warps_per_block
        self.kernel(mOut, mScale, mIn, num_rows, use_pdl).launch(
            grid=[blocks, 1, 1],
            block=[self.warps_per_block * self.WARP_SIZE, 1, 1],
            stream=stream,
            use_pdl=use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mScale: cute.Tensor,
        mIn: cute.Tensor,
        num_rows: Int32,
        use_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        warp_size = const_expr(self.WARP_SIZE)
        warps_per_block = const_expr(self.warps_per_block)
        vecs_per_lane = const_expr(self.vecs_per_lane)

        warp_id = tidx // warp_size
        lane_id = tidx % warp_size
        row = bidx * warps_per_block + warp_id

        # PDL: Signal dependent kernels can start launching early.
        # They will wait() before reading our output, so this is safe.
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if row < num_rows:
            in_row = mIn[row, None]
            out_row = mOut[row, None]
            in_ptr = in_row.iterator.toint()
            out_ptr = out_row.iterator.toint()

            thread_max = Float32(0.0)

            # Load vectors 0-3 (always present for both hidden=1024 and hidden=2048)
            vec0_offset = lane_id * 8 * 2
            ptr0 = cute.make_ptr(self.dtype, in_ptr + vec0_offset, cute.AddressSpace.gmem, assumed_align=16)
            v0_p0, v0_p1, v0_p2, v0_p3 = load_b128(ptr0)
            thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v0_p0, v0_p1, v0_p2, v0_p3))

            vec1_offset = (lane_id + warp_size) * 8 * 2
            ptr1 = cute.make_ptr(self.dtype, in_ptr + vec1_offset, cute.AddressSpace.gmem, assumed_align=16)
            v1_p0, v1_p1, v1_p2, v1_p3 = load_b128(ptr1)
            thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v1_p0, v1_p1, v1_p2, v1_p3))

            vec2_offset = (lane_id + 2 * warp_size) * 8 * 2
            ptr2 = cute.make_ptr(self.dtype, in_ptr + vec2_offset, cute.AddressSpace.gmem, assumed_align=16)
            v2_p0, v2_p1, v2_p2, v2_p3 = load_b128(ptr2)
            thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v2_p0, v2_p1, v2_p2, v2_p3))

            vec3_offset = (lane_id + 3 * warp_size) * 8 * 2
            ptr3 = cute.make_ptr(self.dtype, in_ptr + vec3_offset, cute.AddressSpace.gmem, assumed_align=16)
            v3_p0, v3_p1, v3_p2, v3_p3 = load_b128(ptr3)
            thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v3_p0, v3_p1, v3_p2, v3_p3))

            # Load vectors 4-7 for hidden=2048 (8 vecs/lane) - compile-time specialized
            if const_expr(self.vecs_per_lane == 8):
                vec4_offset = (lane_id + 4 * warp_size) * 8 * 2
                ptr4 = cute.make_ptr(self.dtype, in_ptr + vec4_offset, cute.AddressSpace.gmem, assumed_align=16)
                v4_p0, v4_p1, v4_p2, v4_p3 = load_b128(ptr4)
                thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v4_p0, v4_p1, v4_p2, v4_p3))

                vec5_offset = (lane_id + 5 * warp_size) * 8 * 2
                ptr5 = cute.make_ptr(self.dtype, in_ptr + vec5_offset, cute.AddressSpace.gmem, assumed_align=16)
                v5_p0, v5_p1, v5_p2, v5_p3 = load_b128(ptr5)
                thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v5_p0, v5_p1, v5_p2, v5_p3))

                vec6_offset = (lane_id + 6 * warp_size) * 8 * 2
                ptr6 = cute.make_ptr(self.dtype, in_ptr + vec6_offset, cute.AddressSpace.gmem, assumed_align=16)
                v6_p0, v6_p1, v6_p2, v6_p3 = load_b128(ptr6)
                thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v6_p0, v6_p1, v6_p2, v6_p3))

                vec7_offset = (lane_id + 7 * warp_size) * 8 * 2
                ptr7 = cute.make_ptr(self.dtype, in_ptr + vec7_offset, cute.AddressSpace.gmem, assumed_align=16)
                v7_p0, v7_p1, v7_p2, v7_p3 = load_b128(ptr7)
                thread_max = max_f32(thread_max, absmax_from_packed_bf16x2(v7_p0, v7_p1, v7_p2, v7_p3))

            # Warp reduction
            row_max = warp_reduce_max(thread_max)

            # Compute scale and broadcast via shfl
            fp8_max = Float32(FP8_E4M3_MAX)
            min_scale = Float32(MIN_SCALE)
            scale = row_max / fp8_max
            scale = max_f32(scale, min_scale)

            if lane_id == 0:
                scale_ptr = cute.make_ptr(
                    Float32, mScale.iterator.toint() + row * 4, cute.AddressSpace.gmem, assumed_align=4
                )
                store_f32(scale, scale_ptr)

            # All lanes have the same row_max after warp_reduce_max (butterfly broadcasts)
            # so each lane can compute inv_scale locally - no shuffle needed
            inv_scale = rcp_f32(scale)

            # Quantize and store vectors 0-3
            q0_0, q0_1 = quantize_8_values(v0_p0, v0_p1, v0_p2, v0_p3, inv_scale)
            out0_ptr = cute.make_ptr(Uint8, out_ptr + lane_id * 8, cute.AddressSpace.gmem, assumed_align=8)
            store_u64_from_u32x2(q0_0, q0_1, out0_ptr)

            q1_0, q1_1 = quantize_8_values(v1_p0, v1_p1, v1_p2, v1_p3, inv_scale)
            out1_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
            store_u64_from_u32x2(q1_0, q1_1, out1_ptr)

            q2_0, q2_1 = quantize_8_values(v2_p0, v2_p1, v2_p2, v2_p3, inv_scale)
            out2_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 2 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
            store_u64_from_u32x2(q2_0, q2_1, out2_ptr)

            q3_0, q3_1 = quantize_8_values(v3_p0, v3_p1, v3_p2, v3_p3, inv_scale)
            out3_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 3 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
            store_u64_from_u32x2(q3_0, q3_1, out3_ptr)

            # Quantize and store vectors 4-7 for hidden=2048 - compile-time specialized
            if const_expr(self.vecs_per_lane == 8):
                q4_0, q4_1 = quantize_8_values(v4_p0, v4_p1, v4_p2, v4_p3, inv_scale)
                out4_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 4 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
                store_u64_from_u32x2(q4_0, q4_1, out4_ptr)

                q5_0, q5_1 = quantize_8_values(v5_p0, v5_p1, v5_p2, v5_p3, inv_scale)
                out5_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 5 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
                store_u64_from_u32x2(q5_0, q5_1, out5_ptr)

                q6_0, q6_1 = quantize_8_values(v6_p0, v6_p1, v6_p2, v6_p3, inv_scale)
                out6_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 6 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
                store_u64_from_u32x2(q6_0, q6_1, out6_ptr)

                q7_0, q7_1 = quantize_8_values(v7_p0, v7_p1, v7_p2, v7_p3, inv_scale)
                out7_ptr = cute.make_ptr(Uint8, out_ptr + (lane_id + 7 * warp_size) * 8, cute.AddressSpace.gmem, assumed_align=8)
                store_u64_from_u32x2(q7_0, q7_1, out7_ptr)


@dsl_user_op
def store_u64_from_u32x2(lo: Int32, hi: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store 64 bits from two u32 values with streaming (cache-bypass)."""
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(lo).ir_value(loc=loc, ip=ip),
            Int32(hi).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.v2.b32 [$0], {$1, $2};",  # cs = cache streaming
        "l,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_f32(val: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store a single float32 to global memory."""
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class Fp8QuantBlockPerRow:
    """FP8 quantization kernel with one block per row.

    Better for small batches where we want full block parallelism per row.
    Uses shared memory for block-wide reduction.
    """

    NUM_THREADS = 256
    WARP_SIZE = 32

    def __init__(
        self,
        hidden: int,
        dtype: Type[cutlass.Numeric] = BFloat16,
    ):
        self.hidden = hidden
        self.dtype = dtype
        self.vec_size = 8
        assert hidden % self.vec_size == 0
        self.num_vecs = hidden // self.vec_size
        self.num_warps = self.NUM_THREADS // self.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mScale: cute.Tensor,
        mIn: cute.Tensor,
        num_rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mOut, mScale, mIn, num_rows).launch(
            grid=[num_rows, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mScale: cute.Tensor,
        mIn: cute.Tensor,
        num_rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        hidden = const_expr(self.hidden)
        num_vecs = const_expr(self.num_vecs)
        num_threads = const_expr(self.NUM_THREADS)
        warp_size = const_expr(self.WARP_SIZE)
        num_warps = const_expr(self.num_warps)

        row = bidx
        warp_id = tidx // warp_size
        lane_id = tidx % warp_size

        # Get row pointers
        in_row = mIn[row, None]
        out_row = mOut[row, None]
        in_ptr = in_row.iterator.toint()
        out_ptr = out_row.iterator.toint()

        # Shared memory for inter-warp reduction
        # warp_maxes: one float per warp for reduction
        # scale_broadcast: one float to broadcast scale to all threads
        @cute.struct
        class SharedStorage:
            warp_maxes: cute.struct.Align[cute.struct.MemRange[Float32, self.num_warps], 16]
            scale_broadcast: cute.struct.Align[cute.struct.MemRange[Float32, 1], 16]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)
        warp_maxes = storage.warp_maxes.get_tensor(cute.make_layout((self.num_warps,)))
        scale_broadcast = storage.scale_broadcast.get_tensor(cute.make_layout((1,)))

        # Pass 1: Compute per-row absmax
        thread_max = Float32(0.0)

        for vec_off in cutlass.range_constexpr(num_vecs // num_threads):
            vec_idx = tidx + vec_off * num_threads
            vec_offset = vec_idx * 8 * 2
            ptr = cute.make_ptr(
                self.dtype, in_ptr + vec_offset, cute.AddressSpace.gmem, assumed_align=16
            )
            p0, p1, p2, p3 = load_b128(ptr)
            local_max = absmax_from_packed_bf16x2(p0, p1, p2, p3)
            thread_max = max_f32(thread_max, local_max)

        # Warp reduction first
        warp_max = warp_reduce_max(thread_max)

        # Store warp max to shared memory
        if lane_id == 0:
            warp_maxes[warp_id] = warp_max

        cute.arch.sync_threads()

        # Thread 0 does final reduction
        if tidx == 0:
            row_max = Float32(0.0)
            for i in cutlass.range(num_warps, unroll_full=True):
                row_max = max_f32(row_max, warp_maxes[i])

            # Compute and store scale
            fp8_max = Float32(FP8_E4M3_MAX)
            min_scale_val = Float32(MIN_SCALE)
            scale = row_max / fp8_max
            scale = max_f32(scale, min_scale_val)

            # Store scale to output and shared memory
            scale_ptr = cute.make_ptr(
                Float32, mScale.iterator.toint() + row * 4, cute.AddressSpace.gmem, assumed_align=4
            )
            store_f32(scale, scale_ptr)
            scale_broadcast[0] = scale

        cute.arch.sync_threads()

        # All threads read scale
        scale = scale_broadcast[0]
        inv_scale = rcp_f32(scale)

        # Pass 2: Quantize and store
        for vec_off in cutlass.range_constexpr(num_vecs // num_threads):
            vec_idx = tidx + vec_off * num_threads
            vec_offset = vec_idx * 8 * 2

            in_vec_ptr = cute.make_ptr(
                self.dtype, in_ptr + vec_offset, cute.AddressSpace.gmem, assumed_align=16
            )
            p0, p1, p2, p3 = load_b128(in_vec_ptr)

            q0, q1 = quantize_8_values(p0, p1, p2, p3, inv_scale)

            out_vec_ptr = cute.make_ptr(
                Uint8, out_ptr + vec_idx * 8, cute.AddressSpace.gmem, assumed_align=8
            )
            store_u64_from_u32x2(q0, q1, out_vec_ptr)


# Cache for compiled kernels: (hidden, warps_per_block) -> compiled kernel
_compile_cache: dict = {}


def _to_cute_2d_bf16(t: "torch.Tensor") -> cute.Tensor:
    """Convert 2D bf16 tensor to CuTe tensor."""
    return (
        from_dlpack(t.detach(), assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order(), divisibility=8)
    )


def _to_cute_2d_u8(t: "torch.Tensor") -> cute.Tensor:
    """Convert 2D uint8 tensor to CuTe tensor."""
    return (
        from_dlpack(t.detach(), assumed_align=8, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order(), divisibility=8)
    )


def _to_cute_1d_f32(t: "torch.Tensor") -> cute.Tensor:
    """Convert 1D f32 tensor to CuTe tensor with dynamic shape."""
    return (
        from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True)
        .mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order(), divisibility=1)
    )


def _can_use_single_pass(hidden: int) -> bool:
    """Check if single-pass kernel can be used for given hidden dimension.

    Single-pass kernel supports:
    - hidden=1024: 4 vectors per lane (16 i32 registers)
    - hidden=2048: 8 vectors per lane (32 i32 registers)
    """
    if hidden % 8 != 0:
        return False
    num_vecs = hidden // 8
    vecs_per_lane = num_vecs // 32
    return vecs_per_lane in (4, 8)  # hidden=1024 or hidden=2048


def _get_warps_per_block(num_rows: int, hidden: int) -> int:
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


def fp8_quant_cute(
    out_bits: "torch.Tensor",
    out_scale: "torch.Tensor",
    inp: "torch.Tensor",
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
    import torch

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

    use_single_pass = _can_use_single_pass(hidden)

    if use_single_pass:
        # Single-pass kernel - choose warps_per_block based on batch size
        warps_per_block = _get_warps_per_block(num_rows, hidden)
        cache_key = (hidden, warps_per_block, use_pdl)

        if cache_key not in _compile_cache:
            mIn = _to_cute_2d_bf16(inp)
            mOut = _to_cute_2d_u8(out_bits)
            mScale = _to_cute_1d_f32(out_scale)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            kernel = Fp8QuantSinglePass(hidden=hidden, dtype=BFloat16, warps_per_block=warps_per_block)
            _compile_cache[cache_key] = cute.compile(
                kernel, mOut, mScale, mIn, num_rows, stream_fake, use_pdl,
                options="--enable-tvm-ffi"
            )

        _compile_cache[cache_key](out_bits, out_scale, inp, num_rows)
    else:
        # Two-pass warp kernel - works for any hidden dimension
        cache_key = (hidden, "warp")

        if cache_key not in _compile_cache:
            mIn = _to_cute_2d_bf16(inp)
            mOut = _to_cute_2d_u8(out_bits)
            mScale = _to_cute_1d_f32(out_scale)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            kernel = Fp8QuantWarpPerRow(hidden=hidden, dtype=BFloat16)
            _compile_cache[cache_key] = cute.compile(
                kernel, mOut, mScale, mIn, num_rows, stream_fake,
                options="--enable-tvm-ffi"
            )

        _compile_cache[cache_key](out_bits, out_scale, inp, num_rows)


__all__ = [
    "Fp8QuantWarpPerRow",
    "Fp8QuantBlockPerRow",
    "Fp8QuantSinglePass",
    "fp8_quant_cute",
]
