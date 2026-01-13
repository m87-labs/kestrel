"""CuTe MoE kernel utilities, PTX helpers, and initialization."""

import math
from typing import Any, Dict, Tuple, Type

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack

from kestrel_kernels.flash_attn.cute import utils as fa_utils
from kestrel_kernels.cute_moe.config import CuteMoeConfig, _get_cuda_arch

from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


# =============================================================================
# PTX Inline Assembly Helpers
# =============================================================================


@dsl_user_op
def bitcast_bf16_to_i32(bf16_val, *, loc=None, ip=None) -> Int32:
    """Bitcast a single BF16 value to i32 (zero-extended).

    This is needed because:
    1. Int32(bf16_val) does float-to-int conversion, not bitcast
    2. Our smem holds packed FP8 data in "BF16" slots (16 bits = 2 packed FP8)

    The returned i32 has the 16-bit value in the low bits and zeros in the high bits.
    This works with cvt_fp8x2_lo_to_bf16x2 which extracts only the low 16 bits.
    """
    from cutlass._mlir.dialects import arith

    # Get the IR value (bfloat16 type)
    ir_val = bf16_val.ir_value(loc=loc, ip=ip)
    # Bitcast bfloat16 → i16 (same size, just type change)
    i16_val = llvm.bitcast(T.i16(), ir_val, loc=loc, ip=ip)
    # Zero-extend i16 → i32
    i32_val = arith.extui(T.i32(), i16_val, loc=loc, ip=ip)
    return Int32(i32_val)


@dsl_user_op
def bitcast_f16_to_i32(f16_val, *, loc=None, ip=None) -> Int32:
    """Bitcast a single F16 value to i32 (zero-extended).

    Same as bitcast_bf16_to_i32 but for F16. Both are 16-bit floats.
    The returned i32 has the 16-bit value in the low bits and zeros in the high bits.
    """
    from cutlass._mlir.dialects import arith

    ir_val = f16_val.ir_value(loc=loc, ip=ip)
    i16_val = llvm.bitcast(T.i16(), ir_val, loc=loc, ip=ip)
    i32_val = arith.extui(T.i32(), i16_val, loc=loc, ip=ip)
    return Int32(i32_val)


@dsl_user_op
def pack_b16x2_to_b32(lo: Int32, hi: Int32, *, loc=None, ip=None) -> Int32:
    """Pack two 16-bit values into one 32-bit value.

    Input: lo and hi are 32-bit values where only the low 16 bits are used.
    Output: 32-bit value with {hi[15:0], lo[15:0]}.
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo16, hi16;
            mov.b32 {lo16, _}, $1;
            mov.b32 {hi16, _}, $2;
            mov.b32 $0, {lo16, hi16};
        }
        """,
        "=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def unpack_b32_lo16(packed: Int32, *, loc=None, ip=None) -> Int32:
    """Extract low 16 bits from 32-bit value, zero-extended to 32 bits."""
    result = llvm.inline_asm(
        T.i32(),
        [Int32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo16, hi16;
            mov.b32 {lo16, hi16}, $1;
            mov.b32 $0, {lo16, 0};
        }
        """,
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def unpack_b32_hi16(packed: Int32, *, loc=None, ip=None) -> Int32:
    """Extract high 16 bits from 32-bit value, zero-extended to 32 bits."""
    result = llvm.inline_asm(
        T.i32(),
        [Int32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo16, hi16;
            mov.b32 {lo16, hi16}, $1;
            mov.b32 $0, {hi16, 0};
        }
        """,
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def cvt_fp8x2_to_bf16x2(packed_fp8x2: Int32, *, loc=None, ip=None) -> Int32:
    """Convert 2 E4M3 FP8 values (from low 16 bits of 32-bit reg) to 2 packed BF16 values.

    Input: 32-bit register with 2 packed FP8 values in the low 16 bits (high 16 bits ignored)
    Output: 32-bit register containing 2 packed BF16 values

    This is the main conversion function. Use bitcast_bf16_to_i32 to prepare the input.
    Conversion chain: E4M3 → F16 → F32 → BF16
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(packed_fp8x2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            .reg .b32 f16x2;
            .reg .f16 f16_0, f16_1;
            .reg .f32 f32_0, f32_1;
            .reg .b16 bf16_0, bf16_1;

            mov.b32 {lo, hi}, $1;
            cvt.rn.f16x2.e4m3x2 f16x2, lo;
            mov.b32 {f16_0, f16_1}, f16x2;
            cvt.f32.f16 f32_0, f16_0;
            cvt.f32.f16 f32_1, f16_1;
            cvt.rn.bf16.f32 bf16_0, f32_0;
            cvt.rn.bf16.f32 bf16_1, f32_1;
            mov.b32 $0, {bf16_0, bf16_1};
        }
        """,
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def cvt_fp8x2_to_f16x2(packed_fp8x2: Int32, *, loc=None, ip=None) -> Int32:
    """Convert 2 E4M3 FP8 values to 2 packed F16 values (FAST - single PTX instruction).

    Input: 32-bit register with 2 packed FP8 values in the low 16 bits
    Output: 32-bit register containing 2 packed F16 values

    This is much faster than cvt_fp8x2_to_bf16x2 as it's a single instruction.
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(packed_fp8x2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $1;
            cvt.rn.f16x2.e4m3x2 $0, lo;
        }
        """,
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def cvt_fp8x4_to_f16x4_lo_hi(packed_fp8x4: Int32, *, loc=None, ip=None):
    """Convert 4 E4M3 FP8 values to 4 F16 values, returning lo and hi pairs.

    Input: 32-bit register with 4 packed FP8 values (each FP8 is 8 bits)
    Output: (lo_f16x2, hi_f16x2) - two i32 registers each with 2 packed F16

    This matches Triton's F2FP.F16.E4M3.UNPACK_B + F2FP.F16.E4M3.UNPACK_B.H1 pattern,
    avoiding the PRMT overhead from separate lo/hi extraction.
    """
    result = llvm.inline_asm(
        T.i64(),  # Pack both i32 results into i64
        [Int32(packed_fp8x4).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo_fp8x2, hi_fp8x2;
            .reg .b32 lo_f16x2, hi_f16x2;
            mov.b32 {lo_fp8x2, hi_fp8x2}, $1;
            cvt.rn.f16x2.e4m3x2 lo_f16x2, lo_fp8x2;
            cvt.rn.f16x2.e4m3x2 hi_f16x2, hi_fp8x2;
            mov.b64 $0, {lo_f16x2, hi_f16x2};
        }
        """,
        "=l,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def unpack_i64_lo32(packed_i64, *, loc=None, ip=None) -> Int32:
    """Extract low 32 bits from i64."""
    from cutlass._mlir.dialects import arith
    i32_val = arith.trunci(T.i32(), packed_i64, loc=loc, ip=ip)
    return Int32(i32_val)


@dsl_user_op
def unpack_i64_hi32(packed_i64, *, loc=None, ip=None) -> Int32:
    """Extract high 32 bits from i64."""
    from cutlass._mlir.dialects import arith
    shift_amt = arith.constant(T.i64(), 32, loc=loc, ip=ip)
    shifted = arith.shrui(packed_i64, shift_amt, loc=loc, ip=ip)
    i32_val = arith.trunci(T.i32(), shifted, loc=loc, ip=ip)
    return Int32(i32_val)


@dsl_user_op
def i32_to_f16x2(packed_i32: Int32, *, loc=None, ip=None):
    """Bitcast i32 to 2 packed F16 (returns tuple of 2 f16 values for CuTe fragment)."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith
    ir_val = Int32(packed_i32).ir_value(loc=loc, ip=ip)
    f16_type = ir.F16Type.get()
    vec_type = ir.VectorType.get([2], f16_type)
    vec_f16x2 = llvm.bitcast(vec_type, ir_val, loc=loc, ip=ip)
    idx_0 = arith.constant(T.i32(), 0, loc=loc, ip=ip)
    idx_1 = arith.constant(T.i32(), 1, loc=loc, ip=ip)
    f16_lo = llvm.extractelement(vec_f16x2, idx_0, loc=loc, ip=ip)
    f16_hi = llvm.extractelement(vec_f16x2, idx_1, loc=loc, ip=ip)
    return f16_lo, f16_hi


@dsl_user_op
def cvt_bf16_to_f16(bf16_val, *, loc=None, ip=None):
    """Convert a single BF16 value to F16 (single PTX instruction).

    Input: BF16 value
    Output: F16 value

    Uses cvt.rn.f16.bf16 which is a single instruction on SM90.
    """
    ir_bf16 = bf16_val.ir_value(loc=loc, ip=ip)
    ir_i16 = llvm.bitcast(T.i16(), ir_bf16, loc=loc, ip=ip)

    result_i16 = llvm.inline_asm(
        T.i16(),
        [ir_i16],
        "cvt.rn.f16.bf16 $0, $1;",
        "=h,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    result_f16 = llvm.bitcast(T.f16(), result_i16, loc=loc, ip=ip)
    return result_f16


@dsl_user_op
def load_bf16x2_as_i32(ptr, *, loc=None, ip=None) -> Int32:
    """Load 2 consecutive BF16 values from gmem as an Int32.

    Input: pointer to gmem (Int64 address)
    Output: Int32 containing {bf16[1], bf16[0]}

    Uses ld.global.b32 for a single 32-bit load.
    """
    result = llvm.inline_asm(
        T.i32(),
        [ptr],
        "ld.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def cvt_bf16x2_to_f16x2(packed_bf16x2: Int32, *, loc=None, ip=None) -> Int32:
    """Convert 2 packed BF16 values to 2 packed F16 values (2 PTX instructions).

    Input: 32-bit register with 2 packed BF16 values {bf16[1], bf16[0]}
    Output: 32-bit register containing 2 packed F16 values {f16[1], f16[0]}
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(packed_bf16x2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 bf16_0, bf16_1, f16_0, f16_1;
            mov.b32 {bf16_0, bf16_1}, $1;
            cvt.rn.f16.bf16 f16_0, bf16_0;
            cvt.rn.f16.bf16 f16_1, bf16_1;
            mov.b32 $0, {f16_0, f16_1};
        }
        """,
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def store_smem_b32(value: Int32, smem_ptr, *, loc=None, ip=None) -> None:
    """Store 32 bits to shared memory.

    Input: value (Int32), smem_ptr (Int64 address)
    """
    if hasattr(smem_ptr, 'ir_value'):
        ptr_ir = smem_ptr.ir_value(loc=loc, ip=ip)
    else:
        ptr_ir = smem_ptr

    llvm.inline_asm(
        None,
        [ptr_ir, Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_smem_u32(smem_ptr, *, loc=None, ip=None) -> Int32:
    """Load 32 bits from shared memory.

    Input: smem_ptr - Int64 address in shared memory
    Output: Int32 with loaded value
    """
    if hasattr(smem_ptr, 'ir_value'):
        ptr_ir = smem_ptr.ir_value(loc=loc, ip=ip)
    else:
        ptr_ir = smem_ptr

    result = llvm.inline_asm(
        T.i32(),
        [ptr_ir],
        "ld.shared.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def byte_perm_u32(a: Int32, b: Int32, selector: int, *, loc=None, ip=None) -> Int32:
    """Byte permutation using PTX prmt instruction.

    Selects 4 bytes from the concatenation of b and a (8 bytes total).
    Byte indices: a[0..3] = indices 0..3, b[0..3] = indices 4..7.

    The selector is a 16-bit immediate where each nibble (4 bits) specifies
    which source byte to place in the corresponding result byte position.

    Example: selector=0x5140 produces result[0]=a[0], result[1]=b[0],
             result[2]=a[1], result[3]=b[1]

    Input: a, b - Int32 source values, selector - 16-bit immediate
    Output: Int32 with permuted bytes
    """
    ir_a = Int32(a).ir_value(loc=loc, ip=ip)
    ir_b = Int32(b).ir_value(loc=loc, ip=ip)

    result = llvm.inline_asm(
        T.i32(),
        [ir_a, ir_b],
        f"prmt.b32 $0, $1, $2, {selector};",
        "=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def warp_sync(*, loc=None, ip=None) -> None:
    """Synchronize all threads within a warp."""
    llvm.inline_asm(
        None,
        [],
        "bar.warp.sync 0xffffffff;",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_smem_u16(smem_ptr, *, loc=None, ip=None) -> Int32:
    """Load 16 bits from shared memory, zero-extended to 32 bits.

    Input: smem_ptr - Int64 address in shared memory
    Output: Int32 with loaded 16-bit value zero-extended
    """
    # Convert smem_ptr to IR value if it has ir_value method
    if hasattr(smem_ptr, 'ir_value'):
        ptr_ir = smem_ptr.ir_value(loc=loc, ip=ip)
    else:
        ptr_ir = smem_ptr

    result = llvm.inline_asm(
        T.i32(),
        [ptr_ir],
        "ld.shared.u16 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def store_smem_u16(value: Int32, smem_ptr, *, loc=None, ip=None) -> None:
    """Store low 16 bits to shared memory.

    Input: value - Int32 with value in low 16 bits, smem_ptr - Int64 address
    """
    # Convert smem_ptr to IR value if it has ir_value method
    if hasattr(smem_ptr, 'ir_value'):
        ptr_ir = smem_ptr.ir_value(loc=loc, ip=ip)
    else:
        ptr_ir = smem_ptr

    # Note: with no output, operands are numbered starting at $0
    # $0 = ptr_ir (l = 64-bit), $1 = value (r = 32-bit)
    llvm.inline_asm(
        None,
        [ptr_ir, Int32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 v16; mov.b32 {v16, _}, $1; st.shared.b16 [$0], v16; }",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvt_fp8x2_to_f16_both(packed_fp8_in_f16, *, loc=None, ip=None):
    """Convert 2 packed FP8 to 2 separate F16 values using LLVM vector extraction.

    Input: F16 value that actually holds 2 packed FP8 values (from ldmatrix on packed smem)
    Output: Tuple of (f16_lo, f16_hi) - both converted F16 values

    Uses LLVM vector<2 x f16> and extractelement to avoid PRMT overhead from trunci/shrui.
    """
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith

    # Bitcast f16 → i16 (LLVM inline asm can't handle f16 directly)
    ir_f16 = packed_fp8_in_f16.ir_value(loc=loc, ip=ip)
    ir_i16 = llvm.bitcast(T.i16(), ir_f16, loc=loc, ip=ip)

    # PTX: convert 2 FP8 → 2 F16 packed in 32-bit register
    result_i32 = llvm.inline_asm(
        T.i32(),
        [ir_i16],
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        "=r,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    # Bitcast i32 → vector<2 x f16> using MLIR's native VectorType
    f16_type = ir.F16Type.get()
    vec_type = ir.VectorType.get([2], f16_type)
    vec_f16x2 = llvm.bitcast(vec_type, result_i32, loc=loc, ip=ip)

    # Extract elements from vector (result type inferred from vector)
    idx_0 = arith.constant(T.i32(), 0, loc=loc, ip=ip)
    idx_1 = arith.constant(T.i32(), 1, loc=loc, ip=ip)
    f16_lo = llvm.extractelement(vec_f16x2, idx_0, loc=loc, ip=ip)
    f16_hi = llvm.extractelement(vec_f16x2, idx_1, loc=loc, ip=ip)

    return f16_lo, f16_hi


@dsl_user_op
def cvt_fp8x4_to_f16x4_packed(packed_fp8x4: Int32, *, loc=None, ip=None):
    """Convert 4 packed FP8 values to 2 packed F16x2 values (Int32 → two Int32).

    Input: Int32 containing 4 packed E4M3 FP8 values (each 8 bits)
    Output: Tuple of (lo_f16x2, hi_f16x2) - two Int32 values, each containing 2 packed F16

    This avoids all bitcast overhead by working entirely with Int32 registers.
    Uses two F2FP instructions (one for low 2 FP8, one for high 2 FP8).
    """
    ir_i32 = Int32(packed_fp8x4).ir_value(loc=loc, ip=ip)

    # PTX: Split into lo/hi 16-bit halves, convert each to f16x2
    # Low 2 FP8 → 2 F16 packed in Int32
    lo_i32 = llvm.inline_asm(
        T.i32(),
        [ir_i32],
        "{ .reg .b16 lo; mov.b32 {lo, _}, $1; cvt.rn.f16x2.e4m3x2 $0, lo; }",
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    # High 2 FP8 → 2 F16 packed in Int32
    hi_i32 = llvm.inline_asm(
        T.i32(),
        [ir_i32],
        "{ .reg .b16 hi; mov.b32 {_, hi}, $1; cvt.rn.f16x2.e4m3x2 $0, hi; }",
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    return Int32(lo_i32), Int32(hi_i32)


# Keep the separate functions for compatibility but have them call the combined one
@dsl_user_op
def cvt_fp8x2_to_f16_lo(packed_fp8_in_f16, *, loc=None, ip=None):
    """Convert low FP8 from packed pair to F16.

    Note: For best performance, use cvt_fp8x2_to_f16_both to get both values
    with a single conversion instruction.
    """
    from cutlass._mlir.dialects import arith

    ir_f16 = packed_fp8_in_f16.ir_value(loc=loc, ip=ip)
    ir_i16 = llvm.bitcast(T.i16(), ir_f16, loc=loc, ip=ip)

    result_i32 = llvm.inline_asm(
        T.i32(),
        [ir_i16],
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        "=r,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    lo_i16 = arith.trunci(T.i16(), result_i32, loc=loc, ip=ip)
    result_f16 = llvm.bitcast(T.f16(), lo_i16, loc=loc, ip=ip)
    return result_f16


@dsl_user_op
def cvt_fp8x2_to_f16_hi(packed_fp8_in_f16, *, loc=None, ip=None):
    """Convert high FP8 from packed pair to F16.

    Note: For best performance, use cvt_fp8x2_to_f16_both to get both values
    with a single conversion instruction.
    """
    from cutlass._mlir.dialects import arith

    ir_f16 = packed_fp8_in_f16.ir_value(loc=loc, ip=ip)
    ir_i16 = llvm.bitcast(T.i16(), ir_f16, loc=loc, ip=ip)

    result_i32 = llvm.inline_asm(
        T.i32(),
        [ir_i16],
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        "=r,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )

    shift_16 = arith.constant(T.i32(), 16, loc=loc, ip=ip)
    shifted = arith.shrui(result_i32, shift_16, loc=loc, ip=ip)
    hi_i16 = arith.trunci(T.i16(), shifted, loc=loc, ip=ip)
    result_f16 = llvm.bitcast(T.f16(), hi_i16, loc=loc, ip=ip)
    return result_f16


@dsl_user_op
def extract_f16_lo(packed_f16x2: Int32, *, loc=None, ip=None):
    """Extract the low F16 from a 32-bit value containing 2 packed F16.

    Input: 32-bit register with {f16_hi, f16_lo}
    Output: Single F16 value (the low one)
    """
    from cutlass._mlir.dialects import arith

    ir_val = Int32(packed_f16x2).ir_value(loc=loc, ip=ip)
    # Truncate to i16 (keeps low 16 bits)
    i16_val = arith.trunci(T.i16(), ir_val, loc=loc, ip=ip)
    # Bitcast i16 → f16
    f16_val = llvm.bitcast(T.f16(), i16_val, loc=loc, ip=ip)
    return f16_val


@dsl_user_op
def extract_f16_hi(packed_f16x2: Int32, *, loc=None, ip=None):
    """Extract the high F16 from a 32-bit value containing 2 packed F16.

    Input: 32-bit register with {f16_hi, f16_lo}
    Output: Single F16 value (the high one)
    """
    from cutlass._mlir.dialects import arith

    ir_val = Int32(packed_f16x2).ir_value(loc=loc, ip=ip)
    # Shift right by 16 to get high 16 bits in low position
    shift_amt = arith.constant(T.i32(), 16, loc=loc, ip=ip)
    shifted = arith.shrui(ir_val, shift_amt, loc=loc, ip=ip)
    # Truncate to i16
    i16_val = arith.trunci(T.i16(), shifted, loc=loc, ip=ip)
    # Bitcast i16 → f16
    f16_val = llvm.bitcast(T.f16(), i16_val, loc=loc, ip=ip)
    return f16_val


@dsl_user_op
def cvt_fp8x2_to_bf16_lo_hi(packed_fp8x2: Int32, *, loc=None, ip=None):
    """Convert 2 FP8 to 2 separate BF16 values (combined operation for speed).

    Input: 32-bit value with 2 packed FP8 in low 16 bits
    Returns: Tuple of (bf16_lo, bf16_hi) as separate 16-bit values in i32 registers

    This combines the conversion and extraction into one PTX block to reduce overhead.
    Returns two i32 values where each has a BF16 in the low 16 bits.
    """
    # Returns (lo, hi) packed as i64, then caller unpacks
    result = llvm.inline_asm(
        T.i64(),
        [Int32(packed_fp8x2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 fp8_lo, fp8_hi;
            .reg .b32 f16x2;
            .reg .f16 f16_0, f16_1;
            .reg .f32 f32_0, f32_1;
            .reg .b16 bf16_0, bf16_1;
            .reg .b32 out_lo, out_hi;

            mov.b32 {fp8_lo, fp8_hi}, $1;
            cvt.rn.f16x2.e4m3x2 f16x2, fp8_lo;
            mov.b32 {f16_0, f16_1}, f16x2;
            cvt.f32.f16 f32_0, f16_0;
            cvt.f32.f16 f32_1, f16_1;
            cvt.rn.bf16.f32 bf16_0, f32_0;
            cvt.rn.bf16.f32 bf16_1, f32_1;
            mov.b32 out_lo, {bf16_0, 0};
            mov.b32 out_hi, {bf16_1, 0};
            mov.b64 $0, {out_lo, out_hi};
        }
        """,
        "=l,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def unpack_bf16_pair_lo(packed_i64, *, loc=None, ip=None):
    """Extract the low BF16 from packed i64 result of cvt_fp8x2_to_bf16_lo_hi."""
    from cutlass._mlir.dialects import arith

    ir_val = packed_i64
    # Truncate i64 to i32 (keeps low 32 bits)
    i32_val = arith.trunci(T.i32(), ir_val, loc=loc, ip=ip)
    # Truncate to i16
    i16_val = arith.trunci(T.i16(), i32_val, loc=loc, ip=ip)
    # Bitcast to bf16
    bf16_val = llvm.bitcast(T.bf16(), i16_val, loc=loc, ip=ip)
    return bf16_val


@dsl_user_op
def unpack_bf16_pair_hi(packed_i64, *, loc=None, ip=None):
    """Extract the high BF16 from packed i64 result of cvt_fp8x2_to_bf16_lo_hi."""
    from cutlass._mlir.dialects import arith

    ir_val = packed_i64
    # Shift right by 32 to get high 32 bits
    shift_amt = arith.constant(T.i64(), 32, loc=loc, ip=ip)
    shifted = arith.shrui(ir_val, shift_amt, loc=loc, ip=ip)
    # Truncate to i32
    i32_val = arith.trunci(T.i32(), shifted, loc=loc, ip=ip)
    # Truncate to i16
    i16_val = arith.trunci(T.i16(), i32_val, loc=loc, ip=ip)
    # Bitcast to bf16
    bf16_val = llvm.bitcast(T.bf16(), i16_val, loc=loc, ip=ip)
    return bf16_val


@dsl_user_op
def extract_bf16_lo(packed_bf16x2: Int32, *, loc=None, ip=None):
    """Extract the low BF16 from a 32-bit value containing 2 packed BF16.

    Input: 32-bit register with {bf16_hi, bf16_lo}
    Output: Single BF16 value (the low one)
    """
    from cutlass._mlir.dialects import arith

    ir_val = Int32(packed_bf16x2).ir_value(loc=loc, ip=ip)
    # Truncate to i16 (keeps low 16 bits)
    i16_val = arith.trunci(T.i16(), ir_val, loc=loc, ip=ip)
    # Bitcast i16 → bfloat16
    bf16_val = llvm.bitcast(T.bf16(), i16_val, loc=loc, ip=ip)
    return bf16_val


@dsl_user_op
def extract_bf16_hi(packed_bf16x2: Int32, *, loc=None, ip=None):
    """Extract the high BF16 from a 32-bit value containing 2 packed BF16.

    Input: 32-bit register with {bf16_hi, bf16_lo}
    Output: Single BF16 value (the high one)
    """
    from cutlass._mlir.dialects import arith

    ir_val = Int32(packed_bf16x2).ir_value(loc=loc, ip=ip)
    # Shift right by 16 to get high 16 bits in low position
    shift_amt = arith.constant(T.i32(), 16, loc=loc, ip=ip)
    shifted = arith.shrui(ir_val, shift_amt, loc=loc, ip=ip)
    # Truncate to i16
    i16_val = arith.trunci(T.i16(), shifted, loc=loc, ip=ip)
    # Bitcast i16 → bfloat16
    bf16_val = llvm.bitcast(T.bf16(), i16_val, loc=loc, ip=ip)
    return bf16_val


@dsl_user_op
def store_streaming_b32(value: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store 32 bits (e.g. 2xBF16) with cache streaming hint (st.global.cs).

    Bypasses L1 and marks for early L2 eviction - useful for scattered write-only stores.
    """
    llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.cs.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_streaming_b128(
    v0: Int32, v1: Int32, v2: Int32, v3: Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    """Store 128 bits (e.g. 8xBF16) with cache streaming hint (st.global.cs.v4.b32).

    Stores 4x32-bit values in a single 128-bit transaction.
    Bypasses L1 and marks for early L2 eviction - useful for scattered write-only stores.
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
def shfl_sync_idx_b32(value: Int32, src_lane: Int32, *, loc=None, ip=None) -> Int32:
    """Warp shuffle - read value from src_lane within the warp.

    Uses shfl.sync.idx.b32 with full mask (0xffffffff) for all threads participating.
    The fourth operand 0x1f means width=32 (full warp).
    The fifth operand 0xffffffff is the membership mask (all threads participate).
    """
    result = llvm.inline_asm(
        T.i32(),
        [Int32(value).ir_value(loc=loc, ip=ip), Int32(src_lane).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        "=r,r,r",
        has_side_effects=True,  # Synchronization point
        loc=loc,
        ip=ip,
    )
    return Int32(result)


def tiled_copy_2d_bypass(
    dtype: Type[cutlass.Numeric], major_mode_size: int, num_threads: int
) -> cute.TiledCopy:
    """Like copy_utils.tiled_copy_2d but with L1 cache bypass for async copies.

    Uses LoadCacheMode.GLOBAL which generates cp.async.cg (bypass L1, cache in L2),
    matching Triton's LDGSTS.E.BYPASS.128 pattern for gathered/scattered access.
    """
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


# =============================================================================
# WGMMA Helper
# =============================================================================


@cute.jit
def _wgmma_gemm_no_fence(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


# =============================================================================
# Global State and Initialization
# =============================================================================


_CUTLASS_INITIALIZED = False
_CUTE_KERNEL_ATTRS_SET: set[str] = set()
_DEVICE_CACHE_CONFIG_SET = False

# Precompiled kernel registry
_precompiled_cache: Dict[Tuple[str, CuteMoeConfig, int, int], Any] = {}
_precompiled_cache_fp8: Dict[Tuple[str, CuteMoeConfig], Any] = {}

from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module


def _load_precompiled_kernel(kind: str, config: CuteMoeConfig, N: int, K: int, use_pdl: bool = False):
    """Load a precompiled kernel if available, return None otherwise."""
    compile_key = (kind, config, N, K, use_pdl)

    # Check if already loaded (use appropriate cache based on dtype)
    cache = _precompiled_cache_fp8 if config.dtype == "fp8" else _precompiled_cache
    if compile_key in cache:
        return cache[compile_key]

    # Build filename and function name
    arch = get_cuda_arch()
    dtype_suffix = "_fp8" if config.dtype == "fp8" else ""
    kernel_suffix = "_wgmma" if config.kernel_type == "wgmma" else ""
    pdl_suffix = "_pdl" if use_pdl else ""
    base_name = (
        f"cute_moe_{kind}_m{config.block_m}_n{config.block_n}_k{config.block_k}"
        f"_N{N}_K{K}_w{config.num_warps}_s{config.num_stages}{dtype_suffix}{kernel_suffix}{pdl_suffix}_{arch}"
    )
    filename = f"{base_name}.so"
    function_name = base_name

    # Load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    kernel_fn = getattr(mod, function_name)
    cache[compile_key] = kernel_fn
    return kernel_fn


def _ensure_cutlass_initialized() -> None:
    global _CUTLASS_INITIALIZED
    if _CUTLASS_INITIALIZED:
        return
    # The upstream helper `cutlass.cuda.initialize_cuda_context()` uses `cuCtxCreate`
    # (new context), which is incompatible with PyTorch/Triton tensors already
    # allocated in the process. We must use the device *primary* context instead.
    res, = cuda.cuInit(0)
    if res != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed with {res}")
    res, cur_ctx = cuda.cuCtxGetCurrent()
    if res != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed with {res}")
    if int(cur_ctx) == 0:
        res, dev = cuda.cuDeviceGet(int(torch.cuda.current_device()))
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed with {res}")
        res, primary_ctx = cuda.cuDevicePrimaryCtxRetain(dev)
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain failed with {res}")
        res, = cuda.cuCtxSetCurrent(primary_ctx)
        if res != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed with {res}")
    _CUTLASS_INITIALIZED = True


def _set_compiled_kernel_shared_carveout(compiled: Any, *, carveout_pct: int = 100) -> None:
    """Best-effort: prefer shared memory over L1 for the compiled kernel."""
    try:
        import cuda.bindings.runtime as cuda_rt

        jit_module = getattr(compiled, "jit_module", None)
        libs = getattr(jit_module, "cuda_library", None)
        if not libs:
            return
        lib = libs[0]

        kernel_info = getattr(compiled, "kernel_info", None)
        if not kernel_info:
            return

        dev = int(torch.cuda.current_device())
        for name in kernel_info.keys():
            if not isinstance(name, str):
                continue
            if name in _CUTE_KERNEL_ATTRS_SET:
                continue
            err, kernel = cuda_rt.cudaLibraryGetKernel(lib, name.encode())
            if int(err) != 0:
                continue
            # Match Triton's cache preference (and increase effective shared memory capacity).
            # This is separate from the carveout attribute and affects NCU's "Function Cache Configuration".
            try:
                cuda_rt.cudaFuncSetCacheConfig(
                    kernel, cuda_rt.cudaFuncCache.cudaFuncCachePreferShared
                )
            except Exception:
                pass
            cuda_rt.cudaKernelSetAttributeForDevice(
                kernel,
                cuda_rt.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout,
                int(carveout_pct),
                dev,
            )
            _CUTE_KERNEL_ATTRS_SET.add(name)
    except Exception:
        # Never fail the op because of optional tuning.
        return


def _maybe_set_device_cache_config() -> None:
    """Best-effort: prefer shared memory over L1 on SM90."""
    global _DEVICE_CACHE_CONFIG_SET
    if _DEVICE_CACHE_CONFIG_SET:
        return
    try:
        import cuda.bindings.runtime as cuda_rt

        cuda_rt.cudaDeviceSetCacheConfig(cuda_rt.cudaFuncCache.cudaFuncCachePreferShared)
        _DEVICE_CACHE_CONFIG_SET = True
    except Exception:
        return


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================


def _to_cute_tensor_1d_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 1D tensor")
    return from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_1d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous 1D tensor")
    return from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_scalar_i32(t: torch.Tensor) -> cute.Tensor:
    if t.dtype != torch.int32 or t.ndim != 1 or t.numel() != 1 or not t.is_contiguous():
        raise ValueError("Expected contiguous int32 tensor with shape (1,)")
    return from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


def _to_cute_tensor_2d_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 2 or not t.is_contiguous():
        raise ValueError("Expected row-major contiguous 2D tensor")
    # The MoE kernel relies on 128-bit vectorized accesses (vec_size=8 for BF16/FP16).
    # Add compactness/divisibility hints so the compiler can prove alignment for cp.async.
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=1, alignment=assumed_align, divisibility=8, enable_tvm_ffi=True
    )


def _to_cute_tensor_2d_contig_u8(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor (got {t.dtype})")
    if t.ndim != 2 or not t.is_contiguous():
        raise ValueError("Expected row-major contiguous 2D uint8 tensor")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=1, alignment=assumed_align, divisibility=16, enable_tvm_ffi=True
    )


def _to_cute_tensor_3d_last_contig(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.ndim != 3 or t.stride(-1) != 1:
        raise ValueError("Expected 3D tensor contiguous in the last dim")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=2, alignment=assumed_align, divisibility=8, enable_tvm_ffi=True
    )


def _to_cute_tensor_3d_last_contig_u8(t: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor (got {t.dtype})")
    if t.ndim != 3 or t.stride(-1) != 1:
        raise ValueError("Expected 3D uint8 tensor contiguous in the last dim")
    return fa_utils.convert_from_dlpack(
        t.detach(), leading_dim=2, alignment=assumed_align, divisibility=16, enable_tvm_ffi=True
    )
