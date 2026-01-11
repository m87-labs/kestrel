#!/usr/bin/env python3
"""Precompile Flash Attention kernel variants for AOT deployment.

Variants are precompiled for Moondream's specific configurations:
- Text forward: bf16, hd64, causal + prefix-lm mask, paged_kv_non_tma
- Text decode: bf16 (with bf16 or FP8 KV cache), hd64, causal
- Vision forward: bf16, hd72, non-causal, non-paged
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link, PARALLEL_COMPILE, COMPILE_WORKERS


@dataclass(frozen=True)
class FlashAttnVariant:
    """A single Flash Attention kernel variant to precompile."""

    kernel_type: Literal["forward", "decode"]
    dtype_name: str  # "bf16"
    dtype_kv_name: str | None  # "bf16", "fp8_e4m3", or None (same as dtype)
    head_dim: int
    causal: bool
    mask_mod_name: str | None  # "prefix_lm_730" or None
    paged_kv_non_tma: bool

    @property
    def dtype(self):
        return BFloat16 if self.dtype_name == "bf16" else Float16

    @property
    def dtype_kv(self):
        if self.dtype_kv_name is None:
            return None
        if self.dtype_kv_name == "bf16":
            return BFloat16
        if self.dtype_kv_name == "f16":
            return Float16
        if self.dtype_kv_name == "fp8_e4m3":
            return cutlass.Float8E4M3FN
        raise ValueError(f"Unknown dtype_kv: {self.dtype_kv_name}")

    def filename(self, arch: str) -> str:
        """Generate filename for the precompiled kernel."""
        dtype_kv_str = f"_{self.dtype_kv_name}" if self.dtype_kv_name else ""
        causal_str = "_causal" if self.causal else ""
        mask_str = f"_{self.mask_mod_name}" if self.mask_mod_name else ""
        paged_str = "_paged" if self.paged_kv_non_tma else ""
        return (
            f"flash_attn_{self.kernel_type}_sm90_{self.dtype_name}{dtype_kv_str}_"
            f"hd{self.head_dim}{causal_str}{mask_str}{paged_str}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        """Generate function name for the precompiled kernel."""
        return self.filename(arch).replace(".so", "")


# Moondream-specific variants
VARIANTS = [
    # Text forward (causal, paged, bf16 KV)
    FlashAttnVariant("forward", "bf16", None, 64, True, None, True),
    # Text forward (causal, paged, FP8 KV)
    FlashAttnVariant("forward", "bf16", "fp8_e4m3", 64, True, None, True),
    # Text forward (prefix-lm mask, paged, bf16 KV)
    FlashAttnVariant("forward", "bf16", None, 64, False, "prefix_lm_730", True),
    # Text forward (prefix-lm mask, paged, FP8 KV)
    FlashAttnVariant("forward", "bf16", "fp8_e4m3", 64, False, "prefix_lm_730", True),
    # Text decode (bf16 KV)
    FlashAttnVariant("decode", "bf16", "bf16", 64, True, None, False),
    # Text decode (FP8 KV)
    FlashAttnVariant("decode", "bf16", "fp8_e4m3", 64, True, None, False),
    # Vision forward (non-paged, non-causal)
    FlashAttnVariant("forward", "bf16", None, 72, False, None, False),
]


def _arch_supports_sm90(arch: str) -> bool:
    """Check if architecture supports SM90 features (Hopper+)."""
    if not arch.startswith("sm"):
        return False
    # Strip any suffix like 'a' from sm90a
    version_str = arch[2:].rstrip("abcdefghijklmnopqrstuvwxyz")
    version = int(version_str)
    return version >= 90


def _make_fake_tensor(dtype, shape, divisibility=8, assumed_align=16):
    """Create a fake tensor for compilation.

    Note: Fake tensors use symbolic dimensions for shape/stride, allowing
    cute.compile to generate dynamic kernels that work with runtime tensor sizes.
    """
    if dtype is None:
        return None

    # Create symbolic strides with divisibility hints
    strides = []
    for i in range(len(shape) - 1):
        strides.append(cute.sym_int64(divisibility=divisibility))
    strides.append(1)  # Last dimension is contiguous

    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=tuple(strides),
        assumed_align=assumed_align,
    )


def _make_fake_tensor_1d_int32(shape):
    """Create a fake 1D int32 tensor matching to_cute_tensor(..., assumed_align=4, leading_dim=0)."""
    return cute.runtime.make_fake_tensor(
        cutlass.Int32,
        shape,
        stride=(1,),  # 1D tensor has trivial stride
        assumed_align=4,  # Matches JIT path for seqused_k/page_table
    )


def compile_forward_variant(
    variant: FlashAttnVariant, arch: str, output_dir: Path
) -> tuple[FlashAttnVariant, Path | None, str | None]:
    """Compile a forward kernel variant."""
    try:
        dtype_kv_str = variant.dtype_kv_name or variant.dtype_name
        print(f"Compiling forward: dtype={variant.dtype_name} dtype_kv={dtype_kv_str} head_dim={variant.head_dim} "
              f"causal={variant.causal} mask={variant.mask_mod_name} paged={variant.paged_kv_non_tma}")

        from kestrel_kernels.flash_attn.cute.flash_fwd import FlashAttentionForwardSm90

        # Get mask function if needed
        mask_mod = None
        if variant.mask_mod_name == "prefix_lm_730":
            from kestrel_kernels.flash_attn.cute.mask_definitions import cute_prefix_lm_mask_730
            mask_mod = cute_prefix_lm_mask_730

        # Create kernel instance
        # For FP8 KV, pass dtype_kv to the kernel
        dtype_kv = variant.dtype_kv if variant.dtype_kv_name else None
        fa_fwd = FlashAttentionForwardSm90(
            variant.dtype,
            variant.head_dim,
            variant.head_dim,  # head_dim_v = head_dim
            qhead_per_kvhead=1,  # MHA
            is_causal=variant.causal,
            is_local=False,
            pack_gqa=False,
            tile_m=128,
            tile_n=128,
            num_stages=2,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            mask_mod=mask_mod,
            score_mod=None,
            has_aux_tensors=False,
            paged_kv_non_tma=variant.paged_kv_non_tma,
            dtype_kv=dtype_kv,
        )

        # Create symbolic dimensions
        batch = cute.sym_int()
        seqlen_q = cute.sym_int()
        num_heads = cute.sym_int()
        head_dim = variant.head_dim

        # Create fake tensors matching interface.py's to_cute_tensor()
        q_tensor = _make_fake_tensor(variant.dtype, (batch, seqlen_q, num_heads, head_dim))
        o_tensor = _make_fake_tensor(variant.dtype, (batch, seqlen_q, num_heads, head_dim))

        # For FP8 KV, K/V are passed as Uint8 (bit-level representation)
        is_fp8_kv = variant.dtype_kv_name == "fp8_e4m3"
        kv_tensor_dtype = cutlass.Uint8 if is_fp8_kv else variant.dtype

        if variant.paged_kv_non_tma:
            # Paged K/V: shape is (n_pages, page_size, n_kv_heads, head_dim)
            n_pages = cute.sym_int()
            page_size = cute.sym_int()
            max_num_pages_per_seq = cute.sym_int()
            k_tensor = _make_fake_tensor(kv_tensor_dtype, (n_pages, page_size, num_heads, head_dim))
            v_tensor = _make_fake_tensor(kv_tensor_dtype, (n_pages, page_size, num_heads, head_dim))
            # page_table uses assumed_align=4 in JIT (to_cute_tensor(..., assumed_align=4, leading_dim=1))
            page_table = _make_fake_tensor(cutlass.Int32, (batch, max_num_pages_per_seq), assumed_align=4)
            # seqused_k uses assumed_align=4 in JIT (to_cute_tensor(..., assumed_align=4, leading_dim=0))
            seqused_k = _make_fake_tensor_1d_int32((batch,))
        else:
            # Non-paged K/V: shape is (batch, seqlen_k, n_kv_heads, head_dim)
            seqlen_k = cute.sym_int()
            k_tensor = _make_fake_tensor(kv_tensor_dtype, (batch, seqlen_k, num_heads, head_dim))
            v_tensor = _make_fake_tensor(kv_tensor_dtype, (batch, seqlen_k, num_heads, head_dim))
            page_table = None
            seqused_k = None

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Compile with same signature as interface.py
        # Use 1.0 for scale values since None doesn't work for scalar float parameters
        compiled = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            None,  # lse_tensor
            1.0,   # softmax_scale
            1.0,   # k_scale_val (use 1.0 for precompilation)
            1.0,   # v_scale_val (use 1.0 for precompilation)
            stream,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            seqused_k,  # seqused_k (tensor for paged, None for non-paged)
            page_table,  # page_table (tensor for paged, None for non-paged)
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # sparse_tensors
            None,  # aux_tensors
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "flash_fwd_tmp")

        print(f"Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        import traceback
        return (variant, None, f"{e}\n{traceback.format_exc()}")


def compile_decode_variant(
    variant: FlashAttnVariant, arch: str, output_dir: Path
) -> tuple[FlashAttnVariant, Path | None, str | None]:
    """Compile a decode kernel variant."""
    try:
        dtype_kv = variant.dtype_kv or variant.dtype
        print(f"Compiling decode: dtype={variant.dtype_name} dtype_kv={variant.dtype_kv_name} "
              f"head_dim={variant.head_dim} causal={variant.causal}")

        from kestrel_kernels.flash_attn.cute.flash_decode_sm90 import FlashAttentionDecodeSm90

        # Determine tile_size_per_bdx based on FP8 status
        is_fp8 = variant.dtype_kv_name == "fp8_e4m3"
        tile_size_per_bdx = 2 if is_fp8 else 4

        # Create kernel instance
        fa_decode = FlashAttentionDecodeSm90(
            variant.dtype,
            variant.head_dim,
            qhead_per_kvhead=1,  # MHA
            dtype_kv=dtype_kv,
            is_causal=variant.causal,
            is_local=False,
            tile_size_per_bdx=tile_size_per_bdx,
        )

        # Create symbolic dimensions
        batch = cute.sym_int()
        seqlen_q = cute.sym_int()
        num_heads = cute.sym_int()
        head_dim = variant.head_dim

        # Decode uses paged KV with page_size=1
        # K/V shapes: (num_pages, page_size, n_kv_heads, head_dim)
        num_pages = cute.sym_int()
        page_size = cute.sym_int()
        max_kv_len = cute.sym_int()

        # Create fake tensors
        q_tensor = _make_fake_tensor(variant.dtype, (batch, seqlen_q, num_heads, head_dim))

        # For decode, K/V may be FP8 (stored as uint8)
        # Shape is (num_pages, page_size, n_kv_heads, head_dim) for paged KV
        kv_dtype = dtype_kv if not is_fp8 else cutlass.Uint8
        k_tensor = _make_fake_tensor(kv_dtype, (num_pages, page_size, num_heads, head_dim))
        v_tensor = _make_fake_tensor(kv_dtype, (num_pages, page_size, num_heads, head_dim))
        o_tensor = _make_fake_tensor(variant.dtype, (batch, seqlen_q, num_heads, head_dim))

        # Decode requires page_table and seqused_k
        # Use assumed_align=4 to match JIT path (to_cute_tensor(..., assumed_align=4))
        page_table = _make_fake_tensor(cutlass.Int32, (batch, max_kv_len), assumed_align=4)
        seqused_k = _make_fake_tensor_1d_int32((batch,))

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Compile with same signature as interface.py
        # Use 1.0 for scale values since None doesn't work for scalar float parameters
        compiled = cute.compile(
            fa_decode,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            None,  # lse_tensor
            1.0,   # softmax_scale
            1.0,   # k_scale_val (use 1.0 for precompilation)
            1.0,   # v_scale_val (use 1.0 for precompilation)
            stream,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            seqused_k,  # seqused_k (required for paged decode)
            page_table,  # page_table (required for paged decode)
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # sparse_tensors
            None,  # aux_tensors
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "flash_decode_tmp")

        print(f"Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        import traceback
        return (variant, None, f"{e}\n{traceback.format_exc()}")


def compile_variant(
    args: tuple[FlashAttnVariant, str, Path]
) -> tuple[FlashAttnVariant, Path | None, str | None]:
    """Compile a single variant."""
    variant, arch, output_dir = args
    if variant.kernel_type == "forward":
        return compile_forward_variant(variant, arch, output_dir)
    elif variant.kernel_type == "decode":
        return compile_decode_variant(variant, arch, output_dir)
    else:
        return (variant, None, f"Unknown kernel type: {variant.kernel_type}")


def main():
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    # Check if architecture supports Flash Attention precompilation
    if not _arch_supports_sm90(arch):
        print(f"Architecture {arch} does not support SM90+ features.")
        print("Flash Attention precompilation requires SM90 (Hopper) or newer.")
        print("Skipping Flash Attention precompilation.")
        return

    output_dir = get_precompiled_dir()
    print(f"Output directory: {output_dir}")
    print(f"Total variants to compile: {len(VARIANTS)}")

    # Compile all variants
    failed = []
    succeeded = []

    if PARALLEL_COMPILE and len(VARIANTS) > 1:
        # Use spawn context to avoid CUDA context issues with fork
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        print(f"Compiling {len(VARIANTS)} variants in parallel with {COMPILE_WORKERS} workers")

        with ProcessPoolExecutor(max_workers=COMPILE_WORKERS, mp_context=ctx) as executor:
            args = [(variant, arch, output_dir) for variant in VARIANTS]
            for result_variant, so_path, error in executor.map(compile_variant, args):
                if error:
                    failed.append((result_variant, error))
                else:
                    succeeded.append((result_variant, so_path))
    else:
        print(f"Compiling {len(VARIANTS)} variants sequentially")
        for variant in VARIANTS:
            result_variant, so_path, error = compile_variant((variant, arch, output_dir))
            if error:
                failed.append((result_variant, error))
            else:
                succeeded.append((result_variant, so_path))

    print(f"\nFlash Attention precompilation complete:")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed variants:")
        for variant, error in failed:
            print(f"  {variant.kernel_type} hd={variant.head_dim}: {error[:300]}")
        # Don't exit with error - Flash Attention precompilation is optional
        # sys.exit(1)


if __name__ == "__main__":
    main()
