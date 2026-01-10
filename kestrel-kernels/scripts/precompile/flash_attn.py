#!/usr/bin/env python3
"""Precompile Flash Attention kernel variants for AOT deployment.

Currently supports:
- SM90 forward path (Hopper)
- SM90 decode path (Hopper)
- SM100 forward path (Blackwell)

Variants are precompiled for common configurations:
- dtypes: bf16, f16
- head_dim: 64, 128
- causal: True, False
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import BFloat16, Float16

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link


@dataclass(frozen=True)
class FlashAttnVariant:
    """A single Flash Attention kernel variant to precompile."""

    kernel_type: Literal["forward", "decode"]
    sm_version: Literal["sm90", "sm100"]
    dtype_name: str  # "bf16" or "f16"
    head_dim: int
    head_dim_v: int | None  # For asymmetric head dims (DeepSeek)
    causal: bool
    qhead_per_kvhead: int = 1  # 1 for MHA, >1 for GQA

    @property
    def dtype(self):
        return BFloat16 if self.dtype_name == "bf16" else Float16

    def filename(self, arch: str) -> str:
        hdv = f"_hdv{self.head_dim_v}" if self.head_dim_v else ""
        causal_str = "_causal" if self.causal else ""
        gqa = f"_gqa{self.qhead_per_kvhead}" if self.qhead_per_kvhead > 1 else ""
        return (
            f"flash_attn_{self.kernel_type}_{self.sm_version}_{self.dtype_name}_"
            f"hd{self.head_dim}{hdv}{causal_str}{gqa}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        hdv = f"_hdv{self.head_dim_v}" if self.head_dim_v else ""
        causal_str = "_causal" if self.causal else ""
        gqa = f"_gqa{self.qhead_per_kvhead}" if self.qhead_per_kvhead > 1 else ""
        return (
            f"flash_attn_{self.kernel_type}_{self.sm_version}_{self.dtype_name}_"
            f"hd{self.head_dim}{hdv}{causal_str}{gqa}_{arch}"
        )


def _arch_supports_sm90(arch: str) -> bool:
    """Check if architecture supports SM90 features (Hopper+)."""
    if not arch.startswith("sm"):
        return False
    version = int(arch[2:])
    return version >= 90


def _arch_supports_sm100(arch: str) -> bool:
    """Check if architecture supports SM100 features (Blackwell)."""
    if not arch.startswith("sm"):
        return False
    version = int(arch[2:])
    return version >= 100


def generate_precompile_variants(arch: str) -> list[FlashAttnVariant]:
    """Generate Flash Attention variants based on target architecture."""
    variants = []

    dtypes = ["bf16", "f16"]
    head_dims = [64, 128]
    causals = [False, True]

    # SM90 (Hopper) forward and decode variants
    if _arch_supports_sm90(arch):
        for dtype in dtypes:
            for hd in head_dims:
                for causal in causals:
                    # Forward path
                    variants.append(FlashAttnVariant(
                        kernel_type="forward",
                        sm_version="sm90",
                        dtype_name=dtype,
                        head_dim=hd,
                        head_dim_v=None,
                        causal=causal,
                    ))
                    # Decode path
                    variants.append(FlashAttnVariant(
                        kernel_type="decode",
                        sm_version="sm90",
                        dtype_name=dtype,
                        head_dim=hd,
                        head_dim_v=None,
                        causal=causal,
                    ))

    # SM100 (Blackwell) forward variants
    if _arch_supports_sm100(arch):
        for dtype in dtypes:
            for hd in head_dims:
                for causal in causals:
                    variants.append(FlashAttnVariant(
                        kernel_type="forward",
                        sm_version="sm100",
                        dtype_name=dtype,
                        head_dim=hd,
                        head_dim_v=None,
                        causal=causal,
                    ))

    return variants


def compile_variant(
    variant: FlashAttnVariant, arch: str, output_dir: Path
) -> tuple[FlashAttnVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    try:
        print(
            f"Compiling: {variant.kernel_type} {variant.sm_version} "
            f"dtype={variant.dtype_name} head_dim={variant.head_dim} "
            f"causal={variant.causal}"
        )

        # Import kernel classes
        if variant.sm_version == "sm90":
            if variant.kernel_type == "forward":
                from kestrel_kernels.flash_attn.cute.flash_fwd import FlashAttentionForwardSm90
                kernel_cls = FlashAttentionForwardSm90
            else:  # decode
                from kestrel_kernels.flash_attn.cute.flash_decode_sm90 import FlashAttentionDecodeSm90
                kernel_cls = FlashAttentionDecodeSm90
        elif variant.sm_version == "sm100":
            from kestrel_kernels.flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
            kernel_cls = FlashAttentionForwardSm100
        else:
            raise ValueError(f"Unsupported SM version: {variant.sm_version}")

        # Create kernel instance
        head_dim_v = variant.head_dim_v or variant.head_dim

        if variant.kernel_type == "decode":
            fa_kernel = kernel_cls(
                variant.dtype,
                variant.head_dim,
                variant.qhead_per_kvhead,
                is_causal=variant.causal,
            )
        elif variant.sm_version == "sm90":
            fa_kernel = kernel_cls(
                variant.dtype,
                variant.head_dim,
                head_dim_v,
                variant.qhead_per_kvhead,
                is_causal=variant.causal,
                tile_m=128,
                tile_n=128,
                num_stages=2,
                num_threads=384,
            )
        else:  # sm100
            fa_kernel = kernel_cls(
                variant.head_dim,
                head_dim_v,
                qhead_per_kvhead=variant.qhead_per_kvhead,
                is_causal=variant.causal,
            )

        # Create fake tensors for compilation
        # This is simplified - real compilation needs proper tensor shapes
        batch_sym = cute.sym_int()
        seqlen_q_sym = cute.sym_int()
        seqlen_k_sym = cute.sym_int()
        num_heads_sym = cute.sym_int()
        head_dim_static = variant.head_dim
        head_dim_v_static = head_dim_v

        # Q: (batch, seqlen_q, num_heads, head_dim)
        q_fake = cute.runtime.make_fake_tensor(
            variant.dtype,
            (batch_sym, seqlen_q_sym, num_heads_sym, head_dim_static),
            stride=(
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                1,
            ),
            assumed_align=16,
        )
        # K: (batch, seqlen_k, num_heads_kv, head_dim)
        k_fake = cute.runtime.make_fake_tensor(
            variant.dtype,
            (batch_sym, seqlen_k_sym, num_heads_sym, head_dim_static),
            stride=(
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                1,
            ),
            assumed_align=16,
        )
        # V: (batch, seqlen_k, num_heads_kv, head_dim_v)
        v_fake = cute.runtime.make_fake_tensor(
            variant.dtype,
            (batch_sym, seqlen_k_sym, num_heads_sym, head_dim_v_static),
            stride=(
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                1,
            ),
            assumed_align=16,
        )
        # O: (batch, seqlen_q, num_heads, head_dim_v)
        o_fake = cute.runtime.make_fake_tensor(
            variant.dtype,
            (batch_sym, seqlen_q_sym, num_heads_sym, head_dim_v_static),
            stride=(
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                1,
            ),
            assumed_align=16,
        )

        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Compile
        # Note: This is a simplified compilation - real Flash Attention
        # has many more arguments (softmax_scale, LSE, cu_seqlens, etc.)
        # Full implementation requires matching the exact signature in interface.py
        compiled = cute.compile(
            fa_kernel,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            None,  # lse
            1.0,  # softmax_scale
            None,  # k_scale
            None,  # v_scale
            stream_fake,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            -1,  # window_size_left
            -1,  # window_size_right
            None,  # learnable_sink
            None,  # sparse_tensors
            None,  # aux_tensors
            options="--enable-tvm-ffi",
        )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "flash_attn_tmp")

        print(f"Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        import traceback
        return (variant, None, f"{e}\n{traceback.format_exc()}")


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

    # Generate variants based on architecture
    variants = generate_precompile_variants(arch)
    print(f"Total variants to compile: {len(variants)}")

    if not variants:
        print("No Flash Attention variants to compile for this architecture.")
        return

    # Compile all variants
    failed = []
    succeeded = []

    for variant in variants:
        result_variant, so_path, error = compile_variant(variant, arch, output_dir)
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
            print(f"  {variant.kernel_type} {variant.sm_version} hd={variant.head_dim}: {error[:200]}")
        # Don't exit with error - Flash Attention precompilation is optional
        # sys.exit(1)

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
