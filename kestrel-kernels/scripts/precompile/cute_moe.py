#!/usr/bin/env python3
"""Precompile CuTe MoE kernel variants for AOT deployment.

Reads tuned configs from JSON files in kestrel_kernels/configs/ and compiles
all unique kernel variants needed.
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link, PARALLEL_COMPILE, COMPILE_WORKERS


@dataclass(frozen=True)
class MoeVariant:
    """A single CuTe MoE kernel variant to precompile."""

    kind: str  # "up" or "down"
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    N: int
    K: int
    dtype: str = "bf16"  # "bf16" or "fp8"
    kernel_type: str = "warp"  # "warp" or "wgmma"
    use_pdl: bool = False  # Programmatic Dependent Launch (FP8 only)

    @property
    def mul_routed_weight(self) -> bool:
        return self.kind == "down"

    @property
    def top_k(self) -> int:
        return 1 if self.kind == "down" else 8

    def filename(self, arch: str) -> str:
        dtype_suffix = "_fp8" if self.dtype == "fp8" else ""
        kernel_suffix = "_wgmma" if self.kernel_type == "wgmma" else ""
        pdl_suffix = "_pdl" if self.use_pdl else ""
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_N{self.N}_K{self.K}_w{self.num_warps}_s{self.num_stages}{dtype_suffix}{kernel_suffix}{pdl_suffix}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        dtype_suffix = "_fp8" if self.dtype == "fp8" else ""
        kernel_suffix = "_wgmma" if self.kernel_type == "wgmma" else ""
        pdl_suffix = "_pdl" if self.use_pdl else ""
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_N{self.N}_K{self.K}_w{self.num_warps}_s{self.num_stages}{dtype_suffix}{kernel_suffix}{pdl_suffix}_{arch}"
        )


def _parse_model_dims(config_name: str) -> tuple[int, int]:
    match = re.match(r"cute_moe_E\d+_H(\d+)_I(\d+)_", config_name)
    if not match:
        raise ValueError(f"Could not parse H/I dims from config name: {config_name}")
    return int(match.group(1)), int(match.group(2))


def _parse_dtype_from_filename(config_name: str) -> str:
    """Parse dtype (bf16 or fp8) from config filename."""
    if "_fp8_" in config_name:
        return "fp8"
    return "bf16"


def load_variants_from_configs() -> list[MoeVariant]:
    """Load all unique variants from JSON config files."""
    configs_dir = Path(__file__).parent.parent.parent / "python" / "kestrel_kernels" / "configs"
    variants: set[MoeVariant] = set()

    for config_file in configs_dir.glob("cute_moe_*.json"):
        print(f"Loading configs from: {config_file.name}")
        H, I = _parse_model_dims(config_file.name)
        dtype = _parse_dtype_from_filename(config_file.name)
        with open(config_file) as f:
            data = json.load(f)

        for kind in ("up", "down"):
            kind_configs = data.get(kind, {})
            if kind == "up":
                N_dim, K_dim = 2 * I, H
            else:
                N_dim, K_dim = H, I
            for token_count, cfg in kind_configs.items():
                # Non-PDL variant (always needed)
                variant = MoeVariant(
                    kind=kind,
                    block_m=cfg["block_m"],
                    block_n=cfg["block_n"],
                    block_k=cfg["block_k"],
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                    N=N_dim,
                    K=K_dim,
                    dtype=dtype,
                    kernel_type=cfg["kernel_type"],
                    use_pdl=False,
                )
                variants.add(variant)

                # PDL variant (for FP8 kernels only - used with fp8_quant)
                if dtype == "fp8":
                    variant_pdl = MoeVariant(
                        kind=kind,
                        block_m=cfg["block_m"],
                        block_n=cfg["block_n"],
                        block_k=cfg["block_k"],
                        num_warps=cfg["num_warps"],
                        num_stages=cfg["num_stages"],
                        N=N_dim,
                        K=K_dim,
                        dtype=dtype,
                        kernel_type=cfg["kernel_type"],
                        use_pdl=True,
                    )
                    variants.add(variant_pdl)

    return sorted(
        variants, key=lambda v: (v.dtype, v.use_pdl, v.kind, v.N, v.K, v.block_m, v.block_n, v.block_k)
    )


def compile_variant(args: tuple[MoeVariant, str, Path]) -> tuple[MoeVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    variant, arch, output_dir = args

    # Import here to avoid issues with multiprocessing fork
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Float8E4M3FN

    from kestrel_kernels.cute_moe import CuteMoeConfig
    from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_warp import _FusedMoeMatmulCuTe
    from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_wgmma import _FusedMoeMatmulCuTeWgmmaBf16
    from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_wgmma import _FusedMoeMatmulCuTeFp8
    from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_warp import _FusedMoeMatmulCuTeWarpFp8

    try:
        dtype_str = f" ({variant.dtype})" if variant.dtype == "fp8" else ""
        kernel_str = f" [{variant.kernel_type}]"
        pdl_str = " [PDL]" if variant.use_pdl else ""
        print(
            f"[{os.getpid()}] Compiling: {variant.kind}{dtype_str}{kernel_str}{pdl_str} N={variant.N} K={variant.K} "
            f"m={variant.block_m} n={variant.block_n} k={variant.block_k} "
            f"w={variant.num_warps} s={variant.num_stages}"
        )

        config = CuteMoeConfig(
            block_m=variant.block_m,
            block_n=variant.block_n,
            block_k=variant.block_k,
            num_warps=variant.num_warps,
            num_stages=variant.num_stages,
            dtype=variant.dtype,
            kernel_type=variant.kernel_type,
        )

        # Symbolic dimensions for dynamic shapes
        M_in_sym = cute.sym_int()
        M_out_sym = cute.sym_int()
        K_static = variant.K
        N_static = variant.N
        E_sym = cute.sym_int()
        EM_sym = cute.sym_int()
        EM_blocks_sym = cute.sym_int()
        TW_sym = cute.sym_int()

        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        if variant.dtype == "fp8":
            # FP8 kernel with separate scale tensors
            fp8_dtype = Float8E4M3FN
            out_dtype = BFloat16

            # Select kernel class based on kernel_type
            fp8_cls = (
                _FusedMoeMatmulCuTeFp8
                if variant.kernel_type == "wgmma"
                else _FusedMoeMatmulCuTeWarpFp8
            )
            op = fp8_cls(
                out_dtype,
                fp8_dtype,
                config,
                mul_routed_weight=variant.mul_routed_weight,
                top_k=variant.top_k,
                N=variant.N,
                K=variant.K,
            )

            # A_bits: (M_in, K) uint8
            a_bits_fake = cute.runtime.make_fake_tensor(
                cute.Uint8, (M_in_sym, K_static),
                stride=(cute.sym_int64(divisibility=16), 1),
                assumed_align=16,
            )
            # A_scale: (M_in,) float32
            a_scale_fake = cute.runtime.make_fake_tensor(
                Float32, (M_in_sym,),
                stride=(1,),
                assumed_align=4,
            )
            # B_bits: (E, N, K) uint8
            b_bits_fake = cute.runtime.make_fake_tensor(
                cute.Uint8, (E_sym, N_static, K_static),
                stride=(cute.sym_int64(divisibility=16), cute.sym_int64(divisibility=16), 1),
                assumed_align=16,
            )
            # B_scale: (E, N) float32
            b_scale_fake = cute.runtime.make_fake_tensor(
                Float32, (E_sym, N_static),
                stride=(cute.sym_int64(divisibility=8), 1),
                assumed_align=4,
            )
            # C: (M_out, N) bfloat16
            c_fake = cute.runtime.make_fake_tensor(
                out_dtype, (M_out_sym, N_static),
                stride=(cute.sym_int64(divisibility=8), 1),
                assumed_align=16,
            )
            # topk_weights: (TW,) bfloat16
            topk_w_fake = cute.runtime.make_fake_tensor(
                out_dtype, (TW_sym,),
                stride=(1,),
                assumed_align=16,
            )
            # sorted_token_ids: (EM,)
            sorted_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (EM_sym,),
                stride=(1,),
                assumed_align=4,
            )
            # expert_ids: (EM_blocks,)
            expert_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (EM_blocks_sym,),
                stride=(1,),
                assumed_align=4,
            )
            # num_tokens_post_padded: (1,)
            post_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (1,),
                stride=(1,),
                assumed_align=4,
            )

            compiled = cute.compile(
                op,
                a_bits_fake,
                a_scale_fake,
                b_bits_fake,
                b_scale_fake,
                c_fake,
                topk_w_fake,
                sorted_fake,
                expert_fake,
                post_fake,
                stream_fake,
                variant.use_pdl,  # PDL support for FP8 kernels
                options="--enable-tvm-ffi",
            )
        else:
            # BF16 kernel
            dtype = BFloat16
            # Select kernel class based on kernel_type
            op_cls = (
                _FusedMoeMatmulCuTeWgmmaBf16
                if variant.kernel_type == "wgmma"
                else _FusedMoeMatmulCuTe
            )
            op = op_cls(
                dtype,
                config,
                mul_routed_weight=variant.mul_routed_weight,
                top_k=variant.top_k,
                N=variant.N,
                K=variant.K,
            )

            # A: (M_in, K) row-major
            a_fake = cute.runtime.make_fake_tensor(
                dtype, (M_in_sym, K_static),
                stride=(cute.sym_int64(divisibility=8), 1),
                assumed_align=16,
            )
            # B: (E, N, K) last-dim contiguous
            b_fake = cute.runtime.make_fake_tensor(
                dtype, (E_sym, N_static, K_static),
                stride=(cute.sym_int64(divisibility=8), cute.sym_int64(divisibility=8), 1),
                assumed_align=16,
            )
            # C: (M_out, N) row-major
            c_fake = cute.runtime.make_fake_tensor(
                dtype, (M_out_sym, N_static),
                stride=(cute.sym_int64(divisibility=8), 1),
                assumed_align=16,
            )
            # topk_weights: (TW,)
            topk_w_fake = cute.runtime.make_fake_tensor(
                dtype, (TW_sym,),
                stride=(1,),
                assumed_align=16,
            )
            # sorted_token_ids: (EM,)
            sorted_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (EM_sym,),
                stride=(1,),
                assumed_align=4,
            )
            # expert_ids: (EM_blocks,)
            expert_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (EM_blocks_sym,),
                stride=(1,),
                assumed_align=4,
            )
            # num_tokens_post_padded: (1,)
            post_fake = cute.runtime.make_fake_tensor(
                cute.Int32, (1,),
                stride=(1,),
                assumed_align=4,
            )

            compiled = cute.compile(
                op,
                a_fake,
                b_fake,
                c_fake,
                topk_w_fake,
                sorted_fake,
                expert_fake,
                post_fake,
                stream_fake,
                options="--enable-tvm-ffi",
            )

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "cute_moe_tmp")

        print(f"[{os.getpid()}] Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        return (variant, None, str(e))


def main():
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    # Load variants from config files
    variants = load_variants_from_configs()
    print(f"Found {len(variants)} unique kernel variants to compile")

    if not variants:
        print("No config files found! Add JSON configs to kestrel_kernels/configs/")
        sys.exit(1)

    output_dir = get_precompiled_dir()

    failed = []
    succeeded = []

    if PARALLEL_COMPILE and len(variants) > 1:
        # Use spawn context to avoid CUDA context issues with fork
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        print(f"Compiling {len(variants)} variants in parallel with {COMPILE_WORKERS} workers")

        with ProcessPoolExecutor(max_workers=COMPILE_WORKERS, mp_context=ctx) as executor:
            args = [(variant, arch, output_dir) for variant in variants]
            for result_variant, so_path, error in executor.map(compile_variant, args):
                if error:
                    failed.append((result_variant, error))
                else:
                    succeeded.append((result_variant, so_path))
    else:
        print(f"Compiling {len(variants)} variants sequentially")
        for variant in variants:
            result_variant, so_path, error = compile_variant((variant, arch, output_dir))
            if error:
                failed.append((result_variant, error))
            else:
                succeeded.append((result_variant, so_path))

    print(f"\nCuTe MoE precompilation complete:")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed variants:")
        for variant, error in failed:
            print(f"  {variant.kind} m={variant.block_m} n={variant.block_n}: {error}")
        sys.exit(1)

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
