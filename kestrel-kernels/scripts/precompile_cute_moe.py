#!/usr/bin/env python3
"""Precompile CuTe MoE kernel variants for AOT deployment.

Reads tuned configs from JSON files in kestrel_kernels/configs/ and compiles
all unique kernel variants needed.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


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

    @property
    def mul_routed_weight(self) -> bool:
        return self.kind == "down"

    @property
    def top_k(self) -> int:
        return 1 if self.kind == "down" else 8

    def filename(self, arch: str) -> str:
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_N{self.N}_K{self.K}_w{self.num_warps}_s{self.num_stages}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_N{self.N}_K{self.K}_w{self.num_warps}_s{self.num_stages}_{arch}"
        )


def _parse_model_dims(config_name: str) -> tuple[int, int]:
    match = re.match(r"cute_moe_E\d+_H(\d+)_I(\d+)_", config_name)
    if not match:
        raise ValueError(f"Could not parse H/I dims from config name: {config_name}")
    return int(match.group(1)), int(match.group(2))


def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper)."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def load_variants_from_configs() -> list[MoeVariant]:
    """Load all unique variants from JSON config files."""
    configs_dir = Path(__file__).parent.parent / "python" / "kestrel_kernels" / "configs"
    variants: set[MoeVariant] = set()

    for config_file in configs_dir.glob("cute_moe_*.json"):
        print(f"Loading configs from: {config_file.name}")
        H, I = _parse_model_dims(config_file.name)
        with open(config_file) as f:
            data = json.load(f)

        for kind in ("up", "down"):
            kind_configs = data.get(kind, {})
            if kind == "up":
                N_dim, K_dim = 2 * I, H
            else:
                N_dim, K_dim = H, I
            for token_count, cfg in kind_configs.items():
                variant = MoeVariant(
                    kind=kind,
                    block_m=cfg["block_m"],
                    block_n=cfg["block_n"],
                    block_k=cfg["block_k"],
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                    N=N_dim,
                    K=K_dim,
                )
                variants.add(variant)

    return sorted(
        variants, key=lambda v: (v.kind, v.N, v.K, v.block_m, v.block_n, v.block_k)
    )


def compile_variant(args: tuple[MoeVariant, str, Path]) -> tuple[MoeVariant, Path, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    variant, arch, output_dir = args

    # Import here to avoid issues with multiprocessing fork
    import cutlass.cute as cute
    from cutlass import BFloat16
    from cuda.bindings import driver as cuda

    from kestrel_kernels.cute_moe import (
        CuteMoeConfig,
        _FusedMoeMatmulCuTe,
        _FusedMoeMatmulCuTeWgmmaBf16,
        _should_use_wgmma_bf16,
    )

    try:
        print(
            f"[{os.getpid()}] Compiling: {variant.kind} N={variant.N} K={variant.K} "
            f"m={variant.block_m} n={variant.block_n} k={variant.block_k} "
            f"w={variant.num_warps} s={variant.num_stages}"
        )

        config = CuteMoeConfig(
            block_m=variant.block_m,
            block_n=variant.block_n,
            block_k=variant.block_k,
            num_warps=variant.num_warps,
            num_stages=variant.num_stages,
        )

        # Create the kernel operator
        dtype = BFloat16
        op_cls = (
            _FusedMoeMatmulCuTeWgmmaBf16
            if _should_use_wgmma_bf16(config)
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

        # Create fake tensors for compilation
        # These use symbolic dimensions for dynamic shapes
        M_in_sym = cute.sym_int()
        M_out_sym = cute.sym_int()
        K_static = variant.K
        N_static = variant.N
        E_sym = cute.sym_int()
        EM_sym = cute.sym_int()
        EM_blocks_sym = cute.sym_int()
        TW_sym = cute.sym_int()

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

        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        # Compile
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

        # Export to object file
        obj_path = output_dir / f"cute_moe_tmp_{os.getpid()}.o"
        function_name = variant.function_name(arch)
        compiled.export_to_c(str(obj_path), function_name=function_name)

        # Get runtime libraries and link to shared object
        runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)

        so_filename = variant.filename(arch)
        so_path = output_dir / so_filename

        # Link command
        cmd = [
            "gcc",
            "-shared",
            "-o",
            str(so_path),
            str(obj_path),
            *runtime_libs,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Clean up object file
        obj_path.unlink()

        print(f"[{os.getpid()}] Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        return (variant, None, str(e))


def main():
    # Detect architecture
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    # Load variants from config files
    variants = load_variants_from_configs()
    print(f"Found {len(variants)} unique kernel variants to compile")

    if not variants:
        print("No config files found! Add JSON configs to kestrel_kernels/configs/")
        sys.exit(1)

    # Output directory for precompiled kernels
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "python" / "kestrel_kernels" / "precompiled"
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py in precompiled directory
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Precompiled CuTe DSL kernels."""\n')

    # Compile sequentially - parallel compilation causes issues with CUDA context
    failed = []
    succeeded = []

    for variant in variants:
        result_variant, so_path, error = compile_variant((variant, arch, output_dir))
        if error:
            failed.append((result_variant, error))
        else:
            succeeded.append((result_variant, so_path))

    print(f"\nPrecompilation complete:")
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
