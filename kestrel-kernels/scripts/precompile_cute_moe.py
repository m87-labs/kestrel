#!/usr/bin/env python3
"""Precompile CuTe MoE kernel variants for AOT deployment."""

import os
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

    @property
    def mul_routed_weight(self) -> bool:
        return self.kind == "down"

    @property
    def top_k(self) -> int:
        return 1 if self.kind == "down" else 8

    def filename(self, arch: str) -> str:
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_w{self.num_warps}_s{self.num_stages}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        return (
            f"cute_moe_{self.kind}_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_w{self.num_warps}_s{self.num_stages}_{arch}"
        )


def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper)."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def get_cuda_arch_num() -> int:
    """Get the CUDA architecture as a number (e.g., 90 for Hopper)."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# All variants needed for BF16 decode (1-64 tokens) + prefill (up to 512 tokens)
#
# Derived from _HARDCODED_CONFIGS in kestrel/fused_moe/module.py:
# - For "up": num_stages used as-is
# - For "down": num_stages capped at min(2, num_stages)
#
# Plus special configs for num_tokens <= 8

PRECOMPILE_VARIANTS: list[MoeVariant] = [
    # Special configs for num_tokens <= 8
    MoeVariant("up", 16, 128, 128, 2, 1),
    MoeVariant("down", 16, 64, 256, 4, 1),
    # Config for tokens=1 (also 32, 48, 64 use same config)
    MoeVariant("up", 16, 256, 128, 4, 3),
    MoeVariant("down", 16, 256, 128, 4, 2),
    # Config for tokens=2
    MoeVariant("up", 16, 64, 128, 4, 4),
    MoeVariant("down", 16, 64, 128, 4, 2),
    # Config for tokens=4
    MoeVariant("up", 16, 64, 256, 4, 3),
    MoeVariant("down", 16, 64, 256, 4, 2),
    # Config for tokens=8
    MoeVariant("up", 16, 32, 256, 4, 2),
    MoeVariant("down", 16, 32, 256, 4, 2),
    # Config for tokens=16
    MoeVariant("up", 16, 32, 128, 4, 5),
    MoeVariant("down", 16, 32, 128, 4, 2),
    # Config for tokens=24
    MoeVariant("up", 16, 128, 256, 4, 2),
    MoeVariant("down", 16, 128, 256, 4, 2),
    # Config for tokens=96
    MoeVariant("up", 32, 256, 128, 4, 3),
    MoeVariant("down", 32, 256, 128, 4, 2),
    # Config for tokens=128
    MoeVariant("up", 32, 128, 128, 4, 3),
    MoeVariant("down", 32, 128, 128, 4, 2),
    # Config for tokens=256
    MoeVariant("up", 64, 64, 64, 4, 3),
    MoeVariant("down", 64, 64, 64, 4, 2),
    # Config for tokens=512
    MoeVariant("up", 128, 128, 64, 8, 3),
    MoeVariant("down", 128, 128, 64, 8, 2),
]


def compile_variant(args: tuple[MoeVariant, str, Path]) -> tuple[MoeVariant, Path, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    variant, arch, output_dir = args

    # Import here to avoid issues with multiprocessing fork
    import cutlass.cute as cute
    from cutlass import BFloat16
    from cuda.bindings import driver as cuda

    from kestrel_kernels.cute_moe import CuteMoeConfig, _FusedMoeMatmulCuTe

    try:
        print(f"[{os.getpid()}] Compiling: {variant.kind} m={variant.block_m} n={variant.block_n} "
              f"k={variant.block_k} w={variant.num_warps} s={variant.num_stages}")

        config = CuteMoeConfig(
            block_m=variant.block_m,
            block_n=variant.block_n,
            block_k=variant.block_k,
            num_warps=variant.num_warps,
            num_stages=variant.num_stages,
        )

        # Create the kernel operator
        dtype = BFloat16
        op = _FusedMoeMatmulCuTe(
            dtype, config,
            mul_routed_weight=variant.mul_routed_weight,
            top_k=variant.top_k,
        )

        # Create fake tensors for compilation
        # These use symbolic dimensions for dynamic shapes
        # IMPORTANT: Each dimension that can vary independently needs its own symbol
        M_in_sym = cute.sym_int()  # input num_tokens (A.shape[0])
        M_out_sym = cute.sym_int()  # output assignments (C.shape[0], may differ for "up" kernel)
        K_sym = cute.sym_int()  # input dim
        N_sym = cute.sym_int()  # output dim
        E_sym = cute.sym_int()  # num_experts
        EM_sym = cute.sym_int()  # sorted_token_ids length
        EM_blocks_sym = cute.sym_int()  # expert_ids length (EM / block_m)
        TW_sym = cute.sym_int()  # topk_weights length (independent, may be 0 for "up")

        # A: (M_in, K) row-major
        a_fake = cute.runtime.make_fake_tensor(
            dtype, (M_in_sym, K_sym),
            stride=(cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        # B: (E, N, K) last-dim contiguous
        b_fake = cute.runtime.make_fake_tensor(
            dtype, (E_sym, N_sym, K_sym),
            stride=(cute.sym_int64(divisibility=8), cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        # C: (M_out, N) row-major - for "up" kernel M_out = num_tokens * top_k
        c_fake = cute.runtime.make_fake_tensor(
            dtype, (M_out_sym, N_sym),
            stride=(cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        # topk_weights: (TW,) - independent dimension, may be 0 for "up" kernels
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
        # expert_ids: (EM_blocks,) where EM_blocks = EM / block_m
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

        # Compile using same approach as topk - auto-detect GPU arch
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
    print(f"Compiling {len(PRECOMPILE_VARIANTS)} kernel variants...")

    # Output directory for precompiled kernels
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "python" / "kestrel_kernels" / "precompiled"
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py in precompiled directory
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Precompiled CuTe DSL kernels."""\n')

    # Compile sequentially (like topk) - parallel compilation causes issues with CUDA context
    failed = []
    succeeded = []

    for variant in PRECOMPILE_VARIANTS:
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
