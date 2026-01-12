#!/usr/bin/env python3
"""Precompile MoE align kernel variants for AOT deployment."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import Int32, Int64

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from kestrel_kernels.moe_align.kernel import (
    MoeAlignCuTeConfig,
    _MoeAlignBlockSizeCuTe,
    _MoeAlignBlockSizeCuTeLarge,
    _MoeAlignBlockSizeCuTeLora,
    _MoeAlignBlockSizeCuTeLargeLora,
)

from .utils import get_cuda_arch, get_precompiled_dir, compile_and_link, PARALLEL_COMPILE, COMPILE_WORKERS


# Path to config JSON files
_CONFIGS_DIR = Path(__file__).parent.parent.parent / "python" / "kestrel_kernels" / "configs"


@dataclass(frozen=True)
class MoeAlignVariant:
    """A single MoE align kernel variant to precompile."""

    kernel_type: str  # "small", "large", "lora_small", "lora_large"
    topk_dtype: type  # Int32 or Int64
    topk: int
    num_experts: int
    block_size: int
    has_expert_map: bool

    def filename(self, arch: str) -> str:
        dtype_name = "i32" if self.topk_dtype == Int32 else "i64"
        expert_map_str = "emap" if self.has_expert_map else "noemap"
        return (
            f"moe_align_{self.kernel_type}_{dtype_name}_k{self.topk}_"
            f"e{self.num_experts}_b{self.block_size}_{expert_map_str}_{arch}.so"
        )

    def function_name(self, arch: str) -> str:
        dtype_name = "i32" if self.topk_dtype == Int32 else "i64"
        expert_map_str = "emap" if self.has_expert_map else "noemap"
        return (
            f"moe_align_{self.kernel_type}_{dtype_name}_k{self.topk}_"
            f"e{self.num_experts}_b{self.block_size}_{expert_map_str}_{arch}"
        )


def get_block_m_values_from_configs() -> set[int]:
    """Extract all unique block_m values from cute_moe config JSON files."""
    block_m_values: set[int] = set()

    if not _CONFIGS_DIR.exists():
        print(f"Warning: Config directory {_CONFIGS_DIR} does not exist")
        return block_m_values

    # Load all config files regardless of arch (like cute_moe.py does)
    config_files = list(_CONFIGS_DIR.glob("cute_moe_*.json"))

    if not config_files:
        print(f"Warning: No config files found in {_CONFIGS_DIR}")
        return block_m_values

    for config_file in config_files:
        try:
            with open(config_file) as f:
                configs = json.load(f)

            # Extract block_m from both "up" and "down" sections
            for section in ("up", "down"):
                section_configs = configs.get(section, {})
                for cfg in section_configs.values():
                    if "block_m" in cfg:
                        block_m_values.add(cfg["block_m"])

            print(f"Loaded block_m values from {config_file.name}")
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}")

    return block_m_values


def generate_precompile_variants() -> list[MoeAlignVariant]:
    """Generate precompile variants based on block_m values from config files."""
    block_m_values = get_block_m_values_from_configs()

    if not block_m_values:
        # Fallback to default values if no configs found
        print("Warning: No block_m values found in configs, using defaults")
        block_m_values = {16, 32, 64, 128}

    print(f"Block_m values from configs: {sorted(block_m_values)}")

    variants = []

    # Generate all kernel types for each block_size and dtype
    kernel_types = ["small", "large", "lora_small", "lora_large"]
    dtypes = [Int32, Int64]

    # MoE variants (topk=8, num_experts=64)
    for block_size in sorted(block_m_values):
        for dtype in dtypes:
            for kernel_type in kernel_types:
                variants.append(
                    MoeAlignVariant(kernel_type, dtype, 8, 64, block_size, False)
                )

    # LoRA on dense layers (topk=1, num_experts=1) - needed for non-MoE models
    # Only need lora_small and lora_large variants, and only int32
    for block_size in sorted(block_m_values):
        variants.append(
            MoeAlignVariant("lora_small", Int32, 1, 1, block_size, False)
        )
        variants.append(
            MoeAlignVariant("lora_large", Int32, 1, 1, block_size, False)
        )

    return variants


def compile_variant(
    args: tuple[MoeAlignVariant, str, Path]
) -> tuple[MoeAlignVariant, Path | None, str | None]:
    """Compile a single variant. Returns (variant, output_path, error_or_none)."""
    variant, arch, output_dir = args

    # Import here to avoid issues with multiprocessing spawn
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    try:
        print(
            f"Compiling: type={variant.kernel_type}, "
            f"dtype={'int64' if variant.topk_dtype == Int64 else 'int32'}, "
            f"topk={variant.topk}, num_experts={variant.num_experts}, "
            f"block_size={variant.block_size}, has_expert_map={variant.has_expert_map}"
        )

        cfg = MoeAlignCuTeConfig()
        t_sym = cute.sym_int()

        # Create fake tensors based on kernel type
        topk_ids_fake = cute.runtime.make_fake_tensor(
            variant.topk_dtype,
            (t_sym, variant.topk),
            stride=(variant.topk, 1),
            assumed_align=variant.topk_dtype.width // 8,
        )
        sorted_fake = cute.runtime.make_fake_tensor(
            Int32,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=4,
        )
        expert_ids_fake = cute.runtime.make_fake_tensor(
            Int32,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=4,
        )
        post_fake = cute.runtime.make_fake_tensor(
            Int32,
            (1,) if variant.kernel_type in ("small", "large") else (cute.sym_int(),),
            stride=(1,),
            assumed_align=4,
        )
        expert_map_fake = cute.runtime.make_fake_tensor(
            Int32,
            (variant.num_experts,),
            stride=(1,),
            assumed_align=4,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        if variant.kernel_type == "small":
            op = _MoeAlignBlockSizeCuTe(
                num_experts=variant.num_experts,
                block_size=variant.block_size,
                has_expert_map=variant.has_expert_map,
                config=cfg,
            )
            compiled = cute.compile(
                op,
                topk_ids_fake,
                sorted_fake,
                expert_ids_fake,
                post_fake,
                expert_map_fake,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        elif variant.kernel_type == "large":
            cumsum_fake = cute.runtime.make_fake_tensor(
                Int32,
                (variant.num_experts,),
                stride=(1,),
                assumed_align=4,
            )
            op = _MoeAlignBlockSizeCuTeLarge(
                num_experts=variant.num_experts,
                block_size=variant.block_size,
                has_expert_map=variant.has_expert_map,
                config=cfg,
            )
            compiled = cute.compile(
                op,
                topk_ids_fake,
                sorted_fake,
                expert_ids_fake,
                post_fake,
                expert_map_fake,
                cumsum_fake,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        elif variant.kernel_type == "lora_small":
            token_lora_fake = cute.runtime.make_fake_tensor(
                Int32,
                (t_sym,),
                stride=(1,),
                assumed_align=4,
            )
            sorted_stride_fake = cute.runtime.make_fake_tensor(
                Int32,
                (1,),
                stride=(1,),
                assumed_align=4,
            )
            expert_stride_fake = cute.runtime.make_fake_tensor(
                Int32,
                (1,),
                stride=(1,),
                assumed_align=4,
            )
            op = _MoeAlignBlockSizeCuTeLora(
                num_experts=variant.num_experts,
                block_size=variant.block_size,
                top_k=variant.topk,
                has_expert_map=variant.has_expert_map,
                config=cfg,
            )
            compiled = cute.compile(
                op,
                topk_ids_fake,
                token_lora_fake,
                sorted_fake,
                expert_ids_fake,
                post_fake,
                sorted_stride_fake,
                expert_stride_fake,
                expert_map_fake,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        elif variant.kernel_type == "lora_large":
            token_lora_fake = cute.runtime.make_fake_tensor(
                Int32,
                (t_sym,),
                stride=(1,),
                assumed_align=4,
            )
            sorted_stride_fake = cute.runtime.make_fake_tensor(
                Int32,
                (1,),
                stride=(1,),
                assumed_align=4,
            )
            expert_stride_fake = cute.runtime.make_fake_tensor(
                Int32,
                (1,),
                stride=(1,),
                assumed_align=4,
            )
            cumsum_fake = cute.runtime.make_fake_tensor(
                Int32,
                (cute.sym_int(),),
                stride=(1,),
                assumed_align=4,
            )
            op = _MoeAlignBlockSizeCuTeLargeLora(
                num_experts=variant.num_experts,
                block_size=variant.block_size,
                top_k=variant.topk,
                has_expert_map=variant.has_expert_map,
                config=cfg,
            )
            compiled = cute.compile(
                op,
                topk_ids_fake,
                token_lora_fake,
                sorted_fake,
                expert_ids_fake,
                post_fake,
                sorted_stride_fake,
                expert_stride_fake,
                expert_map_fake,
                cumsum_fake,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        else:
            raise ValueError(f"Unknown kernel type: {variant.kernel_type}")

        # Link to shared object
        so_path = output_dir / variant.filename(arch)
        compile_and_link(compiled, variant.function_name(arch), so_path, "moe_align_tmp")

        print(f"Created: {so_path.name}")
        return (variant, so_path, None)

    except Exception as e:
        return (variant, None, str(e))


def main():
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    output_dir = get_precompiled_dir()

    # Generate variants based on config files
    variants = generate_precompile_variants()
    print(f"Total variants to compile: {len(variants)}")

    # Compile all variants for the current architecture
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

    print(f"\nMoE align precompilation complete:")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed variants:")
        for variant, error in failed:
            print(f"  {variant.kernel_type}: {error}")
        sys.exit(1)

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
