#!/usr/bin/env python3
"""Precompile MoE align kernel variants for AOT deployment."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure torch is imported first (for libtorch.so)
import torch

import cutlass.cute as cute
from cutlass import Int32, Int64

# Insert package path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from kestrel_kernels.moe_align_cute import (
    MoeAlignCuTeConfig,
    _MoeAlignBlockSizeCuTe,
    _MoeAlignBlockSizeCuTeLarge,
    _MoeAlignBlockSizeCuTeLora,
    _MoeAlignBlockSizeCuTeLargeLora,
)

# Path to config JSON files
_CONFIGS_DIR = Path(__file__).parent.parent / "python" / "kestrel_kernels" / "configs"


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


def get_cuda_arch() -> str:
    """Get the CUDA architecture string (e.g., 'sm90' for Hopper)."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def get_block_m_values_from_configs(arch: str) -> set[int]:
    """Extract all unique block_m values from cute_moe config JSON files.

    Args:
        arch: CUDA architecture string (e.g., 'sm90')

    Returns:
        Set of unique block_m values found across all config files for the given arch.
    """
    block_m_values: set[int] = set()

    if not _CONFIGS_DIR.exists():
        print(f"Warning: Config directory {_CONFIGS_DIR} does not exist")
        return block_m_values

    # Find all config files for this architecture
    pattern = f"cute_moe_*_{arch}.json"
    config_files = list(_CONFIGS_DIR.glob(pattern))

    if not config_files:
        print(f"Warning: No config files found matching {pattern} in {_CONFIGS_DIR}")
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


def generate_precompile_variants(arch: str) -> list[MoeAlignVariant]:
    """Generate precompile variants based on block_m values from config files.

    Args:
        arch: CUDA architecture string (e.g., 'sm90')

    Returns:
        List of MoeAlignVariant to precompile.
    """
    block_m_values = get_block_m_values_from_configs(arch)

    if not block_m_values:
        # Fallback to default values if no configs found
        print("Warning: No block_m values found in configs, using defaults")
        block_m_values = {16, 32, 64, 128}

    print(f"Block_m values from configs: {sorted(block_m_values)}")

    variants = []

    # Generate all kernel types for each block_size and dtype
    kernel_types = ["small", "large", "lora_small", "lora_large"]
    dtypes = [Int32, Int64]

    for block_size in sorted(block_m_values):
        for dtype in dtypes:
            for kernel_type in kernel_types:
                variants.append(
                    MoeAlignVariant(kernel_type, dtype, 8, 64, block_size, False)
                )

    return variants


def compile_variant(variant: MoeAlignVariant, arch: str, output_dir: Path) -> Path:
    """Compile a single variant and return the output path."""
    print(
        f"Compiling: type={variant.kernel_type}, dtype={'int64' if variant.topk_dtype == Int64 else 'int32'}, "
        f"topk={variant.topk}, num_experts={variant.num_experts}, block_size={variant.block_size}, "
        f"has_expert_map={variant.has_expert_map}"
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

    # Export to object file
    obj_path = output_dir / "moe_align_tmp.o"
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
    print(f"Linking: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Clean up object file
    obj_path.unlink()

    print(f"Created: {so_path}")
    return so_path


def main():
    # Detect architecture
    arch = get_cuda_arch()
    print(f"Detected CUDA architecture: {arch}")

    # Output directory for precompiled kernels
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "python" / "kestrel_kernels" / "precompiled"
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py in precompiled directory
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Precompiled CuTe DSL kernels."""\n')

    # Generate variants based on config files
    variants = generate_precompile_variants(arch)
    print(f"Total variants to compile: {len(variants)}")

    # Compile all variants for the current architecture
    failed = []
    succeeded = []

    for variant in variants:
        try:
            so_path = compile_variant(variant, arch, output_dir)
            succeeded.append((variant, so_path))
        except Exception as e:
            failed.append((variant, str(e)))
            print(f"Failed to compile {variant.kernel_type}: {e}")

    print(f"\nPrecompilation complete:")
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
