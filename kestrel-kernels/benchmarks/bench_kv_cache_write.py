"""Benchmark FP8 KV cache write kernel vs vLLM.

The kernel writes BF16 key/value tensors to FP8 paged KV cache with quantization.

Used in text decoder attention:
- n_kv_heads = 32
- head_dim = 64
- block_size = 16 (typical paged attention block)
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.kv_cache_write import reshape_and_cache_flash
from vllm import _custom_ops as vllm_ops


def pytorch_kv_cache_write_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_size: int,
) -> None:
    """Vectorized PyTorch implementation for FP8 KV cache write."""
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size
    # Quantize and scatter
    key_cache[block_indices, block_offsets] = (key.float() / k_scale).to(torch.float8_e4m3fn)
    value_cache[block_indices, block_offsets] = (value.float() / v_scale).to(torch.float8_e4m3fn)


def run_benchmark(
    num_tokens: int,
    num_kv_heads: int = 32,
    head_dim: int = 64,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float, float, float]:
    """Run benchmark and return (kestrel_us, vllm_us, pytorch_us, vs_vllm, vs_pytorch)."""
    device = torch.device("cuda")

    # Allocate enough blocks for the tokens
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    key = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=dtype, device=device)
    value = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=dtype, device=device)

    # FP8 cache: [num_blocks, block_size, num_kv_heads, head_dim]
    key_cache_kestrel = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    value_cache_kestrel = torch.zeros_like(key_cache_kestrel)

    key_cache_vllm = torch.zeros_like(key_cache_kestrel)
    value_cache_vllm = torch.zeros_like(key_cache_vllm)

    key_cache_pytorch = torch.zeros_like(key_cache_kestrel)
    value_cache_pytorch = torch.zeros_like(key_cache_pytorch)

    # Sequential slot mapping (no padding)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warmup kestrel
    for _ in range(warmup):
        reshape_and_cache_flash(
            key, value,
            key_cache_kestrel.view(torch.uint8),
            value_cache_kestrel.view(torch.uint8),
            slot_mapping,
            "fp8_e4m3",
            k_scale, v_scale,
        )
    torch.cuda.synchronize()

    # Warmup vLLM
    for _ in range(warmup):
        vllm_ops.reshape_and_cache_flash(
            key, value,
            key_cache_vllm,
            value_cache_vllm,
            slot_mapping,
            "fp8",
            k_scale, v_scale,
        )
    torch.cuda.synchronize()

    # Warmup PyTorch
    for _ in range(warmup):
        pytorch_kv_cache_write_fp8(
            key, value,
            key_cache_pytorch,
            value_cache_pytorch,
            slot_mapping,
            k_scale, v_scale,
            block_size,
        )
    torch.cuda.synchronize()

    # Benchmark kestrel kernel
    kestrel_timer = benchmark.Timer(
        stmt="reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, 'fp8_e4m3', k_scale, v_scale)",
        globals={
            "reshape_and_cache_flash": reshape_and_cache_flash,
            "key": key,
            "value": value,
            "key_cache": key_cache_kestrel.view(torch.uint8),
            "value_cache": value_cache_kestrel.view(torch.uint8),
            "slot_mapping": slot_mapping,
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark vLLM
    vllm_timer = benchmark.Timer(
        stmt="vllm_reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, 'fp8', k_scale, v_scale)",
        globals={
            "vllm_reshape_and_cache_flash": vllm_ops.reshape_and_cache_flash,
            "key": key,
            "value": value,
            "key_cache": key_cache_vllm,
            "value_cache": value_cache_vllm,
            "slot_mapping": slot_mapping,
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
    )
    vllm_us = vllm_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_kv_cache_write_fp8(key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale, block_size)",
        globals={
            "pytorch_kv_cache_write_fp8": pytorch_kv_cache_write_fp8,
            "key": key,
            "value": value,
            "key_cache": key_cache_pytorch,
            "value_cache": value_cache_pytorch,
            "slot_mapping": slot_mapping,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "block_size": block_size,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, vllm_us, pytorch_us, vllm_us / kestrel_us, pytorch_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 KV cache write kernel")
    parser.add_argument("--num-kv-heads", type=int, default=32, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--block-size", type=int, default=16, help="Block size")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking FP8 KV cache write (heads={args.num_kv_heads}, head_dim={args.head_dim}, block={args.block_size})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Decode: typically 1-8 tokens at a time
    # Prefill: up to max_context (4096)
    token_sizes = [1, 8, 64, 256, 1024, 4096]

    header = f"{'Tokens':>6} {'Kestrel':>10} {'vLLM':>10} {'PyTorch':>10} {'vs vLLM':>10} {'vs PyTorch':>10}"
    print("=" * 62)
    print(header)
    print("=" * 62)

    for num_tokens in token_sizes:
        kestrel_us, vllm_us, pytorch_us, vs_vllm, vs_pytorch = run_benchmark(
            num_tokens,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            num_runs=args.num_runs,
        )
        print(f"{num_tokens:>6} {kestrel_us:>9.1f}us {vllm_us:>9.1f}us {pytorch_us:>9.1f}us {vs_vllm:>9.2f}x {vs_pytorch:>9.1f}x")

    print("=" * 62)


if __name__ == "__main__":
    main()
