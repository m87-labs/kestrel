"""Benchmark moe_align_block_size kernel vs vLLM.

Prepares sorted token indices for block-sparse MoE operations.
Given topk_ids [num_tokens, topk], outputs:
- sorted_token_ids: tokens sorted by expert assignment
- expert_ids: expert index for each block
- num_tokens_post_pad: total tokens after padding

Used in MoE layers:
- num_experts = 64
- topk = 8
- block_size = 128
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.moe_align import moe_align_block_size as kestrel_moe_align
from vllm._custom_ops import moe_align_block_size as vllm_moe_align_op


def vllm_moe_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor,
) -> None:
    """vLLM moe_align_block_size wrapper matching kestrel's signature."""
    vllm_moe_align_op(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        None,  # expert_map - vLLM ignores unless ignore_invalid_experts=True
    )


def run_benchmark(
    num_tokens: int,
    num_experts: int = 64,
    topk: int = 8,
    block_size: int = 128,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run benchmark and return (kestrel_us, vllm_us, speedup)."""
    device = torch.device("cuda")

    # Create test tensors
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)

    # Output tensors for kestrel
    max_num_tokens_padded = num_tokens * topk + num_experts * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_kestrel = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids_kestrel = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_post_kestrel = torch.empty((1,), dtype=torch.int32, device=device)
    expert_map = torch.empty((0,), dtype=torch.int32, device=device)

    # Output tensors for vLLM
    sorted_vllm = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids_vllm = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_post_vllm = torch.empty((1,), dtype=torch.int32, device=device)

    # Warmup kestrel
    for _ in range(warmup):
        kestrel_moe_align(topk_ids, num_experts, block_size, sorted_kestrel, expert_ids_kestrel, num_post_kestrel, expert_map)
    torch.cuda.synchronize()

    # Warmup vLLM
    for _ in range(warmup):
        vllm_moe_align(topk_ids, num_experts, block_size, sorted_vllm, expert_ids_vllm, num_post_vllm, expert_map)
    torch.cuda.synchronize()

    # Benchmark kestrel
    kestrel_timer = benchmark.Timer(
        stmt="kestrel_moe_align(topk_ids, num_experts, block_size, sorted_out, expert_ids, num_post, expert_map)",
        globals={
            "kestrel_moe_align": kestrel_moe_align,
            "topk_ids": topk_ids,
            "num_experts": num_experts,
            "block_size": block_size,
            "sorted_out": sorted_kestrel,
            "expert_ids": expert_ids_kestrel,
            "num_post": num_post_kestrel,
            "expert_map": expert_map,
        },
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark vLLM
    vllm_timer = benchmark.Timer(
        stmt="vllm_moe_align(topk_ids, num_experts, block_size, sorted_out, expert_ids, num_post, expert_map)",
        globals={
            "vllm_moe_align": vllm_moe_align,
            "topk_ids": topk_ids,
            "num_experts": num_experts,
            "block_size": block_size,
            "sorted_out": sorted_vllm,
            "expert_ids": expert_ids_vllm,
            "num_post": num_post_vllm,
            "expert_map": expert_map,
        },
    )
    vllm_us = vllm_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, vllm_us, vllm_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark moe_align_block_size kernel")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--block-size", type=int, default=128, help="Block size")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking moe_align_block_size (num_experts={args.num_experts}, topk={args.topk}, block_size={args.block_size})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    print("=" * 55)
    header = f"{'Context':>10} {'Tokens':>6} {'Kestrel':>10} {'vLLM':>10} {'vs vLLM':>10}"
    print(header)
    print("-" * 55)

    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
        ("long", 1024),
    ]

    for context, num_tokens in test_sizes:
        kestrel_us, vllm_us, speedup = run_benchmark(
            num_tokens,
            num_experts=args.num_experts,
            topk=args.topk,
            block_size=args.block_size,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {kestrel_us:>9.1f}us {vllm_us:>9.1f}us {speedup:>9.2f}x")

    print("=" * 55)


if __name__ == "__main__":
    main()
