"""Benchmark MoE sum kernel vs PyTorch.

The kernel computes: out = sum(input, dim=1) for MoE expert outputs.
Input shape: [num_tokens, topk, hidden_dim]
Output shape: [num_tokens, hidden_dim]

Used in text decoder MoE layers:
- topk = 8 experts per token
- hidden_dim = 2048
- decode: 1 token, prefill: 740 tokens
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.moe_sum import moe_sum


def run_benchmark(
    num_tokens: int,
    topk: int = 8,
    hidden_dim: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, speedup)."""
    device = torch.device("cuda")

    input_tensor = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device=device)
    out_cuda = torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)
    out_pytorch = torch.empty_like(out_cuda)

    # Warmup
    for _ in range(warmup):
        moe_sum(input_tensor, out_cuda)
        torch.sum(input_tensor, dim=1, out=out_pytorch)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="moe_sum(input_tensor, out)",
        globals={
            "moe_sum": moe_sum,
            "input_tensor": input_tensor,
            "out": out_cuda,
        },
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="torch.sum(input_tensor, dim=1, out=out)",
        globals={
            "torch": torch,
            "input_tensor": input_tensor,
            "out": out_pytorch,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, pytorch_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE sum kernel")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts (default: 8)")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension (default: 2048)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking MoE sum (topk={args.topk}, hidden={args.hidden_dim})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    print("=" * 50)
    header = f"{'Context':>10} {'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 50)

    # Text decoder MoE contexts
    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
        ("long", 1024),
    ]

    for context, num_tokens in test_sizes:
        cuda_us, pytorch_us, speedup = run_benchmark(
            num_tokens,
            topk=args.topk,
            hidden_dim=args.hidden_dim,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {speedup:>9.2f}x")

    print("=" * 50)


if __name__ == "__main__":
    main()
