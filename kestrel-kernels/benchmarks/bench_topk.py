"""Benchmark topk kernel vs quack and PyTorch.

Bitonic top-k selection with optional fused softmax.

Used in MoE routing:
- N = 64 experts
- k = 8 top experts per token
- With softmax for expert weights
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.topk import topk_fwd
from quack.topk import topk as quack_topk


def pytorch_topk_softmax(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager implementation of top-k with softmax."""
    values, indices = torch.topk(x, k, dim=-1)
    values = torch.softmax(values, dim=-1)
    return values, indices


def run_benchmark(
    num_tokens: int,
    N: int = 64,
    k: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float, float, float]:
    """Run benchmark and return (kestrel_us, quack_us, pytorch_us, vs_quack, vs_pytorch)."""
    device = torch.device("cuda")

    x = torch.randn((num_tokens, N), dtype=dtype, device=device)

    # Warmup kestrel
    for _ in range(warmup):
        topk_fwd(x, k, softmax=True)
    torch.cuda.synchronize()

    # Warmup quack
    for _ in range(warmup):
        quack_topk(x, k)
    torch.cuda.synchronize()

    # Warmup PyTorch
    for _ in range(warmup):
        pytorch_topk_softmax(x, k)
    torch.cuda.synchronize()

    # Benchmark kestrel
    kestrel_timer = benchmark.Timer(
        stmt="topk_fwd(x, k, softmax=True)",
        globals={
            "topk_fwd": topk_fwd,
            "x": x,
            "k": k,
        },
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark quack (no softmax, so we add it separately for fair comparison)
    # But actually quack doesn't have fused softmax, so we benchmark quack alone
    # to show the advantage of fusion
    quack_timer = benchmark.Timer(
        stmt="quack_topk(x, k)",
        globals={
            "quack_topk": quack_topk,
            "x": x,
            "k": k,
        },
    )
    quack_us = quack_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch (topk + softmax)
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_topk_softmax(x, k)",
        globals={
            "pytorch_topk_softmax": pytorch_topk_softmax,
            "x": x,
            "k": k,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, quack_us, pytorch_us, quack_us / kestrel_us, pytorch_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark topk kernel")
    parser.add_argument("--N", type=int, default=64, help="Number of experts (default: 64)")
    parser.add_argument("--k", type=int, default=8, help="Top-k (default: 8)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking topk (N={args.N}, k={args.k})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    print("Note: Kestrel includes fused softmax, quack does not")
    print()

    print("=" * 70)
    header = f"{'Context':>10} {'Tokens':>6} {'Kestrel':>10} {'Quack':>10} {'PyTorch':>10} {'vs Quack':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 70)

    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
        ("long", 1024),
    ]

    for context, num_tokens in test_sizes:
        kestrel_us, quack_us, pytorch_us, vs_quack, vs_pytorch = run_benchmark(
            num_tokens,
            N=args.N,
            k=args.k,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {kestrel_us:>9.1f}us {quack_us:>9.1f}us {pytorch_us:>9.1f}us {vs_quack:>9.2f}x {vs_pytorch:>9.1f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
