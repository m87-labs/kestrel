"""Benchmark fused MLP kernel vs PyTorch.

The kernel computes: out = residual + (gelu(x @ W1.T + b1) @ W2.T + b2)

Used in vision encoder MLP:
- in_dim = 1152 (enc_dim)
- hidden_dim = 4304
- 729 patches per crop (27x27), up to 13 crops
"""

import argparse

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from kestrel_kernels.fused_mlp import fused_mlp_gelu_bias_residual_cuda


def pytorch_fused_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    """Reference PyTorch implementation."""
    out.copy_(F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2) + residual)


def run_benchmark(
    num_tokens: int,
    in_dim: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, speedup)."""
    device = torch.device("cuda")

    w1_scale = in_dim**-0.5
    w2_scale = hidden_dim**-0.5

    x = torch.randn((num_tokens, in_dim), dtype=dtype, device=device)
    w1 = torch.randn((hidden_dim, in_dim), dtype=dtype, device=device) * w1_scale
    b1 = torch.zeros((hidden_dim,), dtype=dtype, device=device)
    w2 = torch.randn((in_dim, hidden_dim), dtype=dtype, device=device) * w2_scale
    b2 = torch.zeros((in_dim,), dtype=dtype, device=device)
    residual = torch.randn((num_tokens, in_dim), dtype=dtype, device=device)
    out_cuda = torch.empty((num_tokens, in_dim), dtype=dtype, device=device)
    out_pytorch = torch.empty((num_tokens, in_dim), dtype=dtype, device=device)
    hidden = torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        fused_mlp_gelu_bias_residual_cuda(out_cuda, hidden, x, w1, b1, w2, b2, residual)
        pytorch_fused_mlp(x, w1, b1, w2, b2, residual, out_pytorch, hidden)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="fused_mlp_gelu_bias_residual_cuda(out, hidden, x, w1, b1, w2, b2, residual)",
        globals={
            "fused_mlp_gelu_bias_residual_cuda": fused_mlp_gelu_bias_residual_cuda,
            "out": out_cuda,
            "hidden": hidden,
            "x": x,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "residual": residual,
        },
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_fused_mlp(x, w1, b1, w2, b2, residual, out, hidden)",
        globals={
            "pytorch_fused_mlp": pytorch_fused_mlp,
            "out": out_pytorch,
            "hidden": hidden,
            "x": x,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "residual": residual,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, pytorch_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused MLP kernel")
    parser.add_argument("--in-dim", type=int, default=1152, help="Input dimension (default: 1152)")
    parser.add_argument("--hidden-dim", type=int, default=4304, help="Hidden dimension (default: 4304)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking fused MLP (in={args.in_dim}, hidden={args.hidden_dim})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Vision encoder: 729 patches per crop, up to 13 crops
    token_sizes = [729, 1458, 2916, 5832, 9477]  # 1, 2, 4, 8, 13 crops

    header = f"{'Crops':>5} {'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'vs Eager':>10}"
    print("=" * 45)
    print(header)
    print("=" * 45)

    crops = [1, 2, 4, 8, 13]
    for num_crops, num_tokens in zip(crops, token_sizes):
        cuda_us, pytorch_us, speedup = run_benchmark(
            num_tokens, args.in_dim, args.hidden_dim, num_runs=args.num_runs
        )
        print(f"{num_crops:>5} {num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {speedup:>9.2f}x")

    print("=" * 45)


if __name__ == "__main__":
    main()
