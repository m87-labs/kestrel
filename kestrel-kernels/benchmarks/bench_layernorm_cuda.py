"""Benchmark LayerNorm kernel vs PyTorch.

The kernel computes: out = (x - mean) / sqrt(var + eps) * weight + bias

Used in both vision encoder and text decoder:
- Vision encoder: N=1152, 729 patches per crop (up to 13 crops)
- Text decoder: N=2048, decode (1 token) or prefill (740 tokens for image)
"""

import argparse

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from kestrel_kernels.layernorm_cuda import layernorm_bias_cuda, layernorm_bias_reload_cuda


def pytorch_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference PyTorch implementation."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def run_benchmark(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
    variant: str = "auto",
) -> tuple[float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, speedup)."""
    device = torch.device("cuda")
    eps = 1e-5

    x = torch.randn((num_tokens, hidden_dim), dtype=dtype, device=device)
    weight = torch.ones((hidden_dim,), dtype=dtype, device=device)
    bias = torch.zeros((hidden_dim,), dtype=dtype, device=device)
    out_cuda = torch.empty_like(x)

    # Select kernel variant
    if variant == "auto":
        use_reload = hidden_dim == 1152
    else:
        use_reload = variant == "reload_x"

    kernel_fn = layernorm_bias_reload_cuda if use_reload else layernorm_bias_cuda

    # Warmup
    for _ in range(warmup):
        kernel_fn(out_cuda, x, weight, bias, eps)
        pytorch_layernorm(x, weight, bias, eps)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="kernel_fn(out, x, weight, bias, eps)",
        globals={
            "kernel_fn": kernel_fn,
            "out": out_cuda,
            "x": x,
            "weight": weight,
            "bias": bias,
            "eps": eps,
        },
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="F.layer_norm(x, (x.shape[-1],), weight, bias, eps)",
        globals={
            "F": F,
            "x": x,
            "weight": weight,
            "bias": bias,
            "eps": eps,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, pytorch_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark LayerNorm kernel")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Vision encoder: N=1152, 729 patches per crop, up to 13 crops
    print("=" * 55)
    print("Vision Encoder LayerNorm (N=1152, reload_x variant)")
    print("=" * 55)
    header = f"{'Crops':>5} {'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 55)

    vision_sizes = [(1, 729), (2, 1458), (4, 2916), (8, 5832), (13, 9477)]
    for num_crops, num_tokens in vision_sizes:
        cuda_us, pytorch_us, speedup = run_benchmark(
            num_tokens, 1152, num_runs=args.num_runs, variant="reload_x"
        )
        print(f"{num_crops:>5} {num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {speedup:>9.2f}x")

    print()

    # Text decoder: N=2048
    print("=" * 55)
    print("Text Decoder LayerNorm (N=2048, default variant)")
    print("=" * 55)
    header = f"{'Context':>10} {'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 55)

    text_sizes = [
        ("decode", 1),
        ("prefill", 740),
    ]
    for context, num_tokens in text_sizes:
        cuda_us, pytorch_us, speedup = run_benchmark(
            num_tokens, 2048, num_runs=args.num_runs, variant="default"
        )
        print(f"{context:>10} {num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {speedup:>9.2f}x")

    print("=" * 55)


if __name__ == "__main__":
    main()
