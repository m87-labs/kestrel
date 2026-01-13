"""Benchmark fused linear + bias + residual kernel vs PyTorch and torch.compile.

The kernel computes: out = residual + (x @ W.T + bias)

Used in vision encoder attention projection:
- hidden = 1152 (enc_dim)
- 729 patches per crop (27x27), up to 13 crops
"""

import argparse

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from kestrel_kernels.fused_linear_residual import fused_linear_bias_residual_cuda


def pytorch_fused_linear_residual(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, residual: torch.Tensor, out: torch.Tensor
) -> None:
    """Reference PyTorch implementation."""
    out.copy_(F.linear(x, w, b) + residual)


@torch.compile(mode="max-autotune")
def _compiled_fused_linear_residual_impl(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Compiled implementation with cudagraphs."""
    return F.linear(x, w, b) + residual


def compiled_fused_linear_residual(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, residual: torch.Tensor, out: torch.Tensor
) -> None:
    """torch.compile version."""
    out.copy_(_compiled_fused_linear_residual_impl(x, w, b, residual))


def run_benchmark(
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, compile_us, speedup_vs_pytorch, speedup_vs_compile)."""
    device = torch.device("cuda")

    # Weight scale for numerical stability
    w_scale = hidden**-0.5

    x = torch.randn((num_tokens, hidden), dtype=dtype, device=device)
    w = torch.randn((hidden, hidden), dtype=dtype, device=device) * w_scale
    b = torch.randn((hidden,), dtype=dtype, device=device) * w_scale
    residual = torch.randn((num_tokens, hidden), dtype=dtype, device=device)
    out_cuda = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
    out_pytorch = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
    out_compile = torch.empty((num_tokens, hidden), dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        fused_linear_bias_residual_cuda(out_cuda, x, w, b, residual)
        pytorch_fused_linear_residual(x, w, b, residual, out_pytorch)
    for _ in range(warmup * 5):
        compiled_fused_linear_residual(x, w, b, residual, out_compile)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="fused_linear_bias_residual_cuda(out, x, w, b, residual)",
        globals={
            "fused_linear_bias_residual_cuda": fused_linear_bias_residual_cuda,
            "out": out_cuda,
            "x": x,
            "w": w,
            "b": b,
            "residual": residual,
        },
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_fused_linear_residual(x, w, b, residual, out)",
        globals={
            "pytorch_fused_linear_residual": pytorch_fused_linear_residual,
            "out": out_pytorch,
            "x": x,
            "w": w,
            "b": b,
            "residual": residual,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    # Benchmark torch.compile
    compile_timer = benchmark.Timer(
        stmt="compiled_fused_linear_residual(x, w, b, residual, out)",
        globals={
            "compiled_fused_linear_residual": compiled_fused_linear_residual,
            "out": out_compile,
            "x": x,
            "w": w,
            "b": b,
            "residual": residual,
        },
    )
    compile_us = compile_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, compile_us, pytorch_us / cuda_us, compile_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused linear + bias + residual kernel")
    parser.add_argument("--hidden", type=int, default=1152, help="Hidden dimension (default: 1152)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking fused linear + bias + residual (hidden={args.hidden})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Vision encoder: 729 patches per crop (27x27), up to 13 crops
    token_sizes = [729, 1458, 2916, 5832, 9477]  # 1, 2, 4, 8, 13 crops

    header = f"{'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'Compile':>10} {'vs PyTorch':>10}"
    print("=" * 50)
    print(header)
    print("=" * 50)

    for num_tokens in token_sizes:
        cuda_us, pytorch_us, compile_us, speedup_py, speedup_comp = run_benchmark(
            num_tokens, args.hidden, num_runs=args.num_runs
        )
        print(f"{num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {compile_us:>9.1f}us {speedup_py:>9.1f}x")

    print("=" * 50)


if __name__ == "__main__":
    main()
