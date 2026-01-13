"""Benchmark GELU residual activation kernel vs PyTorch and torch.compile.

The activation kernel computes: GELU(h) * (g + 1)
where h and g are the two halves of the input tensor.

Sizes are based on Moondream model configuration:
- hidden = 1024 (expert_inner_dim)
- Prefill: 729 tokens (vision patches), up to 4096 (max context)
- Decode: 1-64 tokens (batch size)
"""

import argparse
import math

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.activation import gelu_residual_cuda


def pytorch_gelu_residual(x: torch.Tensor, out: torch.Tensor) -> None:
    """Reference PyTorch implementation."""
    hidden = x.shape[1] // 2
    h = x[:, :hidden]
    g = x[:, hidden:]
    gelu = 0.5 * h * (1.0 + torch.erf(h * (1.0 / math.sqrt(2.0))))
    out.copy_(gelu * (g + 1.0))


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compiled_gelu_residual_impl(x: torch.Tensor) -> torch.Tensor:
    """Compiled GELU residual implementation."""
    hidden = x.shape[1] // 2
    h = x[:, :hidden]
    g = x[:, hidden:]
    gelu = 0.5 * h * (1.0 + torch.erf(h * (1.0 / math.sqrt(2.0))))
    return gelu * (g + 1.0)


def compiled_gelu_residual(x: torch.Tensor, out: torch.Tensor) -> None:
    """torch.compile version."""
    out.copy_(_compiled_gelu_residual_impl(x))


def run_benchmark(
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, compile_us, speedup_vs_pytorch, speedup_vs_compile)."""
    device = torch.device("cuda")

    x = torch.randn((num_tokens, hidden * 2), dtype=dtype, device=device)
    out_cuda = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
    out_pytorch = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
    out_compile = torch.empty((num_tokens, hidden), dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        gelu_residual_cuda(out_cuda, x)
        pytorch_gelu_residual(x, out_pytorch)
    for _ in range(warmup * 5):
        compiled_gelu_residual(x, out_compile)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="gelu_residual_cuda(out, x)",
        globals={"gelu_residual_cuda": gelu_residual_cuda, "out": out_cuda, "x": x},
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_gelu_residual(x, out)",
        globals={"pytorch_gelu_residual": pytorch_gelu_residual, "out": out_pytorch, "x": x},
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    # Benchmark torch.compile
    compile_timer = benchmark.Timer(
        stmt="compiled_gelu_residual(x, out)",
        globals={"compiled_gelu_residual": compiled_gelu_residual, "out": out_compile, "x": x},
    )
    compile_us = compile_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, compile_us, pytorch_us / cuda_us, compile_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark GELU residual activation kernel")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking GELU residual activation (hidden={args.hidden})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    token_sizes = [1, 64, 740, 1024, 2048]

    header = f"{'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'Compile':>10} {'vs PyTorch':>10}"
    print("=" * 50)
    print(header)
    print("=" * 50)

    for num_tokens in token_sizes:
        cuda_us, pytorch_us, compile_us, speedup_py, speedup_comp = run_benchmark(
            num_tokens, args.hidden, num_runs=args.num_runs
        )
        print(f"{num_tokens:>6} {cuda_us:>9.2f}us {pytorch_us:>9.2f}us {compile_us:>9.2f}us {speedup_py:>9.1f}x")

    print("=" * 50)


if __name__ == "__main__":
    main()
