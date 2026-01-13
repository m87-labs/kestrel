"""Benchmark FP8 row-wise quantization kernel vs PyTorch.

The kernel quantizes BF16 tensors to FP8 (e4m3fn) with per-row dynamic scaling.
Used in MoE to quantize activations before the down projection.

Sizes are based on Moondream MoE configuration:
- hidden = 1024 (intermediate_size)
- topk = 8 (experts per token)
- Rows = num_tokens * topk
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.fp8_quant import fp8_e4m3fn_rowwise_quant_cuda

FP8_E4M3_MAX = 448.0


def pytorch_fp8_rowwise_quant(
    x: torch.Tensor,
    out_bits: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    """Reference PyTorch implementation of per-row FP8 quantization."""
    # Per-row absmax
    absmax = x.abs().amax(dim=1)
    # Compute scale
    scale = absmax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-6)
    out_scale.copy_(scale)
    # Quantize
    x_scaled = x / scale.unsqueeze(1)
    x_clamped = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    out_bits.copy_(x_clamped.to(torch.float8_e4m3fn).view(torch.uint8))


def run_benchmark(
    num_rows: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run benchmark. Returns (kestrel_us, pytorch_us, speedup)."""
    device = torch.device("cuda")

    x = torch.randn((num_rows, hidden), dtype=dtype, device=device)

    # Kestrel outputs
    out_bits_kestrel = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_kestrel = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # PyTorch outputs
    out_bits_pytorch = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_pytorch = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        fp8_e4m3fn_rowwise_quant_cuda(out_bits_kestrel, out_scale_kestrel, x)
        pytorch_fp8_rowwise_quant(x, out_bits_pytorch, out_scale_pytorch)
    torch.cuda.synchronize()

    # Benchmark Kestrel
    kestrel_timer = benchmark.Timer(
        stmt="fp8_e4m3fn_rowwise_quant_cuda(out_bits, out_scale, x)",
        globals={
            "fp8_e4m3fn_rowwise_quant_cuda": fp8_e4m3fn_rowwise_quant_cuda,
            "out_bits": out_bits_kestrel,
            "out_scale": out_scale_kestrel,
            "x": x,
        },
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_fp8_rowwise_quant(x, out_bits, out_scale)",
        globals={
            "pytorch_fp8_rowwise_quant": pytorch_fp8_rowwise_quant,
            "out_bits": out_bits_pytorch,
            "out_scale": out_scale_pytorch,
            "x": x,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, pytorch_us, pytorch_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 row-wise quantization")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking FP8 row-wise quantization (hidden={args.hidden})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Test sizes: num_tokens * topk for MoE
    # topk=8 for Moondream
    test_configs = [
        ("decode", 1, 8),       # 1 token * 8 experts
        ("batch 4", 4, 32),     # 4 tokens * 8 experts
        ("batch 16", 16, 128),  # 16 tokens * 8 experts
        ("prefill", 740, 5920), # 740 tokens * 8 experts
    ]

    print("=" * 55)
    print("FP8 Row-wise Quantization (per-token dynamic scaling)")
    print("=" * 55)
    header = f"{'Context':>10} {'Rows':>6} {'Kestrel':>12} {'PyTorch':>12} {'vs PyTorch':>12}"
    print(header)
    print("-" * 55)

    for context, num_tokens, num_rows in test_configs:
        kestrel_us, pytorch_us, speedup = run_benchmark(
            num_rows, args.hidden, num_runs=args.num_runs
        )
        print(
            f"{context:>10} {num_rows:>6} "
            f"{kestrel_us:>11.1f}us {pytorch_us:>11.1f}us "
            f"{speedup:>11.1f}x"
        )

    print("=" * 55)


if __name__ == "__main__":
    main()
