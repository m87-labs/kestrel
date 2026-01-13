"""Benchmark FP8 row-wise quantization: CuTe DSL vs CUDA kernel vs PyTorch.

Sizes are based on Moondream MoE configuration:
- hidden = 1024 (intermediate_size)
- topk = 8 (experts per token)
- Rows = num_tokens * topk
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.fp8_quant import fp8_e4m3fn_rowwise_quant_cuda
from kestrel_kernels.fp8_quant_cute import fp8_quant_cute

FP8_E4M3_MAX = 448.0


def pytorch_fp8_rowwise_quant(
    x: torch.Tensor,
    out_bits: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    """Reference PyTorch implementation of per-row FP8 quantization."""
    absmax = x.abs().amax(dim=1)
    scale = absmax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-6)
    out_scale.copy_(scale)
    x_scaled = x / scale.unsqueeze(1)
    x_clamped = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    out_bits.copy_(x_clamped.to(torch.float8_e4m3fn).view(torch.uint8))


def run_benchmark(
    num_rows: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
    use_cuda_graph: bool = False,
) -> tuple[float, float, float]:
    """Run benchmark. Returns (cute_us, cuda_us, pytorch_us)."""
    device = torch.device("cuda")

    x = torch.randn((num_rows, hidden), dtype=dtype, device=device)

    # CuTe outputs
    out_bits_cute = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_cute = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # CUDA outputs
    out_bits_cuda = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_cuda = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # PyTorch outputs
    out_bits_pytorch = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_pytorch = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        fp8_quant_cute(out_bits_cute, out_scale_cute, x)
        fp8_e4m3fn_rowwise_quant_cuda(out_bits_cuda, out_scale_cuda, x)
        pytorch_fp8_rowwise_quant(x, out_bits_pytorch, out_scale_pytorch)
    torch.cuda.synchronize()

    if use_cuda_graph:
        # Capture kernels in CUDA graphs to eliminate Python overhead
        g_cute = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_cute):
            fp8_quant_cute(out_bits_cute, out_scale_cute, x)

        cute_timer = benchmark.Timer(
            stmt="g.replay()",
            globals={"g": g_cute},
        )
        cute_us = cute_timer.timeit(num_runs).mean * 1e6

        # CUDA kernel with graph
        g_cuda = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_cuda):
            fp8_e4m3fn_rowwise_quant_cuda(out_bits_cuda, out_scale_cuda, x)

        cuda_timer = benchmark.Timer(
            stmt="g.replay()",
            globals={"g": g_cuda},
        )
        cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

        # PyTorch with CUDA graph
        g_pytorch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_pytorch):
            pytorch_fp8_rowwise_quant(x, out_bits_pytorch, out_scale_pytorch)

        pytorch_timer = benchmark.Timer(
            stmt="g.replay()",
            globals={"g": g_pytorch},
        )
        pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6
    else:
        # Benchmark CuTe
        cute_timer = benchmark.Timer(
            stmt="fp8_quant_cute(out_bits, out_scale, x)",
            globals={
                "fp8_quant_cute": fp8_quant_cute,
                "out_bits": out_bits_cute,
                "out_scale": out_scale_cute,
                "x": x,
            },
        )
        cute_us = cute_timer.timeit(num_runs).mean * 1e6

        # Benchmark CUDA
        cuda_timer = benchmark.Timer(
            stmt="fp8_e4m3fn_rowwise_quant_cuda(out_bits, out_scale, x)",
            globals={
                "fp8_e4m3fn_rowwise_quant_cuda": fp8_e4m3fn_rowwise_quant_cuda,
                "out_bits": out_bits_cuda,
                "out_scale": out_scale_cuda,
                "x": x,
            },
        )
        cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

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

    return cute_us, cuda_us, pytorch_us


def verify_correctness(num_rows: int, hidden: int):
    """Verify CuTe kernel produces correct results vs CUDA kernel."""
    device = torch.device("cuda")
    x = torch.randn((num_rows, hidden), dtype=torch.bfloat16, device=device)

    # CuTe outputs
    out_bits_cute = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_cute = torch.empty((num_rows,), dtype=torch.float32, device=device)

    # CUDA outputs
    out_bits_cuda = torch.empty((num_rows, hidden), dtype=torch.uint8, device=device)
    out_scale_cuda = torch.empty((num_rows,), dtype=torch.float32, device=device)

    fp8_quant_cute(out_bits_cute, out_scale_cute, x)
    fp8_e4m3fn_rowwise_quant_cuda(out_bits_cuda, out_scale_cuda, x)

    # Check scales match
    scale_match = torch.allclose(out_scale_cute, out_scale_cuda, rtol=1e-3, atol=1e-6)

    # Check quantized values match
    bits_match = torch.equal(out_bits_cute, out_bits_cuda)

    return scale_match, bits_match


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 quantization: CuTe vs CUDA")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--verify", action="store_true", help="Verify correctness first")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graphs to eliminate Python overhead")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking FP8 row-wise quantization (hidden={args.hidden})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    if args.cuda_graph:
        print("Mode: CUDA Graph (no Python overhead)")
    print()

    if args.verify:
        print("Verifying correctness...")
        for rows in [8, 32, 128, 5920]:
            scale_ok, bits_ok = verify_correctness(rows, args.hidden)
            status = "OK" if (scale_ok and bits_ok) else "FAIL"
            print(f"  rows={rows}: scale={scale_ok}, bits={bits_ok} -> {status}")
        print()

    test_configs = [
        ("decode", 1, 8),
        ("batch 4", 4, 32),
        ("batch 16", 16, 128),
        ("prefill", 740, 5920),
    ]

    print("=" * 75)
    print("FP8 Row-wise Quantization (per-token dynamic scaling)")
    print("=" * 75)
    header = f"{'Context':>10} {'Rows':>6} {'CuTe':>10} {'CUDA':>10} {'PyTorch':>10} {'vs CUDA':>10} {'vs PT':>10}"
    print(header)
    print("-" * 75)

    for context, num_tokens, num_rows in test_configs:
        cute_us, cuda_us, pytorch_us = run_benchmark(
            num_rows, args.hidden, num_runs=args.num_runs, use_cuda_graph=args.cuda_graph
        )
        vs_cuda = cuda_us / cute_us
        vs_pytorch = pytorch_us / cute_us
        print(
            f"{context:>10} {num_rows:>6} "
            f"{cute_us:>9.1f}us {cuda_us:>9.1f}us {pytorch_us:>9.1f}us "
            f"{vs_cuda:>9.2f}x {vs_pytorch:>9.1f}x"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()
