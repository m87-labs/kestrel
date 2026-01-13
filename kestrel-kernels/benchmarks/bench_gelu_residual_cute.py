"""Benchmark and test GELU residual CuTe DSL kernel.

Compares:
1. CuTe DSL kernel
2. Original CUDA kernel (if available)
3. PyTorch reference
"""

import argparse
import sys
import os
import importlib.util

import torch
import torch.utils.benchmark as benchmark


def _import_cuda_kernel():
    """Import CUDA kernel from installed package."""
    try:
        from kestrel_kernels.activation import gelu_residual_cuda
        return gelu_residual_cuda
    except ImportError:
        return None


def _import_cute_kernel():
    """Import CuTe DSL kernel from local source."""
    source_path = os.path.join(os.path.dirname(__file__), "..", "python")
    spec = importlib.util.spec_from_file_location(
        "gelu_residual_kernel",
        os.path.join(source_path, "kestrel_kernels", "gelu_residual", "kernel.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gelu_residual_cute


# Module-level imports
_gelu_residual_cuda = _import_cuda_kernel()
HAS_CUDA_KERNEL = _gelu_residual_cuda is not None


def pytorch_gelu_residual(x: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation."""
    hidden = x.shape[-1] // 2
    h = x[..., :hidden]
    g = x[..., hidden:]
    return torch.nn.functional.gelu(h, approximate="none") * (g + 1)


def run_benchmark(
    num_tokens: int,
    hidden: int,
    num_runs: int = 100,
    warmup: int = 10,
    use_cuda_graph: bool = False,
) -> dict:
    """Run benchmark comparing implementations."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create input tensor (standard layout)
    x = torch.randn((num_tokens, 2 * hidden), dtype=dtype, device=device)

    # Output tensors
    out_cute = torch.empty((num_tokens, hidden), dtype=dtype, device=device)

    # Use CUDA kernel from module-level import
    has_cuda = HAS_CUDA_KERNEL
    gelu_residual_cuda = _gelu_residual_cuda
    out_cuda = torch.empty((num_tokens, hidden), dtype=dtype, device=device) if has_cuda else None

    # Import CuTe DSL kernel from local source
    gelu_residual_cute = _import_cute_kernel()

    # Warmup and correctness check
    for _ in range(warmup):
        gelu_residual_cute(out_cute, x)
        if has_cuda:
            gelu_residual_cuda(out_cuda, x)
    torch.cuda.synchronize()

    expected = pytorch_gelu_residual(x)

    # Check correctness
    cute_err = (out_cute - expected).abs().max().item()
    cuda_err = (out_cuda - expected).abs().max().item() if has_cuda else float('nan')

    results = {
        "cute_max_err": cute_err,
        "cuda_max_err": cuda_err,
    }

    if use_cuda_graph:
        # Capture kernels in CUDA graphs to eliminate Python overhead
        g_cute = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_cute):
            gelu_residual_cute(out_cute, x)
        torch.cuda.synchronize()

        cute_timer = benchmark.Timer(
            stmt="g.replay()",
            globals={"g": g_cute},
        )
        results["cute_us"] = cute_timer.timeit(num_runs).mean * 1e6

        if has_cuda:
            g_cuda = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cuda):
                gelu_residual_cuda(out_cuda, x)
            torch.cuda.synchronize()

            cuda_timer = benchmark.Timer(
                stmt="g.replay()",
                globals={"g": g_cuda},
            )
            results["cuda_us"] = cuda_timer.timeit(num_runs).mean * 1e6
        else:
            results["cuda_us"] = float('nan')

        # PyTorch with CUDA graph
        out_pytorch = torch.empty((num_tokens, hidden), dtype=dtype, device=device)
        g_pytorch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_pytorch):
            hidden_half = x.shape[-1] // 2
            h = x[..., :hidden_half]
            g = x[..., hidden_half:]
            out_pytorch.copy_(torch.nn.functional.gelu(h, approximate="none") * (g + 1))
        torch.cuda.synchronize()

        pytorch_timer = benchmark.Timer(
            stmt="g.replay()",
            globals={"g": g_pytorch},
        )
        results["pytorch_us"] = pytorch_timer.timeit(num_runs).mean * 1e6
    else:
        # Benchmark CuTe DSL
        cute_timer = benchmark.Timer(
            stmt="gelu_residual_cute(out, x)",
            globals={"gelu_residual_cute": gelu_residual_cute, "out": out_cute, "x": x},
        )
        results["cute_us"] = cute_timer.timeit(num_runs).mean * 1e6

        # Benchmark CUDA (if available)
        if has_cuda:
            cuda_timer = benchmark.Timer(
                stmt="gelu_residual_cuda(out, x)",
                globals={"gelu_residual_cuda": gelu_residual_cuda, "out": out_cuda, "x": x},
            )
            results["cuda_us"] = cuda_timer.timeit(num_runs).mean * 1e6
        else:
            results["cuda_us"] = float('nan')

        # Benchmark PyTorch
        pytorch_timer = benchmark.Timer(
            stmt="pytorch_gelu_residual(x)",
            globals={"pytorch_gelu_residual": pytorch_gelu_residual, "x": x},
        )
        results["pytorch_us"] = pytorch_timer.timeit(num_runs).mean * 1e6

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark GELU residual kernels")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graphs to eliminate Python overhead")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking GELU residual (hidden={args.hidden})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    if args.cuda_graph:
        print("Mode: CUDA Graph (no Python overhead)")
    print()

    # Test sizes matching MoE usage
    # For MoE: rows = num_tokens * topk (topk=8 for Moondream)
    test_configs = [
        ("decode", 1, 8),       # 1 token * 8 experts
        ("batch 4", 4, 32),    # 4 tokens * 8 experts
        ("batch 16", 16, 128),  # 16 tokens * 8 experts
        ("prefill", 740, 5920), # 740 tokens * 8 experts
    ]

    print("=" * 85)
    print("GELU Residual: out = GELU(x) * (y + 1)")
    print("=" * 85)
    header = f"{'Context':>10} {'Rows':>6} {'CuTe':>12} {'CUDA':>12} {'PyTorch':>12} {'vs CUDA':>10}"
    print(header)
    print("-" * 85)

    for context, num_tokens, num_rows in test_configs:
        results = run_benchmark(num_rows, args.hidden, num_runs=args.num_runs, use_cuda_graph=args.cuda_graph)
        cuda_str = f"{results['cuda_us']:>10.1f}us" if not results['cuda_us'] != results['cuda_us'] else "       N/A"
        speedup_str = f"{results['cuda_us'] / results['cute_us']:>9.2f}x" if not results['cuda_us'] != results['cuda_us'] else "      N/A"
        print(
            f"{context:>10} {num_rows:>6} "
            f"{results['cute_us']:>10.1f}us {cuda_str} {results['pytorch_us']:>10.1f}us "
            f"{speedup_str}"
        )

    print("=" * 85)
    print()
    print("Correctness (max absolute error):")
    print(f"  CuTe: {results['cute_max_err']:.2e}")
    if not results['cuda_max_err'] != results['cuda_max_err']:
        print(f"  CUDA: {results['cuda_max_err']:.2e}")


if __name__ == "__main__":
    main()
