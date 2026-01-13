"""Benchmark cute_moe FP8 kernels vs vLLM Triton.

MoE matrix multiplications for Moondream:
- num_experts = 64
- topk = 8
- hidden_size = 2048
- intermediate_size = 1024
"""

import argparse
import os
import sys

import torch
import torch.utils.benchmark as benchmark

# Add kestrel to path for imports (parent of kestrel-kernels)
_KESTREL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _KESTREL_ROOT)

from kestrel.fused_moe import ExpertWeights, FusedMoEModule
from kestrel.fused_moe.module import preallocate_shared_moe_workspaces

# vLLM imports
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


def run_benchmark_fp8(
    num_tokens: int,
    num_experts: int = 64,
    topk: int = 8,
    hidden_size: int = 2048,
    intermediate_size: int = 1024,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run FP8 benchmark with CUDA graphs. Returns (kestrel_us, vllm_us, speedup)."""
    device = torch.device("cuda")

    # Create topk_ids and topk_weights
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    topk_weights = torch.randn((num_tokens, topk), dtype=torch.bfloat16, device=device).softmax(dim=-1)

    # FP8 weight tensors as float8_e4m3fn for vLLM
    w_up_fp8 = torch.randn((num_experts, intermediate_size * 2, hidden_size), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    w_up_scale = torch.ones((num_experts, intermediate_size * 2), dtype=torch.float32, device=device)
    w_down_fp8 = torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    w_down_scale = torch.ones((num_experts, hidden_size), dtype=torch.float32, device=device)

    # BF16 input
    x_bf16 = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    # Create kestrel FusedMoEModule
    up_experts = ExpertWeights(num_experts, hidden_size, intermediate_size * 2, dtype=torch.bfloat16)
    up_experts.weight = torch.nn.Parameter(w_up_fp8.view(torch.uint8), requires_grad=False)
    up_experts.scale = w_up_scale

    down_experts = ExpertWeights(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16)
    down_experts.weight = torch.nn.Parameter(w_down_fp8.view(torch.uint8), requires_grad=False)
    down_experts.scale = w_down_scale

    kestrel_moe = FusedMoEModule(
        up_experts,
        down_experts,
        top_k=topk,
        hidden_size=intermediate_size,
        input_size=hidden_size,
        num_experts=num_experts,
    )

    # vLLM quant config
    quant_config = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        per_act_token_quant=True,
        block_shape=None,
        w1_scale=w_up_scale,
        w2_scale=w_down_scale,
    )

    # Warmup and capture CUDA graph for kestrel
    s_kestrel = torch.cuda.Stream()
    s_kestrel.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_kestrel):
        for _ in range(warmup):
            kestrel_moe(x_bf16, topk_weights, topk_ids)
    torch.cuda.current_stream().wait_stream(s_kestrel)

    g_kestrel = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_kestrel):
        kestrel_moe(x_bf16, topk_weights, topk_ids)
    torch.cuda.synchronize()

    # Warmup and capture CUDA graph for vLLM
    s_vllm = torch.cuda.Stream()
    s_vllm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_vllm):
        for _ in range(warmup):
            fused_experts(x_bf16, w_up_fp8, w_down_fp8, topk_weights, topk_ids, quant_config=quant_config)
    torch.cuda.current_stream().wait_stream(s_vllm)

    g_vllm = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_vllm):
        fused_experts(x_bf16, w_up_fp8, w_down_fp8, topk_weights, topk_ids, quant_config=quant_config)
    torch.cuda.synchronize()

    # Benchmark kestrel CUDA graph
    kestrel_timer = benchmark.Timer(
        stmt="g.replay()",
        globals={"g": g_kestrel},
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark vLLM CUDA graph
    vllm_timer = benchmark.Timer(
        stmt="g.replay()",
        globals={"g": g_vllm},
    )
    vllm_us = vllm_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, vllm_us, vllm_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark cute_moe FP8 kernels")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Moondream dimensions
    num_experts = 64
    topk = 8
    hidden_size = 2048
    intermediate_size = 1024

    print(f"Benchmarking cute_moe FP8 (E={num_experts}, k={topk}, H={hidden_size}, I={intermediate_size})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    print("Full MoE layer: up projection + activation + down projection + sum")
    print()

    print("=" * 60)
    print("FP8 MoE Kernels (W8A8) with CUDA Graphs")
    print("=" * 60)
    header = f"{'Context':>10} {'Tokens':>6} {'Kestrel':>12} {'vLLM (Triton)':>15} {'Speedup':>10}"
    print(header)
    print("-" * 60)

    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
    ]

    # Pre-allocate shared MoE workspaces with max token count
    max_tokens = max(t[1] for t in test_sizes)
    preallocate_shared_moe_workspaces(
        max_num_tokens=max_tokens,
        top_k=topk,
        hidden_size=intermediate_size,
        input_size=hidden_size,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    for context, num_tokens in test_sizes:
        kestrel_us, vllm_us, speedup = run_benchmark_fp8(
            num_tokens,
            num_experts=num_experts,
            topk=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {kestrel_us:>11.1f}us {vllm_us:>14.1f}us {speedup:>9.2f}x")

    print("=" * 60)


if __name__ == "__main__":
    main()
