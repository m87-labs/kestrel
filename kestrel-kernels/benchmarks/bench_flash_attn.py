"""Benchmark flash_attn kernels vs FlashInfer (FP8 KV cache).

Moondream attention parameters:
- num_heads = 32
- head_dim = 64
- page_size = 1 (for decode fastpath)
"""

import argparse
import math
import os
import sys

import torch
import torch.utils.benchmark as benchmark

# Add kestrel to path for imports (parent of kestrel-kernels)
_KESTREL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _KESTREL_ROOT)

import flashinfer
from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def run_decode_benchmark(
    batch_size: int,
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 64,
    page_size: int = 1,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run FP8 KV cache decode benchmark with CUDA graphs. Returns (kestrel_us, flashinfer_us, speedup)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    num_pages = _ceil_div(kv_len, page_size)
    total_pages = batch_size * num_pages + 1

    perm = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages) + 1
    page_table = perm.to(device=device, dtype=torch.int32)

    q = torch.randn((batch_size, 1, num_heads, head_dim), device=device, dtype=dtype)

    # FP8 KV cache
    k_scale, v_scale = 0.5, 0.5
    k_bf16 = torch.randn((total_pages, num_heads, page_size, head_dim), device=device, dtype=dtype)
    v_bf16 = torch.randn_like(k_bf16)
    k_fp8 = (k_bf16 / k_scale).to(fp8_dtype)
    v_fp8 = (v_bf16 / v_scale).to(fp8_dtype)

    k_nhd = k_fp8.permute(0, 2, 1, 3)
    v_nhd = v_fp8.permute(0, 2, 1, 3)

    seqlens_k = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # FlashInfer setup
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    fi_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND")

    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * num_pages
    kv_indices = page_table.reshape(-1).to(torch.int32)
    kv_last_page_len = torch.full((batch_size,), page_size, device=device, dtype=torch.int32)

    fi_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_heads,
        num_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=fp8_dtype,
    )

    # Warmup kestrel
    for _ in range(warmup):
        _flash_attn_fwd(q, k_nhd, v_nhd, page_table=page_table, seqused_k=seqlens_k,
                        softmax_scale=sm_scale, causal=True, k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Warmup FlashInfer
    q_fi = q.squeeze(1)
    for _ in range(warmup):
        fi_wrapper.run(q_fi, (k_fp8, v_fp8), k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Capture CUDA graph for kestrel
    g_kestrel = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_kestrel):
        _flash_attn_fwd(q, k_nhd, v_nhd, page_table=page_table, seqused_k=seqlens_k,
                        softmax_scale=sm_scale, causal=True, k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Capture CUDA graph for FlashInfer
    g_fi = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_fi):
        fi_wrapper.run(q_fi, (k_fp8, v_fp8), k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Benchmark
    kestrel_timer = benchmark.Timer(stmt="g.replay()", globals={"g": g_kestrel})
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    fi_timer = benchmark.Timer(stmt="g.replay()", globals={"g": g_fi})
    fi_us = fi_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, fi_us, fi_us / kestrel_us


def run_prefill_benchmark(
    batch_size: int,
    seq_len: int,
    num_heads: int = 32,
    head_dim: int = 64,
    page_size: int = 1,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run FP8 KV cache prefill benchmark. Returns (kestrel_us, flashinfer_us, speedup)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    num_pages = _ceil_div(seq_len, page_size)
    total_pages = batch_size * num_pages + 1

    perm = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages) + 1
    page_table = perm.to(device=device, dtype=torch.int32)

    # Q, K, V for prefill - all have seq_len tokens
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype)

    # FP8 KV cache
    k_scale, v_scale = 0.5, 0.5
    k_bf16 = torch.randn((total_pages, num_heads, page_size, head_dim), device=device, dtype=dtype)
    v_bf16 = torch.randn_like(k_bf16)
    k_fp8 = (k_bf16 / k_scale).to(fp8_dtype)
    v_fp8 = (v_bf16 / v_scale).to(fp8_dtype)

    k_nhd = k_fp8.permute(0, 2, 1, 3)
    v_nhd = v_fp8.permute(0, 2, 1, 3)

    seqlens_k = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # FlashInfer setup - use fa2 backend like old code
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    fi_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="fa2")

    # Cumulative sequence lengths for Q
    qo_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * seq_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * num_pages
    kv_indices = page_table.reshape(-1).to(torch.int32)
    kv_last_page_len = torch.full((batch_size,), page_size, device=device, dtype=torch.int32)

    fi_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_heads,
        num_heads,
        head_dim,
        page_size,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=fp8_dtype,
    )

    # Warmup kestrel
    for _ in range(warmup):
        _flash_attn_fwd(q, k_nhd, v_nhd, page_table=page_table, seqused_k=seqlens_k,
                        softmax_scale=sm_scale, causal=True, k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Warmup FlashInfer
    q_fi = q.view(batch_size * seq_len, num_heads, head_dim)
    for _ in range(warmup):
        fi_wrapper.run(q_fi, (k_fp8, v_fp8), k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()

    # Benchmark kestrel (no CUDA graph for prefill)
    def run_kestrel():
        _flash_attn_fwd(q, k_nhd, v_nhd, page_table=page_table, seqused_k=seqlens_k,
                        softmax_scale=sm_scale, causal=True, k_scale=k_scale, v_scale=v_scale)

    kestrel_timer = benchmark.Timer(stmt="run_kestrel()", globals={"run_kestrel": run_kestrel})
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark FlashInfer (no CUDA graph for prefill)
    def run_fi():
        fi_wrapper.run(q_fi, (k_fp8, v_fp8), k_scale=k_scale, v_scale=v_scale)

    fi_timer = benchmark.Timer(stmt="run_fi()", globals={"run_fi": run_fi})
    fi_us = fi_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, fi_us, fi_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark flash_attn kernels")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    num_heads = 32
    head_dim = 64

    print(f"Benchmarking flash_attn FP8 KV (heads={num_heads}, head_dim={head_dim})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Decode benchmarks (FP8 KV)
    print("=" * 65)
    print("Paged Decode FP8 KV (page_size=1, with CUDA Graphs)")
    print("=" * 65)
    header = f"{'Batch':>6} {'KV Len':>8} {'Kestrel':>12} {'FlashInfer':>12} {'Speedup':>10}"
    print(header)
    print("-" * 65)

    decode_configs = [
        (1, 740),
        (1, 1024),
        (1, 2048),
        (4, 740),
        (8, 512),
        (16, 256),
        (32, 128),
    ]

    for batch_size, kv_len in decode_configs:
        kestrel_us, fi_us, speedup = run_decode_benchmark(
            batch_size, kv_len, num_heads=num_heads, head_dim=head_dim,
            num_runs=args.num_runs,
        )
        print(f"{batch_size:>6} {kv_len:>8} {kestrel_us:>11.1f}us {fi_us:>11.1f}us {speedup:>9.2f}x")

    print("=" * 65)

    # Prefill benchmarks (FP8 KV)
    print()
    print("=" * 65)
    print("Paged Prefill FP8 KV (page_size=1)")
    print("=" * 65)
    header = f"{'Batch':>6} {'Seq Len':>8} {'Kestrel':>12} {'FlashInfer':>12} {'Speedup':>10}"
    print(header)
    print("-" * 65)

    prefill_configs = [
        (1, 740),   # Single crop vision + question
        (1, 1024),
        (1, 2048),
        (4, 512),
        (8, 256),
    ]

    for batch_size, seq_len in prefill_configs:
        kestrel_us, fi_us, speedup = run_prefill_benchmark(
            batch_size, seq_len, num_heads=num_heads, head_dim=head_dim,
            num_runs=args.num_runs,
        )
        print(f"{batch_size:>6} {seq_len:>8} {kestrel_us:>11.1f}us {fi_us:>11.1f}us {speedup:>9.2f}x")

    print("=" * 65)


if __name__ == "__main__":
    main()
