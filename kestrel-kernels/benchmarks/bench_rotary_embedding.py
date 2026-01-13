"""Benchmark rotary embedding kernel vs vLLM and PyTorch.

Applies rotary position embedding to query and key tensors.

Used in text decoder attention:
- n_heads = 32, n_kv_heads = 32
- head_dim = 64
- decode: 1 token, prefill: 740 tokens
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.rotary_embedding import rotary_embedding
from vllm import _custom_ops as vllm_ops


def pytorch_rotary_embedding(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """PyTorch eager implementation of rotary embedding (GPT-NeoX style)."""
    # positions: [B, S], q/k: [B, S, H, D], cos_sin_cache: [max_pos, rot_dim]
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2

    # Gather cos/sin for each position
    pos_flat = positions.view(-1)  # [B*S]
    cos_sin = cos_sin_cache[pos_flat]  # [B*S, rot_dim]
    cos = cos_sin[:, :embed_dim].view(positions.shape[0], positions.shape[1], 1, embed_dim)
    sin = cos_sin[:, embed_dim:].view(positions.shape[0], positions.shape[1], 1, embed_dim)

    # GPT-NeoX style: first half and second half of head_dim
    q_rot = q[..., :embed_dim]
    q_pass = q[..., embed_dim:]
    k_rot = k[..., :embed_dim]
    k_pass = k[..., embed_dim:]

    # Apply rotation
    q[..., :embed_dim] = q_rot * cos - q_pass * sin
    q[..., embed_dim:] = q_pass * cos + q_rot * sin
    k[..., :embed_dim] = k_rot * cos - k_pass * sin
    k[..., embed_dim:] = k_pass * cos + k_rot * sin


def run_benchmark(
    num_tokens: int,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    head_dim: int = 64,
    max_position: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float, float, float]:
    """Run benchmark and return (kestrel_us, vllm_us, pytorch_us, vs_vllm, vs_pytorch)."""
    device = torch.device("cuda")
    rot_dim = head_dim

    # Create test tensors
    positions = torch.arange(num_tokens, dtype=torch.long, device=device).view(1, num_tokens)

    # Kestrel tensors [B, S, H, D]
    q_kestrel = torch.randn((1, num_tokens, n_heads, head_dim), dtype=dtype, device=device)
    k_kestrel = torch.randn((1, num_tokens, n_kv_heads, head_dim), dtype=dtype, device=device)

    # vLLM tensors [num_tokens, H, D]
    q_vllm = torch.randn((num_tokens, n_heads, head_dim), dtype=dtype, device=device)
    k_vllm = torch.randn((num_tokens, n_kv_heads, head_dim), dtype=dtype, device=device)

    # PyTorch tensors [B, S, H, D]
    q_pytorch = torch.randn((1, num_tokens, n_heads, head_dim), dtype=dtype, device=device)
    k_pytorch = torch.randn((1, num_tokens, n_kv_heads, head_dim), dtype=dtype, device=device)

    # Kestrel cos_sin_cache: [max_pos, rot_dim] with fp32
    cos_sin_cache_kestrel = torch.randn((max_position, rot_dim), dtype=torch.float32, device=device)

    # vLLM cos_sin_cache: same layout but native dtype
    cos_sin_cache_vllm = cos_sin_cache_kestrel.to(dtype)

    # PyTorch cos_sin_cache: native dtype
    cos_sin_cache_pytorch = cos_sin_cache_kestrel.to(dtype)

    # vLLM positions: [num_tokens]
    positions_vllm = positions.view(-1)

    # Warmup kestrel
    for _ in range(warmup):
        rotary_embedding(positions, q_kestrel, k_kestrel, head_dim, cos_sin_cache_kestrel)
    torch.cuda.synchronize()

    # Warmup vLLM
    for _ in range(warmup):
        vllm_ops.rotary_embedding(positions_vllm, q_vllm, k_vllm, head_dim, cos_sin_cache_vllm, True)
    torch.cuda.synchronize()

    # Warmup PyTorch
    for _ in range(warmup):
        pytorch_rotary_embedding(positions, q_pytorch, k_pytorch, cos_sin_cache_pytorch)
    torch.cuda.synchronize()

    # Benchmark kestrel kernel
    kestrel_timer = benchmark.Timer(
        stmt="rotary_embedding(positions, q, k, head_dim, cos_sin_cache)",
        globals={
            "rotary_embedding": rotary_embedding,
            "positions": positions,
            "q": q_kestrel,
            "k": k_kestrel,
            "head_dim": head_dim,
            "cos_sin_cache": cos_sin_cache_kestrel,
        },
    )
    kestrel_us = kestrel_timer.timeit(num_runs).mean * 1e6

    # Benchmark vLLM
    vllm_timer = benchmark.Timer(
        stmt="vllm_rotary_embedding(positions, q, k, head_dim, cos_sin_cache, True)",
        globals={
            "vllm_rotary_embedding": vllm_ops.rotary_embedding,
            "positions": positions_vllm,
            "q": q_vllm,
            "k": k_vllm,
            "head_dim": head_dim,
            "cos_sin_cache": cos_sin_cache_vllm,
        },
    )
    vllm_us = vllm_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_rotary_embedding(positions, q, k, cos_sin_cache)",
        globals={
            "pytorch_rotary_embedding": pytorch_rotary_embedding,
            "positions": positions,
            "q": q_pytorch,
            "k": k_pytorch,
            "cos_sin_cache": cos_sin_cache_pytorch,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return kestrel_us, vllm_us, pytorch_us, vllm_us / kestrel_us, pytorch_us / kestrel_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark rotary embedding kernel")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=32, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking rotary embedding (n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    print("=" * 75)
    header = f"{'Context':>10} {'Tokens':>6} {'Kestrel':>10} {'vLLM':>10} {'PyTorch':>10} {'vs vLLM':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 75)

    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
    ]

    for context, num_tokens in test_sizes:
        kestrel_us, vllm_us, pytorch_us, vs_vllm, vs_pytorch = run_benchmark(
            num_tokens,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {kestrel_us:>9.1f}us {vllm_us:>9.1f}us {pytorch_us:>9.1f}us {vs_vllm:>9.2f}x {vs_pytorch:>9.1f}x")

    print("=" * 75)


if __name__ == "__main__":
    main()
