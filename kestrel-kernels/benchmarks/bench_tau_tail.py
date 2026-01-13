"""Benchmark tau_tail kernel vs PyTorch.

Applies TAU attention scaling to Q and V in packed QKV:
- scale_q = tanh(tok_q_lin) + tau_pos_table[position]
- scale_v = tanh(tok_v_lin) + tau_pos_table[position]
- Q *= scale_q, V *= scale_v

Used in text decoder attention:
- n_heads = 32, head_dim = 64
- decode: 1 token, prefill: 740 tokens
"""

import argparse

import torch
import torch.utils.benchmark as benchmark

from kestrel_kernels.tau_tail_ops import tau_tail_apply_into


def pytorch_tau_tail(
    qkv: torch.Tensor,
    tok_qv_lin: torch.Tensor,
    tau_pos_table: torch.Tensor,
    position_ids: torch.Tensor,
    n_heads: int,
    head_dim: int,
) -> None:
    """Reference PyTorch implementation of tau tail."""
    bsz, seq_len = qkv.shape[:2]
    q_dim = n_heads * head_dim

    # Views into Q and V
    q = qkv[..., :q_dim].view(bsz, seq_len, n_heads, head_dim)
    v = qkv[..., 2 * q_dim:].view(bsz, seq_len, n_heads, head_dim)

    # Compute scales
    tok_qv = tok_qv_lin.tanh()
    tok_q, tok_v = tok_qv.split(n_heads, dim=-1)
    tau_pos = tau_pos_table[position_ids]

    # Apply scaling
    q.mul_((tok_q + tau_pos).unsqueeze(-1))
    v.mul_((tok_v + tau_pos).unsqueeze(-1))


def run_benchmark(
    num_tokens: int,
    n_heads: int = 32,
    head_dim: int = 64,
    max_context: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    num_runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float, float]:
    """Run benchmark and return (cuda_us, pytorch_us, speedup)."""
    device = torch.device("cuda")

    bsz = 1
    q_dim = n_heads * head_dim
    qkv_dim = 3 * q_dim

    # Test tensors
    qkv_cuda = torch.randn((bsz, num_tokens, qkv_dim), dtype=dtype, device=device)
    qkv_pytorch = torch.randn((bsz, num_tokens, qkv_dim), dtype=dtype, device=device)
    tok_qv_lin = torch.randn((bsz, num_tokens, 2 * n_heads), dtype=dtype, device=device)
    tau_pos_table = torch.randn((max_context, n_heads), dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.long, device=device).view(1, num_tokens)

    # Warmup CUDA
    for _ in range(warmup):
        tau_tail_apply_into(
            qkv_out=qkv_cuda,
            tok_qv_lin=tok_qv_lin,
            tau_pos_table=tau_pos_table,
            position_ids=position_ids,
        )
    torch.cuda.synchronize()

    # Warmup PyTorch
    for _ in range(warmup):
        pytorch_tau_tail(qkv_pytorch, tok_qv_lin, tau_pos_table, position_ids, n_heads, head_dim)
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    cuda_timer = benchmark.Timer(
        stmt="tau_tail_apply_into(qkv_out=qkv, tok_qv_lin=tok_qv_lin, tau_pos_table=tau_pos_table, position_ids=position_ids)",
        globals={
            "tau_tail_apply_into": tau_tail_apply_into,
            "qkv": qkv_cuda,
            "tok_qv_lin": tok_qv_lin,
            "tau_pos_table": tau_pos_table,
            "position_ids": position_ids,
        },
    )
    cuda_us = cuda_timer.timeit(num_runs).mean * 1e6

    # Benchmark PyTorch
    pytorch_timer = benchmark.Timer(
        stmt="pytorch_tau_tail(qkv, tok_qv_lin, tau_pos_table, position_ids, n_heads, head_dim)",
        globals={
            "pytorch_tau_tail": pytorch_tau_tail,
            "qkv": qkv_pytorch,
            "tok_qv_lin": tok_qv_lin,
            "tau_pos_table": tau_pos_table,
            "position_ids": position_ids,
            "n_heads": n_heads,
            "head_dim": head_dim,
        },
    )
    pytorch_us = pytorch_timer.timeit(num_runs).mean * 1e6

    return cuda_us, pytorch_us, pytorch_us / cuda_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark tau_tail kernel")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"Benchmarking tau_tail (n_heads={args.n_heads}, head_dim={args.head_dim})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    print("=" * 50)
    header = f"{'Context':>10} {'Tokens':>6} {'CUDA':>10} {'PyTorch':>10} {'vs PyTorch':>10}"
    print(header)
    print("-" * 50)

    test_sizes = [
        ("decode", 1),
        ("batch 4", 4),
        ("batch 16", 16),
        ("prefill", 740),
        ("long", 1024),
    ]

    for context, num_tokens in test_sizes:
        cuda_us, pytorch_us, speedup = run_benchmark(
            num_tokens,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            num_runs=args.num_runs,
        )
        print(f"{context:>10} {num_tokens:>6} {cuda_us:>9.1f}us {pytorch_us:>9.1f}us {speedup:>9.2f}x")

    print("=" * 50)


if __name__ == "__main__":
    main()
