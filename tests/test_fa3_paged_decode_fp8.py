"""Correctness test: FA3 paged decode with FP8 KV cache matches BF16 KV within tolerance."""

from __future__ import annotations

import math

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("FA3 Cute-DSL kernels require SM90+")
    return torch.device("cuda", torch.cuda.current_device())


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def test_fa3_paged_decode_fp8_kv_matches_bf16(device: torch.device) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    # page_size==1 dispatches the SM90 decode fastpath kernels.
    page_size = 1
    num_heads = 8
    head_dim = 64
    prefill_len = 740  # crosses multiple pages
    decode_steps = 8

    # Use non-trivial scales (store fp8(x / scale)).
    k_scale = 0.5
    v_scale = 0.5

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)

    # Logical page -> physical page mapping (matches runtime PageTable semantics).
    perm_cpu = torch.randperm(num_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)

    # Physical KV storage is HND: [num_pages, H, page_size, D] (matches runtime cache layout).
    k_bf16_hnd = torch.zeros((num_pages, num_heads, page_size, head_dim), device=device, dtype=dtype)
    v_bf16_hnd = torch.zeros_like(k_bf16_hnd)
    for t in range(total_len):
        logical_page = t // page_size
        offset = t % page_size
        physical_page = page_table_cpu[logical_page]
        k_bf16_hnd[physical_page, :, offset].copy_(k_flat[t])
        v_bf16_hnd[physical_page, :, offset].copy_(v_flat[t])

    # FP8 cache stores fp8(x / scale).
    k_fp8_hnd = (k_bf16_hnd / k_scale).to(fp8_dtype)
    v_fp8_hnd = (v_bf16_hnd / v_scale).to(fp8_dtype)

    # FA3 expects NHD: [num_pages, page_size, H, D] (use permuted views).
    k_bf16 = k_bf16_hnd.permute(0, 2, 1, 3)
    v_bf16 = v_bf16_hnd.permute(0, 2, 1, 3)
    k_fp8 = k_fp8_hnd.permute(0, 2, 1, 3)
    v_fp8 = v_fp8_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((1,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    for i in range(decode_steps):
        kv_len = prefill_len + i + 1
        q_step = q_flat[prefill_len + i].view(1, num_heads, head_dim)
        fa3_seqlens_k[0] = kv_len

        out_bf16, _ = _flash_attn_fwd(
            q_step.view(1, 1, num_heads, head_dim),
            k_bf16,
            v_bf16,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            paged_kv_non_tma=True,
            softmax_scale=fa3_sm_scale,
            causal=True,
        )
        out_fp8, _ = _flash_attn_fwd(
            q_step.view(1, 1, num_heads, head_dim),
            k_fp8,
            v_fp8,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            paged_kv_non_tma=True,
            softmax_scale=fa3_sm_scale,
            causal=True,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        torch.testing.assert_close(out_fp8[:, 0], out_bf16[:, 0], rtol=5e-2, atol=2e-2)
