"""Correctness tests: paged FA3 decode matches torch SDPA (bf16/FP8 KV cache)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


# Use sm90_device fixture from conftest.py, aliased as 'device' for compatibility
@pytest.fixture
def device(sm90_device):
    return sm90_device


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


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("page_size", [1, 7, 16, 64, 96, 128, 192, 256])
def test_fa3_paged_decode_matches_sdpa_bf16(
    device: torch.device, head_dim: int, batch_size: int, page_size: int
) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    num_heads = 8
    prefill_len = 740  # crosses multiple pages
    decode_steps = 8 if head_dim == 64 else 4

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)
    max_seq_len = num_pages * page_size

    # Logical page -> physical page mapping (matches runtime PageTable semantics).
    #
    # Note: production PageTable reserves physical page 0 as a "zero page" for invalid accesses.
    total_pages = batch_size * num_pages + 1
    perm_cpu = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages) + 1
    page_table = perm_cpu.to(device=device, dtype=torch.int32)

    q_flat = torch.randn(
        (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
    )
    k_flat = torch.randn_like(q_flat)
    v_flat = torch.randn_like(q_flat)

    # Physical KV storage is HND: [total_pages, H, page_size, D] (including the reserved page 0).
    # FA3 expects NHD on input, but we can pass a view with permuted strides to avoid any copies
    # and match the runtime cache layout.
    k_hnd = torch.zeros(
        (total_pages, num_heads, page_size, head_dim), device=device, dtype=dtype
    )
    v_hnd = torch.zeros_like(k_hnd)

    k_padded = torch.zeros(
        (batch_size, max_seq_len, num_heads, head_dim), device=device, dtype=dtype
    )
    v_padded = torch.zeros_like(k_padded)
    k_padded[:, :total_len].copy_(k_flat)
    v_padded[:, :total_len].copy_(v_flat)
    k_pages_hnd = (
        k_padded.view(batch_size, num_pages, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages, num_heads, page_size, head_dim)
    )
    v_pages_hnd = (
        v_padded.view(batch_size, num_pages, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages, num_heads, page_size, head_dim)
    )
    page_table_flat = page_table.to(dtype=torch.long).reshape(-1)
    k_hnd[page_table_flat] = k_pages_hnd
    v_hnd[page_table_flat] = v_pages_hnd

    # FA3 expects NHD: [num_pages, page_size, H, D] (view-backed by HND storage).
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((batch_size,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)
    # Avoid passing a different Q base pointer each decode step (CuTe runtime can cache the
    # pointer from the first invocation). This mirrors production decode, which uses stable
    # staging buffers per CUDA graph.
    q_buf = torch.empty((batch_size, 1, num_heads, head_dim), device=device, dtype=dtype)

    for i in range(decode_steps):
        kv_len = prefill_len + i + 1
        q_step = q_flat[:, prefill_len + i]
        q_buf[:, 0].copy_(q_step)

        fa3_seqlens_k.fill_(kv_len)
        out_fa3, _ = _flash_attn_fwd(
            q_buf,
            k_nhd_view,
            v_nhd_view,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            softmax_scale=fa3_sm_scale,
            causal=True,
        )
        out_fa3_step = out_fa3[:, 0]

        k_dense = k_flat[:, :kv_len].to(torch.float32)
        v_dense = v_flat[:, :kv_len].to(torch.float32)
        q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [B, H, 1, D]
        k_sdpa = k_dense.permute(0, 2, 1, 3)  # [B, H, kv, D]
        v_sdpa = v_dense.permute(0, 2, 1, 3)  # [B, H, kv, D]
        out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
        out_ref_step = out_ref.squeeze(2).to(dtype)

        # BF16 decode differs slightly across kernels; bounds validated on H100.
        torch.testing.assert_close(out_fa3_step, out_ref_step, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("qhead_per_kvhead", [2, 4])
def test_fa3_paged_decode_fastpath_gqa_matches_sdpa_bf16(
    device: torch.device, head_dim: int, qhead_per_kvhead: int
) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    num_kv_heads = 4
    num_q_heads = num_kv_heads * qhead_per_kvhead
    prefill_len = 513
    decode_steps = 4
    page_size = 1

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)

    perm_cpu = torch.randperm(num_pages, device="cpu") + 1
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros(
        (num_pages + 1, num_kv_heads, page_size, head_dim), device=device, dtype=dtype
    )
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(total_len):
        logical_page = t
        physical_page = page_table_cpu[logical_page]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((1,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    for i in range(decode_steps):
        kv_len = prefill_len + i + 1
        q_step = q_flat[prefill_len + i].view(1, num_q_heads, head_dim)
        fa3_seqlens_k[0] = kv_len
        out_fa3, _ = _flash_attn_fwd(
            q_step.view(1, 1, num_q_heads, head_dim),
            k_nhd_view,
            v_nhd_view,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            softmax_scale=fa3_sm_scale,
            causal=True,
        )
        assert _flash_attn_fwd._debug_last_impl == "sm90_decode"
        out_fa3_step = out_fa3[:, 0]

        k_dense = k_flat[:kv_len].to(torch.float32)
        v_dense = v_flat[:kv_len].to(torch.float32)
        q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [1, Hq, 1, D]
        k_sdpa = k_dense.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, kv, D]
        v_sdpa = v_dense.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, kv, D]
        k_sdpa = k_sdpa.repeat_interleave(qhead_per_kvhead, dim=1)  # [1, Hq, kv, D]
        v_sdpa = v_sdpa.repeat_interleave(qhead_per_kvhead, dim=1)  # [1, Hq, kv, D]
        out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
        out_ref_step = out_ref.squeeze(2).to(dtype)

        torch.testing.assert_close(out_fa3_step, out_ref_step, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("head_dim", [64])
def test_fa3_paged_decode_fastpath_sliding_window_matches_sdpa_bf16(
    device: torch.device, head_dim: int
) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    qhead_per_kvhead = 4
    num_kv_heads = 2
    num_q_heads = num_kv_heads * qhead_per_kvhead
    prefill_len = 740
    decode_steps = 4
    page_size = 1
    window_size_left = 128
    window_size_right = 0

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)

    perm_cpu = torch.randperm(num_pages, device="cpu") + 1
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros(
        (num_pages + 1, num_kv_heads, page_size, head_dim), device=device, dtype=dtype
    )
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(total_len):
        logical_page = t
        physical_page = page_table_cpu[logical_page]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((1,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    for i in range(decode_steps):
        kv_len = prefill_len + i + 1
        q_step = q_flat[prefill_len + i].view(1, num_q_heads, head_dim)
        fa3_seqlens_k[0] = kv_len
        out_fa3, _ = _flash_attn_fwd(
            q_step.view(1, 1, num_q_heads, head_dim),
            k_nhd_view,
            v_nhd_view,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            softmax_scale=fa3_sm_scale,
            causal=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        assert _flash_attn_fwd._debug_last_impl == "sm90_decode"
        out_fa3_step = out_fa3[:, 0]

        q_pos = kv_len - 1
        start = max(0, q_pos - window_size_left)
        end = min(kv_len, q_pos + window_size_right + 1)
        k_dense = k_flat[start:end].to(torch.float32)
        v_dense = v_flat[start:end].to(torch.float32)
        q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [1, Hq, 1, D]
        k_sdpa = k_dense.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, kv, D]
        v_sdpa = v_dense.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, kv, D]
        k_sdpa = k_sdpa.repeat_interleave(qhead_per_kvhead, dim=1)  # [1, Hq, kv, D]
        v_sdpa = v_sdpa.repeat_interleave(qhead_per_kvhead, dim=1)  # [1, Hq, kv, D]
        out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
        out_ref_step = out_ref.squeeze(2).to(dtype)

        torch.testing.assert_close(out_fa3_step, out_ref_step, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("kv_len", [1, 5, 16, 48])
def test_fa3_paged_decode_fastpath_small_kv_len_matches_sdpa_bf16(
    device: torch.device, head_dim: int, batch_size: int, kv_len: int
) -> None:
    """Regression: ensure decode fastpath is finite for small KV lengths.

    The SM90 decode kernel uses a tz-parallel softmax merge. When kv_len is small,
    some tz planes have no valid tokens; we must not produce NaNs in that case.
    """
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    num_heads = 8
    page_size = 1
    prefill_len = kv_len - 1
    decode_steps = 1  # always decode 1 token (seqlen_q == 1)

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)

    # Logical page -> physical page mapping (matches runtime PageTable semantics).
    #
    # Note: production PageTable reserves physical page 0 as a "zero page" for invalid accesses.
    total_pages = batch_size * num_pages + 1
    perm_cpu = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages) + 1
    page_table = perm_cpu.to(device=device, dtype=torch.int32)

    q_flat = torch.randn(
        (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
    )
    k_flat = torch.randn_like(q_flat)
    v_flat = torch.randn_like(q_flat)

    # Physical KV storage is HND: [total_pages, H, page_size, D] (including reserved page 0).
    k_hnd = torch.zeros(
        (total_pages, num_heads, page_size, head_dim), device=device, dtype=dtype
    )
    v_hnd = torch.zeros_like(k_hnd)

    # Copy dense K/V into the paged backing store using the random page mapping.
    k_pages_hnd = (
        k_flat.view(batch_size, num_pages, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages, num_heads, page_size, head_dim)
    )
    v_pages_hnd = (
        v_flat.view(batch_size, num_pages, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages, num_heads, page_size, head_dim)
    )
    page_table_flat = page_table.to(dtype=torch.long).reshape(-1)
    k_hnd[page_table_flat] = k_pages_hnd
    v_hnd[page_table_flat] = v_pages_hnd

    # FA3 expects NHD: [num_pages, page_size, H, D] (view-backed by HND storage).
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    # Use a stable Q base pointer (mirrors production decode + avoids CuTe pointer caching issues).
    q_buf = torch.empty((batch_size, 1, num_heads, head_dim), device=device, dtype=dtype)
    q_step = q_flat[:, prefill_len]
    q_buf[:, 0].copy_(q_step)
    out_fa3, _ = _flash_attn_fwd(
        q_buf,
        k_nhd_view,
        v_nhd_view,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
    )
    assert _flash_attn_fwd._debug_last_impl == "sm90_decode"
    assert not torch.isnan(out_fa3).any(), "FA3 decode fastpath produced NaNs"

    k_dense = k_flat[:, :kv_len].to(torch.float32)
    v_dense = v_flat[:, :kv_len].to(torch.float32)
    q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [B, H, 1, D]
    k_sdpa = k_dense.permute(0, 2, 1, 3)  # [B, H, kv, D]
    v_sdpa = v_dense.permute(0, 2, 1, 3)  # [B, H, kv, D]
    out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
    out_ref_step = out_ref.squeeze(2).to(dtype)

    torch.testing.assert_close(out_fa3[:, 0], out_ref_step, rtol=1e-2, atol=2e-3)
