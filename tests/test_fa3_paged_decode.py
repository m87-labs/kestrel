"""Correctness test: paged FA3 decode matches torch SDPA (bf16 KV cache)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


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


@pytest.mark.parametrize(
    "head_dim,causal",
    [
        # Typical decode shape.
        (64, True),
        # Exercises the interface heuristic that selects tile_n=192 on SM90 when
        # head_dim==128 and attention is noncausal.
        (128, False),
    ],
)
@pytest.mark.parametrize("page_size", [1, 7, 16, 64, 96, 128, 192, 256])
def test_fa3_paged_decode_matches_sdpa_bf16(
    device: torch.device, head_dim: int, causal: bool, page_size: int
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
    perm_cpu = torch.randperm(num_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)

    # Physical KV storage is HND: [num_pages, H, page_size, D].
    # FA3 currently expects NHD on input, but we can pass a view with permuted strides to avoid
    # any copies and match the runtime cache layout.
    k_hnd = torch.zeros((num_pages, num_heads, page_size, head_dim), device=device, dtype=dtype)
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(total_len):
        logical_page = t // page_size
        offset = t % page_size
        physical_page = page_table_cpu[logical_page]
        k_hnd[physical_page, :, offset].copy_(k_flat[t])
        v_hnd[physical_page, :, offset].copy_(v_flat[t])

    # FA3 expects NHD: [num_pages, page_size, H, D] (view-backed by HND storage).
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((1,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    for i in range(decode_steps):
        kv_len = prefill_len + i + 1
        q_step = q_flat[prefill_len + i].view(1, num_heads, head_dim)

        fa3_seqlens_k[0] = kv_len
        out_fa3, _ = _flash_attn_fwd(
            q_step.view(1, 1, num_heads, head_dim),
            k_nhd_view,
            v_nhd_view,
            page_table=page_table,
            seqused_k=fa3_seqlens_k,
            softmax_scale=fa3_sm_scale,
            causal=causal,
        )
        out_fa3_step = out_fa3[:, 0]

        k_dense = k_flat[:kv_len].to(torch.float32)
        v_dense = v_flat[:kv_len].to(torch.float32)
        q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [1, H, 1, D]
        k_sdpa = k_dense.permute(1, 0, 2).unsqueeze(0)  # [1, H, kv, D]
        v_sdpa = v_dense.permute(1, 0, 2).unsqueeze(0)  # [1, H, kv, D]
        out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
        out_ref_step = out_ref.squeeze(2).to(dtype)

        # BF16 decode differs slightly across kernels; bounds validated on H100.
        torch.testing.assert_close(out_fa3_step, out_ref_step, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
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

    perm_cpu = torch.randperm(num_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros((num_pages, num_kv_heads, page_size, head_dim), device=device, dtype=dtype)
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


@pytest.mark.parametrize("head_dim", [64, 128])
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

    perm_cpu = torch.randperm(num_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros((num_pages, num_kv_heads, page_size, head_dim), device=device, dtype=dtype)
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


@pytest.mark.parametrize("kv_len", [1, 16, 48])
def test_fa3_paged_decode_fastpath_small_kv_len_matches_sdpa_bf16(
    device: torch.device, kv_len: int
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
    head_dim = 64
    num_heads = 8
    page_size = 1
    prefill_len = max(kv_len - 1, 0)
    decode_steps = 1

    total_len = prefill_len + decode_steps
    num_pages = _ceil_div(total_len, page_size)

    perm_cpu = torch.randperm(num_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, num_pages)
    page_table_cpu = perm_cpu.tolist()

    q_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((total_len, num_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros((num_pages, num_heads, page_size, head_dim), device=device, dtype=dtype)
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(total_len):
        logical_page = t
        physical_page = page_table_cpu[logical_page]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)
    q_step = q_flat[prefill_len].view(1, num_heads, head_dim)
    out_fa3, _ = _flash_attn_fwd(
        q_step.view(1, 1, num_heads, head_dim),
        k_nhd_view,
        v_nhd_view,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
    )
    assert _flash_attn_fwd._debug_last_impl == "sm90_decode"
    assert not torch.isnan(out_fa3).any(), "FA3 decode fastpath produced NaNs"

    k_dense = k_flat[:kv_len].to(torch.float32)
    v_dense = v_flat[:kv_len].to(torch.float32)
    q_sdpa = q_step.to(torch.float32).unsqueeze(2)  # [1, H, 1, D]
    k_sdpa = k_dense.permute(1, 0, 2).unsqueeze(0)  # [1, H, kv, D]
    v_sdpa = v_dense.permute(1, 0, 2).unsqueeze(0)  # [1, H, kv, D]
    out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
    out_ref_step = out_ref.squeeze(2).to(dtype)

    torch.testing.assert_close(out_fa3[:, 0], out_ref_step, rtol=1e-2, atol=2e-3)
