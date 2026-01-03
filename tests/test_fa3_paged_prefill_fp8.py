"""Correctness tests: paged FA3 prefill matches torch SDPA (FP8 KV cache)."""

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


def _build_paged_kv(
    *,
    k_dense: torch.Tensor,  # [B, S, H, D]
    v_dense: torch.Tensor,  # [B, S, H, D]
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k_dense.shape != v_dense.shape:
        raise ValueError("k_dense and v_dense shapes must match")
    if k_dense.ndim != 4:
        raise ValueError(f"Expected BSHD inputs; got {k_dense.shape}")
    if page_size <= 0:
        raise ValueError("page_size must be > 0")

    batch_size, seqlen, num_heads, head_dim = k_dense.shape
    num_pages_per_seq = _ceil_div(seqlen, page_size)
    max_seqlen = num_pages_per_seq * page_size
    # Reserve physical page 0 as a "zero page" for invalid accesses.
    total_pages = batch_size * num_pages_per_seq + 1

    perm_cpu = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages_per_seq) + 1
    page_table = perm_cpu.to(device=k_dense.device, dtype=torch.int32)

    k_hnd = torch.zeros(
        (total_pages, num_heads, page_size, head_dim), device=k_dense.device, dtype=k_dense.dtype
    )
    v_hnd = torch.zeros_like(k_hnd)

    k_padded = torch.zeros(
        (batch_size, max_seqlen, num_heads, head_dim), device=k_dense.device, dtype=k_dense.dtype
    )
    v_padded = torch.zeros_like(k_padded)
    k_padded[:, :seqlen].copy_(k_dense)
    v_padded[:, :seqlen].copy_(v_dense)

    # [B, pages, page, H, D] -> HND: [B*pages, H, page, D]
    k_pages_hnd = (
        k_padded.view(batch_size, num_pages_per_seq, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages_per_seq, num_heads, page_size, head_dim)
    )
    v_pages_hnd = (
        v_padded.view(batch_size, num_pages_per_seq, page_size, num_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size * num_pages_per_seq, num_heads, page_size, head_dim)
    )

    page_table_flat = page_table.to(dtype=torch.long).reshape(-1)
    k_hnd[page_table_flat] = k_pages_hnd
    v_hnd[page_table_flat] = v_pages_hnd

    # FA3 expects NHD: [pages, page, H, D] (view-backed by HND storage).
    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)
    return page_table, k_nhd_view, v_nhd_view


def _sdpa_math(
    q: torch.Tensor,  # [B, S, H, D]
    k: torch.Tensor,  # [B, S, H, D]
    v: torch.Tensor,  # [B, S, H, D]
    *,
    is_causal: bool,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q_4d = q.permute(0, 2, 1, 3).to(torch.float32)
    k_4d = k.permute(0, 2, 1, 3).to(torch.float32)
    v_4d = v.permute(0, 2, 1, 3).to(torch.float32)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        out = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=attn_mask, is_causal=is_causal
        )
    return out.permute(0, 2, 1, 3).to(dtype=q.dtype)


def test_fa3_paged_prefill_fp8_causal_matches_sdpa(device: torch.device) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    batch_size = 2
    num_heads = 4
    head_dim = 64
    seqlen = 130
    page_size = 16

    # Non-trivial scales (cache stores fp8(x / scale)).
    k_scale = 0.5
    v_scale = 0.5

    q = torch.randn((batch_size, seqlen, num_heads, head_dim), device=device, dtype=dtype)
    k_dense = torch.randn_like(q)
    v_dense = torch.randn_like(q)

    page_table, k_paged_bf16, v_paged_bf16 = _build_paged_kv(
        k_dense=k_dense, v_dense=v_dense, page_size=page_size
    )
    k_paged_fp8 = (k_paged_bf16 / k_scale).to(fp8_dtype)
    v_paged_fp8 = (v_paged_bf16 / v_scale).to(fp8_dtype)

    # Tell FA3 to ignore padded tail tokens (varlen K only; Q is already exact length).
    seqused_k = torch.full((batch_size,), seqlen, device=device, dtype=torch.int32)

    out_fa3, _ = _flash_attn_fwd(
        q,
        k_paged_fp8,
        v_paged_fp8,
        page_table=page_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        paged_kv_non_tma=True,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert _flash_attn_fwd._debug_last_impl == "fwd"

    k_ref = (k_dense / k_scale).to(fp8_dtype).to(torch.float32) * k_scale
    v_ref = (v_dense / v_scale).to(fp8_dtype).to(torch.float32) * v_scale
    out_ref = _sdpa_math(q, k_ref, v_ref, is_causal=True)

    torch.testing.assert_close(out_fa3, out_ref, rtol=6e-2, atol=3e-2)


def test_fa3_paged_prefill_fp8_prefixlm_matches_sdpa(device: torch.device) -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd
    from kestrel_kernels.flash_attn.cute.mask_definitions import cute_prefix_lm_mask_730

    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    batch_size = 1
    num_heads = 4
    head_dim = 64
    seqlen = 740  # crosses the 730-token prefix boundary
    page_size = 1  # keep fixed-length to allow mask_mod (varlen + mask_mod not supported yet)

    k_scale = 0.5
    v_scale = 0.5

    q = torch.randn((batch_size, seqlen, num_heads, head_dim), device=device, dtype=dtype)
    k_dense = torch.randn_like(q)
    v_dense = torch.randn_like(q)

    page_table, k_paged_bf16, v_paged_bf16 = _build_paged_kv(
        k_dense=k_dense, v_dense=v_dense, page_size=page_size
    )
    k_paged_fp8 = (k_paged_bf16 / k_scale).to(fp8_dtype)
    v_paged_fp8 = (v_paged_bf16 / v_scale).to(fp8_dtype)

    out_fa3, _ = _flash_attn_fwd(
        q,
        k_paged_fp8,
        v_paged_fp8,
        page_table=page_table,
        softmax_scale=1.0 / math.sqrt(head_dim),
        mask_mod=cute_prefix_lm_mask_730,
        paged_kv_non_tma=True,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert _flash_attn_fwd._debug_last_impl == "fwd"

    prefix_len = 730
    q_idx = torch.arange(seqlen, device=device).view(seqlen, 1)
    kv_idx = torch.arange(seqlen, device=device).view(1, seqlen)
    allowed = ((q_idx < prefix_len) & (kv_idx < prefix_len)) | (q_idx >= kv_idx)
    # Use an additive mask to avoid bool semantics differences across torch versions.
    attn_mask = torch.where(allowed, 0.0, -float("inf")).to(torch.float32)

    k_ref = (k_dense / k_scale).to(fp8_dtype).to(torch.float32) * k_scale
    v_ref = (v_dense / v_scale).to(fp8_dtype).to(torch.float32) * v_scale
    out_ref = _sdpa_math(q, k_ref, v_ref, is_causal=False, attn_mask=attn_mask)

    torch.testing.assert_close(out_fa3, out_ref, rtol=6e-2, atol=3e-2)
