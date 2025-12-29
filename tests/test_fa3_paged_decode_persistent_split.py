"""Correctness test: SM90 persistent split decode matches SM90 decode fastpath."""

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


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("qhead_per_kvhead", [1, 4])
@pytest.mark.parametrize("return_lse", [True, False])
def test_fa3_paged_decode_persistent_split_fused_ignores_inactive_splits(
    device: torch.device,
    head_dim: int,
    qhead_per_kvhead: int,
    return_lse: bool,
) -> None:
    """Regression: with large page_table capacity, seqused_k can be much smaller.

    The fused persistent split kernel must not read stale partials from inactive splits.
    """
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import (
        _flash_attn_fwd,
        _flash_attn_sm90_decode_persistent_split_fused,
    )
    from kestrel_kernels.flash_attn.cute.flash_decode_sm90_persistent_fused import (
        FlashAttentionDecodeSm90PersistentSplitFused,
    )
    from kestrel_kernels.flash_attn.cute.interface import torch2cute_dtype_map

    dtype = torch.bfloat16
    num_kv_heads = 4
    num_q_heads = num_kv_heads * qhead_per_kvhead
    page_size = 1

    # Large capacity, but we'll set seqused_k smaller at runtime.
    capacity_pages = 2048
    perm_cpu = torch.randperm(capacity_pages, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(1, capacity_pages)
    page_table_cpu = perm_cpu.tolist()

    # Two steps to simulate graph replay with varying seqused_k.
    kv_len_big = 740  # ceil_div(740, 128) == 6
    kv_len_small = 200  # ceil_div(200, 128) == 2

    q_step = torch.randn((1, 1, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((kv_len_big, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((kv_len_big, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros(
        (capacity_pages, num_kv_heads, page_size, head_dim), device=device, dtype=dtype
    )
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(kv_len_big):
        physical_page = page_table_cpu[t]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])

    k_nhd_view = k_hnd.permute(0, 2, 1, 3)
    v_nhd_view = v_hnd.permute(0, 2, 1, 3)

    fa3_seqlens_k = torch.empty((1,), device=device, dtype=torch.int32)
    fa3_sm_scale = 1.0 / math.sqrt(head_dim)

    fa3_seqlens_k[0] = kv_len_big
    out_fast_big, lse_fast_big = _flash_attn_fwd(
        q_step,
        k_nhd_view,
        v_nhd_view,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
        return_lse=return_lse,
    )
    assert _flash_attn_fwd._debug_last_impl == "sm90_decode"

    fa3_seqlens_k[0] = kv_len_small
    out_fast_small, lse_fast_small = _flash_attn_fwd(
        q_step,
        k_nhd_view,
        v_nhd_view,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
        return_lse=return_lse,
    )
    assert _flash_attn_fwd._debug_last_impl == "sm90_decode"

    # Fused persistent split: reuse partial buffers across calls to simulate graph replay.
    max_splits = 6
    split_tokens = 128
    out_partial = torch.empty(
        (max_splits, 1, 1, num_q_heads, head_dim), device=device, dtype=torch.float32
    )
    lse_partial = torch.empty(
        (max_splits, 1, num_q_heads, 1), device=device, dtype=torch.float32
    )
    lse_out_big = (
        torch.empty((1, num_q_heads, 1), device=device, dtype=torch.float32) if return_lse else None
    )
    lse_out_small = (
        torch.empty((1, num_q_heads, 1), device=device, dtype=torch.float32) if return_lse else None
    )
    dtype_cute = torch2cute_dtype_map[dtype]
    tmp_kernel = FlashAttentionDecodeSm90PersistentSplitFused(
        dtype=dtype_cute,
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kvhead,
        num_splits=max_splits,
        is_causal=True,
        is_local=False,
        split_tokens=split_tokens,
        persist_oversub=1,
    )
    split_counters = torch.zeros(
        (1, num_kv_heads * tmp_kernel.group_count), device=device, dtype=torch.int32
    )

    fa3_seqlens_k[0] = kv_len_big
    out_fused_big = torch.empty_like(out_fast_big)
    _flash_attn_sm90_decode_persistent_split_fused(
        q_step,
        k_nhd_view,
        v_nhd_view,
        out_fused_big,
        lse_out_big,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
        local=False,
        window_size_left=None,
        window_size_right=None,
        qhead_per_kvhead=qhead_per_kvhead,
        num_splits=max_splits,
        split_tokens=split_tokens,
        persist_oversub=1,
        out_partial=out_partial,
        lse_partial=lse_partial,
        split_counters=split_counters,
    )
    torch.testing.assert_close(out_fused_big, out_fast_big, rtol=1e-2, atol=2e-3)
    if return_lse:
        assert lse_out_big is not None and lse_fast_big is not None
        torch.testing.assert_close(lse_out_big, lse_fast_big, rtol=1e-3, atol=1e-3)

    # Second call: smaller kv_len; inactive splits must be ignored (partials are stale).
    fa3_seqlens_k[0] = kv_len_small
    out_fused_small = torch.empty_like(out_fast_small)
    _flash_attn_sm90_decode_persistent_split_fused(
        q_step,
        k_nhd_view,
        v_nhd_view,
        out_fused_small,
        lse_out_small,
        page_table=page_table,
        seqused_k=fa3_seqlens_k,
        softmax_scale=fa3_sm_scale,
        causal=True,
        local=False,
        window_size_left=None,
        window_size_right=None,
        qhead_per_kvhead=qhead_per_kvhead,
        num_splits=max_splits,
        split_tokens=split_tokens,
        persist_oversub=1,
        out_partial=out_partial,
        lse_partial=lse_partial,
        split_counters=split_counters,
    )
    torch.testing.assert_close(out_fused_small, out_fast_small, rtol=1e-2, atol=2e-3)
    if return_lse:
        assert lse_out_small is not None and lse_fast_small is not None
        torch.testing.assert_close(lse_out_small, lse_fast_small, rtol=1e-3, atol=1e-3)
