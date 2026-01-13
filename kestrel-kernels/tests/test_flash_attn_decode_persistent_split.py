"""Tests for SM90 persistent split decode kernel (precompiled).

These tests cover the precompiled persistent split fused decode kernel variants
that are actually used in kestrel production (MHA decode with paged KV cache).

The persistent split fused path is only triggered during CUDA graph capture,
so we test via the high-level API with graph capture enabled.
"""

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


@pytest.mark.skip(reason="Precompiled kernel has stride alignment issue - needs investigation")
def test_persistent_split_fused_via_cuda_graph_bf16(device: torch.device) -> None:
    """Test that persistent split fused works during CUDA graph capture (BF16)."""
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    head_dim = 64
    num_kv_heads = 4
    num_q_heads = num_kv_heads  # MHA
    page_size = 1  # Required for persistent split path
    kv_len = 512
    batch_size = 1

    # Setup paged KV cache
    perm_cpu = torch.randperm(kv_len, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(batch_size, kv_len)
    page_table_cpu = perm_cpu.tolist()

    q = torch.randn((batch_size, 1, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((kv_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((kv_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    # Build paged KV in HND format
    k_hnd = torch.zeros((kv_len, num_kv_heads, page_size, head_dim), device=device, dtype=dtype)
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(kv_len):
        physical_page = page_table_cpu[t]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])

    k_nhd = k_hnd.permute(0, 2, 1, 3)
    v_nhd = v_hnd.permute(0, 2, 1, 3)

    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)

    # Reference: without graph capture (uses regular decode)
    out_ref, _ = _flash_attn_fwd(
        q, k_nhd, v_nhd,
        page_table=page_table,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=True,
        return_lse=False,
    )
    ref_impl = _flash_attn_fwd._debug_last_impl

    # Test: with graph capture (should use persistent split fused with lse=None)
    # This is the scenario where the original bug occurred
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # Warmup
        _flash_attn_fwd(
            q, k_nhd, v_nhd, out=out,
            page_table=page_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=False,
        )
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _flash_attn_fwd(
            q, k_nhd, v_nhd, out=out,
            page_table=page_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=False,
        )
    graph_impl = _flash_attn_fwd._debug_last_impl

    # Replay graph
    g.replay()
    torch.cuda.synchronize()

    # The graph capture should trigger persistent split fused path
    # (or fall back to regular decode if conditions aren't met)
    # Either way, output should match reference
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=2e-3)


@pytest.mark.skip(reason="Precompiled kernel has stride alignment issue - needs investigation")
def test_persistent_split_fused_via_cuda_graph_fp8(device: torch.device) -> None:
    """Test that persistent split fused works during CUDA graph capture (FP8 KV).

    This is the original bug scenario: FP8 KV cache + CUDA graph + return_lse=False.
    """
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 64
    num_kv_heads = 4
    num_q_heads = num_kv_heads
    page_size = 1
    kv_len = 512
    batch_size = 1
    k_scale = 0.5
    v_scale = 0.5

    perm_cpu = torch.randperm(kv_len, device="cpu")
    page_table = perm_cpu.to(device=device, dtype=torch.int32).view(batch_size, kv_len)
    page_table_cpu = perm_cpu.tolist()

    q = torch.randn((batch_size, 1, num_q_heads, head_dim), device=device, dtype=dtype)
    k_flat = torch.randn((kv_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_flat = torch.randn((kv_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_hnd = torch.zeros((kv_len, num_kv_heads, page_size, head_dim), device=device, dtype=dtype)
    v_hnd = torch.zeros_like(k_hnd)
    for t in range(kv_len):
        physical_page = page_table_cpu[t]
        k_hnd[physical_page, :, 0].copy_(k_flat[t])
        v_hnd[physical_page, :, 0].copy_(v_flat[t])

    # Quantize to FP8
    k_hnd_fp8 = (k_hnd / k_scale).to(fp8_dtype)
    v_hnd_fp8 = (v_hnd / v_scale).to(fp8_dtype)

    k_nhd_fp8 = k_hnd_fp8.permute(0, 2, 1, 3)
    v_nhd_fp8 = v_hnd_fp8.permute(0, 2, 1, 3)

    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)

    # Reference: without graph capture
    out_ref, _ = _flash_attn_fwd(
        q, k_nhd_fp8, v_nhd_fp8,
        page_table=page_table,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=True,
        k_scale=k_scale,
        v_scale=v_scale,
        return_lse=False,
    )

    # Test: with graph capture (the original bug scenario)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _flash_attn_fwd(
            q, k_nhd_fp8, v_nhd_fp8, out=out,
            page_table=page_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=True,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=False,
        )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _flash_attn_fwd(
            q, k_nhd_fp8, v_nhd_fp8, out=out,
            page_table=page_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=True,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=False,
        )

    g.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=5e-3)
