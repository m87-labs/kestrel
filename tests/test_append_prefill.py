"""Tests for append prefill attention in text decoder.

Append prefill computes attention for suffix tokens against the full KV cache
(cached prefix + new suffix). This is used for prefix cache hits where we skip
recomputing attention for the cached prefix.

The key insight is FA3's causal mask uses (seqlen_k - seqlen_q) to right-align
Q with K, which naturally handles the case where:
- Q contains only suffix tokens (length = suffix_len)
- K/V contains full cache (length = total_len)
- Causal mask allows Q[i] to attend to K[0..skip_positions+i]
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


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _build_paged_kv_cache(
    *,
    k_dense: torch.Tensor,  # [B, S, H, D]
    v_dense: torch.Tensor,  # [B, S, H, D]
    page_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged KV cache from dense K/V tensors.

    Returns (page_table, k_paged, v_paged) where k_paged and v_paged are
    in the format expected by FA3: [pages, page_size, heads, dim].
    """
    if k_dense.shape != v_dense.shape:
        raise ValueError("k_dense and v_dense shapes must match")
    if k_dense.ndim != 4:
        raise ValueError(f"Expected BSHD inputs; got {k_dense.shape}")

    batch_size, seqlen, num_heads, head_dim = k_dense.shape
    num_pages_per_seq = _ceil_div(seqlen, page_size)
    max_seqlen = num_pages_per_seq * page_size
    total_pages = batch_size * num_pages_per_seq + 1  # +1 for reserved page 0

    # Create randomized page table (simulates non-contiguous allocation)
    perm_cpu = torch.randperm(total_pages - 1, device="cpu").view(batch_size, num_pages_per_seq) + 1
    page_table = perm_cpu.to(device=k_dense.device, dtype=torch.int32)

    # Create paged cache in HND format [pages, heads, page_size, dim]
    k_hnd = torch.zeros(
        (total_pages, num_heads, page_size, head_dim), device=k_dense.device, dtype=k_dense.dtype
    )
    v_hnd = torch.zeros_like(k_hnd)

    # Pad and copy to paged format
    k_padded = torch.zeros(
        (batch_size, max_seqlen, num_heads, head_dim), device=k_dense.device, dtype=k_dense.dtype
    )
    v_padded = torch.zeros_like(k_padded)
    k_padded[:, :seqlen].copy_(k_dense)
    v_padded[:, :seqlen].copy_(v_dense)

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

    # FA3 expects NHD: [pages, page_size, heads, dim]
    k_nhd = k_hnd.permute(0, 2, 1, 3)
    v_nhd = v_hnd.permute(0, 2, 1, 3)
    return page_table, k_nhd, v_nhd


class TestAppendPrefillAttention:
    """Tests for append prefill attention correctness."""

    def test_append_attention_matches_full_prefill_suffix(self, device: torch.device) -> None:
        """Append attention output matches full prefill output for suffix tokens.

        This is the core correctness test: when we reuse cached prefix KV and
        only compute attention for suffix tokens, the output should match what
        we'd get from full prefill.
        """
        torch.manual_seed(42)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

        dtype = torch.bfloat16
        batch_size = 1
        num_heads = 4
        head_dim = 64

        # Typical cache hit scenario: 730 prefix (BOS + image), 20 suffix
        prefix_len = 730
        suffix_len = 20
        total_len = prefix_len + suffix_len
        page_size = 1

        # Generate Q/K/V for full sequence
        q_full = torch.randn(
            (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
        )
        k_dense = torch.randn_like(q_full)
        v_dense = torch.randn_like(q_full)

        # Build paged KV cache
        page_table, k_paged, v_paged = _build_paged_kv_cache(
            k_dense=k_dense, v_dense=v_dense, page_size=page_size
        )
        seqused_k = torch.full((batch_size,), total_len, device=device, dtype=torch.int32)

        # Full prefill: compute attention for all positions
        out_full, _ = _flash_attn_fwd(
            q_full, k_paged, v_paged,
            page_table=page_table,
            seqused_k=seqused_k,
            causal=True,
            paged_kv_non_tma=True,
        )
        out_suffix_ref = out_full[:, prefix_len:, :, :]

        # Append prefill: only compute attention for suffix
        q_suffix = q_full[:, prefix_len:, :, :]
        out_append, _ = _flash_attn_fwd(
            q_suffix, k_paged, v_paged,
            page_table=page_table,
            seqused_k=seqused_k,  # Still total_len - K/V contains full cache
            causal=True,
            paged_kv_non_tma=True,
        )

        # Outputs should match
        torch.testing.assert_close(out_append, out_suffix_ref, rtol=1e-3, atol=1e-3)

    def test_append_attention_various_prefix_lengths(self, device: torch.device) -> None:
        """Append attention works correctly for various prefix/suffix lengths."""
        torch.manual_seed(42)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

        dtype = torch.bfloat16
        batch_size = 1
        num_heads = 4
        head_dim = 64
        page_size = 1

        # Test various prefix lengths including non-tile-aligned values
        test_cases = [
            (1, 10),      # Minimal prefix (just BOS)
            (100, 50),    # Moderate prefix
            (128, 20),    # Tile-aligned prefix (tile_m=128)
            (137, 30),    # Non-tile-aligned prefix
            (730, 5),     # Moondream image prefix, minimal suffix
            (730, 100),   # Moondream image prefix, longer suffix
        ]

        for prefix_len, suffix_len in test_cases:
            total_len = prefix_len + suffix_len

            q_full = torch.randn(
                (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
            )
            k_dense = torch.randn_like(q_full)
            v_dense = torch.randn_like(q_full)

            page_table, k_paged, v_paged = _build_paged_kv_cache(
                k_dense=k_dense, v_dense=v_dense, page_size=page_size
            )
            seqused_k = torch.full((batch_size,), total_len, device=device, dtype=torch.int32)

            # Reference: full prefill
            out_full, _ = _flash_attn_fwd(
                q_full, k_paged, v_paged,
                page_table=page_table,
                seqused_k=seqused_k,
                causal=True,
                paged_kv_non_tma=True,
            )
            out_suffix_ref = out_full[:, prefix_len:, :, :]

            # Test: append prefill
            q_suffix = q_full[:, prefix_len:, :, :]
            out_append, _ = _flash_attn_fwd(
                q_suffix, k_paged, v_paged,
                page_table=page_table,
                seqused_k=seqused_k,
                causal=True,
                paged_kv_non_tma=True,
            )

            torch.testing.assert_close(
                out_append, out_suffix_ref, rtol=1e-3, atol=1e-3,
                msg=f"Mismatch for prefix_len={prefix_len}, suffix_len={suffix_len}"
            )

    def test_append_attention_single_suffix_token(self, device: torch.device) -> None:
        """Append attention works for single-token suffix (full prompt match case).

        When the entire prompt is cached, we still need to compute at least one
        suffix token (to avoid empty Q). This tests that edge case.
        """
        torch.manual_seed(42)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

        dtype = torch.bfloat16
        batch_size = 1
        num_heads = 4
        head_dim = 64
        prefix_len = 730
        suffix_len = 1  # Single token
        total_len = prefix_len + suffix_len
        page_size = 1

        q_full = torch.randn(
            (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
        )
        k_dense = torch.randn_like(q_full)
        v_dense = torch.randn_like(q_full)

        page_table, k_paged, v_paged = _build_paged_kv_cache(
            k_dense=k_dense, v_dense=v_dense, page_size=page_size
        )
        seqused_k = torch.full((batch_size,), total_len, device=device, dtype=torch.int32)

        # Reference: full prefill
        out_full, _ = _flash_attn_fwd(
            q_full, k_paged, v_paged,
            page_table=page_table,
            seqused_k=seqused_k,
            causal=True,
            paged_kv_non_tma=True,
        )
        out_suffix_ref = out_full[:, prefix_len:, :, :]

        # Test: append with single token
        q_suffix = q_full[:, prefix_len:, :, :]
        assert q_suffix.shape[1] == 1, "Should be single token"

        out_append, _ = _flash_attn_fwd(
            q_suffix, k_paged, v_paged,
            page_table=page_table,
            seqused_k=seqused_k,
            causal=True,
            paged_kv_non_tma=True,
        )

        torch.testing.assert_close(out_append, out_suffix_ref, rtol=1e-3, atol=1e-3)

    def test_append_attention_with_fp8_kv_cache(self, device: torch.device) -> None:
        """Append attention works correctly with FP8 quantized KV cache."""
        torch.manual_seed(42)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd

        dtype = torch.bfloat16
        fp8_dtype = torch.float8_e4m3fn
        batch_size = 1
        num_heads = 4
        head_dim = 64
        prefix_len = 200
        suffix_len = 50
        total_len = prefix_len + suffix_len
        page_size = 1

        k_scale = 0.5
        v_scale = 0.5

        q_full = torch.randn(
            (batch_size, total_len, num_heads, head_dim), device=device, dtype=dtype
        )
        k_dense = torch.randn_like(q_full)
        v_dense = torch.randn_like(q_full)

        page_table, k_paged_bf16, v_paged_bf16 = _build_paged_kv_cache(
            k_dense=k_dense, v_dense=v_dense, page_size=page_size
        )

        # Quantize to FP8
        k_paged_fp8 = (k_paged_bf16 / k_scale).to(fp8_dtype)
        v_paged_fp8 = (v_paged_bf16 / v_scale).to(fp8_dtype)

        seqused_k = torch.full((batch_size,), total_len, device=device, dtype=torch.int32)

        # Reference: full prefill with FP8
        out_full, _ = _flash_attn_fwd(
            q_full, k_paged_fp8, v_paged_fp8,
            page_table=page_table,
            seqused_k=seqused_k,
            causal=True,
            paged_kv_non_tma=True,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out_suffix_ref = out_full[:, prefix_len:, :, :]

        # Test: append with FP8
        q_suffix = q_full[:, prefix_len:, :, :]
        out_append, _ = _flash_attn_fwd(
            q_suffix, k_paged_fp8, v_paged_fp8,
            page_table=page_table,
            seqused_k=seqused_k,
            causal=True,
            paged_kv_non_tma=True,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # FP8 has higher tolerance due to quantization
        torch.testing.assert_close(out_append, out_suffix_ref, rtol=5e-2, atol=3e-2)


class TestAppendPrefillOrchestration:
    """Tests for orchestrating append prefill at the text decoder level.

    These tests verify that the text decoder can be correctly configured for
    append prefill mode by the runtime.
    """

    def test_position_ids_offset_for_append(self, device: torch.device) -> None:
        """Verify position_ids should start from skip_positions for append mode.

        In append mode:
        - position_ids = [skip_positions, skip_positions+1, ..., total_len-1]
        - This is used for:
          1. RoPE: suffix tokens get correct rotary positions
          2. KV cache write: suffix K/V written to correct slots
        """
        torch.manual_seed(42)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        # This is more of a documentation test showing the expected position_ids
        skip_positions = 730
        suffix_len = 10
        total_len = skip_positions + suffix_len

        # For append mode, position_ids should be:
        position_ids = torch.arange(
            skip_positions, total_len, device=device, dtype=torch.int32
        )

        assert position_ids.shape[0] == suffix_len
        assert position_ids[0].item() == skip_positions
        assert position_ids[-1].item() == total_len - 1

    def test_seqused_k_should_be_total_len(self, device: torch.device) -> None:
        """Verify seqused_k should be total_len (not suffix_len) for append mode.

        seqused_k tells FA3 how many K/V positions to attend to.
        For append mode, this must be total_len so suffix Q attends to full cache.
        """
        skip_positions = 730
        suffix_len = 10
        total_len = skip_positions + suffix_len

        # seqused_k should be total_len for append mode
        seqused_k = torch.full((1,), total_len, device=device, dtype=torch.int32)

        assert seqused_k.item() == total_len
        assert seqused_k.item() != suffix_len  # Common mistake


class TestAppendPrefillInvariants:
    """Tests for invariants that must hold for append prefill to work correctly."""

    def test_causal_mask_offset_formula(self, device: torch.device) -> None:
        """Verify the causal mask offset formula (seqlen_k - seqlen_q).

        FA3's causal mask uses offset = seqlen_k - seqlen_q to right-align Q with K.
        For append mode with prefix_len=730, suffix_len=10:
        - seqlen_q = 10
        - seqlen_k = 740
        - offset = 740 - 10 = 730

        This means Q[0] (at local position 0) attends to K[0..730] (global positions),
        which is exactly positions 0 to skip_positions.
        """
        prefix_len = 730
        suffix_len = 10
        total_len = prefix_len + suffix_len

        seqlen_q = suffix_len
        seqlen_k = total_len
        offset = seqlen_k - seqlen_q

        assert offset == prefix_len

        # Q[i] at local position i corresponds to global position (prefix_len + i)
        # With causal mask, Q[i] can attend to K[0..(prefix_len + i)]
        for i in range(suffix_len):
            local_q_pos = i
            global_q_pos = prefix_len + i
            max_k_pos = global_q_pos  # Causal: can attend up to own position

            # Verify this is what we expect
            assert max_k_pos == prefix_len + local_q_pos
