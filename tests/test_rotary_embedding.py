"""Correctness tests for rotary_embedding."""

import pytest
import torch

from kestrel.ops.rotary_embedding import precompute_freqs_cis, rotary_embedding_cuda


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _rotary_reference_neox_fp32(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference matching vixtral-train (fp32 math, cast back once).

    vixtral-train stores freqs in fp32 and multiplies bf16 by fp32 (promoting to fp32),
    then casts back to bf16.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must be float32")

    rot_dim = int(cos_sin_cache.size(1))
    assert rot_dim % 2 == 0
    embed_dim = rot_dim // 2

    cache = cos_sin_cache.index_select(0, positions.reshape(-1))  # [T, rot_dim]
    cos = cache[:, :embed_dim].view(*positions.shape, 1, embed_dim)
    sin = cache[:, embed_dim:].view(*positions.shape, 1, embed_dim)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x0 = x[..., :embed_dim].float()
        x1 = x[..., embed_dim : 2 * embed_dim].float()
        out0 = (x0 * cos - x1 * sin).to(dtype=x.dtype)
        out1 = (x1 * cos + x0 * sin).to(dtype=x.dtype)
        return torch.cat([out0, out1, x[..., 2 * embed_dim :]], dim=-1)

    return apply(q), apply(k)


@pytest.mark.parametrize(
    "bsz,seqlen",
    [
        (2, 17),  # triggers split-head 2-offset-per-thread kernel (num_tokens <= 64)
        (2, 50),  # triggers 2-offset-per-thread kernel (num_tokens > 64)
        (2, 65),  # triggers 2-offset-per-thread kernel (num_tokens > 64)
    ],
)
def test_rotary_embedding_matches_reference(
    device: torch.device, bsz: int, seqlen: int
) -> None:
    torch.manual_seed(0)

    num_heads = 32
    head_size = 64
    rot_dim = 32
    max_pos = 256

    positions = torch.randint(0, max_pos, (bsz, seqlen), device=device, dtype=torch.int64)
    q = torch.randn((bsz, seqlen, num_heads, head_size), device=device, dtype=torch.bfloat16)
    k = torch.randn((bsz, seqlen, num_heads, head_size), device=device, dtype=torch.bfloat16)
    cos_sin_cache_fp32 = precompute_freqs_cis(
        rot_dim, max_pos, dtype=torch.float32, device=device
    )

    q_in = q.clone()
    k_in = k.clone()
    rotary_embedding_cuda(positions, q, k, head_size, cos_sin_cache_fp32)

    # vixtral-train computes RoPE in fp32 (freqs are fp32), then casts once to bf16.
    q_fp32, k_fp32 = _rotary_reference_neox_fp32(
        positions, q_in, k_in, cos_sin_cache_fp32
    )
    torch.testing.assert_close(q, q_fp32, atol=0, rtol=0)
    torch.testing.assert_close(k, k_fp32, atol=0, rtol=0)


def test_rotary_embedding_rejects_non_contiguous_query(device: torch.device) -> None:
    positions = torch.zeros((1, 1), device=device, dtype=torch.int64)
    q_full = torch.zeros((1, 1, 1, 16), device=device, dtype=torch.bfloat16)
    q = q_full[..., ::2]  # shape [..., 8] but non-contiguous
    assert not q.is_contiguous()
    k = torch.zeros((1, 1, 1, 8), device=device, dtype=torch.bfloat16)
    cos_sin_cache = precompute_freqs_cis(8, 1, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="query must be contiguous"):
        rotary_embedding_cuda(positions, q, k, 8, cos_sin_cache)


def test_rotary_embedding_rejects_dtype_mismatch(device: torch.device) -> None:
    positions = torch.zeros((1, 1), device=device, dtype=torch.int64)
    q = torch.zeros((1, 1, 1, 8), device=device, dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 1, 8), device=device, dtype=torch.bfloat16)
    cos_sin_cache = precompute_freqs_cis(8, 1, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="cos_sin_cache must be float32"):
        rotary_embedding_cuda(positions, q, k, 8, cos_sin_cache)
