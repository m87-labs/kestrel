"""Tests for fused tau tail (tanh + tau_pos gather + Q/V scaling)."""

import pytest
import torch

from kestrel_kernels.tau_tail_ops import tau_tail_apply_into


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


DTYPES = (torch.float16, torch.bfloat16)
RTOL = 5e-2
ATOL = 5e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [1, 17, 257])
def test_tau_tail_matches_pytorch(device, dtype: torch.dtype, seq_len: int) -> None:
    torch.manual_seed(0)

    bsz = 2
    n_heads = 32
    head_dim = 64
    q_dim = n_heads * head_dim
    qkv_dim = 3 * q_dim
    max_context = 512

    qkv_init = torch.randn((bsz, seq_len, qkv_dim), device=device, dtype=dtype)
    tok_qv_lin = torch.randn((bsz, seq_len, 2 * n_heads), device=device, dtype=dtype)

    tau_pos_table = torch.randn((max_context, n_heads), device=device, dtype=dtype)
    position_ids = torch.randint(
        0, max_context, (bsz, seq_len), device=device, dtype=torch.long
    )

    qkv_ref = qkv_init.clone()
    q_ref = qkv_ref[..., :q_dim].view(bsz, seq_len, n_heads, head_dim)
    v_ref = qkv_ref[..., 2 * q_dim :].view(bsz, seq_len, n_heads, head_dim)

    tok_qv = tok_qv_lin.clone()
    tok_qv.tanh_()
    tok_q, tok_v = tok_qv.split(n_heads, dim=-1)
    tau_pos = tau_pos_table[position_ids]
    q_ref.mul_((tok_q + tau_pos).unsqueeze(-1))
    v_ref.mul_((tok_v + tau_pos).unsqueeze(-1))

    qkv_out = qkv_init.clone()
    tau_tail_apply_into(
        qkv_out=qkv_out,
        tok_qv_lin=tok_qv_lin,
        tau_pos_table=tau_pos_table,
        position_ids=position_ids,
    )

    torch.testing.assert_close(qkv_out, qkv_ref, rtol=RTOL, atol=ATOL)
