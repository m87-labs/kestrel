from __future__ import annotations

import torch

from kestrel_kernels.tau_tail import tau_tail_apply_cuda


def tau_tail_apply_into(
    *,
    qkv_out: torch.Tensor,
    tok_qv_lin: torch.Tensor,
    tau_pos_table: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    """Apply tau scaling in-place to Q and V views inside `qkv_out`.

    This implements the "tau tail" after `tok_qv_lin = linear(gelu(qkv_out), wqwv)`:
      - tok_qv = tanh(tok_qv_lin)
      - tau_pos = tau_pos_table[position_ids]
      - q *= (tok_q + tau_pos)[..., None]
      - v *= (tok_v + tau_pos)[..., None]

    Notes:
      - Intended for inference (no backward).
      - `qkv_out` must be shaped (B, S, 3 * n_heads * head_dim) (i.e. no GQA).
      - Uses a custom CUDA kernel to fuse tanh + gather + Q/V scaling.
    """
    if qkv_out.ndim != 3:
        raise ValueError(f"qkv_out must be rank-3 (B,S,C), got {tuple(qkv_out.shape)}")
    if tok_qv_lin.ndim != 3:
        raise ValueError(
            f"tok_qv_lin must be rank-3 (B,S,2H), got {tuple(tok_qv_lin.shape)}"
        )
    if tau_pos_table.ndim != 2:
        raise ValueError(
            f"tau_pos_table must be rank-2 (max_context,H), got {tuple(tau_pos_table.shape)}"
        )
    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must be rank-2 (B,S), got {tuple(position_ids.shape)}"
        )

    if qkv_out.shape[:2] != tok_qv_lin.shape[:2]:
        raise ValueError(
            f"qkv_out and tok_qv_lin must match in (B,S), got {tuple(qkv_out.shape[:2])} vs {tuple(tok_qv_lin.shape[:2])}"
        )
    if qkv_out.shape[:2] != position_ids.shape:
        raise ValueError(
            f"qkv_out and position_ids must match in (B,S), got {tuple(qkv_out.shape[:2])} vs {tuple(position_ids.shape)}"
        )

    if qkv_out.dtype != tok_qv_lin.dtype or qkv_out.dtype != tau_pos_table.dtype:
        raise ValueError("qkv_out/tok_qv_lin/tau_pos_table must have the same dtype")

    if position_ids.dtype not in (torch.int64,):
        raise ValueError(f"position_ids must be int64, got {position_ids.dtype}")

    tau_tail_apply_cuda(qkv_out, tok_qv_lin, tau_pos_table, position_ids)


__all__ = [
    "tau_tail_apply_cuda",
    "tau_tail_apply_into",
]

