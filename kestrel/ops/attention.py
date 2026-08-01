"""Model-independent attention tensor operations."""

from __future__ import annotations

import torch
from kestrel_kernels import get_runtime


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand grouped K/V heads without materializing the broadcast dimension."""
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        kv_heads,
        repeats,
        sequence_length,
        head_dim,
    )
    return expanded.reshape(
        batch,
        kv_heads * repeats,
        sequence_length,
        head_dim,
    )


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense or packed attention through the device runtime."""

    q, k, v = (
        tensor.transpose(1, 2).contiguous()
        for tensor in (query, key, value)
    )
    arguments = {"causal": causal, "softmax_scale": scaling}
    if cu_seqlens is not None:
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != query.device
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError(
                "packed row boundaries must be contiguous int32 on-device")
        q, k, v = (tensor.flatten(0, 1) for tensor in (q, k, v))
        arguments.update(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
    out, _ = get_runtime().attention.flash_attn_fwd(q, k, v, **arguments)
    if cu_seqlens is not None:
        out = out.reshape(query.shape[0], query.shape[2], *out.shape[-2:])
    return out.contiguous()


__all__ = ["dense_attention", "repeat_kv"]
