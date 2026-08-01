"""Model-independent inference attention operations."""

from __future__ import annotations

from typing import Any

import torch
from kestrel_kernels import get_runtime


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend over ``[batch, heads, sequence, dim]`` or packed rows."""
    if cu_seqlens is not None and (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != query.device
        or cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("packed row boundaries must be contiguous int32 on-device")
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    arguments: dict[str, Any] = {
        "causal": causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "softmax_scale": scaling,
    }
    if cu_seqlens is not None:
        q, k, v = (tensor.flatten(0, 1) for tensor in (q, k, v))
        arguments.update(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
    out, _ = get_runtime().attention.flash_attn_fwd(q, k, v, **arguments)
    if cu_seqlens is not None:
        out = out.reshape(query.shape[0], query.shape[2], *out.shape[-2:])
    return out


def paged_attention(
    query: torch.Tensor,
    *,
    paged_kv_layer: Any,
    page_table: torch.Tensor,
    paged_kv_seqlens_k: torch.Tensor,
    scaling: float,
    paged_kv_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Attend batched or packed queries against a paged K/V layer."""
    q = query.transpose(1, 2)
    seqused_q = paged_kv_seqlens_q
    if cu_seqlens_q is not None:
        q = q.flatten(0, 1)
        seqused_q = None
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"paged attention requires fp16/bf16 query, got {q.dtype}")

    out, _ = get_runtime().attention.flash_attn_fwd(
        q,
        paged_kv_layer.k_cache.permute(0, 2, 1, 3),
        paged_kv_layer.v_cache.permute(0, 2, 1, 3),
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_k=paged_kv_seqlens_k,
        paged_kv_non_tma=True,
        causal=sliding_window is None,
        window_size_left=(
            sliding_window - 1 if sliding_window is not None else None
        ),
        window_size_right=0 if sliding_window is not None else None,
        softmax_scale=scaling,
        k_scale=getattr(paged_kv_layer, "k_scale", None),
        v_scale=getattr(paged_kv_layer, "v_scale", None),
    )
    if cu_seqlens_q is not None:
        out = out.reshape(query.shape[0], query.shape[2], *out.shape[-2:])
    return out


__all__ = [
    "dense_attention",
    "paged_attention",
]
