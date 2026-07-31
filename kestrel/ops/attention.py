"""Model-independent inference attention operations."""

from __future__ import annotations

from typing import Any

import torch
from kestrel_kernels import get_runtime
from torch.nn import functional as F


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, width = hidden_states.shape
    return (
        hidden_states[:, :, None]
        .expand(batch, heads, repeats, sequence, width)
        .reshape(batch, heads * repeats, sequence, width)
    )


def bidirectional_padding_mask(
    valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    blocked = torch.full(
        (), torch.finfo(dtype).min, dtype=dtype, device=valid.device
    )
    return torch.where(valid[:, None, None, :], 0.0, blocked).expand(
        valid.shape[0], 1, valid.shape[1], valid.shape[1]
    )


def _window_mask(
    query: torch.Tensor,
    key_length: int,
    *,
    causal: bool,
    left: int,
    right: int | None,
) -> torch.Tensor:
    query_length = query.shape[-2]
    query_positions = (
        torch.arange(query_length, device=query.device)
        + key_length
        - query_length
    )
    key_positions = torch.arange(key_length, device=query.device)
    keep = key_positions[None] >= query_positions[:, None] - left
    if causal:
        keep &= key_positions[None] <= query_positions[:, None]
    if right is not None:
        keep &= key_positions[None] <= query_positions[:, None] + right
    return torch.where(
        keep,
        0.0,
        torch.full(
            (), torch.finfo(query.dtype).min,
            dtype=query.dtype, device=query.device,
        ),
    )[None, None]


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    attention_mask: torch.Tensor | None,
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
    if cu_seqlens is not None and query.device.type != "cuda":
        boundaries = cu_seqlens.tolist()
        if (
            boundaries[:1] != [0]
            or boundaries[-1:] != [query.shape[-2]]
            or any(a >= b for a, b in zip(boundaries[:-1], boundaries[1:]))
        ):
            raise ValueError(
                "packed row boundaries must strictly partition every token"
            )
        return torch.cat(
            [
                dense_attention(
                    query[..., start:end, :],
                    key[..., start:end, :],
                    value[..., start:end, :],
                    num_key_value_groups,
                    None,
                    scaling,
                    causal,
                    window_size_left,
                    window_size_right,
                )
                for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            ],
            dim=1,
        )

    if (
        attention_mask is None
        and query.device.type == "cuda"
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        q, k, v = (
            tensor.transpose(1, 2).contiguous()
            for tensor in (query, key, value)
        )
        arguments: dict[str, Any] = {
            "causal": causal,
            "window_size_left": window_size_left,
            "window_size_right": window_size_right,
            "softmax_scale": scaling,
        }
        if cu_seqlens is not None:
            q, k, v = (
                tensor.flatten(0, 1) for tensor in (q, k, v)
            )
            arguments.update(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
            )
        out, _ = get_runtime().attention.flash_attn_fwd(q, k, v, **arguments)
        if cu_seqlens is not None:
            out = out.reshape(query.shape[0], query.shape[2], *out.shape[-2:])
        return out.contiguous()

    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)
    if attention_mask is None and window_size_left is not None:
        attention_mask = _window_mask(
            query,
            key.shape[-2],
            causal=causal,
            left=window_size_left,
            right=window_size_right,
        )
    elif attention_mask is None and causal:
        attention_mask = _window_mask(
            query,
            key.shape[-2],
            causal=True,
            left=key.shape[-2],
            right=None,
        )
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous()


def paged_attention(
    query: torch.Tensor,
    *,
    paged_kv_layer: Any,
    page_table: torch.Tensor,
    paged_kv_seqlens_k: torch.Tensor,
    scaling: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Attend one token against a paged K/V layer."""
    out, _ = get_runtime().attention.flash_attn_fwd(
        query.transpose(1, 2).contiguous(),
        paged_kv_layer.k_cache.permute(0, 2, 1, 3),
        paged_kv_layer.v_cache.permute(0, 2, 1, 3),
        page_table=page_table,
        seqused_k=paged_kv_seqlens_k,
        paged_kv_non_tma=True,
        causal=sliding_window is None,
        window_size_left=(
            sliding_window - 1 if sliding_window is not None else None
        ),
        window_size_right=0 if sliding_window is not None else None,
        softmax_scale=scaling,
        k_scale=paged_kv_layer.k_scale,
        v_scale=paged_kv_layer.v_scale,
    )
    return out.contiguous()


def variable_length_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    used_key_lengths: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Attend over rows whose valid prefixes are given by ``used_key_lengths``."""
    if (
        used_key_lengths.dtype != torch.int32
        or used_key_lengths.device != query.device
        or used_key_lengths.shape != query.shape[:1]
    ):
        raise ValueError("used-K lengths must be on-device int32 [batch]")
    if query.device.type in ("cuda", "mps") and query.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        out, _ = get_runtime().attention.flash_attn_fwd(
            query,
            key,
            value,
            seqused_k=used_key_lengths,
            causal=False,
            softmax_scale=scaling,
        )
        return out

    query = query.transpose(1, 2)
    key = repeat_kv(key.transpose(1, 2), num_key_value_groups)
    value = repeat_kv(value.transpose(1, 2), num_key_value_groups)
    positions = torch.arange(query.shape[-2], device=query.device)
    mask = bidirectional_padding_mask(
        positions[None] < used_key_lengths[:, None],
        dtype=query.dtype,
    )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scaling,
    ).transpose(1, 2).contiguous()


__all__ = [
    "bidirectional_padding_mask",
    "dense_attention",
    "paged_attention",
    "repeat_kv",
    "variable_length_attention",
]
