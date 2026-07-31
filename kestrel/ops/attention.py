"""Model-independent inference attention operations."""

from __future__ import annotations

from typing import Any

import torch
from kestrel_kernels import get_runtime
from torch.nn import functional as F


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
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


def bidirectional_padding_mask(
    valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``[batch, 1, query, key]`` additive key-padding mask."""
    batch, sequence_length = valid.shape
    mask = torch.where(
        valid[:, None, None, :],
        torch.zeros((), dtype=dtype, device=valid.device),
        torch.full(
            (),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=valid.device,
        ),
    )
    return mask.expand(batch, 1, sequence_length, sequence_length)


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
    """Run dense attention over ``[batch, heads, sequence, dim]`` inputs."""
    if cu_seqlens is not None:
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != query.device
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError(
                "packed attention cumulative lengths must be contiguous int32 "
                f"on {query.device}, got {cu_seqlens.dtype} "
                f"{cu_seqlens.device} {tuple(cu_seqlens.shape)}"
            )
    if cu_seqlens is not None and query.device.type != "cuda":
        boundaries = cu_seqlens.tolist()
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != query.shape[-2]
            or any(start >= end for start, end in zip(
                boundaries[:-1], boundaries[1:], strict=True
            ))
        ):
            raise ValueError(
                "packed attention cumulative lengths must begin at zero, end "
                "at the packed token count, and contain nonempty rows"
            )
        outputs = [
            dense_attention(
                query[..., start:end, :],
                key[..., start:end, :],
                value[..., start:end, :],
                num_key_value_groups,
                attention_mask=None,
                scaling=scaling,
                causal=causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
            for start, end in zip(
                boundaries[:-1],
                boundaries[1:],
                strict=True,
            )
        ]
        return torch.cat(outputs, dim=1)

    if (
        attention_mask is None
        and query.device.type == "cuda"
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        query_bshd = query.transpose(1, 2).contiguous()
        key_bshd = key.transpose(1, 2).contiguous()
        value_bshd = value.transpose(1, 2).contiguous()
        if cu_seqlens is not None:
            query_bshd = query_bshd.flatten(0, 1)
            key_bshd = key_bshd.flatten(0, 1)
            value_bshd = value_bshd.flatten(0, 1)
            if (
                query_bshd.ndim != 3
                or key_bshd.ndim != 3
                or value_bshd.ndim != 3
            ):
                raise RuntimeError(
                    "packed FlashAttention inputs must be [tokens, heads, dim]"
                )
        arguments = {
            "causal": causal,
            "window_size_left": window_size_left,
            "window_size_right": window_size_right,
            "softmax_scale": scaling,
        }
        if cu_seqlens is not None:
            arguments["cu_seqlens_q"] = cu_seqlens
            arguments["cu_seqlens_k"] = cu_seqlens
        out, _ = get_runtime().attention.flash_attn_fwd(
            query_bshd,
            key_bshd,
            value_bshd,
            **arguments,
        )
        if cu_seqlens is not None:
            out = out.reshape(query.shape[0], query.shape[2], *out.shape[-2:])
        return out.contiguous()

    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    if attention_mask is None and window_size_left is not None:
        query_length = query.shape[-2]
        key_length = key_states.shape[-2]
        query_positions = (
            torch.arange(query_length, device=query.device)
            + key_length
            - query_length
        )
        key_positions = torch.arange(key_length, device=query.device)
        keep = (
            key_positions[None, :]
            >= query_positions[:, None] - window_size_left
        )
        if causal:
            keep &= key_positions[None, :] <= query_positions[:, None]
        if window_size_right is not None:
            keep &= (
                key_positions[None, :]
                <= query_positions[:, None] + window_size_right
            )
        attention_mask = torch.where(
            keep,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full(
                (),
                torch.finfo(query.dtype).min,
                dtype=query.dtype,
                device=query.device,
            ),
        )[None, None, :, :]

    if query.device.type in ("cuda", "mps") and query.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        out = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=causal and attention_mask is None,
            scale=scaling,
        )
        return out.transpose(1, 2).contiguous()

    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is None and causal:
        query_length = query.shape[-2]
        key_length = key_states.shape[-2]
        query_positions = (
            torch.arange(query_length, device=query.device)
            + key_length
            - query_length
        )
        key_positions = torch.arange(key_length, device=query.device)
        keep = key_positions[None, :] <= query_positions[:, None]
        attention_mask = torch.where(
            keep,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full(
                (),
                torch.finfo(query.dtype).min,
                dtype=query.dtype,
                device=query.device,
            ),
        )[None, None, :, :]
    if attention_mask is not None:
        scores = scores + attention_mask
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    out = torch.matmul(probabilities, value_states)
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
    """Run single-token attention against a paged KV layer."""
    query_bshd = query.transpose(1, 2).contiguous()
    key_cache = paged_kv_layer.k_cache.permute(0, 2, 1, 3)
    value_cache = paged_kv_layer.v_cache.permute(0, 2, 1, 3)
    out, _ = get_runtime().attention.flash_attn_fwd(
        query_bshd,
        key_cache,
        value_cache,
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
    """Run noncausal attention over packed rows with per-row used lengths."""
    if (
        used_key_lengths.dtype != torch.int32
        or used_key_lengths.shape != query.shape[:1]
    ):
        raise ValueError(
            "used-K lengths must be int32 [batch], got "
            f"{used_key_lengths.dtype} {tuple(used_key_lengths.shape)} "
            f"for query {tuple(query.shape)}"
        )
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
    key_states = repeat_kv(key.transpose(1, 2), num_key_value_groups)
    value_states = repeat_kv(value.transpose(1, 2), num_key_value_groups)
    positions = torch.arange(query.shape[-2], device=query.device)
    valid = positions.unsqueeze(0) < used_key_lengths.unsqueeze(1)
    mask = bidirectional_padding_mask(valid, dtype=query.dtype)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    probabilities = F.softmax(
        scores + mask,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    out = torch.matmul(probabilities, value_states)
    return out.transpose(1, 2).contiguous()


__all__ = [
    "bidirectional_padding_mask",
    "dense_attention",
    "paged_attention",
    "repeat_kv",
    "variable_length_attention",
]
