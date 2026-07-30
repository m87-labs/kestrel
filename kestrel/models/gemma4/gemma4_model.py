"""Gemma 4 model implementation."""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from kestrel_kernels import get_runtime
from kestrel.runtime.bounded_projection import (
    PackedBoundedProjections,
)
from torch import nn
from torch.nn import functional as F

from .gemma4_config import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    attention_kv_heads,
)
from ._model_utils import SimpleDynamicCache, get_activation

_dense_runtime = get_runtime().dense
_rotary_runtime = get_runtime().rotary
_kestrel_rmsnorm = _dense_runtime.rmsnorm
_kestrel_gated_activation_into = _dense_runtime.gated_activation_into
_prepare_neox_rotary = _rotary_runtime.prepare_neox
_apply_neox_rotary = _rotary_runtime.apply_neox


class Gemma4RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        else:
            self.register_buffer(
                "weight",
                torch.ones(dim, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _kestrel_rmsnorm(hidden_states, self.weight, self.eps)


def _rope_default_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Standard RoPE: ``1 / base ** (i / dim)`` for even ``i``.

    ``partial_rotary_factor`` shortens the rotated portion (the rest of
    the head dim passes through un-rotated). ``factor`` applies linear
    scaling at the end (``inv_freq /= factor``).
    """
    dim = int(head_dim * partial_rotary_factor)
    inv = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv / float(factor)


def _rope_proportional_inv_freq(
    head_dim: int,
    base: float,
    *,
    partial_rotary_factor: float = 1.0,
    factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Proportional RoPE used by Gemma 4 for the global-attention layers.

    Rotates the first ``rope_angles = int(partial_rotary_factor * head_dim // 2)``
    frequency pairs and leaves the remaining ``head_dim // 2 - rope_angles``
    pairs un-rotated (zero inv_freq → cos=1, sin=0 → identity). The
    rotated frequencies use ``head_dim`` (not the truncated dim) in the
    denominator — that's the key difference from "default" partial RoPE.

    Output tensor has length ``head_dim // 2``.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_rot = 1.0 / (
        base
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64, device=device).float()
            / head_dim
        )
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv = torch.cat(
            (inv_rot, torch.zeros(nope_angles, dtype=torch.float32, device=device)), dim=0
        )
    else:
        inv = inv_rot
    return inv / float(factor)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class Gemma4TextRotaryEmbedding(nn.Module):

    def __init__(self, config: Gemma4TextConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_types: set[str] = set(config.layer_types or [])
        # Plain attribute (not a buffer) — see class docstring.
        self.inv_freq: dict[str, torch.Tensor] = {}
        self.attention_scaling: dict[str, float] = {}
        self._init_tables(device=device)

    def _init_tables(self, device: Optional[torch.device] = None) -> None:
        for layer_type in sorted(self.layer_types):
            params = self.config.rope_parameters.get(layer_type)
            if params is None:
                continue
            rope_type = params.get("rope_type", "default")
            base = float(params["rope_theta"])
            partial = float(params.get("partial_rotary_factor", 1.0))
            factor = float(params.get("factor", 1.0))

            # HF: only "proportional" RoPE uses global_head_dim; "default"
            # always uses head_dim (even on full_attention layers).
            if layer_type == "full_attention" and rope_type == "proportional":
                head_dim = self.config.global_head_dim
            else:
                head_dim = self.config.head_dim

            if rope_type == "default":
                inv = _rope_default_inv_freq(
                    head_dim, base, partial_rotary_factor=partial, factor=factor, device=device
                )
            elif rope_type == "proportional":
                inv = _rope_proportional_inv_freq(
                    head_dim, base, partial_rotary_factor=partial, factor=factor, device=device
                )
            else:
                raise ValueError(f"Unsupported rope_type {rope_type!r} for layer_type {layer_type!r}")

            self.inv_freq[layer_type] = inv
            self.attention_scaling[layer_type] = 1.0

    def _ensure_device(self, device: torch.device) -> None:
        for k, v in self.inv_freq.items():
            if v.device != device:
                self.inv_freq[k] = v.to(device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_device(x.device)
        inv = self.inv_freq[layer_type]
        scaling = self.attention_scaling[layer_type]

        inv_expanded = inv[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()

        # Force float32 for the rotary math — Gemma's downstream
        # numerics are sensitive to bf16 rounding in this region.
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_expanded @ pos_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * scaling
            sin = emb.sin() * scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def _attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    causal: bool,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
) -> torch.Tensor:
    if (
        attention_mask is None
        and query.device.type == "cuda"
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        out, _ = get_runtime().attention.flash_attn_fwd(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softmax_scale=scaling,
        )
        return out.contiguous()

    key_states = _repeat_kv(key, num_key_value_groups)
    value_states = _repeat_kv(value, num_key_value_groups)

    if attention_mask is None and window_size_left is not None:
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]
        q_pos = torch.arange(q_len, device=query.device) + (kv_len - q_len)
        kv_pos = torch.arange(kv_len, device=query.device)
        keep = kv_pos[None, :] >= q_pos[:, None] - window_size_left
        if window_size_right is not None:
            keep &= kv_pos[None, :] <= q_pos[:, None] + window_size_right
        attention_mask = torch.where(
            keep,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full(
                (), torch.finfo(query.dtype).min,
                dtype=query.dtype, device=query.device,
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

    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is None and causal:
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]
        q_pos = torch.arange(q_len, device=query.device) + (kv_len - q_len)
        kv_pos = torch.arange(kv_len, device=query.device)
        keep = kv_pos[None, :] <= q_pos[:, None]
        attention_mask = torch.where(
            keep,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full(
                (), torch.finfo(query.dtype).min,
                dtype=query.dtype, device=query.device,
            ),
        )[None, None, :, :]
    if attention_mask is not None:
        attn = attn + attention_mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, value_states)
    return out.transpose(1, 2).contiguous()


def _paged_attention_forward(
    query: torch.Tensor,
    *,
    paged_kv_layer: Any,
    page_table: torch.Tensor,
    paged_kv_seqlens_k: torch.Tensor,
    scaling: float,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    from kestrel_kernels import get_runtime

    q_bshd = query.transpose(1, 2).contiguous()
    k_cache = paged_kv_layer.k_cache.permute(0, 2, 1, 3)
    v_cache = paged_kv_layer.v_cache.permute(0, 2, 1, 3)
    out, _ = get_runtime().attention.flash_attn_fwd(
        q_bshd,
        k_cache,
        v_cache,
        page_table=page_table,
        seqused_k=paged_kv_seqlens_k,
        paged_kv_non_tma=True,
        causal=sliding_window is None,
        window_size_left=(sliding_window - 1) if sliding_window is not None else None,
        window_size_right=0 if sliding_window is not None else None,
        softmax_scale=scaling,
        k_scale=getattr(paged_kv_layer, "k_scale", None),
        v_scale=getattr(paged_kv_layer, "v_scale", None),
    )
    return out.contiguous()


class Gemma4TextAttention(nn.Module):

    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        layer_types = config.layer_types or []
        self.layer_type = layer_types[layer_idx] if layer_types else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = (
            config.global_head_dim if (not self.is_sliding and config.global_head_dim) else config.head_dim
        )
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_kv_heads = attention_kv_heads(config, is_sliding=self.is_sliding)
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = config.num_attention_heads // num_kv_heads
        self.scaling = 1.0
        # Shared-KV layer detection: the last ``num_kv_shared_layers``
        # layers reuse upstream K/V and skip their own K/V machinery.
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_shared >= 0 and config.num_kv_shared_layers > 0

        # Are we the last non-shared layer of our type? If so we publish
        # K/V into ``shared_kv_states`` for downstream shared layers.
        prev_layers = layer_types[:first_shared]
        if prev_layers and self.layer_type in prev_layers and not self.is_kv_shared_layer:
            self.store_full_length_kv = (
                layer_idx
                == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            )
        else:
            self.store_full_length_kv = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
            self.k_proj = nn.Linear(
                config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any,
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        shared_paged_kv_states: dict[str, Any],
        past_key_values: Optional[SimpleDynamicCache] = None,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        paged_kv_use_sliding_window: bool = True,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)

        paged_kv_layer = None
        if past_key_values is not None and hasattr(past_key_values, "get_paged_layer"):
            paged_kv_layer = past_key_values.get_paged_layer(self.layer_idx)

        shared_paged_kv_layer = (
            shared_paged_kv_states.get(self.layer_type)
            if self.is_kv_shared_layer
            else None
        )
        key_states: Optional[torch.Tensor] = None
        value_states: Optional[torch.Tensor] = None
        if self.is_kv_shared_layer:
            query_states, _ = _apply_neox_rotary(
                query_states, None, position_embeddings
            )
            query_states = query_states.transpose(1, 2)
            # Pull pre-computed K/V from the same-type non-shared layer.
            if shared_paged_kv_layer is not None:
                paged_kv_layer = shared_paged_kv_layer
            else:
                key_states, value_states = shared_kv_states[self.layer_type]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            )
            key_states = self.k_norm(key_states)
            query_states, key_states = _apply_neox_rotary(
                query_states, key_states, position_embeddings
            )
            query_states = query_states.transpose(1, 2)
            assert key_states is not None
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            if paged_kv_layer is not None:
                if (
                    cache_position_ids is None
                    or slot_mapping is None
                    or page_table is None
                    or paged_kv_seqlens_k is None
                ):
                    raise RuntimeError("Gemma paged KV update requires decode metadata")
                assert key_states is not None and value_states is not None
                paged_kv_layer.update(
                    input_pos=cache_position_ids,
                    k_val=key_states.transpose(1, 2),
                    v_val=value_states.transpose(1, 2),
                    slot_mapping=slot_mapping,
                )
            else:
                assert key_states is not None and value_states is not None
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
        if self.store_full_length_kv:
            if paged_kv_layer is not None:
                shared_paged_kv_states[self.layer_type] = paged_kv_layer
            else:
                assert key_states is not None and value_states is not None
                shared_kv_states[self.layer_type] = (key_states, value_states)

        if paged_kv_layer is not None:
            attn_out = _paged_attention_forward(
                query_states,
                paged_kv_layer=paged_kv_layer,
                page_table=page_table,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                scaling=self.scaling,
                sliding_window=self.sliding_window if paged_kv_use_sliding_window else None,
            )
        else:
            assert key_states is not None and value_states is not None
            attn_out = _attention_forward(
                query_states,
                key_states,
                value_states,
                num_key_value_groups=self.num_key_value_groups,
                attention_mask=attention_mask,
                scaling=self.scaling,
                causal=not self.is_sliding,
                window_size_left=(self.sliding_window - 1) if self.is_sliding else None,
                window_size_right=0 if self.is_sliding else None,
            )
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_out)


class Gemma4TextMLP(nn.Module):

    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared = layer_idx >= first_shared > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)
        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = get_activation(config.hidden_activation)
        self.gated_activation = (
            "gelu_tanh"
            if config.hidden_activation == "gelu_pytorch_tanh"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if self.gated_activation is None:
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            hidden = self.act_fn(gate) * up
        else:
            hidden = torch.empty(
                (*gate_up.shape[:-1], self.intermediate_size),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            _kestrel_gated_activation_into(
                hidden,
                gate_up,
                activation=self.gated_activation,
                layout="contiguous",
            )
        return self.down_proj(hidden)


class Gemma4TextDecoderLayer(nn.Module):

    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = get_activation(config.hidden_activation)
            self.per_layer_input_gate = nn.Linear(
                self.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, self.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )

        # MoE branch (used by 26B-A4B only). Built lazily / skipped for
        # dense variants where ``enable_moe_block`` is False.
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            raise NotImplementedError(
                "MoE Gemma 4 (26B-A4B) is not yet vendored; this scaffold "
                "covers the dense variants (E2B / E4B / 31B)."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor],
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        shared_paged_kv_states: dict[str, Any],
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleDynamicCache] = None,
        **attention_kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            shared_paged_kv_states=shared_paged_kv_states,
            past_key_values=past_key_values,
            **attention_kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4TextScaledWordEmbedding(nn.Embedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def _mask_neg_value(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


@dataclass
class Gemma4TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[SimpleDynamicCache]


@dataclass
class Gemma4CausalLMOutput:
    logits: torch.Tensor
    past_key_values: Optional[SimpleDynamicCache]


class Gemma4TextModel(nn.Module):

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id or 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types or [])

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.tensor(2.0**-0.5),
                persistent=False,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        per = self.embed_tokens_per_layer(input_ids)
        return per.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        proj = self.per_layer_projection_norm(proj)
        if per_layer_inputs is None:
            return proj
        return (proj + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleDynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        prebuilt_masks: Optional[dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        paged_kv_use_sliding_window: bool = True,
    ) -> Gemma4TextModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("per_layer_inputs requires inputs_embeds (not input_ids)")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                assert input_ids is not None
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            past_key_values = SimpleDynamicCache()

        seq_len = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_len
            position_ids = position_ids.unsqueeze(0)

        masks = prebuilt_masks or {}
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            cos, sin = self.rotary_emb(inputs_embeds, position_ids, layer_type)
            position_embeddings[layer_type] = _prepare_neox_rotary(cos, sin)

        hidden_states = inputs_embeds
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = UserDict()
        shared_paged_kv_states: dict[str, Any] = {}
        for i, layer in enumerate(self.layers):
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            layer_type = self.config.layer_types[i]
            hidden_states = layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=masks.get(layer_type),
                shared_paged_kv_states=shared_paged_kv_states,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position_ids=cache_position_ids,
                slot_mapping=slot_mapping,
                page_table=page_table,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                paged_kv_use_sliding_window=paged_kv_use_sliding_window,
            )

        hidden_states = self.norm(hidden_states)
        return Gemma4TextModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Gemma4ForCausalLM(nn.Module):

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # HF ties the LM head to the (unscaled) token embedding weight.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleDynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        prebuilt_masks: Optional[dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        paged_kv_use_sliding_window: bool = True,
    ) -> Gemma4CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            prebuilt_masks=prebuilt_masks,
            use_cache=use_cache,
            cache_position_ids=cache_position_ids,
            slot_mapping=slot_mapping,
            page_table=page_table,
            paged_kv_seqlens_k=paged_kv_seqlens_k,
            paged_kv_use_sliding_window=paged_kv_use_sliding_window,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return Gemma4CausalLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

class Gemma4ClippableLinear(nn.Module):

    def __init__(
        self,
        config: Gemma4VisionConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        return self.forward_bounded_input(hidden_states)

    def forward_bounded_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class Gemma4VisionPatchEmbedder(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        position_embeddings = (
            self.position_embedding_table[0, clamped_positions[..., 0]]
            + self.position_embedding_table[1, clamped_positions[..., 1]]
        )
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


class Gemma4VisionPooler(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]})."
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionMLP(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = Gemma4ClippableLinear(
            config,
            self.hidden_size,
            2 * self.intermediate_size,
        )
        self.down_proj = Gemma4ClippableLinear(config, self.intermediate_size, self.hidden_size)
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "fused vision MLP requires hidden_activation='gelu_pytorch_tanh', "
                f"got {config.hidden_activation!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        hidden = torch.empty(
            (*gate_up.shape[:-1], self.intermediate_size),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        _kestrel_gated_activation_into(
            hidden,
            gate_up,
            activation="gelu_tanh",
            layout="contiguous",
        )
        return self.down_proj(hidden)


def _vision_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _vision_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_vision_rotate_half(x) * sin)


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Splits the head_dim into ``ndim`` blocks and applies RoPE per-axis.

    ``ndim`` is ``position_ids.shape[-1]`` (2 for image x/y). Each block
    has length ``2 * (head_dim // (2 * ndim))``; remaining channels (if
    head_dim isn't divisible cleanly) pass through unrotated.
    """
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "num_rotated_channels_per_dim must be > 0;"
            f" got {num_rotated_channels_per_dim} (channels={num_input_channels}, ndim={ndim})"
        )
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _vision_apply_rope(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4VisionRotaryEmbedding(nn.Module):

    def __init__(self, config: Gemma4VisionConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters.get("rope_type", "default")
        if self.rope_type != "default":
            raise ValueError(f"Vision RoPE only supports rope_type='default', got {self.rope_type!r}")
        base = float(config.rope_parameters["rope_theta"])
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # Per HF: the reference impl computes RoPE freqs independently for each
        # spatial dimension using head_dim // ndim (ndim=2 for x/y), so each axis
        # gets the same frequency range — not a global inv_freq split.
        spatial_dim = head_dim // 2
        self.spatial_dim = spatial_dim
        inv = 1.0 / (
            base
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.int64, device=device).float() / spatial_dim)
        )
        self.inv_freq = inv
        self.attention_scaling = 1.0

    def _ensure_device(self, device: torch.device) -> None:
        if self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_device(x.device)
        inv_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        device_type = x.device.type if x.device.type != "mps" else "cpu"

        all_cos, all_sin = [], []
        for i in range(2):
            dim_pos = position_ids[:, :, i]
            dim_pos_expanded = dim_pos[:, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_expanded @ dim_pos_expanded).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


def _vision_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv, n_rep, slen, head_dim)
        .reshape(batch, num_kv * n_rep, slen, head_dim)
    )


def _vision_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    seqused_k: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    if seqused_k.dtype != torch.int32 or seqused_k.shape != query.shape[:1]:
        raise ValueError(
            "vision used-K lengths must be int32 [batch], got "
            f"{seqused_k.dtype} {tuple(seqused_k.shape)} for query {tuple(query.shape)}"
        )
    if query.device.type in ("cuda", "mps") and query.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        out, _ = get_runtime().attention.flash_attn_fwd(
            query,
            key,
            value,
            seqused_k=seqused_k,
            causal=False,
            softmax_scale=scaling,
        )
        return out

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    key_states = _vision_repeat_kv(key, num_key_value_groups)
    value_states = _vision_repeat_kv(value, num_key_value_groups)
    positions = torch.arange(query.shape[-2], device=query.device)
    valid = positions.unsqueeze(0) < seqused_k.unsqueeze(1)
    attention_mask = _build_bidirectional_mask(valid, dtype=query.dtype)
    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn = attn + attention_mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, value_states)
    return out.transpose(1, 2).contiguous()


class Gemma4VisionAttention(nn.Module):

    def __init__(self, config: Gemma4VisionConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0
        self.qkv_proj = PackedBoundedProjections(
            config.hidden_size,
            (
                config.num_attention_heads * self.head_dim,
                config.num_key_value_heads * self.head_dim,
                config.num_key_value_heads * self.head_dim,
            ),
            source_names=("q_proj", "k_proj", "v_proj"),
            use_bounds=config.use_clipped_linears,
        )
        self.o_proj = Gemma4ClippableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqused_k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states, key_states, value_states = self.qkv_proj(hidden_states)
        query_states = query_states.view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids)

        key_states = key_states.view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids)

        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states)

        attn_out = _vision_attention_forward(
            query_states,
            key_states,
            value_states,
            num_key_value_groups=self.num_key_value_groups,
            seqused_k=seqused_k,
            scaling=self.scaling,
        )
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_out)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqused_k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seqused_k=seqused_k,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _build_bidirectional_mask(
    valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Bidirectional additive mask: ``[B, 1, Q, K]``.

    ``valid`` is ``[B, S]`` bool, True for valid patches. Output blocks
    keys that are padding (per row); on a query that is itself padding,
    HF's behaviour is to let the row attend anywhere (padding rows are
    discarded downstream).
    """
    # Keys that are padding get masked out for every query.
    B, S = valid.shape
    neg = _mask_neg_value(dtype)
    # [B, 1, 1, S] additive: 0 where valid, neg where padding
    mask_kv = torch.where(
        valid[:, None, None, :],
        torch.zeros((), dtype=dtype, device=valid.device),
        torch.full((), neg, dtype=dtype, device=valid.device),
    )
    return mask_kv.expand(B, 1, S, S)


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config, i) for i in range(self.num_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)
        seqused_k = attention_mask.sum(dim=-1, dtype=torch.int32)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                seqused_k=seqused_k,
                position_ids=pixel_position_ids,
            )
        return hidden_states


@dataclass
class Gemma4VisionOutput:
    last_hidden_state: torch.Tensor


class Gemma4VisionModel(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        if config.standardize:
            self.register_buffer("std_bias", torch.empty(config.hidden_size))
            self.register_buffer("std_scale", torch.empty(config.hidden_size))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> Gemma4VisionOutput:
        pooling_kernel_size = self.config.pooling_kernel_size
        # HF derives ``output_length`` from the input grid divided by
        # pooling_kernel_size**2 along the patch axis (axis -2 of
        # pixel_values *as the processor returns it* — see HF source).
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        encoded = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoded,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return Gemma4VisionOutput(last_hidden_state=hidden_states)


class Gemma4VisionEmbedder(nn.Module):

    def __init__(
        self,
        vision_config: Gemma4VisionConfig,
        text_config: Gemma4TextConfig,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_config.hidden_size
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = nn.Linear(
            self.vision_hidden_size, self.text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            self.vision_hidden_size,
            eps=vision_config.rms_norm_eps,
            with_scale=False,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


@dataclass
class Gemma4ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[SimpleDynamicCache]
    image_hidden_states: Optional[torch.Tensor] = None


@dataclass
class Gemma4CausalLMOutput:
    logits: torch.Tensor
    past_key_values: Optional[SimpleDynamicCache]
    image_hidden_states: Optional[torch.Tensor] = None


class Gemma4Model(nn.Module):

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        text_cfg = config.text_config
        self.vocab_size = text_cfg.vocab_size

        self.language_model = Gemma4TextModel(text_cfg)
        self.vocab_size_per_layer_input = text_cfg.vocab_size_per_layer_input

        if config.vision_config is not None:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.embed_vision = Gemma4VisionEmbedder(config.vision_config, text_cfg)
        else:
            self.vision_tower = None
            self.embed_vision = None

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.embed_tokens

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image patches and project into text-embedding space."""
        if self.vision_tower is None:
            raise RuntimeError("vision_config is None; image inputs not supported")
        vision_out = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return self.embed_vision(vision_out.last_hidden_state)

    def image_placeholder_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids == self.config.image_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleDynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        prebuilt_masks: Optional[dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        paged_kv_use_sliding_window: bool = True,
    ) -> Gemma4ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        # Build placeholder masks before we trample input_ids.
        if input_ids is not None:
            image_mask = self.image_placeholder_mask(input_ids)

            # The image token id may be OOV for embed_tokens. Replace it with
            # pad before lookup, then scatter the real vision features.
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = self.config.text_config.pad_token_id or 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        else:
            image_mask = torch.zeros_like(inputs_embeds[..., 0], dtype=torch.bool)
            llm_input_ids = None

        # PLE inputs: derived from input_ids (with multimodal slots → pad).
        if per_layer_inputs is None and self.config.text_config.hidden_size_per_layer_input:
            if llm_input_ids is not None:
                per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids)
            # If only inputs_embeds was provided, language_model.forward
            # will reverse-lookup to recover ids — but that's expensive.
            # Production callers should pass per_layer_inputs directly
            # when feeding raw embeds.

        # Merge image features.
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            scatter_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, image_features)

        out = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            prebuilt_masks=prebuilt_masks,
            use_cache=use_cache,
            cache_position_ids=cache_position_ids,
            slot_mapping=slot_mapping,
            page_table=page_table,
            paged_kv_seqlens_k=paged_kv_seqlens_k,
            paged_kv_use_sliding_window=paged_kv_use_sliding_window,
        )
        return Gemma4ModelOutput(
            last_hidden_state=out.last_hidden_state,
            past_key_values=out.past_key_values,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Gemma4ForConditionalGeneration(nn.Module):

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        text_cfg = config.text_config
        self.model = Gemma4Model(config)
        self.vocab_size = text_cfg.vocab_size
        self.lm_head = nn.Linear(text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # HF ties the LM head to the (unscaled) embed_tokens weight.
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleDynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        prebuilt_masks: Optional[dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> Gemma4CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            prebuilt_masks=prebuilt_masks,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.text_config.final_logit_softcapping is not None:
            cap = self.config.text_config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return Gemma4CausalLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            image_hidden_states=outputs.image_hidden_states,
        )

__all__ = [
    "Gemma4ForConditionalGeneration",
    "Gemma4Model",
    "SimpleDynamicCache",
]
