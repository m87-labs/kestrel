"""DFlash draft model — a faithful port of the z-lab DFlash reference modeling.

The drafter is a small dense, all-full-attention transformer (the checkpoint
declares ``model_type: qwen3``) that block-predicts ``block_size`` tokens in one
non-causal forward, conditioned on the target model's hidden states. It is *not*
the Qwen 3.5 hybrid: it carries no Gated DeltaNet and uses **standard Qwen3 1D
RoPE** (not the target's MRoPE) and **standard RMSNorm weights** (no offset
folding). We therefore port the reference rather than reuse the target's fused
blocks, so the forward matches the reference exactly.

Module attribute names mirror the checkpoint keys (``fc``, ``hidden_norm``,
``layers.N.self_attn.{q,k,v,o}_proj`` / ``{q,k}_norm``, ``layers.N.mlp`` ...,
``norm``) so the checkpoint loads via ``load_state_dict`` with no key mapping.

This module is intentionally embed/LM-head free: like the reference it consumes a
pre-embedded ``noise_embedding`` and the concatenated ``target_hidden`` and
returns the final hidden state. Tying to the target's embedding/LM head and
paged-KV / kestrel-kernel integration land in a later PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DFlashConfig:
    """Subset of the checkpoint ``config.json`` the drafter needs."""

    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: tuple[int, ...] = (1, 8, 15, 22, 29)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DFlashConfig":
        dflash = data.get("dflash_config", {}) or {}
        # Fall back to the dataclass defaults whenever a (nested) key is absent,
        # so a checkpoint that omits ``dflash_config`` — or just ``mask_token_id``
        # / ``target_layer_ids`` within it — still yields a usable config rather
        # than ``mask_token_id=None`` / ``target_layer_ids=()``.
        defaults = cls()
        target_layer_ids = dflash.get("target_layer_ids")
        return cls(
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get(
                "head_dim", data["hidden_size"] // data["num_attention_heads"]
            ),
            vocab_size=data["vocab_size"],
            rms_norm_eps=data.get("rms_norm_eps", defaults.rms_norm_eps),
            rope_theta=data.get("rope_theta", defaults.rope_theta),
            block_size=data.get("block_size", defaults.block_size),
            mask_token_id=dflash.get("mask_token_id", defaults.mask_token_id),
            target_layer_ids=(
                defaults.target_layer_ids
                if target_layer_ids is None
                else tuple(target_layer_ids)
            ),
        )


class DFlashRMSNorm(nn.Module):
    """Standard Qwen3 RMSNorm (fp32 reduction, no weight offset)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DFlash RoPE: queries cover only the trailing ``q_len`` positions of the
    cos/sin cache (the query block); keys cover the full context+block span."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashRotaryEmbedding(nn.Module):
    """Standard Qwen3 1D rotary embedding (default rope, full rotary)."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [bsz, seq] covering context + query block.
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # [bsz, seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


class DFlashAttention(nn.Module):
    """Non-causal attention over [context-KV ++ query block], context K/V derived
    from the fused target hidden states (recomputed per layer, matching the
    reference; the cached-injection optimization is a later PR)."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [b, H, q, d]

        k = torch.cat([self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1)
        v = torch.cat([self.v_proj(target_hidden), self.v_proj(hidden_states)], dim=1)
        k = k.view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [b, Hkv, ctx+q, d]
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling  # non-causal
        if attn_mask is not None:
            attn = attn + attn_mask  # mask padded context positions (graphed fixed ctx)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, position_embeddings, attn_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    """Core DFlash drafter (embed/LM-head free).

    ``target_hidden`` is the concatenation over ``target_layer_ids`` of the
    target's hidden states at each context position, shape
    ``[bsz, ctx_len, len(target_layer_ids) * hidden_size]``. ``noise_embedding``
    is the embedded query block ``[bsz, q_len, hidden_size]``. ``position_ids``
    covers ``ctx_len + q_len`` positions. Returns ``[bsz, q_len, hidden_size]``.
    """

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DFlashRotaryEmbedding(config)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, attn_mask)
        return self.norm(hidden_states)
