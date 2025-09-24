"""Text transformer implementation for Moondream.

Adapted from the Moondream project (Apache-2.0).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention

from .config import TextConfig
from .layers import (
    build_dense_mlp,
    build_moe_mlp,
    layer_norm,
    mlp,
    moe_mlp,
    LayerNormWeights,
    LinearWeights,
    MLPWeights,
)
from .rope import apply_rotary_emb, precompute_freqs_cis


def text_encoder(input_ids: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return F.embedding(input_ids, module.wte)


def attn(
    x: torch.Tensor,
    module: nn.Module,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[nn.Module],
    attn_mask: Optional[torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Optional[dict] = None,
    flex_block_mask_slice=None,
) -> torch.Tensor:
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    if position_ids.ndim == 1:
        position_matrix = position_ids.view(-1, 1)
    elif position_ids.ndim == 2:
        position_matrix = position_ids
    else:
        raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")

    qkv_out = module.qkv(x)
    if lora is not None:
        qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(module, "tau") and module.tau is not None:
        tok_feat = F.gelu(qkv_out)
        tok_q = torch.tanh(torch.matmul(tok_feat, module.tau["wq"].t())).permute(0, 2, 1)
        tok_v = torch.tanh(torch.matmul(tok_feat, module.tau["wv"].t())).permute(0, 2, 1)

        pos = position_matrix.to(q.dtype) + 1
        pos_log = pos.log().unsqueeze(1)  # (B,1,S)
        alpha = module.tau["alpha"].view(1, -1, 1)
        tau_pos = 1 + (torch.sigmoid(alpha * pos_log) - 0.5)  # (B,H,S)

        tau_q = (tok_q + tau_pos).unsqueeze(-1)
        tau_v = (tok_v + tau_pos).unsqueeze(-1)
        q = q * tau_q
        v = v * tau_v

    q = apply_rotary_emb(q.to(torch.float32), freqs_cis, position_ids, n_heads).to(q.dtype)
    k = apply_rotary_emb(k.to(torch.float32), freqs_cis, position_ids, n_kv_heads).to(k.dtype)

    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)

    if flex_block_mask_slice is not None:
        torch._assert(n_heads == n_kv_heads, "Grouped query attention not supported")
        out = flex_attention(q, k, v, block_mask=flex_block_mask_slice)
    else:
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
        )

    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out0 = module.proj(out)
    if lora is not None:
        out1 = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
        out = out0 + out1
    else:
        out = out0
    return out


def text_decoder(
    x: torch.Tensor,
    module: nn.Module,
    attn_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    config: TextConfig,
    lora: Optional[dict] = None,
    flex_block_mask_slice=None,
) -> torch.Tensor:
    for i, block in enumerate(module.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            mlp_lora = layer_lora["mlp"]
            attn_lora = layer_lora["attn"]
        else:
            mlp_lora = None
            attn_lora = None

        ln_weights = LayerNormWeights(weight=block.ln.weight, bias=block.ln.bias)
        x_norm = layer_norm(x, ln_weights)
        attn_out = attn(
            x_norm,
            block.attn,
            module.freqs_cis,
            block.kv_cache,
            attn_mask,
            config.n_heads,
            config.n_kv_heads,
            position_ids,
            lora=attn_lora,
            flex_block_mask_slice=flex_block_mask_slice,
        )

        if config.moe is not None and i >= config.moe.start_layer:
            mlp_out = moe_mlp(x_norm, block.mlp, config.moe.experts_per_token)
        else:
            mlp_weights = MLPWeights(
                fc1=LinearWeights(
                    weight=block.mlp["fc1"].weight, bias=block.mlp["fc1"].bias
                ),
                fc2=LinearWeights(
                    weight=block.mlp["fc2"].weight, bias=block.mlp["fc2"].bias
                ),
            )
            mlp_out = mlp(x_norm, mlp_weights, lora=mlp_lora)

        x = x + attn_out + mlp_out

    return x


def lm_head(hidden: torch.Tensor, module: nn.Module, indices: Optional[torch.Tensor] = None):
    hidden_last = hidden[:, -1, :]
    post_ln = LayerNormWeights(weight=module.post_ln.weight, bias=module.post_ln.bias)
    hidden_norm = layer_norm(hidden_last, post_ln)
    if indices is not None:
        weights = module.lm_head.weight[indices]
        bias = module.lm_head.bias[indices]
        logits = F.linear(hidden_norm, weights, bias)
    else:
        logits = module.lm_head(hidden_norm)
    return logits


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    qkv_dim = int(config.dim * (1 + 2 * config.n_kv_heads / config.n_heads))
    if config.group_size is not None:
        raise NotImplementedError("Quantized linear layers are not supported yet")

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(config.dim, qkv_dim, dtype=dtype),
                                    "proj": nn.Linear(config.dim, config.dim, dtype=dtype),
                                    "tau": nn.ParameterDict(
                                        {
                                            "wq": nn.Parameter(
                                                torch.empty(config.n_heads, qkv_dim, dtype=dtype)
                                            ),
                                            "wv": nn.Parameter(
                                                torch.empty(config.n_heads, qkv_dim, dtype=dtype)
                                            ),
                                            "alpha": nn.Parameter(
                                                torch.empty(config.n_heads, dtype=dtype)
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "mlp": (
                                build_moe_mlp(
                                    config.dim,
                                    config.moe.expert_inner_dim,
                                    config.moe.num_experts,
                                    dtype,
                                )
                                if config.moe is not None
                                and i >= config.moe.start_layer
                                else build_dense_mlp(config.dim, config.ff_dim, dtype)
                            ),
                        }
                    )
                    for i in range(config.n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.dim, dtype=dtype),
            "lm_head": nn.Linear(config.dim, config.vocab_size, dtype=dtype),
        }
    )

    text.wte = nn.Parameter(torch.empty(config.vocab_size, config.dim, dtype=dtype))
    text.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(config.dim // (2 * config.n_heads), config.max_context),
        persistent=False,
    )

    for block in text["blocks"]:
        block.kv_cache = None

    return text


__all__ = [
    "text_encoder",
    "text_decoder",
    "lm_head",
    "attn",
    "build_text_model",
]
