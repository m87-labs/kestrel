"""Text transformer implementation for Moondream.

Adapted from the Moondream project (Apache-2.0).
"""


from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .lora_workspace import TextLoRAWorkspace
from ..ops import apply_rotary_emb, precompute_freqs_cis
from .flashinfer import (
    FlashInferBatchMetadata,
    FlashInferDecodeContext,
    FlashInferPrefillBatchMetadata,
    FlashInferPrefillContext,
    run_flashinfer_prefill,
)


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
    flashinfer_state: Optional[
        tuple[FlashInferDecodeContext, FlashInferBatchMetadata, bool]
    ] = None,
    flashinfer_prefill_state: Optional[
        tuple[FlashInferPrefillContext, FlashInferPrefillBatchMetadata]
    ] = None,
    mode: Literal["prefill", "decode"] = "decode",
    *,
    slot_mapping: torch.Tensor,
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

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(module, "tau") and module.tau is not None:
        tok_feat = F.gelu(qkv_out)
        tok_q = torch.tanh(torch.matmul(tok_feat, module.tau["wq"].t())).permute(
            0, 2, 1
        )
        tok_v = torch.tanh(torch.matmul(tok_feat, module.tau["wv"].t())).permute(
            0, 2, 1
        )

        pos = position_matrix.to(q.dtype) + 1
        pos_log = pos.log().unsqueeze(1)  # (B,1,S)
        alpha = module.tau["alpha"].view(1, -1, 1)
        tau_pos = 1 + (torch.sigmoid(alpha * pos_log) - 0.5)  # (B,H,S)

        q.mul_((tok_q + tau_pos).unsqueeze(-1))
        v.mul_((tok_v + tau_pos).unsqueeze(-1))

    cos = freqs_cis[..., 0][position_ids]
    sin = freqs_cis[..., 1][position_ids]
    if cos.ndim == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q, k = apply_rotary_emb(
        q,
        k,
        cos.contiguous(),
        sin.contiguous(),
    )

    flash_ctx = None
    metadata = None
    use_graph = False
    if flashinfer_state is not None:
        if len(flashinfer_state) == 3:
            flash_ctx, metadata, use_graph = flashinfer_state  # type: ignore[misc]
        else:
            raise ValueError(
                f"Unexpected flashinfer_state tuple length {len(flashinfer_state)}"
            )

    prefill_ctx = None
    prefill_metadata = None
    if flashinfer_prefill_state is not None:
        if len(flashinfer_prefill_state) != 2:
            raise ValueError(
                "flashinfer_prefill_state must contain a context and metadata tuple"
            )
        prefill_ctx, prefill_metadata = flashinfer_prefill_state

    if kv_cache is not None:
        kv_result = kv_cache.update(position_ids, k, v, slot_mapping=slot_mapping)
    else:
        kv_result = (k, v)

    if flash_ctx is not None:
        if kv_cache is None:
            raise RuntimeError("FlashInfer decode requires a KV cache")
        torch._assert(q.shape[2] == 1, "FlashInfer decode expects q_len == 1")
        q_heads = q.view(bsz, n_heads, head_dim)
        k_cache = kv_cache.cache.k_cache  # type: ignore[attr-defined]
        v_cache = kv_cache.cache.v_cache  # type: ignore[attr-defined]
        k_scale = getattr(kv_cache.cache, "k_scale", None)  # type: ignore[attr-defined]
        v_scale = getattr(kv_cache.cache, "v_scale", None)  # type: ignore[attr-defined]
        attn_heads = flash_ctx.run(
            q_heads,
            (k_cache, v_cache),
            use_graph=use_graph,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out = attn_heads.unsqueeze(2)
    elif prefill_ctx is not None:
        if kv_cache is None:
            raise RuntimeError("FlashInfer prefill requires a KV cache")
        if prefill_metadata is None:
            raise RuntimeError("FlashInfer prefill requires metadata")
        q_heads = q.transpose(1, 2).reshape(-1, n_heads, head_dim).contiguous()
        paged_cache = kv_cache.cache  # type: ignore[attr-defined]
        k_scale = getattr(paged_cache, "k_scale", None)
        v_scale = getattr(paged_cache, "v_scale", None)
        out_buf = torch.empty_like(q_heads)
        attn_heads = run_flashinfer_prefill(
            prefill_ctx,
            q_heads,
            (paged_cache.k_cache, paged_cache.v_cache),
            k_scale=k_scale,
            v_scale=v_scale,
            out=out_buf,
        )
        out = (
            attn_heads.view(bsz, q_len, n_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
    else:
        k_attn, v_attn = kv_result
        out = F.scaled_dot_product_attention(
            q,
            k_attn,
            v_attn,
            attn_mask=attn_mask,
            enable_gqa=n_heads != n_kv_heads,
            is_causal=(mode == "prefill") and attn_mask is None,
        )

    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    return module.proj(out)


def text_decoder(
    x: torch.Tensor,
    module: nn.Module,
    attn_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    config: TextConfig,
    *,
    slot_mapping: torch.Tensor,
    flashinfer_ctx: Optional[FlashInferDecodeContext] = None,
    flashinfer_metadata: Optional[FlashInferBatchMetadata] = None,
    use_flashinfer: bool = False,
    use_graph: bool = False,
    flashinfer_prefill_ctx: Optional[FlashInferPrefillContext] = None,
    flashinfer_prefill_metadata: Optional[FlashInferPrefillBatchMetadata] = None,
    use_flashinfer_prefill: bool = False,
    mode: Literal["prefill", "decode"] = "decode",
    lora_workspace: TextLoRAWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    for i, block in enumerate(module.blocks):
        ln_weights = LayerNormWeights(weight=block.ln.weight, bias=block.ln.bias)
        x_norm = layer_norm(x, ln_weights)

        flash_state = None
        if (
            use_flashinfer
            and flashinfer_ctx is not None
            and flashinfer_metadata is not None
        ):
            flash_state = (flashinfer_ctx, flashinfer_metadata, use_graph)

        prefill_state = None
        if (
            mode == "prefill"
            and use_flashinfer_prefill
            and flashinfer_prefill_ctx is not None
            and flashinfer_prefill_metadata is not None
        ):
            prefill_state = (flashinfer_prefill_ctx, flashinfer_prefill_metadata)

        attn_out = attn(
            x_norm,
            block.attn,
            module.freqs_cis,
            block.kv_cache,
            attn_mask,
            config.n_heads,
            config.n_kv_heads,
            position_ids,
            flashinfer_state=flash_state,
            flashinfer_prefill_state=prefill_state,
            mode=mode,
            slot_mapping=slot_mapping,
        )

        if config.moe is not None and i >= config.moe.start_layer:
            moe_workspace = lora_workspace.moe_layer(i) if lora_workspace else None
            mlp_out = moe_mlp(
                x_norm,
                block.mlp,
                config.moe.experts_per_token,
                mode=mode,
                lora_workspace=moe_workspace,
                lora_slot_ids=lora_slot_ids,
            )
        else:
            mlp_weights = MLPWeights(
                fc1=LinearWeights(
                    weight=block.mlp["fc1"].weight, bias=block.mlp["fc1"].bias
                ),
                fc2=LinearWeights(
                    weight=block.mlp["fc2"].weight, bias=block.mlp["fc2"].bias
                ),
            )
            dense_workspace = lora_workspace.dense_layer(i) if lora_workspace else None
            mlp_out = mlp(
                x_norm,
                mlp_weights,
                lora_workspace=dense_workspace,
                lora_slot_ids=lora_slot_ids,
            )

        x = x + attn_out + mlp_out

    return x


def lm_head(
    hidden: torch.Tensor, module: nn.Module, indices: Optional[torch.Tensor] = None
):
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


def build_text_model(
    config: TextConfig, dtype: torch.dtype, *, device: torch.device | str | None = None
) -> nn.Module:
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
                                    "proj": nn.Linear(
                                        config.dim, config.dim, dtype=dtype
                                    ),
                                    "tau": nn.ParameterDict(
                                        {
                                            "wq": nn.Parameter(
                                                torch.empty(
                                                    config.n_heads, qkv_dim, dtype=dtype
                                                )
                                            ),
                                            "wv": nn.Parameter(
                                                torch.empty(
                                                    config.n_heads, qkv_dim, dtype=dtype
                                                )
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
                                    top_k=config.moe.experts_per_token,
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
        precompute_freqs_cis(
            config.dim // (2 * config.n_heads), config.max_context, device=device
        ),
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
