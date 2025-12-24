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
from ..ops import precompute_freqs_cis
from ..ops.rotary_embedding import rotary_embedding_cuda
from .flashinfer import (
    FlashInferBatchMetadata,
    FlashInferDecodeContext,
)
from kestrel_kernels.flash_attn.cute.interface import _flash_attn_fwd
from kestrel_kernels.flash_attn.cute.mask_definitions import cute_prefix_lm_mask_730
from kestrel_kernels.tau_tail_ops import tau_tail_apply_into


# Avoid expensive runtime hashing in kestrel-kernels' compile key generation.
# TODO: This should become dynamic once prefix length is no longer fixed.
cute_prefix_lm_mask_730.__cute_hash__ = "kestrel_kernels.flash_attn.cute.mask_definitions.cute_prefix_lm_mask_730"


def text_encoder(input_ids: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return F.embedding(input_ids, module.wte)


def build_tau_pos_tables(
    text_module: nn.Module,
    n_heads: int,
    max_context: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """Build tau position tables for all attention blocks.

    Call this after loading/tying weights to precompute the position-dependent
    scaling factors. This avoids small elementwise kernels on the forward path
    and enables CUDA graph compatibility.
    """
    pos = torch.arange(max_context, device=device, dtype=torch.float32) + 1
    pos_log = pos.log().unsqueeze(-1)  # (P, 1)

    for block in text_module["blocks"]:
        attn_module = block["attn"]
        tau = getattr(attn_module, "tau", None)
        if tau is None:
            continue
        alpha = tau["alpha"]
        alpha_fp32 = alpha.to(dtype=torch.float32, device=device).view(1, -1)
        tau_pos_table = (torch.sigmoid(alpha_fp32 * pos_log) + 0.5).to(dtype=dtype)
        attn_module._tau_pos_table = tau_pos_table  # type: ignore[attr-defined]


def attn(
    x: torch.Tensor,
    module: nn.Module,
    cos_sin_cache: torch.Tensor,
    kv_cache: Optional[nn.Module],
    attn_mask: Optional[torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    flashinfer_state: Optional[
        tuple[FlashInferDecodeContext, FlashInferBatchMetadata, bool]
    ] = None,
    mode: Literal["prefill", "decode"] = "decode",
    *,
    slot_mapping: torch.Tensor,
    use_prefix_attn: bool = False,
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
    if hasattr(module, "tau") and module.tau is not None:
        tau_wqwv = module.tau["wqwv"]
        tok_qv_lin = F.linear(F.gelu(qkv_out), tau_wqwv)
        # _tau_pos_table is built by build_tau_pos_tables() after weight loading.
        tau_tail_apply_into(
            qkv_out=qkv_out,
            tok_qv_lin=tok_qv_lin,
            tau_pos_table=module._tau_pos_table,  # type: ignore[attr-defined]
            position_ids=position_matrix,
        )

    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim)
    k = k.view(bsz, q_len, n_kv_heads, head_dim)
    v = v.view(bsz, q_len, n_kv_heads, head_dim)

    # Apply in-place rotary embedding (GPT-NeoX style) on bshd tensors.
    # This avoids materializing per-step cos/sin tensors and keeps everything on-GPU.
    rotary_embedding_cuda(position_matrix, q, k, head_dim, cos_sin_cache)

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

    if kv_cache is not None:
        kv_cache.update(position_ids, k, v, slot_mapping=slot_mapping)

    if flash_ctx is not None:
        if kv_cache is None:
            raise RuntimeError("FlashInfer decode requires a KV cache")
        torch._assert(q.shape[1] == 1, "FlashInfer decode expects q_len == 1")
        q_heads = q[:, 0]
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
        out = attn_heads.unsqueeze(1)
    elif mode == "prefill":
        if not x.is_cuda:
            raise RuntimeError("FA3 prefill requires CUDA tensors")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype for FA3 prefill: {q.dtype}")
        mask_mod = None
        causal = True
        pack_gqa = None
        if use_prefix_attn:
            # TODO: Make this dynamic (per-request prefix length) when we support
            # non-fixed image prefixes. For now, hardcode Moondream's BOS+image
            # prefix length (730 = 1 + 27*27).
            causal = False
            mask_mod = cute_prefix_lm_mask_730
            pack_gqa = False
        out, _ = _flash_attn_fwd(
            q,
            k,
            v,
            causal=causal,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
        )
    else:
        raise RuntimeError(
            "SDPA fallback removed; FlashInfer decode is required for non-prefill"
        )

    out = out.view(bsz, q_len, d_model)
    return module.proj(out)


def text_decoder(
    x: torch.Tensor,
    module: nn.Module,
    attn_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    config: TextConfig,
    *,
    slot_mapping: torch.Tensor,
    use_prefix_attn: bool = False,
    flashinfer_ctx: Optional[FlashInferDecodeContext] = None,
    flashinfer_metadata: Optional[FlashInferBatchMetadata] = None,
    use_flashinfer: bool = False,
    use_graph: bool = False,
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

        attn_out = attn(
            x_norm,
            block.attn,
            module.cos_sin_cache,
            block.kv_cache,
            attn_mask,
            config.n_heads,
            config.n_kv_heads,
            position_ids,
            flashinfer_state=flash_state,
            mode=mode,
            slot_mapping=slot_mapping,
            use_prefix_attn=use_prefix_attn,
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
                                            "wqwv": nn.Parameter(
                                                torch.empty(
                                                    config.n_heads * 2,
                                                    qkv_dim,
                                                    dtype=dtype,
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
    cos_sin_cache = precompute_freqs_cis(
        config.dim // (2 * config.n_heads),
        config.max_context,
        dtype=torch.float32,
        device=device,
    )
    text.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    for block in text["blocks"]:
        block.kv_cache = None

    return text


__all__ = [
    "text_encoder",
    "text_decoder",
    "lm_head",
    "attn",
    "build_text_model",
    "build_tau_pos_tables",
]
