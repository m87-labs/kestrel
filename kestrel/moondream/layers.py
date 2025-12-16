"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal

from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

from ..fused_moe import ExpertWeights, FusedMoEModule
from ..fused_moe.lora_kernels import apply_moe_lora

# Re-export LoRA for convenience
from .lora import LoRA, MoEMLPLoRA, DenseMLPLoRA  # noqa: F401
from .lora_workspace import DenseLoRALayerWorkspace


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def layer_norm(x: torch.Tensor, w: LayerNormWeights) -> torch.Tensor:
    return F.layer_norm(x, w.bias.shape, w.weight, w.bias)


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return F.linear(x, w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


def apply_dense_lora(
    x: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
) -> None:
    """Apply mixed-slot dense LoRA by treating slots as MoE experts.

    This reuses the fused MoE LoRA kernel by treating adapter slots as "experts"
    with top_k=1. Each token selects exactly one slot (its sequence's adapter),
    and the kernel computes: output += x @ A.T @ B.T for that slot's weights.

    Slot 0 is always zeros in the workspace, so tokens with lora_slot_ids=0
    contribute zero delta (no LoRA applied).

    Args:
        x: Input activations, shape [num_tokens, hidden_dim].
        output: Output tensor to accumulate into, shape [num_tokens, out_dim].
        lora_a: LoRA A weights, shape [max_slots, rank, hidden_dim].
        lora_b: LoRA B weights, shape [max_slots, out_dim, rank].
        lora_slot_ids: Per-token slot indices, shape [num_tokens].
    """
    num_tokens = x.shape[0]
    max_slots = lora_a.shape[0]

    # Shape as [num_tokens, 1] for top_k=1 routing
    topk_ids = lora_slot_ids.view(-1, 1).to(torch.int32)
    topk_weights = torch.ones(num_tokens, 1, dtype=x.dtype, device=x.device)

    # Route tokens by slot ID (treating slots as experts)
    block_size_m = 16
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, max_slots
    )

    # Output needs shape [num_tokens, top_k, out_dim] for apply_moe_lora
    out_dim = output.shape[-1]
    output_3d = output.view(num_tokens, 1, out_dim)

    config = {"BLOCK_SIZE_M": block_size_m}
    apply_moe_lora(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output_3d,
        lora_a=lora_a,
        lora_b=lora_b,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=1,
        config=config,
        mul_routed_weight=False,
    )


def mlp(
    x: torch.Tensor,
    w: MLPWeights,
    *,
    lora_workspace: DenseLoRALayerWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense MLP with optional mixed-slot LoRA.

    Args:
        x: Input tensor, shape [batch, seq_len, dim].
        w: MLP weights (fc1, fc2).
        lora_workspace: Multi-slot LoRA workspace for this layer, or None.
        lora_slot_ids: Per-sequence slot indices, shape [batch], or None.
    """
    B, T, C = x.shape
    use_lora = lora_workspace is not None and lora_slot_ids is not None

    h = linear(x, w.fc1)
    if use_lora:
        # Flatten for LoRA kernel: [batch * seq_len, dim]
        x_flat = x.view(-1, C)
        h_flat = h.view(-1, h.shape[-1])
        # Expand slot IDs for all tokens in each sequence
        slot_ids_expanded = lora_slot_ids.repeat_interleave(T)
        apply_dense_lora(
            x_flat, h_flat, lora_workspace.up_a, lora_workspace.up_b, slot_ids_expanded
        )

    h = gelu_approx(h)
    out = linear(h, w.fc2)
    if use_lora:
        h_flat = h.view(-1, h.shape[-1])
        out_flat = out.view(-1, out.shape[-1])
        apply_dense_lora(
            h_flat, out_flat, lora_workspace.down_a, lora_workspace.down_b, slot_ids_expanded
        )

    return out


def build_dense_mlp(d_model: int, d_ffn: int, dtype: torch.dtype) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "fc1": nn.Linear(d_model, d_ffn, dtype=dtype),
            "fc2": nn.Linear(d_ffn, d_model, dtype=dtype),
        }
    )


def build_moe_mlp(
    d_model: int, d_ffn: int, n_experts: int, dtype: torch.dtype, *, top_k: int
) -> nn.ModuleDict:
    router = nn.Linear(d_model, n_experts, dtype=dtype)
    up_experts = ExpertWeights(n_experts, d_model, d_ffn * 2, dtype=dtype)
    down_experts = ExpertWeights(n_experts, d_ffn, d_model, dtype=dtype)
    fused = FusedMoEModule(
        up_experts,
        down_experts,
        top_k=top_k,
        hidden_size=d_ffn,
        input_size=d_model,
        num_experts=n_experts,
    )
    return nn.ModuleDict({"router": router, "mlp": fused})


def moe_mlp(
    x: torch.Tensor,
    mlp_module: nn.Module,
    experts_per_token: int,
    *,
    mode: Literal["prefill", "decode"] = "decode",
    lora: MoEMLPLoRA | None = None,
) -> torch.Tensor:
    B, T, C = x.shape
    x_flat = x.reshape(-1, C)

    router = mlp_module["router"]
    fused_mlp = mlp_module["mlp"]

    router_logits = router(x_flat)
    topk_logits, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)

    mlp_out = fused_mlp(
        x_flat,
        topk_weights,
        topk_idxs,
        lora,
    ).view(B, T, C)
    return mlp_out


__all__ = [
    "LayerNormWeights",
    "LinearWeights",
    "MLPWeights",
    "layer_norm",
    "mlp",
    "apply_dense_lora",
    "moe_mlp",
    "build_dense_mlp",
    "build_moe_mlp",
    "LoRA",
]
