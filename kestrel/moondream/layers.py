"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal

from ..dense_lora import (
    DenseLoRATorchMLPScratch,
    apply_dense_lora,
    prepare_dense_lora_batch,
)
from ..ops.layernorm_cuda import layernorm_bias
from kestrel_kernels import get_runtime

_KERNELS = get_runtime()

# Re-export LoRA for convenience
from .lora import LoRA, MoEMLPLoRA, DenseMLPLoRA  # noqa: F401
from .lora_workspace import DenseLoRALayerWorkspace, MoELoRALayerWorkspace


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def layer_norm(x: torch.Tensor, w: LayerNormWeights) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16:
        try:
            return layernorm_bias(x, w.weight, w.bias)
        except Exception:
            pass
    return F.layer_norm(x, w.bias.shape, w.weight, w.bias)


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return _KERNELS.linear.linear(x, w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


def mlp(
    x: torch.Tensor,
    w: MLPWeights,
    *,
    lora_workspace: DenseLoRALayerWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
    lora_scratch: DenseLoRATorchMLPScratch | None = None,
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

    prepared_batch = None
    if use_lora:
        prepared_batch = prepare_dense_lora_batch(
            lora_slot_ids,
            segment_len=T,
        )

    h = linear(x, w.fc1)
    if use_lora:
        x_flat = x.view(-1, C)
        lora_delta = torch.zeros_like(h.view(-1, h.shape[-1]))
        assert prepared_batch is not None
        apply_dense_lora(
            x_flat,
            lora_delta,
            lora_workspace.up_a,
            lora_workspace.up_b,
            prepared_batch=prepared_batch,
            scratch=lora_scratch.up if lora_scratch is not None else None,
        )
        h = h + lora_delta.view_as(h)

    h = gelu_approx(h)
    out = linear(h, w.fc2)
    if use_lora:
        h_flat = h.view(-1, h.shape[-1])
        lora_delta = torch.zeros_like(out.view(-1, out.shape[-1]))
        assert prepared_batch is not None
        apply_dense_lora(
            h_flat,
            lora_delta,
            lora_workspace.down_a,
            lora_workspace.down_b,
            prepared_batch=prepared_batch,
            scratch=lora_scratch.down if lora_scratch is not None else None,
        )
        out = out + lora_delta.view_as(out)

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
    from ..fused_moe import ExpertWeights, FusedMoEModule

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
    lora_workspace: MoELoRALayerWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
    single_lora_id: int | None = None,
) -> torch.Tensor:
    B, T, C = x.shape
    x_flat = x.view(-1, C)

    router = mlp_module["router"]
    fused_mlp = mlp_module["mlp"]

    router_logits = router(x_flat)
    topk_weights, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1)
    topk_idxs = topk_idxs.to(torch.int32)

    # Expand slot IDs for all tokens if we have a sequence length > 1
    expanded_slot_ids = None
    if lora_workspace is not None and lora_slot_ids is not None:
        expanded_slot_ids = lora_slot_ids.repeat_interleave(T)

    mlp_out = fused_mlp(
        x_flat,
        topk_weights,
        topk_idxs,
        lora_workspace,
        expanded_slot_ids,
        single_lora_id,
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
