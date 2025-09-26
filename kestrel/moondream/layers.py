"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal, Optional

from ..scattermoe import MLP as ScatterMoEMLP


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


def mlp(x: torch.Tensor, w: MLPWeights, lora: Optional[dict] = None) -> torch.Tensor:
    x0 = linear(x, w.fc1)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc1"]["A"]), lora["fc1"]["B"])
        x = x0 + x1
    else:
        x = x0

    x = gelu_approx(x)

    x0 = linear(x, w.fc2)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc2"]["A"]), lora["fc2"]["B"])
        x = x0 + x1
    else:
        x = x0

    return x


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
    return nn.ModuleDict(
        {
            "router": nn.Linear(d_model, n_experts, dtype=dtype),
            "mlp": ScatterMoEMLP(
                input_size=d_model,
                hidden_size=d_ffn,
                num_experts=n_experts,
                top_k=top_k,
                dtype=dtype,
            ),
        }
    )


def moe_mlp(
    x: torch.Tensor,
    mlp_module: nn.Module,
    experts_per_token: int,
    *,
    mode: Literal["prefill", "decode"] = "decode",
) -> torch.Tensor:
    B, T, C = x.shape
    x_flat = x.reshape(-1, C)

    router = mlp_module["router"]
    scatter_mlp = mlp_module["mlp"]

    router_logits = router(x_flat)
    topk_logits, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)

    num_tokens, top_k = topk_idxs.shape

    if mode == "prefill":
        mlp_out = scatter_mlp(x_flat, topk_weights, topk_idxs).view(B, T, C)
        return mlp_out

    if T == 1:
        w1_weight = scatter_mlp.experts.weight
        w2_weight = scatter_mlp.output_experts.weight

        flat_idxs = topk_idxs.view(-1)
        flat_weights = topk_weights.view(-1)

        w1_selected = w1_weight[flat_idxs]
        w2_selected = w2_weight[flat_idxs]

        x_expanded = x_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, C)

        x1_full = torch.bmm(w1_selected, x_expanded.unsqueeze(-1)).squeeze(-1)
        x1, g = x1_full.chunk(2, dim=-1)
        x1 = F.gelu(x1) * (g + 1)

        expert_outs = torch.bmm(w2_selected, x1.unsqueeze(-1)).squeeze(-1)

        weighted_outs = expert_outs * flat_weights.unsqueeze(-1)
        weighted_outs = weighted_outs.view(num_tokens, top_k, C)

        mlp_out = weighted_outs.sum(dim=1)
        mlp_out = mlp_out.view(B, T, C)
        return mlp_out

    out = x_flat.new_zeros(x_flat.size())
    experts = scatter_mlp.experts.weight
    outputs = scatter_mlp.output_experts.weight

    for expert_id in range(experts.shape[0]):
        token_pos, which_k = (topk_idxs == expert_id).nonzero(as_tuple=True)
        if token_pos.numel() == 0:
            continue

        gate = topk_weights[token_pos, which_k]
        x_sel = x_flat[token_pos]

        w1 = experts[expert_id]
        x1_full = torch.matmul(x_sel, w1.transpose(0, 1))
        x1, g = x1_full.chunk(2, dim=-1)
        x1 = F.gelu(x1) * (g + 1)

        w2 = outputs[expert_id]
        expert_out = torch.matmul(x1, w2.transpose(0, 1)) * gate.unsqueeze(-1)
        out[token_pos] += expert_out

    return out.view(B, T, C)


__all__ = [
    "LayerNormWeights",
    "LinearWeights",
    "MLPWeights",
    "layer_norm",
    "mlp",
    "moe_mlp",
    "build_dense_mlp",
    "build_moe_mlp",
]
