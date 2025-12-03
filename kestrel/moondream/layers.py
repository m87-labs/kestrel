"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal

from ..fused_moe import ExpertWeights, FusedMoEModule

# Re-export LoRA for convenience
from .lora import LoRA, MoEMLPLoRA, DenseMLPLoRA  # noqa: F401


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


def mlp(x: torch.Tensor, w: MLPWeights, lora: DenseMLPLoRA | None = None) -> torch.Tensor:
    h = linear(x, w.fc1)
    if lora is not None:
        h = h + F.linear(F.linear(x, lora.fc1_a), lora.fc1_b) * lora.scale
    h = gelu_approx(h)
    out = linear(h, w.fc2)
    if lora is not None:
        out = out + F.linear(F.linear(h, lora.fc2_a), lora.fc2_b) * lora.scale
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
    "moe_mlp",
    "build_dense_mlp",
    "build_moe_mlp",
    "LoRA",
]
