"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Literal, Mapping

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


def mlp(x: torch.Tensor, w: MLPWeights) -> torch.Tensor:
    x = linear(x, w.fc1)
    x = gelu_approx(x)
    return linear(x, w.fc2)


@dataclass(frozen=True)
class LoRALinear:
    """LoRA weights for a single linear projection."""

    down: torch.Tensor
    up: torch.Tensor
    alpha: float = 1.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        rank = int(self.down.shape[0])
        scale = self.alpha / float(rank)
        down_proj = F.linear(x, self.down)
        return F.linear(down_proj, self.up) * scale


@dataclass(frozen=True)
class LoRA:
    """Container for optional LoRA adapters."""

    vision: Mapping[str, LoRALinear] = field(default_factory=dict)

    @classmethod
    def for_vision(cls, module: nn.Module, rank: int) -> "LoRA":
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        proj_mlp = module.proj_mlp
        fc2 = proj_mlp["fc2"]
        in_dim = fc2.in_features
        out_dim = fc2.out_features
        device = fc2.weight.device
        dtype = fc2.weight.dtype

        down = torch.zeros(rank, in_dim, device=device, dtype=dtype)
        std = 1.0 / float(rank)
        up = torch.randn(out_dim, rank, device=device, dtype=dtype) * std
        adapter = LoRALinear(down=down, up=up, alpha=float(rank))
        return cls(vision={"proj_mlp.fc2": adapter})


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

    num_tokens = topk_idxs.shape[0]

    if mode == "decode":
        scatter_mlp.ensure_workspaces(num_tokens, x_flat.device, x_flat.dtype)

    mlp_out = scatter_mlp(x_flat, topk_weights, topk_idxs).view(B, T, C)
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
    "LoRALinear",
]
