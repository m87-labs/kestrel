"""LoRA (Low-Rank Adaptation) modules for Moondream inference.

This module provides LoRA adapters for the Moondream model, covering:
- Vision projection MLP (optional)
- Text model MLPs (dense layers before MoE, and MoE expert layers)

For text models with MoE (Mixture of Experts), the LoRA rank is distributed
across active experts: each expert receives rank = total_rank / experts_per_token.
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextConfig, VisionConfig


# -----------------------------------------------------------------------------
# Vision LoRA
# -----------------------------------------------------------------------------


class VisionLoRALinear(nn.Module):
    """LoRA adapter for a single linear layer in the vision model."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Parameter(torch.empty(rank, in_features, dtype=dtype))
        self.up = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA output to be added to the base layer output."""
        scale = self.alpha / self.rank
        return F.linear(F.linear(x, self.down), self.up) * scale


class VisionLoRA(nn.Module):
    """LoRA adapter for the vision projection MLP.

    Currently supports LoRA on the proj_mlp.fc2 layer, which projects
    vision features to the text embedding space.
    """

    def __init__(
        self,
        vision_config: VisionConfig,
        rank: int,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.config = vision_config
        self._rank = rank
        self._alpha = alpha

        # proj_mlp.fc2: [proj_inner_dim] -> [proj_out_dim]
        self.proj_fc2 = VisionLoRALinear(
            in_features=vision_config.proj_inner_dim,
            out_features=vision_config.proj_out_dim,
            rank=rank,
            alpha=alpha,
            dtype=dtype,
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def alpha(self) -> float:
        return self._alpha


# -----------------------------------------------------------------------------
# Text LoRA
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TextLoRAConfig:
    """Configuration for text model LoRA adapters.

    Attributes:
        rank: Total LoRA rank. For MoE layers, this is divided by
            experts_per_token to get the per-expert rank.
        alpha: Scaling factor applied to LoRA output (output *= alpha / rank).
    """

    rank: int
    alpha: float = 1.0


class DenseMLPLoRA(nn.Module):
    """LoRA adapter for a dense (non-MoE) MLP layer.

    Covers both up-projection and down-projection.

    Naming convention:
        - up == base MLP fc1
        - down == base MLP fc2
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        rank: int,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._rank = rank
        self._alpha = alpha

        # up: [d_model] -> [d_ffn]
        self.up_a = nn.Parameter(torch.empty(rank, d_model, dtype=dtype))
        self.up_b = nn.Parameter(torch.zeros(d_ffn, rank, dtype=dtype))

        # down: [d_ffn] -> [d_model]
        self.down_a = nn.Parameter(torch.empty(rank, d_ffn, dtype=dtype))
        self.down_b = nn.Parameter(torch.zeros(d_model, rank, dtype=dtype))

        nn.init.kaiming_uniform_(self.up_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_a, a=math.sqrt(5))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def scale(self) -> float:
        return self._alpha / self._rank


class MoEMLPLoRA(nn.Module):
    """LoRA adapter for a Mixture-of-Experts MLP layer.

    Each expert has its own LoRA weights. The up-projection covers both
    gate and up weights (fused as d_expert * 2).
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        num_experts: int,
        rank_per_expert: int,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._rank_per_expert = rank_per_expert
        self._alpha = alpha
        self._num_experts = num_experts

        # up projection: [d_model] -> [d_expert * 2] (gate + up fused)
        self.up_a = nn.Parameter(
            torch.empty(num_experts, rank_per_expert, d_model, dtype=dtype)
        )
        self.up_b = nn.Parameter(
            torch.zeros(num_experts, d_expert * 2, rank_per_expert, dtype=dtype)
        )

        # down projection: [d_expert] -> [d_model]
        self.down_a = nn.Parameter(
            torch.empty(num_experts, rank_per_expert, d_expert, dtype=dtype)
        )
        self.down_b = nn.Parameter(
            torch.zeros(num_experts, d_model, rank_per_expert, dtype=dtype)
        )

        # Initialize A matrices with kaiming, B matrices stay zero
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.up_a[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.down_a[i], a=math.sqrt(5))

    @property
    def rank_per_expert(self) -> int:
        return self._rank_per_expert

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def scale(self) -> float:
        return self._alpha / self._rank_per_expert


class TextLoRA(nn.Module):
    """LoRA adapter covering all MLP layers in the text model.

    For models with MoE, layers before `start_layer` use dense MLPs,
    and layers from `start_layer` onward use MoE. Each layer type
    has its own LoRA structure.
    """

    def __init__(
        self,
        text_config: TextConfig,
        lora_config: TextLoRAConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._lora_config = lora_config

        moe_cfg = text_config.moe
        start_layer = moe_cfg.start_layer if moe_cfg else text_config.n_layers

        # Dense MLP layers: [0, start_layer)
        self.dense = nn.ModuleList(
            [
                DenseMLPLoRA(
                    d_model=text_config.dim,
                    d_ffn=text_config.ff_dim,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                    dtype=dtype,
                )
                for _ in range(start_layer)
            ]
        )

        # MoE layers: [start_layer, n_layers)
        if moe_cfg is not None:
            rank_per_expert = lora_config.rank // moe_cfg.experts_per_token
            if rank_per_expert < 1:
                raise ValueError(
                    f"rank ({lora_config.rank}) must be >= experts_per_token "
                    f"({moe_cfg.experts_per_token})"
                )
            self.moe = nn.ModuleList(
                [
                    MoEMLPLoRA(
                        d_model=text_config.dim,
                        d_expert=moe_cfg.expert_inner_dim,
                        num_experts=moe_cfg.num_experts,
                        rank_per_expert=rank_per_expert,
                        alpha=lora_config.alpha,
                        dtype=dtype,
                    )
                    for _ in range(text_config.n_layers - start_layer)
                ]
            )
            self._rank_per_expert = rank_per_expert
        else:
            self.moe = nn.ModuleList()
            self._rank_per_expert = 0

        self._start_layer = start_layer

    @property
    def lora_config(self) -> TextLoRAConfig:
        return self._lora_config

    @property
    def rank(self) -> int:
        return self._lora_config.rank

    @property
    def alpha(self) -> float:
        return self._lora_config.alpha

    @property
    def rank_per_expert(self) -> int:
        return self._rank_per_expert

    @property
    def start_layer(self) -> int:
        return self._start_layer

    def get_dense_lora(self, layer_idx: int) -> Optional[DenseMLPLoRA]:
        """Get LoRA for a dense layer, or None if layer is MoE."""
        if layer_idx < len(self.dense):
            return self.dense[layer_idx]
        return None

    def get_moe_lora(self, layer_idx: int) -> Optional[MoEMLPLoRA]:
        """Get LoRA for an MoE layer, or None if layer is dense."""
        moe_idx = layer_idx - self._start_layer
        if 0 <= moe_idx < len(self.moe):
            return self.moe[moe_idx]
        return None


# -----------------------------------------------------------------------------
# Unified LoRA Container
# -----------------------------------------------------------------------------


class LoRA(nn.Module):
    """Unified LoRA adapter containing text and optional vision components.

    This is the top-level container passed through the inference pipeline.
    Text LoRA is always required; vision LoRA is optional.
    """

    def __init__(
        self,
        text: TextLoRA,
        vision: Optional[VisionLoRA] = None,
    ) -> None:
        super().__init__()
        self._text = text
        self._vision = vision

        self.add_module("text_lora", text)
        if vision is not None:
            self.add_module("vision_lora", vision)

    @property
    def text(self) -> TextLoRA:
        return self._text

    @property
    def vision(self) -> Optional[VisionLoRA]:
        return self._vision

    @classmethod
    def create(
        cls,
        text_config: TextConfig,
        lora_config: TextLoRAConfig,
        vision_config: Optional[VisionConfig] = None,
        vision_rank: Optional[int] = None,
        vision_alpha: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LoRA":
        """Create a LoRA adapter.

        Args:
            text_config: Text model configuration.
            lora_config: LoRA configuration for text layers.
            vision_config: Vision model configuration (required if vision_rank is set).
            vision_rank: LoRA rank for vision projection. If None, no vision LoRA.
            vision_alpha: Scaling factor for vision LoRA.
            dtype: Data type for LoRA parameters.

        Returns:
            LoRA adapter with text (and optionally vision) components.
        """
        text_lora = TextLoRA(
            text_config=text_config,
            lora_config=lora_config,
            dtype=dtype,
        )

        vision_lora = None
        if vision_rank is not None:
            if vision_config is None:
                raise ValueError("vision_config required when vision_rank is set")
            vision_lora = VisionLoRA(
                vision_config=vision_config,
                rank=vision_rank,
                alpha=vision_alpha,
                dtype=dtype,
            )

        return cls(text=text_lora, vision=vision_lora)


class AdapterProvider(Protocol):
    """Provider interface for loading LoRA adapters by id.

    Phase 1 notes:
    - `config()` is temporary and used for eager CUDA graph capture.
    - `default_adapter()` is used for engine warmup.
    - `get()` must return CUDA tensors compatible with the runtime.
    """

    def config(self) -> TextLoRAConfig:
        """Return the (rank, alpha) contract for all adapters."""

    def default_adapter(self) -> str:
        """Return the adapter id used for warmup and eager init paths."""

    def get(self, adapter: str) -> "LoRA":
        """Return the LoRA module for this adapter id."""


__all__ = [
    # Vision LoRA
    "VisionLoRALinear",
    "VisionLoRA",
    # Text LoRA
    "TextLoRAConfig",
    "DenseMLPLoRA",
    "MoEMLPLoRA",
    "TextLoRA",
    # Unified
    "LoRA",
    # Provider
    "AdapterProvider",
]
