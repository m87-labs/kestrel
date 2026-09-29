"""Inference-only PyTorch DINOv2 model.

The module preserves Hugging Face checkpoint names but intentionally omits
masking, dropout, stochastic depth, checkpointing, training losses, and task
heads. Model dataflow is adapted from Hugging Face Transformers' DINOv2
implementation (Copyright 2023 Meta AI and The HuggingFace Inc. team,
Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import Dinov2Config


@dataclass(frozen=True)
class Dinov2Output:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[1]} channels; expected {self.num_channels}"
            )
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"image dimensions {(height, width)} must be divisible by patch size "
                f"{self.patch_size}"
            )
        hidden_states = self.projection(pixel_values.to(self.projection.weight.dtype))
        return hidden_states.flatten(2).transpose(1, 2)


class Dinov2Embeddings(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.empty(1, config.num_position_embeddings, config.hidden_size)
        )
        self.patch_size = config.patch_size
        # Cache fixed-resolution positions; the parameter version invalidates
        # them after a checkpoint reload. This model supports inference only.
        self._resampled_positions: dict[tuple, torch.Tensor] = {}
        nn.init.trunc_normal_(self.cls_token, std=config.initializer_range)
        nn.init.trunc_normal_(self.position_embeddings, std=config.initializer_range)

    @torch.no_grad()
    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        table = self.position_embeddings
        cache_key = (
            height,
            width,
            table.device,
            table.dtype,
            table.data_ptr(),
            table._version,
        )
        cached = self._resampled_positions.get(cache_key)
        if cached is not None:
            return cached

        base_grid = math.isqrt(num_positions)
        if base_grid * base_grid != num_positions:
            raise ValueError(
                f"position table has {num_positions} patch rows, which is not a square grid"
            )
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        dim = embeddings.shape[-1]

        class_position = self.position_embeddings[:, :1]
        patch_positions = self.position_embeddings[:, 1:].reshape(
            1, base_grid, base_grid, dim
        )
        target_dtype = patch_positions.dtype
        patch_positions = F.interpolate(
            patch_positions.permute(0, 3, 1, 2).to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(target_dtype)
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, -1, dim)
        resampled = torch.cat((class_position, patch_positions), dim=1)
        self._resampled_positions[cache_key] = resampled
        return resampled

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        patch_embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
        return embeddings + self.interpolate_pos_encoding(embeddings, height, width)


class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.head_dim
        self.all_head_size = config.hidden_size
        self.scaling = self.attention_head_size**-0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = hidden_states.shape
        return hidden_states.reshape(
            batch_size,
            tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        attention_probs = F.softmax(attention_weights, dim=-1)
        context = torch.matmul(attention_probs, value).transpose(1, 2).contiguous()
        return context.reshape(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size)


class Dinov2SelfOutput(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class Dinov2Attention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class Dinov2LayerScale(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            torch.full((config.hidden_size,), config.layerscale_value)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="none"))


class Dinov2Layer(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.layer_scale1(
            self.attention(self.norm1(hidden_states))
        )
        return hidden_states + self.layer_scale2(self.mlp(self.norm2(hidden_states)))


class Dinov2Encoder(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            Dinov2Layer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class Dinov2Model(nn.Module):
    """Bare DINOv2 image encoder with the V1 two-tensor output contract."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        config.validate()
        if config.hidden_act != "gelu":
            raise ValueError(f"unsupported hidden_act {config.hidden_act!r}")
        if config.use_swiglu_ffn:
            raise ValueError("SwiGLU DINOv2 checkpoints are outside the inference V1")
        if config.attention_probs_dropout_prob or config.hidden_dropout_prob:
            raise ValueError("dropout-bearing DINOv2 configs are outside the inference V1")
        if config.drop_path_rate:
            raise ValueError("stochastic-depth DINOv2 configs are outside the inference V1")

        self.config = config
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> Dinov2Output:
        if pixel_values is None:
            raise ValueError("pixel_values is required")
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        last_hidden_state = self.layernorm(hidden_states)
        return Dinov2Output(
            last_hidden_state=last_hidden_state,
            pooler_output=last_hidden_state[:, 0, :],
        )


__all__ = [
    "Dinov2Attention",
    "Dinov2Embeddings",
    "Dinov2Encoder",
    "Dinov2Layer",
    "Dinov2LayerScale",
    "Dinov2MLP",
    "Dinov2Model",
    "Dinov2Output",
    "Dinov2PatchEmbeddings",
    "Dinov2SelfAttention",
]
