"""Small PyTorch reference model for Qwen3-TTS generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import Qwen3TTSCodePredictorConfig, Qwen3TTSConfig, Qwen3TTSTalkerConfig

# Inference equations and parameter names follow the Apache-2.0 Qwen3-TTS
# reference implementation and Nari Labs' compact serving implementation.

KeyValue = tuple[torch.Tensor, torch.Tensor]
KeyValueCache = tuple[KeyValue, ...]


@dataclass(frozen=True, slots=True)
class TalkerOutput:
    last_hidden_state: torch.Tensor
    logits: torch.Tensor
    past_key_values: KeyValueCache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (values * self.weight.float()).to(dtype)


TransformerConfig = Qwen3TTSTalkerConfig | Qwen3TTSCodePredictorConfig


class GatedMLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary_factors(
    values: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float,
    mrope_section: tuple[int, ...] | None,
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = values.shape[-1]
    frequencies = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim)
    )

    if position_ids.ndim == 2:
        angles = position_ids.float().unsqueeze(-1) * frequencies
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
        return cos.unsqueeze(1).to(values.dtype), sin.unsqueeze(1).to(values.dtype)

    angles = position_ids.float().unsqueeze(-1) * frequencies
    cos_by_axis = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin_by_axis = torch.cat((angles.sin(), angles.sin()), dim=-1)
    if mrope_section is None:
        cos, sin = cos_by_axis[0], sin_by_axis[0]
    elif mrope_interleaved:
        modality_count = len(mrope_section)

        def interleave(parts: torch.Tensor) -> torch.Tensor:
            combined = parts[0, ..., : head_dim // 2].clone()
            for axis, section in enumerate(mrope_section[1:], 1):
                combined[..., axis : section * modality_count : modality_count] = parts[
                    axis, ..., axis : section * modality_count : modality_count
                ]
            return torch.cat((combined, combined), dim=-1)

        cos, sin = interleave(cos_by_axis), interleave(sin_by_axis)
    else:
        split_sizes = mrope_section * 2
        cos = torch.cat(
            [part[index % len(mrope_section)] for index, part in enumerate(cos_by_axis.split(split_sizes, -1))],
            dim=-1,
        )
        sin = torch.cat(
            [part[index % len(mrope_section)] for index, part in enumerate(sin_by_axis.split(split_sizes, -1))],
            dim=-1,
        )
    return cos.unsqueeze(1).to(values.dtype), sin.unsqueeze(1).to(values.dtype)


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, *, multimodal_rope: bool) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.rope_theta = float(config.rope_theta)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        sections = rope_scaling.get("mrope_section") if multimodal_rope else None
        self.mrope_section = tuple(int(value) for value in sections) if sections else None
        self.mrope_interleaved = bool(rope_scaling.get("interleaved", False))
        bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)
        eps = float(config.rms_norm_eps)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, KeyValue]:
        batch_size, token_count, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch_size, token_count, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, token_count, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, token_count, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = _rotary_factors(
            query,
            position_ids,
            self.rope_theta,
            self.mrope_section,
            self.mrope_interleaved,
        )
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        from kestrel.ops.attention import dense_attention

        attended = dense_attention(
            query,
            key,
            value,
            scaling=self.head_dim**-0.5,
            causal=True,
            cu_seqlens=cu_seq_lens,
        ).reshape(batch_size, token_count, -1)
        return self.o_proj(attended), (key, value)


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, *, multimodal_rope: bool) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(config, multimodal_rope=multimodal_rope)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, KeyValue]:
        attended, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            cu_seq_lens,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class _Backbone(nn.Module):
    def __init__(self, config: TransformerConfig, *, multimodal_rope: bool) -> None:
        super().__init__()
        self.multimodal_rope = multimodal_rope
        self.layers = nn.ModuleList(
            DecoderLayer(config, multimodal_rope=multimodal_rope)
            for _ in range(int(config.num_hidden_layers))
        )
        self.norm = RMSNorm(int(config.hidden_size), float(config.rms_norm_eps))

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, KeyValueCache]:
        positions = position_ids.to(device=input_embeds.device, dtype=torch.long)
        if self.multimodal_rope:
            if positions.ndim == 2:
                positions = positions.unsqueeze(0).expand(3, -1, -1)
            elif positions.shape[0] == 4:
                positions = positions[1:]
        hidden_states = input_embeds
        present: list[KeyValue] = []
        for layer in self.layers:
            hidden_states, layer_present = layer(
                hidden_states,
                positions,
                cu_seq_lens,
            )
            present.append(layer_present)
        return self.norm(hidden_states), tuple(present)


class ResizeMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, input_size, bias=True)
        self.linear_fc2 = nn.Linear(input_size, output_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(hidden_states)))


class Qwen3TTSTalkerInnerModel(_Backbone):
    def __init__(self, config: Qwen3TTSTalkerConfig) -> None:
        super().__init__(config, multimodal_rope=True)
        self.text_embedding = nn.Embedding(int(config.text_vocab_size), int(config.text_hidden_size))
        self.codec_embedding = nn.Embedding(int(config.vocab_size), int(config.hidden_size))


class Qwen3TTSTalkerModel(nn.Module):
    """Checkpoint-compatible Talker with packed eager prefill."""

    def __init__(self, config: Qwen3TTSConfig | Qwen3TTSTalkerConfig) -> None:
        super().__init__()
        talker = config.talker if isinstance(config, Qwen3TTSConfig) else config
        self.config = talker
        self.model = Qwen3TTSTalkerInnerModel(talker)
        self.text_projection = ResizeMLP(int(talker.text_hidden_size), int(talker.hidden_size))
        self.codec_head = nn.Linear(int(talker.hidden_size), int(talker.vocab_size), bias=False)

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> TalkerOutput:
        hidden_states, past_key_values = self.model(
            input_embeds,
            position_ids=position_ids,
            cu_seq_lens=cu_seq_lens,
        )
        return TalkerOutput(
            last_hidden_state=hidden_states,
            logits=self.codec_head(hidden_states),
            past_key_values=past_key_values,
        )


class Qwen3TTSCodePredictorInnerModel(_Backbone):
    def __init__(self, config: Qwen3TTSConfig) -> None:
        talker = config.talker
        predictor = config.code_predictor
        super().__init__(predictor, multimodal_rope=False)
        self.codec_embedding = nn.ModuleList(
            nn.Embedding(int(predictor.vocab_size), int(talker.hidden_size))
            for _ in range(config.num_code_groups - 1)
        )


class Qwen3TTSCodePredictor(nn.Module):
    """Checkpoint-compatible Code Predictor weight tree for generated decode."""

    def __init__(self, config: Qwen3TTSConfig) -> None:
        super().__init__()
        talker = config.talker
        predictor = config.code_predictor
        self.config = predictor
        self.model = Qwen3TTSCodePredictorInnerModel(config)
        self.lm_head = nn.ModuleList(
            nn.Linear(int(predictor.hidden_size), int(predictor.vocab_size), bias=False)
            for _ in range(config.num_code_groups - 1)
        )
        self.small_to_mtp_projection = (
            nn.Linear(
                int(talker.hidden_size),
                int(predictor.hidden_size),
                bias=True,
            )
            if talker.hidden_size != predictor.hidden_size
            else nn.Identity()
        )

__all__ = [
    "KeyValue",
    "KeyValueCache",
    "Qwen3TTSCodePredictor",
    "Qwen3TTSTalkerModel",
    "TalkerOutput",
]
