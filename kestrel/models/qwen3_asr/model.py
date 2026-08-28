"""Inference-only PyTorch definition of Qwen3-ASR.

This is the correctness/reference path. Production decode is bound through the
generated backend separately; keeping the ordinary module makes checkpoint and
kernel equivalence tests independent of Transformers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import AudioEncoderConfig, Qwen3AsrConfig, TextDecoderConfig


KvCache = list[tuple[Tensor, Tensor]]


class RmsNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(value.dtype)


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // self.num_heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden: Tensor, window_lengths: tuple[int, ...]) -> Tensor:
        total = hidden.shape[0]
        shape = (total, self.num_heads, self.head_dim)
        q = self.q_proj(hidden).view(shape)
        k = self.k_proj(hidden).view(shape)
        v = self.v_proj(hidden).view(shape)

        length = window_lengths[0]
        if all(item == length for item in window_lengths):
            batch_shape = (len(window_lengths), length, *shape[1:])
            attended = F.scaled_dot_product_attention(
                q.view(batch_shape).transpose(1, 2),
                k.view(batch_shape).transpose(1, 2),
                v.view(batch_shape).transpose(1, 2),
                scale=self.scale,
            )
            output = attended.transpose(1, 2).reshape(total, -1)
        else:
            q_chunks = q.split(window_lengths)
            k_chunks = k.split(window_lengths)
            v_chunks = v.split(window_lengths)
            groups: dict[int, list[int]] = {}
            for index, item in enumerate(window_lengths):
                groups.setdefault(item, []).append(index)

            outputs: dict[int, Tensor] = {}
            for indices in groups.values():
                attended = F.scaled_dot_product_attention(
                    torch.stack([q_chunks[index] for index in indices]).transpose(1, 2),
                    torch.stack([k_chunks[index] for index in indices]).transpose(1, 2),
                    torch.stack([v_chunks[index] for index in indices]).transpose(1, 2),
                    scale=self.scale,
                )
                for index, chunk in zip(indices, attended.transpose(1, 2), strict=True):
                    outputs[index] = chunk.flatten(1)
            output = torch.cat([outputs[index] for index in range(len(window_lengths))])
        return self.out_proj(output)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, hidden: Tensor, window_lengths: tuple[int, ...]) -> Tensor:
        hidden = hidden + self.self_attn(
            self.self_attn_layer_norm(hidden), window_lengths
        )
        return hidden + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))


def _sinusoids(length: int, channels: int) -> Tensor:
    scale = math.log(10_000) / (channels // 2 - 1)
    inverse = torch.exp(-scale * torch.arange(channels // 2, dtype=torch.float32))
    phase = torch.arange(length, dtype=torch.float32)[:, None] * inverse[None]
    return torch.cat((phase.sin(), phase.cos()), dim=1)


def _conv_output_length(length: Tensor) -> Tensor:
    return torch.where(length > 0, (length + 7) // 8, 0)


def _audio_output_length(length: Tensor, chunk_length: int) -> Tensor:
    full_chunks, remainder = (
        torch.div(length, chunk_length, rounding_mode="floor"),
        length % chunk_length,
    )
    return full_chunks * _conv_output_length(
        torch.tensor(chunk_length, device=length.device)
    ) + _conv_output_length(remainder)


def _conv_output_length_int(length: int) -> int:
    return (length + 7) // 8 if length > 0 else 0


def _audio_output_length_int(length: int, chunk_length: int) -> int:
    full_chunks, remainder = divmod(length, chunk_length)
    return full_chunks * _conv_output_length_int(
        chunk_length
    ) + _conv_output_length_int(remainder)


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, channels, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        frequency_bins = (config.num_mel_bins + 1) // 2
        frequency_bins = (frequency_bins + 1) // 2
        frequency_bins = (frequency_bins + 1) // 2
        self.conv_out = nn.Linear(channels * frequency_bins, config.d_model, bias=False)
        self.layers = nn.ModuleList(
            AudioEncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.register_buffer(
            "positional_embedding",
            _sinusoids(config.max_position_embeddings, config.d_model),
            persistent=False,
        )

    def reset_nonpersistent_buffers(self) -> None:
        self.positional_embedding = _sinusoids(
            self.config.max_position_embeddings, self.config.d_model
        ).to(self.conv_out.weight.device)

    def window_lengths(self, feature_mask: Tensor) -> tuple[int, ...]:
        return tuple(
            length
            for valid_frames in feature_mask.sum(-1).long().tolist()
            for length in self.feature_layout(valid_frames)[1]
        )

    def feature_layout(self, valid_frames: int) -> tuple[int, tuple[int, ...]]:
        chunk_length = self.config.n_window * 2
        audio_tokens = _audio_output_length_int(valid_frames, chunk_length)
        window = _conv_output_length_int(chunk_length) * (
            self.config.n_window_infer // chunk_length
        )
        full, remainder = divmod(audio_tokens, window)
        lengths = [window] * full
        if remainder:
            lengths.append(remainder)
        return audio_tokens, tuple(lengths)

    def forward(
        self,
        features: Tensor,
        feature_mask: Tensor,
        window_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        batch, mel_bins, padded_length = features.shape
        chunk_length = self.config.n_window * 2
        if padded_length % chunk_length:
            raise ValueError(f"feature length must be a multiple of {chunk_length}")
        chunks_per_item = padded_length // chunk_length
        chunks = (
            features.view(batch, mel_bins, chunks_per_item, chunk_length)
            .permute(0, 2, 1, 3)
            .reshape(batch * chunks_per_item, 1, mel_bins, chunk_length)
        )
        hidden = F.gelu(self.conv2d1(chunks))
        hidden = F.gelu(self.conv2d2(hidden))
        hidden = F.gelu(self.conv2d3(hidden))
        chunk_count, channels, frequency, time = hidden.shape
        hidden = self.conv_out(
            hidden.permute(0, 3, 1, 2).reshape(chunk_count, time, channels * frequency)
        )
        hidden = hidden + self.positional_embedding[:time].to(hidden.dtype)

        raw_chunk_lengths = feature_mask.view(batch, chunks_per_item, chunk_length).sum(
            -1
        )
        post_lengths = _conv_output_length(raw_chunk_lengths.long()).flatten()
        valid = torch.arange(time, device=features.device)[None] < post_lengths[:, None]
        hidden = hidden.flatten(0, 1)[valid.flatten()]

        lengths = (
            self.window_lengths(feature_mask)
            if window_lengths is None
            else window_lengths
        )
        for layer in self.layers:
            hidden = layer(hidden, lengths)
        return self.ln_post(hidden)

    def _post_tokens(self, feature_mask: Tensor) -> Tensor:
        return _audio_output_length(
            feature_mask.sum(-1).long(), self.config.n_window * 2
        ).sum()


class MultiModalProjector(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_model)
        self.linear_2 = nn.Linear(config.d_model, config.output_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear_2(F.gelu(self.linear_1(hidden)))


def _rotate_half(value: Tensor) -> Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class TextAttention(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RmsNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RmsNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        cached: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if cached is not None:
            k = torch.cat((cached[0], k), dim=2)
            v = torch.cat((cached[1], v), dim=2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=cached is None and length > 1,
            scale=self.head_dim**-0.5,
            enable_gqa=self.groups > 1,
        )
        output = attended.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(output), (k, v)


class TextMlp(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TextDecoderLayer(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.self_attn = TextAttention(config)
        self.mlp = TextMlp(config)
        self.input_layernorm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        cached: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attention, updated = self.self_attn(
            self.input_layernorm(hidden), cos, sin, cached
        )
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), updated


class TextDecoder(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        inverse = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def reset_nonpersistent_buffers(self) -> None:
        config = self.config
        self.inverse_frequency = (
            1.0
            / (
                config.rope_theta
                ** (
                    torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                    / config.head_dim
                )
            )
        ).to(self.embed_tokens.weight.device)

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        input_embeddings: Tensor | None = None,
        cache: KvCache | None = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, KvCache | None]:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of input_ids or input_embeddings")
        hidden = (
            self.embed_tokens(input_ids)
            if input_embeddings is None
            else input_embeddings
        )
        assert hidden is not None
        offset = 0 if cache is None else cache[0][0].shape[2]
        positions = torch.arange(offset, offset + hidden.shape[1], device=hidden.device)
        frequency = torch.outer(positions.float(), self.inverse_frequency)
        embedding = torch.cat((frequency, frequency), dim=-1)
        cos, sin = embedding.cos().to(hidden.dtype), embedding.sin().to(hidden.dtype)
        updated: KvCache | None = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            hidden, layer_cache = layer(
                hidden,
                cos,
                sin,
                None if cache is None else cache[index],
            )
            if updated is not None:
                updated.append(layer_cache)
        return self.norm(hidden), updated


@dataclass(frozen=True, slots=True)
class Qwen3AsrOutput:
    logits: Tensor
    cache: KvCache


class Qwen3AsrModel(nn.Module):
    def __init__(self, config: Qwen3AsrConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio)
        self.language_model = TextDecoder(config.text)
        self.multi_modal_projector = MultiModalProjector(config.audio)

    def audio_embeddings(
        self,
        features: Tensor,
        feature_mask: Tensor,
        window_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        return self.multi_modal_projector(
            self.audio_tower(features, feature_mask, window_lengths)
        )

    def prefill(
        self,
        input_ids: Tensor,
        features: Tensor,
        feature_mask: Tensor,
        *,
        window_lengths: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, KvCache]:
        embeddings = self.language_model.embed_tokens(input_ids)
        audio = self.audio_embeddings(features, feature_mask, window_lengths).to(
            embeddings.dtype
        )
        mask = input_ids == self.config.audio_token_id
        embeddings = embeddings.masked_scatter(mask.unsqueeze(-1), audio)
        hidden, cache = self.language_model(input_embeddings=embeddings)
        assert cache is not None
        return hidden, cache


class Qwen3AsrForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3AsrConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3AsrModel(config)
        self.lm_head = nn.Linear(
            config.text.hidden_size, config.text.vocab_size, bias=False
        )
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def reset_nonpersistent_buffers(self) -> None:
        self.model.audio_tower.reset_nonpersistent_buffers()
        self.model.language_model.reset_nonpersistent_buffers()

    def prefill(
        self,
        input_ids: Tensor,
        features: Tensor,
        feature_mask: Tensor,
        *,
        window_lengths: tuple[int, ...] | None = None,
        last_indices: Tensor | None = None,
    ) -> Qwen3AsrOutput:
        hidden, cache = self.model.prefill(
            input_ids,
            features,
            feature_mask,
            window_lengths=window_lengths,
        )
        if last_indices is None:
            last_hidden = hidden[:, -1:]
        else:
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            last_hidden = hidden[rows, last_indices].unsqueeze(1)
        return Qwen3AsrOutput(self.lm_head(last_hidden).float(), cache)

    def decode(self, input_ids: Tensor, cache: KvCache) -> Qwen3AsrOutput:
        hidden, updated = self.model.language_model(input_ids=input_ids, cache=cache)
        assert updated is not None
        return Qwen3AsrOutput(self.lm_head(hidden).float(), updated)


__all__ = ["KvCache", "Qwen3AsrForConditionalGeneration", "Qwen3AsrOutput"]
