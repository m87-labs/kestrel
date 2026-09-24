"""Inference-only Qwen3-TTS 12 Hz codec decoder and streaming state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from kestrel_kernels.swiglu import gated_activation_into

from .config import Qwen3TTSCodecConfig, Qwen3TTSConfig

# Equations and checkpoint names follow the Apache-2.0 Qwen3-TTS decoder;
# incremental boundary state follows Nari Labs' Apache-2.0 streaming adapter.

KeyValue = tuple[torch.Tensor, torch.Tensor]
KeyValueCache = tuple[KeyValue, ...]


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Tried library RMSNorm (fallback at eps=1e-5): no consistent codec gain
        # across H100/B200 B1/4/8 F3/12 (0.99–1.00x); keep the direct expression.
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (values * self.weight.float()).to(dtype)


class _MLP(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        hidden, width = self.gate_proj.weight.shape
        self.register_buffer("gate_up_weight", torch.stack((
            self.gate_proj.weight.view(hidden // 8, 8, width),
            self.up_proj.weight.view(hidden // 8, 8, width),
        ), dim=1).reshape(2 * hidden, width))
        del self.gate_proj, self.up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        activated = torch.empty_like(gate_up[..., :gate_up.shape[-1] // 2])
        gated_activation_into(activated, gate_up, activation="silu", layout="interleaved_i8")
        return self.down_proj(activated)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(values: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = values.shape[-1]
    frequencies = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim)
    )
    angles = position_ids.float().unsqueeze(-1) * frequencies
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(1)
    return values * cos.to(values.dtype) + _rotate_half(values) * sin.to(values.dtype)


class _Attention(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: KeyValue,
    ) -> tuple[torch.Tensor, KeyValue]:
        batch_size, token_count, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, token_count, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, token_count, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query = _apply_rope(query, position_ids, self.rope_theta)
        key = _apply_rope(key, position_ids, self.rope_theta)
        key = torch.cat((past_key_value[0], key), dim=2)
        value = torch.cat((past_key_value[1], value), dim=2)

        attended = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(self.num_key_value_groups, dim=1),
            value.repeat_interleave(self.num_key_value_groups, dim=1),
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, -1)
        return self.o_proj(attended), (key, value)


class _LayerScale(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_initial_scale)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.scale * hidden_states


class _TransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = _LayerScale(config)
        self.mlp_layer_scale = _LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: KeyValue,
    ) -> tuple[torch.Tensor, KeyValue]:
        attended, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value,
        )
        hidden_states = hidden_states + self.self_attn_layer_scale(attended)
        hidden_states = hidden_states + self.mlp_layer_scale(
            self.mlp(self.post_attention_layernorm(hidden_states))
        )
        return hidden_states, present


@dataclass(frozen=True, slots=True)
class _TransformerOutput:
    last_hidden_state: torch.Tensor
    past_key_values: KeyValueCache


class _Transformer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(_TransformerLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_key_values: Sequence[KeyValue],
        past_lengths: torch.Tensor,
    ) -> _TransformerOutput:
        hidden_states = self.input_proj(input_embeds)
        retained = past_key_values[0][0].shape[2]
        slots = torch.arange(retained, device=hidden_states.device)
        key_positions = torch.cat(
            (position_ids[:, :1] + slots - retained, position_ids),
            dim=1,
        )
        valid_prior = slots.unsqueeze(0) >= retained - past_lengths.unsqueeze(1)
        valid_keys = torch.cat(
            (valid_prior, torch.ones_like(position_ids, dtype=torch.bool)),
            dim=1,
        )

        query_positions = position_ids.unsqueeze(-1)
        keys = key_positions.unsqueeze(1)
        attention_mask = (
            valid_keys.unsqueeze(1)
            & (keys <= query_positions)
            & (keys > query_positions - self.config.sliding_window)
        ).unsqueeze(1)

        present: list[KeyValue] = []
        for index, layer in enumerate(self.layers):
            layer_past = past_key_values[index]
            hidden_states, layer_present = layer(
                hidden_states,
                position_ids,
                attention_mask,
                layer_past,
            )
            layer_present = (
                layer_present[0][:, :, -retained:],
                layer_present[1][:, :, -retained:],
            )
            present.append(layer_present)

        hidden_states = self.output_proj(self.norm(hidden_states))
        return _TransformerOutput(
            hidden_states,
            tuple(present),
        )


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.padding = (kernel_size - 1) * dilation

class _CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.right_pad = kernel_size - stride

class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = _CausalConv1d(channels, channels, 7, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

class _SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.ones(channels))

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        self.alpha.exp_()
        self.beta.exp_().add_(1.0e-9).reciprocal_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        result = values + torch.sin(
            values * self.alpha.view(1, -1, 1)
        ).square() * self.beta.view(1, -1, 1)
        return result.to(hidden_states.dtype)


class _ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.act1 = _SnakeBeta(channels)
        self.conv1 = _CausalConv1d(channels, channels, 7, dilation=dilation)
        self.act2 = _SnakeBeta(channels)
        # Tried F.linear for these 1x1 convolutions: 0.21-0.60x Conv1d at
        # B1/B8 F12 on H100/B200; keeping Conv1d.
        self.conv2 = _CausalConv1d(channels, channels, 1)

class _DecoderBlock(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig, layer_index: int) -> None:
        super().__init__()
        in_channels = config.decoder_dim // 2**layer_index
        out_channels = config.decoder_dim // 2 ** (layer_index + 1)
        rate = config.upsample_rates[layer_index]
        self.block = nn.ModuleList(
            [
                _SnakeBeta(in_channels),
                _CausalConvTranspose1d(in_channels, out_channels, 2 * rate, rate),
                *(_ResidualUnit(out_channels, dilation) for dilation in (1, 3, 9)),
            ]
        )

class _EuclideanCodebook(nn.Module):
    def __init__(self, dimension: int, codebook_size: int) -> None:
        super().__init__()
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dimension))

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        normalized = self.embedding_sum.float() / self.cluster_usage.float().clamp(
            min=1e-5
        ).unsqueeze(1)
        self.embedding_sum.copy_(normalized)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.embedding_sum)


class _VectorQuantization(nn.Module):
    def __init__(self, dimension: int, codebook_size: int) -> None:
        super().__init__()
        self._codebook = _EuclideanCodebook(dimension, codebook_size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._codebook.decode(codes).transpose(1, 2)


class _ResidualVectorQuantization(nn.Module):
    def __init__(self, dimension: int, codebook_size: int, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _VectorQuantization(dimension, codebook_size) for _ in range(count)
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(
            (layer.decode(layer_codes) for layer, layer_codes in zip(self.layers, codes)),
            start=0,
        )


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig, count: int) -> None:
        super().__init__()
        dimension = config.codebook_dim // 2
        # Tried F.linear for these 1x1 projections: 0.43-0.90x Conv1d at
        # B1/B8 F12 on H100/B200; keeping Conv1d.
        self.input_proj = nn.Conv1d(config.codebook_dim, dimension, 1, bias=False)
        self.output_proj = nn.Conv1d(dimension, config.codebook_dim, 1, bias=False)
        self.vq = _ResidualVectorQuantization(dimension, config.codebook_size, count)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.vq.decode(codes.transpose(0, 1)))


class _SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecConfig) -> None:
        super().__init__()
        self.rvq_first = _ResidualVectorQuantizer(config, 1)
        self.rvq_rest = _ResidualVectorQuantizer(config, config.num_quantizers - 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        hidden_states = self.rvq_first.decode(codes[:, :1])
        return hidden_states + self.rvq_rest.decode(codes[:, 1:])


class Qwen3TTSCodecDecoder(nn.Module):
    """Checkpoint-compatible 12 Hz code-to-waveform decoder."""

    def __init__(self, config: Qwen3TTSConfig | Qwen3TTSCodecConfig) -> None:
        super().__init__()
        codec = config.codec if isinstance(config, Qwen3TTSConfig) else config
        self.config = codec
        self.pre_transformer = _Transformer(codec)
        self.quantizer = _SplitResidualVectorQuantizer(codec)
        self.pre_conv = _CausalConv1d(codec.codebook_dim, codec.latent_dim, 3)
        self.upsample = nn.ModuleList(
            nn.ModuleList(
                [
                    _CausalConvTranspose1d(codec.latent_dim, codec.latent_dim, ratio, ratio),
                    _ConvNeXtBlock(codec.latent_dim),
                ]
            )
            for ratio in codec.upsampling_ratios
        )
        blocks: list[nn.Module] = [
            _CausalConv1d(codec.latent_dim, codec.decoder_dim, 7)
        ]
        blocks.extend(_DecoderBlock(codec, index) for index in range(len(codec.upsample_rates)))
        output_channels = codec.decoder_dim // 2 ** len(codec.upsample_rates)
        blocks.extend((_SnakeBeta(output_channels), _CausalConv1d(output_channels, 1, 7)))
        self.decoder = nn.ModuleList(blocks)

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        """Materialize invariant checkpoint transforms once after loading."""

        for module in self.modules():
            if isinstance(module, (_EuclideanCodebook, _SnakeBeta, _MLP)):
                module.prepare_for_inference()

@dataclass(slots=True)
class IncrementalCodecState:
    """Causal state owned by one streaming request."""

    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)


class Qwen3TTSIncrementalDecoder:
    """Decode only new code frames while preserving exact causal boundaries."""

    def __init__(self, decoder: Qwen3TTSCodecDecoder) -> None:
        self.decoder = decoder
        self.retained_context = decoder.config.sliding_window - 1

    @staticmethod
    def _stack(
        states: Sequence[IncrementalCodecState],
        mapping: str,
        key: str | int,
        shape: tuple[int, ...],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for state in states:
            value = getattr(state, mapping).get(key)
            values.append(reference.new_zeros(shape) if value is None else value)
        return torch.stack(values)

    @classmethod
    def _causal_conv(
        cls,
        module: _CausalConv1d,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        if not module.padding:
            return module.conv(hidden_states).contiguous()
        history = cls._stack(
            states,
            "conv_histories",
            key,
            (hidden_states.shape[1], module.padding),
            hidden_states,
        )
        combined = torch.cat((history, hidden_states), dim=-1)
        output = module.conv(combined).contiguous()
        for row, state in enumerate(states):
            state.conv_histories[key] = combined[row, :, -module.padding :].detach().clone()
        return output

    @classmethod
    def _causal_transconv(
        cls,
        module: _CausalConvTranspose1d,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        conv = module.conv
        expanded = F.conv_transpose1d(
            hidden_states,
            conv.weight,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding,
            groups=conv.groups,
            dilation=conv.dilation,
        )
        if module.right_pad:
            overlap = cls._stack(
                states,
                "transconv_overlaps",
                key,
                (expanded.shape[1], module.right_pad),
                expanded,
            )
            expanded[:, :, : module.right_pad] += overlap
        emitted_size = hidden_states.shape[-1] * conv.stride[0]
        emitted = expanded[:, :, :emitted_size]
        if conv.bias is not None:
            emitted = emitted + conv.bias.view(1, -1, 1)
        if module.right_pad:
            for row, state in enumerate(states):
                state.transconv_overlaps[key] = expanded[row, :, emitted_size:].detach().clone()
        return emitted.contiguous()

    @classmethod
    def _convnext(
        cls,
        module: _ConvNeXtBlock,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = cls._causal_conv(
            module.dwconv,
            hidden_states,
            states,
            f"{key}.dwconv",
        ).transpose(1, 2)
        hidden_states = module.pwconv2(module.act(module.pwconv1(module.norm(hidden_states))))
        return residual + (module.gamma * hidden_states).transpose(1, 2)

    @classmethod
    def _residual_unit(
        cls,
        module: _ResidualUnit,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = cls._causal_conv(
            module.conv1,
            module.act1(hidden_states),
            states,
            f"{key}.conv1",
        )
        hidden_states = cls._causal_conv(
            module.conv2,
            module.act2(hidden_states),
            states,
            f"{key}.conv2",
        )
        return residual + hidden_states

    def _transformer(
        self,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        position_ids: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transformer = self.decoder.pre_transformer
        fresh_frames = hidden_states.shape[1]
        if position_ids is None:
            frame_positions = torch.tensor(
                [state.frame_position for state in states],
                dtype=torch.long,
                device=hidden_states.device,
            )
            position_ids = frame_positions.unsqueeze(1) + torch.arange(
                fresh_frames,
                device=hidden_states.device,
            )
        reference = transformer.layers[0].self_attn.k_proj.weight
        cache_shape = (
            self.decoder.config.num_key_value_heads,
            self.retained_context,
            self.decoder.config.head_dim,
        )
        cache = tuple(
            (
                self._stack(states, "transformer_keys", index, cache_shape, reference),
                self._stack(states, "transformer_values", index, cache_shape, reference),
            )
            for index in range(len(transformer.layers))
        )
        if context_lengths is None:
            context_lengths = torch.tensor(
                [state.transformer_context_length for state in states],
                dtype=torch.long,
                device=hidden_states.device,
            )
        output = transformer(
            hidden_states,
            position_ids=position_ids,
            past_key_values=cache,
            past_lengths=context_lengths,
        )
        for layer_index, (keys, values) in enumerate(output.past_key_values):
            for state, key, value in zip(
                states,
                keys.detach().unbind(0),
                values.detach().unbind(0),
                strict=True,
            ):
                state.transformer_keys[layer_index] = key
                state.transformer_values[layer_index] = value
        for state in states:
            state.frame_position += fresh_frames
            state.transformer_context_length = min(
                self.retained_context,
                state.transformer_context_length + fresh_frames,
            )
        return output.last_hidden_state

    @torch.inference_mode()
    def prepare_transformer_input(
        self,
        codes: torch.Tensor,
        states: Sequence[IncrementalCodecState],
    ) -> torch.Tensor:
        """Decode codebooks and the causal pre-convolution into BTC rows."""

        if codes.ndim != 3 or codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"codes must have shape (batch, {self.decoder.config.num_quantizers}, frames)"
            )
        if codes.shape[0] != len(states):
            raise ValueError("each codec row needs one incremental state")
        if codes.shape[2] == 0:
            return next(self.decoder.parameters()).new_empty(
                (codes.shape[0], 0, self.decoder.config.latent_dim)
            )

        hidden_states = self.decoder.quantizer.decode(codes)
        hidden_states = self._causal_conv(
            self.decoder.pre_conv,
            hidden_states,
            states,
            "pre_conv",
        ).transpose(1, 2)
        return hidden_states

    @torch.inference_mode()
    def prepare_vocoder(
        self,
        codes: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        *,
        position_ids: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode codes through the streaming transformer into BTC latents."""

        hidden_states = self.prepare_transformer_input(codes, states)
        if hidden_states.shape[1] == 0:
            return hidden_states
        return self._transformer(
            hidden_states,
            states,
            position_ids,
            context_lengths,
        )

    @torch.inference_mode()
    def decode_vocoder(
        self,
        hidden_states: torch.Tensor,
        states: Sequence[IncrementalCodecState],
    ) -> torch.Tensor:
        """Decode BTC transformer latents into waveform samples."""

        if (
            hidden_states.ndim != 3
            or hidden_states.shape[2] != self.decoder.config.latent_dim
        ):
            raise ValueError(
                "codec latents must have shape "
                f"(batch, frames, {self.decoder.config.latent_dim})"
            )
        if hidden_states.shape[0] != len(states):
            raise ValueError("each codec row needs one incremental state")
        if hidden_states.shape[1] == 0:
            return hidden_states.new_empty((hidden_states.shape[0], 1, 0))

        hidden_states = hidden_states.transpose(1, 2)

        for index, blocks in enumerate(self.decoder.upsample):
            hidden_states = self._causal_transconv(
                blocks[0], hidden_states, states, f"upsample.{index}.0"
            )
            hidden_states = self._convnext(
                blocks[1], hidden_states, states, f"upsample.{index}.1"
            )

        hidden_states = self._causal_conv(
            self.decoder.decoder[0], hidden_states, states, "decoder.0"
        )
        for block_index, block in enumerate(self.decoder.decoder[1:-2], start=1):
            hidden_states = block.block[0](hidden_states)
            hidden_states = self._causal_transconv(
                block.block[1],
                hidden_states,
                states,
                f"decoder.{block_index}.block.1",
            )
            for unit_index, unit in enumerate(block.block[2:]):
                hidden_states = self._residual_unit(
                    unit,
                    hidden_states,
                    states,
                    f"decoder.{block_index}.block.{unit_index + 2}",
                )
        hidden_states = self.decoder.decoder[-2](hidden_states)
        hidden_states = self._causal_conv(
            self.decoder.decoder[-1],
            hidden_states,
            states,
            f"decoder.{len(self.decoder.decoder) - 1}",
        )
        return hidden_states.clamp(-1, 1)

    @torch.inference_mode()
    def __call__(
        self,
        codes: torch.Tensor,
        states: Sequence[IncrementalCodecState],
        *,
        position_ids: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.prepare_vocoder(
            codes,
            states,
            position_ids=position_ids,
            context_lengths=context_lengths,
        )
        return self.decode_vocoder(latent, states)


__all__ = [
    "IncrementalCodecState",
    "Qwen3TTSCodecDecoder",
    "Qwen3TTSIncrementalDecoder",
]
