"""Inference-only eager Whisper math used as a test correctness oracle.

This module intentionally owns no ``nn.Module`` or ``Parameter`` objects. The
shipping prefill and generated-decode backends consume the same typed weights;
these functional forwards validate their tensor contracts without shipping an
alternate runtime implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from kestrel.models.whisper.config import WhisperTurboConfig
from kestrel.models.whisper.weights import (
    AttentionWeights,
    DecoderLayerWeights,
    EncoderLayerWeights,
    LayerNormWeights,
    LinearWeights,
    WhisperModelWeights,
    validate_whisper_weight_tree,
)


def _linear(value: Tensor, weights: LinearWeights) -> Tensor:
    return F.linear(value, weights.weight, weights.bias)


def _layer_norm(
    value: Tensor,
    weights: LayerNormWeights,
    *,
    eps: float,
) -> Tensor:
    return F.layer_norm(
        value,
        (value.shape[-1],),
        weight=weights.weight,
        bias=weights.bias,
        eps=eps,
    )


def _heads(value: Tensor, num_heads: int) -> Tensor:
    batch, tokens, width = value.shape
    return value.view(batch, tokens, num_heads, width // num_heads)


def _attention_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output_weights: LinearWeights,
    *,
    causal: bool,
) -> Tensor:
    """Attention over ABI-layout K/V: ``[batch, tokens, heads, dim]``."""

    query_heads = query.transpose(1, 2)
    key_heads = key.transpose(1, 2)
    value_heads = value.transpose(1, 2)
    # Queries already carry Whisper's head-dimension scale.
    attended = F.scaled_dot_product_attention(
        query_heads,
        key_heads,
        value_heads,
        is_causal=causal,
        scale=1.0,
    )
    attended = (
        attended.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
    )
    return _linear(attended, output_weights)


def _project_query(
    hidden_states: Tensor,
    weights: AttentionWeights,
    *,
    num_heads: int,
) -> Tensor:
    head_dim = hidden_states.shape[-1] // num_heads
    return _heads(_linear(hidden_states, weights.query) * (head_dim**-0.5), num_heads)


def _project_key_value(
    hidden_states: Tensor,
    weights: AttentionWeights,
    *,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    return (
        _heads(_linear(hidden_states, weights.key), num_heads),
        _heads(_linear(hidden_states, weights.value), num_heads),
    )


def _self_attention(
    hidden_states: Tensor,
    weights: AttentionWeights,
    *,
    num_heads: int,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    query = _project_query(hidden_states, weights, num_heads=num_heads)
    key, value = _project_key_value(hidden_states, weights, num_heads=num_heads)
    output = _attention_output(
        query,
        key,
        value,
        weights.output,
        causal=causal,
    )
    return output, key, value


def _cross_attention(
    hidden_states: Tensor,
    key: Tensor,
    value: Tensor,
    weights: AttentionWeights,
    *,
    num_heads: int,
) -> Tensor:
    query = _project_query(hidden_states, weights, num_heads=num_heads)
    return _attention_output(
        query,
        key,
        value,
        weights.output,
        causal=False,
    )


def _mlp(
    hidden_states: Tensor,
    fc1: LinearWeights,
    fc2: LinearWeights,
) -> Tensor:
    # Whisper's ``gelu`` is the exact erf form, not tanh GELU.
    return _linear(F.gelu(_linear(hidden_states, fc1), approximate="none"), fc2)


def _encoder_layer(
    hidden_states: Tensor,
    weights: EncoderLayerWeights,
    config: WhisperTurboConfig,
) -> Tensor:
    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        weights.self_attention_layer_norm,
        eps=config.layer_norm_eps,
    )
    attention, _key, _value = _self_attention(
        normalized,
        weights.self_attention,
        num_heads=config.encoder_attention_heads,
        causal=False,
    )
    hidden_states = residual + attention

    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        weights.final_layer_norm,
        eps=config.layer_norm_eps,
    )
    hidden_states = residual + _mlp(normalized, weights.fc1, weights.fc2)
    if hidden_states.dtype is torch.float16:
        clamp = torch.finfo(torch.float16).max - 1000
        hidden_states = hidden_states.clamp(min=-clamp, max=clamp)
    return hidden_states


@dataclass(frozen=True, slots=True)
class CrossAttentionKV:
    """Dense reference K/V in ``[layers, batch, source, heads, dim]`` layout."""

    keys: Tensor
    values: Tensor

    def layer(self, index: int) -> tuple[Tensor, Tensor]:
        return self.keys[index], self.values[index]


@dataclass(frozen=True, slots=True)
class DecoderSelfKV:
    """Reference self-K/V for the decoded prefix only."""

    keys: Tensor
    values: Tensor
    length: int


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    logits: Tensor
    hidden_states: Tensor
    self_kv: DecoderSelfKV


class WhisperInferenceModel:
    """Functional eager model with no training or fallback serving path."""

    def __init__(
        self,
        config: WhisperTurboConfig,
        weights: WhisperModelWeights,
    ) -> None:
        self.config = config
        self.weights = weights
        validate_whisper_weight_tree(weights, config)

    @torch.inference_mode()
    def encode(self, input_features: Tensor) -> Tensor:
        encoder = self.weights.encoder
        hidden_states = F.gelu(
            F.conv1d(
                input_features,
                encoder.conv1.weight,
                encoder.conv1.bias,
                stride=encoder.conv1.stride,
                padding=encoder.conv1.padding,
            ),
            approximate="none",
        )
        hidden_states = F.gelu(
            F.conv1d(
                hidden_states,
                encoder.conv2.weight,
                encoder.conv2.bias,
                stride=encoder.conv2.stride,
                padding=encoder.conv2.padding,
            ),
            approximate="none",
        )
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states + encoder.position_embedding
        for layer in encoder.layers:
            hidden_states = _encoder_layer(hidden_states, layer, self.config)
        return _layer_norm(
            hidden_states,
            encoder.final_layer_norm,
            eps=self.config.layer_norm_eps,
        )

    @torch.inference_mode()
    def preproject_cross_kv(
        self,
        encoder_hidden_states: Tensor,
        *,
        storage_dtype: torch.dtype | None = None,
    ) -> CrossAttentionKV:
        keys = []
        values = []
        for layer in self.weights.decoder.layers:
            key, value = _project_key_value(
                encoder_hidden_states,
                layer.cross_attention,
                num_heads=self.config.decoder_attention_heads,
            )
            if storage_dtype is not None:
                key = key.to(dtype=storage_dtype)
                value = value.to(dtype=storage_dtype)
            keys.append(key)
            values.append(value)
        return CrossAttentionKV(
            keys=torch.stack(keys, dim=0).contiguous(),
            values=torch.stack(values, dim=0).contiguous(),
        )

    def _decoder_layer_full(
        self,
        hidden_states: Tensor,
        layer: DecoderLayerWeights,
        cross_key: Tensor,
        cross_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normalized = _layer_norm(
            hidden_states,
            layer.self_attention_layer_norm,
            eps=self.config.layer_norm_eps,
        )
        attention, self_key, self_value = _self_attention(
            normalized,
            layer.self_attention,
            num_heads=self.config.decoder_attention_heads,
            causal=True,
        )
        hidden_states = residual + attention

        residual = hidden_states
        normalized = _layer_norm(
            hidden_states,
            layer.cross_attention_layer_norm,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = residual + _cross_attention(
            normalized,
            cross_key,
            cross_value,
            layer.cross_attention,
            num_heads=self.config.decoder_attention_heads,
        )

        residual = hidden_states
        normalized = _layer_norm(
            hidden_states,
            layer.final_layer_norm,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = residual + _mlp(normalized, layer.fc1, layer.fc2)
        return hidden_states, self_key, self_value

    @torch.inference_mode()
    def decoder_prefix(
        self,
        token_ids: Tensor,
        cross_kv: CrossAttentionKV,
    ) -> DecoderOutput:
        _batch_size, tokens = token_ids.shape

        decoder = self.weights.decoder
        hidden_states = F.embedding(token_ids, decoder.token_embedding)
        hidden_states = hidden_states + decoder.position_embedding[:tokens]
        self_keys = []
        self_values = []
        for index, layer in enumerate(decoder.layers):
            cross_key, cross_value = cross_kv.layer(index)
            hidden_states, self_key, self_value = self._decoder_layer_full(
                hidden_states,
                layer,
                cross_key,
                cross_value,
            )
            self_keys.append(self_key)
            self_values.append(self_value)
        hidden_states = _layer_norm(
            hidden_states,
            decoder.final_layer_norm,
            eps=self.config.layer_norm_eps,
        )
        logits = F.linear(hidden_states, decoder.output_projection)
        cache = DecoderSelfKV(
            keys=torch.stack(self_keys).contiguous(),
            values=torch.stack(self_values).contiguous(),
            length=tokens,
        )
        return DecoderOutput(logits=logits, hidden_states=hidden_states, self_kv=cache)


__all__ = [
    "CrossAttentionKV",
    "DecoderOutput",
    "DecoderSelfKV",
    "WhisperInferenceModel",
]
