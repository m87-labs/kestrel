"""Inference-only Parakeet TDT checkpoint configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParakeetEncoderConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_mel_bins: int
    conv_kernel_size: int
    subsampling_conv_channels: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    subsampling_factor: int
    max_position_embeddings: int
    hidden_act: str


@dataclass(frozen=True, slots=True)
class ParakeetTdtConfig:
    encoder: ParakeetEncoderConfig
    blank_token_id: int
    decoder_hidden_size: int
    durations: tuple[int, ...]
    max_symbols_per_step: int
    num_decoder_layers: int
    pad_token_id: int
    vocab_size: int
    hidden_act: str

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ParakeetTdtConfig":
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        encoder = raw["encoder_config"]
        config = cls(
            encoder=ParakeetEncoderConfig(
                hidden_size=encoder["hidden_size"],
                intermediate_size=encoder["intermediate_size"],
                num_hidden_layers=encoder["num_hidden_layers"],
                num_attention_heads=encoder["num_attention_heads"],
                num_key_value_heads=encoder["num_key_value_heads"],
                num_mel_bins=encoder["num_mel_bins"],
                conv_kernel_size=encoder["conv_kernel_size"],
                subsampling_conv_channels=encoder["subsampling_conv_channels"],
                subsampling_conv_kernel_size=encoder["subsampling_conv_kernel_size"],
                subsampling_conv_stride=encoder["subsampling_conv_stride"],
                subsampling_factor=encoder["subsampling_factor"],
                max_position_embeddings=encoder["max_position_embeddings"],
                hidden_act=encoder["hidden_act"],
            ),
            blank_token_id=raw["blank_token_id"],
            decoder_hidden_size=raw["decoder_hidden_size"],
            durations=tuple(raw["durations"]),
            max_symbols_per_step=raw["max_symbols_per_step"],
            num_decoder_layers=raw["num_decoder_layers"],
            pad_token_id=raw["pad_token_id"],
            vocab_size=raw["vocab_size"],
            hidden_act=raw["hidden_act"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        encoder = self.encoder
        if encoder.hidden_act != "silu" or self.hidden_act != "relu":
            raise ValueError("unsupported Parakeet activation functions")
        if encoder.hidden_size % encoder.num_attention_heads:
            raise ValueError("encoder hidden_size must be divisible by attention heads")
        if encoder.num_attention_heads != encoder.num_key_value_heads:
            raise ValueError("grouped-query Parakeet encoders are not supported")
        if (
            encoder.subsampling_factor not in (2, 4, 8)
            or encoder.subsampling_conv_kernel_size != 3
            or encoder.subsampling_conv_stride != 2
            or encoder.num_mel_bins % encoder.subsampling_factor
        ):
            raise ValueError("unsupported Parakeet subsampling geometry")
        if self.blank_token_id >= self.vocab_size:
            raise ValueError("blank token must be inside the vocabulary")
        if (
            not self.durations
            or self.durations[0] != 0
            or any(
                right <= left for left, right in zip(self.durations, self.durations[1:])
            )
        ):
            raise ValueError("durations must be strictly increasing from zero")


__all__ = ["ParakeetEncoderConfig", "ParakeetTdtConfig"]
