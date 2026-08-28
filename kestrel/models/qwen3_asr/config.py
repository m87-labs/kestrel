"""Inference-only Qwen3-ASR checkpoint configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FORCED_ALIGNER_LANGUAGES = (
    "Chinese",
    "Cantonese",
    "English",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "Russian",
    "Korean",
    "Japanese",
)


@dataclass(frozen=True, slots=True)
class AudioEncoderConfig:
    d_model: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    encoder_layers: int
    downsample_hidden_size: int = 480
    num_mel_bins: int = 128
    max_position_embeddings: int = 13
    n_window: int = 50
    n_window_infer: int = 800
    output_dim: int = 1024
    activation_function: str = "gelu"


@dataclass(frozen=True, slots=True)
class TextDecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float


def _audio_config(raw: dict[str, Any]) -> AudioEncoderConfig:
    return AudioEncoderConfig(
        d_model=raw["d_model"],
        encoder_attention_heads=raw["encoder_attention_heads"],
        encoder_ffn_dim=raw["encoder_ffn_dim"],
        encoder_layers=raw["encoder_layers"],
        downsample_hidden_size=raw["downsample_hidden_size"],
        num_mel_bins=raw["num_mel_bins"],
        max_position_embeddings=raw.get("max_position_embeddings", 13),
        n_window=raw["n_window"],
        n_window_infer=raw["n_window_infer"],
        output_dim=raw["output_dim"],
        activation_function=raw["activation_function"],
    )


def _text_config(raw: dict[str, Any]) -> TextDecoderConfig:
    rope = raw.get("rope_parameters", {})
    return TextDecoderConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=rope.get("rope_theta", raw.get("rope_theta", 1_000_000)),
    )


@dataclass(frozen=True, slots=True)
class Qwen3AsrConfig:
    audio: AudioEncoderConfig
    text: TextDecoderConfig
    audio_token_id: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Qwen3AsrConfig":
        root = Path(path)
        with (root / "config.json").open(encoding="utf-8") as file:
            raw = json.load(file)
        with (root / "generation_config.json").open(encoding="utf-8") as file:
            generation = json.load(file)
        thinker = raw["thinker_config"]
        config = cls(
            audio=_audio_config(thinker["audio_config"]),
            text=_text_config(thinker["text_config"]),
            audio_token_id=thinker["audio_token_id"],
            eos_token_ids=tuple(generation["eos_token_id"]),
            pad_token_id=generation["pad_token_id"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        audio = self.audio
        text = self.text
        if audio.activation_function != "gelu":
            raise ValueError("Qwen3-ASR audio activation must be GELU")
        if audio.d_model % audio.encoder_attention_heads:
            raise ValueError("audio d_model must be divisible by its attention heads")
        if text.head_dim <= 0 or text.num_attention_heads <= 0:
            raise ValueError("text attention geometry must be positive")
        if text.num_attention_heads % text.num_key_value_heads:
            raise ValueError("text query heads must be divisible by KV heads")
        if audio.output_dim != text.hidden_size:
            raise ValueError("audio projector output must match text hidden_size")
        if audio.n_window != 50 or audio.max_position_embeddings < 13:
            raise ValueError("unsupported Qwen3-ASR audio chunk geometry")


@dataclass(frozen=True, slots=True)
class ForcedAlignerConfig:
    audio: AudioEncoderConfig
    text: TextDecoderConfig
    audio_token_id: int
    classify_num: int
    supported_languages: tuple[str, ...]
    timestamp_segment_ms: float
    timestamp_token_id: int

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ForcedAlignerConfig":
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        config = cls(
            audio=_audio_config(raw["audio_config"]),
            text=_text_config(raw["text_config"]),
            audio_token_id=raw["audio_token_id"],
            classify_num=len(raw["id2label"]),
            supported_languages=_FORCED_ALIGNER_LANGUAGES,
            timestamp_segment_ms=float(raw["timestamp_segment_time"]),
            timestamp_token_id=raw["timestamp_token_id"],
        )
        Qwen3AsrConfig(
            config.audio,
            config.text,
            config.audio_token_id,
            (),
            0,
        ).validate()
        return config


__all__ = [
    "AudioEncoderConfig",
    "ForcedAlignerConfig",
    "Qwen3AsrConfig",
    "TextDecoderConfig",
]
