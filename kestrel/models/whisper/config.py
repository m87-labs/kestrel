"""Strict inference-only configuration for Whisper large-v3-turbo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .assets import WhisperAssets


class UnsupportedWhisperConfig(ValueError):
    """The checkpoint is not the exact architecture covered by this runtime."""


def _mismatches(data: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for name, wanted in expected.items():
        if name not in data:
            result.append(f"{name}=<missing> (expected {wanted!r})")
            continue
        actual = data[name]
        if actual != wanted:
            result.append(f"{name}={actual!r} (expected {wanted!r})")
    return result


@dataclass(frozen=True, slots=True)
class WhisperTurboConfig:
    """Only model facts used by inference and native/generated lowering."""

    d_model: int = 1280
    encoder_layers: int = 32
    decoder_layers: int = 4
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120
    max_source_positions: int = 1500
    max_target_positions: int = 448
    num_mel_bins: int = 128
    vocab_size: int = 51866
    pad_token_id: int = 50257
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    decoder_start_token_id: int = 50258
    activation_function: str = "gelu"
    scale_embedding: bool = False
    tie_word_embeddings: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        positive_fields = {
            "d_model": self.d_model,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "decoder_attention_heads": self.decoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "max_source_positions": self.max_source_positions,
            "max_target_positions": self.max_target_positions,
            "num_mel_bins": self.num_mel_bins,
            "vocab_size": self.vocab_size,
        }
        for name, value in positive_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Whisper {name} must be a positive integer")
        for name, heads in (
            ("encoder_attention_heads", self.encoder_attention_heads),
            ("decoder_attention_heads", self.decoder_attention_heads),
        ):
            if self.d_model % heads:
                raise ValueError(f"Whisper d_model must be divisible by {name}")
        for name, token_id in (
            ("pad_token_id", self.pad_token_id),
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("decoder_start_token_id", self.decoder_start_token_id),
        ):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"Whisper {name} must be an integer")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Whisper {name} must be inside the vocabulary")
        if self.activation_function != "gelu":
            raise ValueError("Whisper inference supports exact GELU only")
        if self.scale_embedding:
            raise ValueError("Whisper scaled token embeddings are unsupported")
        if not self.tie_word_embeddings:
            raise ValueError("Whisper inference requires tied token/output embeddings")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0.0:
            raise ValueError("Whisper layer_norm_eps must be finite and positive")

    @property
    def decoder_head_dim(self) -> int:
        return self.d_model // self.decoder_attention_heads

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WhisperTurboConfig":
        if not isinstance(data, Mapping):
            raise TypeError("Whisper config must be a mapping")
        defaults = cls()
        expected = {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
            "is_encoder_decoder": True,
            "d_model": defaults.d_model,
            "encoder_layers": defaults.encoder_layers,
            "decoder_layers": defaults.decoder_layers,
            "encoder_attention_heads": defaults.encoder_attention_heads,
            "decoder_attention_heads": defaults.decoder_attention_heads,
            "encoder_ffn_dim": defaults.encoder_ffn_dim,
            "decoder_ffn_dim": defaults.decoder_ffn_dim,
            "max_source_positions": defaults.max_source_positions,
            "max_target_positions": defaults.max_target_positions,
            "num_mel_bins": defaults.num_mel_bins,
            "vocab_size": defaults.vocab_size,
            "pad_token_id": defaults.pad_token_id,
            "bos_token_id": defaults.bos_token_id,
            "eos_token_id": defaults.eos_token_id,
            "decoder_start_token_id": defaults.decoder_start_token_id,
            "activation_function": defaults.activation_function,
            "scale_embedding": defaults.scale_embedding,
            "use_cache": True,
        }
        bad = _mismatches(data, expected)
        if (
            "num_hidden_layers" in data
            and data["num_hidden_layers"] != defaults.encoder_layers
        ):
            bad.append(
                f"num_hidden_layers={data['num_hidden_layers']!r} "
                f"(expected {defaults.encoder_layers!r})"
            )
        if "tie_word_embeddings" in data and data["tie_word_embeddings"] is not True:
            bad.append(
                f"tie_word_embeddings={data['tie_word_embeddings']!r} (expected True)"
            )
        if bad:
            raise UnsupportedWhisperConfig(
                "Unsupported Whisper checkpoint architecture: " + "; ".join(bad)
            )
        return cls()

    @classmethod
    def from_assets(cls, assets: WhisperAssets) -> "WhisperTurboConfig":
        return cls.from_dict(assets.load_json("config.json"))


@dataclass(frozen=True, slots=True)
class WhisperPreprocessorConfig:
    """Fixed short-form feature geometry for the pinned checkpoint."""

    sampling_rate: int = 16000
    chunk_length: int = 30
    n_fft: int = 400
    hop_length: int = 160
    feature_size: int = 128
    n_samples: int = 480000
    nb_max_frames: int = 3000
    padding_value: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WhisperPreprocessorConfig":
        if not isinstance(data, Mapping):
            raise TypeError("Whisper preprocessor config must be a mapping")
        defaults = cls()
        expected = {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "sampling_rate": defaults.sampling_rate,
            "chunk_length": defaults.chunk_length,
            "n_fft": defaults.n_fft,
            "hop_length": defaults.hop_length,
            "feature_size": defaults.feature_size,
            "n_samples": defaults.n_samples,
            "nb_max_frames": defaults.nb_max_frames,
            "padding_value": defaults.padding_value,
            "padding_side": "right",
            "return_attention_mask": False,
        }
        bad = _mismatches(data, expected)
        if bad:
            raise UnsupportedWhisperConfig(
                "Unsupported Whisper preprocessing geometry: " + "; ".join(bad)
            )
        return cls()

    @classmethod
    def from_assets(cls, assets: WhisperAssets) -> "WhisperPreprocessorConfig":
        return cls.from_dict(assets.load_json("preprocessor_config.json"))


__all__ = [
    "UnsupportedWhisperConfig",
    "WhisperPreprocessorConfig",
    "WhisperTurboConfig",
]
