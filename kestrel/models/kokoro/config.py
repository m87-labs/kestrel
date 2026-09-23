"""Pinned Kokoro-82M inference configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AlbertConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    num_hidden_layers: int
    embedding_size: int = 128
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


@dataclass(frozen=True, slots=True)
class IstftNetConfig:
    upsample_kernel_sizes: tuple[int, ...]
    upsample_rates: tuple[int, ...]
    gen_istft_hop_size: int
    gen_istft_n_fft: int
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    resblock_kernel_sizes: tuple[int, ...]
    upsample_initial_channel: int


@dataclass(frozen=True, slots=True)
class KokoroConfig:
    vocab: Mapping[str, int]
    n_token: int
    hidden_dim: int
    style_dim: int
    n_layer: int
    max_dur: int
    n_mels: int
    text_encoder_kernel_size: int
    albert: AlbertConfig
    istftnet: IstftNetConfig

    @classmethod
    def from_json_file(cls, path: str | Path) -> "KokoroConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise TypeError("Kokoro config must contain a JSON object")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KokoroConfig":
        plbert = _mapping(raw, "plbert")
        istft = _mapping(raw, "istftnet")
        vocab = _mapping(raw, "vocab")
        config = cls(
            vocab={str(symbol): _integer(index, f"vocab[{symbol!r}]") for symbol, index in vocab.items()},
            n_token=_integer(raw.get("n_token"), "n_token"),
            hidden_dim=_integer(raw.get("hidden_dim"), "hidden_dim"),
            style_dim=_integer(raw.get("style_dim"), "style_dim"),
            n_layer=_integer(raw.get("n_layer"), "n_layer"),
            max_dur=_integer(raw.get("max_dur"), "max_dur"),
            n_mels=_integer(raw.get("n_mels"), "n_mels"),
            text_encoder_kernel_size=_integer(
                raw.get("text_encoder_kernel_size"), "text_encoder_kernel_size"
            ),
            albert=AlbertConfig(
                vocab_size=_integer(raw.get("n_token"), "n_token"),
                hidden_size=_integer(plbert.get("hidden_size"), "plbert.hidden_size"),
                num_attention_heads=_integer(
                    plbert.get("num_attention_heads"), "plbert.num_attention_heads"
                ),
                intermediate_size=_integer(
                    plbert.get("intermediate_size"), "plbert.intermediate_size"
                ),
                max_position_embeddings=_integer(
                    plbert.get("max_position_embeddings"),
                    "plbert.max_position_embeddings",
                ),
                num_hidden_layers=_integer(
                    plbert.get("num_hidden_layers"), "plbert.num_hidden_layers"
                ),
            ),
            istftnet=IstftNetConfig(
                upsample_kernel_sizes=_int_tuple(
                    istft.get("upsample_kernel_sizes"),
                    "istftnet.upsample_kernel_sizes",
                ),
                upsample_rates=_int_tuple(
                    istft.get("upsample_rates"), "istftnet.upsample_rates"
                ),
                gen_istft_hop_size=_integer(
                    istft.get("gen_istft_hop_size"),
                    "istftnet.gen_istft_hop_size",
                ),
                gen_istft_n_fft=_integer(
                    istft.get("gen_istft_n_fft"), "istftnet.gen_istft_n_fft"
                ),
                resblock_dilation_sizes=tuple(
                    _int_tuple(values, "istftnet.resblock_dilation_sizes")
                    for values in _sequence(
                        istft.get("resblock_dilation_sizes"),
                        "istftnet.resblock_dilation_sizes",
                    )
                ),
                resblock_kernel_sizes=_int_tuple(
                    istft.get("resblock_kernel_sizes"),
                    "istftnet.resblock_kernel_sizes",
                ),
                upsample_initial_channel=_integer(
                    istft.get("upsample_initial_channel"),
                    "istftnet.upsample_initial_channel",
                ),
            ),
        )
        config.validate_v1()
        return config

    def validate_v1(self) -> None:
        """Reject configs that do not describe the pinned V1 checkpoint."""

        expected = {
            "n_token": 178,
            "hidden_dim": 512,
            "style_dim": 128,
            "n_layer": 3,
            "max_dur": 50,
            "n_mels": 80,
            "text_encoder_kernel_size": 5,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(f"unsupported Kokoro config: {actual}, expected {expected}")
        if len(self.vocab) == 0 or max(self.vocab.values(), default=-1) >= self.n_token:
            raise ValueError("Kokoro vocabulary indices must fit n_token")
        if self.albert != AlbertConfig(178, 768, 12, 2048, 512, 12):
            raise ValueError(f"unsupported Kokoro ALBERT config: {self.albert}")
        expected_istft = IstftNetConfig(
            upsample_kernel_sizes=(20, 12),
            upsample_rates=(10, 6),
            gen_istft_hop_size=5,
            gen_istft_n_fft=20,
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kernel_sizes=(3, 7, 11),
            upsample_initial_channel=512,
        )
        if self.istftnet != expected_istft:
            raise ValueError(f"unsupported Kokoro ISTFTNet config: {self.istftnet}")


def _mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    return tuple(_integer(item, name) for item in _sequence(value, name))


__all__ = ["AlbertConfig", "IstftNetConfig", "KokoroConfig"]
