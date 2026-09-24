"""Configuration for the supported Qwen3-TTS CustomVoice checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
SUPPORTED_CHECKPOINTS: dict[str, str] = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": (
        "85e237c12c027371202489a0ec509ded67b5e4b5"
    ),
    DEFAULT_REPO_ID: DEFAULT_REVISION,
}
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1_920
NUM_CODE_GROUPS = 16
CODE_PREDICTOR_VOCAB_SIZE = 2_048
DEFAULT_MAX_TOKENS = 2_048
_TALKER_DIMENSIONS = {
    "0b6": (1024, 3072),
    "1b7": (2048, 6144),
}


@dataclass(frozen=True, slots=True)
class Qwen3TTSCodePredictorConfig:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = CODE_PREDICTOR_VOCAB_SIZE
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    num_code_groups: int = NUM_CODE_GROUPS
    attention_bias: bool = False


@dataclass(frozen=True, slots=True)
class Qwen3TTSTalkerConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 3072
    text_vocab_size: int = 151936
    text_hidden_size: int = 2048
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    num_code_groups: int = NUM_CODE_GROUPS
    attention_bias: bool = False
    rope_scaling: dict[str, object] = field(
        default_factory=lambda: {
            "interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "type": "default",
        }
    )
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_language_id: dict[str, int] = field(default_factory=dict)
    spk_id: dict[str, int] = field(default_factory=dict)
    spk_is_dialect: dict[str, str | bool] = field(default_factory=dict)
    code_predictor: Qwen3TTSCodePredictorConfig = field(
        default_factory=Qwen3TTSCodePredictorConfig
    )


@dataclass(frozen=True, slots=True)
class Qwen3TTSCodecConfig:
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    max_position_embeddings: int = 8000
    num_quantizers: int = NUM_CODE_GROUPS
    sliding_window: int = 72
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: tuple[int, ...] = (2, 2)
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    layer_scale_initial_scale: float = 0.01


_PREDICTOR_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
    "rms_norm_eps", "rope_theta", "num_code_groups", "attention_bias",
)
_TALKER_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "text_vocab_size",
    "text_hidden_size", "max_position_embeddings", "rms_norm_eps", "rope_theta",
    "num_code_groups", "attention_bias", "rope_scaling", "codec_pad_id",
    "codec_bos_id", "codec_eos_token_id", "codec_think_id", "codec_nothink_id",
    "codec_think_bos_id", "codec_think_eos_id",
)
_CODEC_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "latent_dim", "codebook_dim", "codebook_size",
    "decoder_dim", "max_position_embeddings", "num_quantizers", "sliding_window",
    "rms_norm_eps", "rope_theta", "hidden_act", "attention_bias", "attention_dropout",
    "layer_scale_initial_scale",
)


@dataclass(frozen=True, slots=True)
class Qwen3TTSConfig:
    talker: Qwen3TTSTalkerConfig
    codec: Qwen3TTSCodecConfig = field(default_factory=Qwen3TTSCodecConfig)
    model_size: str = "1b7"
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673

    @property
    def code_predictor(self) -> Qwen3TTSCodePredictorConfig:
        return self.talker.code_predictor

    @property
    def num_code_groups(self) -> int:
        return self.talker.num_code_groups

    @property
    def generated_name(self) -> str:
        major, minor = self.model_size.split("b", 1)
        return f"qwen3_tts_{major}_{minor}b_custom_voice"

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Qwen3TTSConfig":
        root = Path(directory)
        return cls.from_dicts(
            _read_json(root / "config.json"),
            _read_json(root / "speech_tokenizer" / "config.json"),
        )

    @classmethod
    def from_dicts(
        cls, model: Mapping[str, Any], codec: Mapping[str, Any]
    ) -> "Qwen3TTSConfig":
        _require(model, "model", model_type="qwen3_tts", tts_model_type="custom_voice",
                 tokenizer_type="qwen3_tts_tokenizer_12hz")
        model_size = model.get("tts_model_size")
        if not isinstance(model_size, str) or model_size not in _TALKER_DIMENSIONS:
            raise ValueError(
                f"unsupported Qwen3-TTS model.tts_model_size: {model_size!r}"
            )
        talker_dimensions = _TALKER_DIMENSIONS[model_size]
        talker_raw = _child(model, "talker_config")
        predictor_raw = _child(talker_raw, "code_predictor_config")
        predictor = Qwen3TTSCodePredictorConfig()
        _require_fixed(predictor_raw, predictor, _PREDICTOR_FIELDS, "code predictor")
        talker = Qwen3TTSTalkerConfig(
            hidden_size=talker_dimensions[0],
            intermediate_size=talker_dimensions[1],
            code_predictor=predictor,
        )
        _require_fixed(talker_raw, talker, _TALKER_FIELDS, "talker")
        talker = replace(
            talker,
            codec_language_id=_int_map(talker_raw, "codec_language_id"),
            spk_id=_int_map(talker_raw, "spk_id"),
            spk_is_dialect=_dialect_map(talker_raw),
        )
        _validate_domains(talker)

        decoder_raw = _child(codec, "decoder_config")
        codec_config = Qwen3TTSCodecConfig()
        _require(codec, "codec", model_type="qwen3_tts_tokenizer_12hz",
                 output_sample_rate=SAMPLE_RATE, decode_upsample_rate=SAMPLES_PER_FRAME)
        _require_fixed(decoder_raw, codec_config, _CODEC_FIELDS, "codec decoder")
        _require(decoder_raw, "codec decoder", upsample_rates=list(codec_config.upsample_rates),
                 upsampling_ratios=list(codec_config.upsampling_ratios))
        result = cls(talker=talker, codec=codec_config, model_size=model_size)
        _require(model, "model", tts_pad_token_id=result.tts_pad_token_id,
                 tts_bos_token_id=result.tts_bos_token_id,
                 tts_eos_token_id=result.tts_eos_token_id)
        return result


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Qwen3-TTS config not found: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Qwen3-TTS config: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"Qwen3-TTS config must be a mapping: {path}")
    return value


def _child(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"Qwen3-TTS {name} must be a mapping")
    return value


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (not isinstance(actual, bool) and isinstance(actual, (int, float))
                and float(actual) == expected)
    return type(actual) is type(expected) and actual == expected


def _require(values: Mapping[str, Any], path: str, **expected: object) -> None:
    for name, wanted in expected.items():
        actual = values.get(name)
        if not _matches(actual, wanted):
            raise ValueError(
                f"unsupported Qwen3-TTS {path}.{name}: {actual!r}; expected {wanted!r}"
            )


def _require_fixed(
    values: Mapping[str, Any], config: object, names: tuple[str, ...], path: str
) -> None:
    _require(values, path, **{name: getattr(config, name) for name in names})


def _int_map(values: Mapping[str, Any], name: str) -> dict[str, int]:
    raw = _child(values, name)
    if any(not isinstance(key, str) or type(value) is not int for key, value in raw.items()):
        raise TypeError(f"Qwen3-TTS {name} must map strings to integers")
    return dict(raw)


def _dialect_map(values: Mapping[str, Any]) -> dict[str, str | bool]:
    raw = _child(values, "spk_is_dialect")
    if any(not isinstance(key, str) or not isinstance(value, (str, bool))
           for key, value in raw.items()):
        raise TypeError("Qwen3-TTS spk_is_dialect has invalid entries")
    return dict(raw)


def _validate_domains(config: Qwen3TTSTalkerConfig) -> None:
    if not config.spk_id or set(config.spk_id) != set(config.spk_is_dialect):
        raise ValueError("Qwen3-TTS speaker maps are empty or inconsistent")
    if not config.codec_language_id:
        raise ValueError("Qwen3-TTS language map is empty")
    dialects = {value for value in config.spk_is_dialect.values() if isinstance(value, str)}
    if not dialects <= set(config.codec_language_id):
        raise ValueError("Qwen3-TTS speaker map references an unknown dialect")
    ids = (*config.spk_id.values(), *config.codec_language_id.values())
    if len(ids) != len(set(ids)) or any(token < 0 or token >= config.vocab_size for token in ids):
        raise ValueError("Qwen3-TTS speaker/language IDs collide or exceed the vocabulary")
