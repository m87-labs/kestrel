"""Small request values shared by the Qwen3-TTS skill and runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .config import CODE_PREDICTOR_VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class Qwen3TTSSampling:
    """Validated CustomVoice sampling controls."""

    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0

    def __post_init__(self) -> None:
        for name in ("do_sample", "subtalker_dosample"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        for name in ("top_k", "subtalker_top_k"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "temperature",
            "top_p",
            "repetition_penalty",
            "subtalker_temperature",
            "subtalker_top_p",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0.0 or self.subtalker_temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0 or not 0.0 < self.subtalker_top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if self.subtalker_dosample and not (
            0 <= self.subtalker_top_k <= CODE_PREDICTOR_VOCAB_SIZE
        ):
            raise ValueError(
                "subtalker_top_k must lie in [0, 2048] when sampling"
            )

    @property
    def talker_temperature(self) -> float:
        return self.temperature if self.do_sample else 0.0

    @property
    def subtalker_parameters(self) -> tuple[float, int, float]:
        if not self.subtalker_dosample:
            return 0.0, 1, 1.0
        return (
            self.subtalker_temperature,
            self.subtalker_top_k,
            self.subtalker_top_p,
        )


DEFAULT_SAMPLING = Qwen3TTSSampling()


@dataclass(frozen=True, slots=True)
class CustomVoiceRequest:
    text: str
    voice: str = "ryan"
    language: str = "auto"
    instructions: str | None = None
    stream: bool = False
    sampling: Qwen3TTSSampling = DEFAULT_SAMPLING

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        for name in ("voice", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError("instructions must be a string or None")
        if type(self.stream) is not bool:
            raise TypeError("stream must be a boolean")
        if not isinstance(self.sampling, Qwen3TTSSampling):
            raise TypeError("sampling must be Qwen3TTSSampling")

    @property
    def voice_key(self) -> str:
        return self.voice.strip().casefold()

    @property
    def language_key(self) -> str:
        return self.language.strip().casefold()


@dataclass(frozen=True, slots=True)
class EncodedCustomVoiceRequest:
    request: CustomVoiceRequest
    text_token_ids: tuple[int, ...]
    instruction_token_ids: tuple[int, ...]


__all__ = [
    "CustomVoiceRequest",
    "DEFAULT_SAMPLING",
    "EncodedCustomVoiceRequest",
    "Qwen3TTSSampling",
]
