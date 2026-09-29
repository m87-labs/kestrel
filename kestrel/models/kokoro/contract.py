"""Kokoro synthesis request validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


_PROMPT_KEYS = frozenset({"phonemes", "voice", "speed", "stream"})


@dataclass(frozen=True, slots=True)
class KokoroSynthesisRequest:
    phonemes: str
    voice: str = "af_heart"
    speed: float = 1.0
    stream: bool = False

    @classmethod
    def from_mapping(
        cls, prompt: Mapping[str, object]
    ) -> "KokoroSynthesisRequest":
        if not isinstance(prompt, Mapping):
            raise TypeError("synthesize inputs must be a mapping")
        if "speaker" in prompt:
            raise ValueError("use voice, not speaker, for speech synthesis")
        if "text" in prompt:
            raise ValueError(
                "Kokoro requires phonemes, not text; phonemize text before synthesis"
            )
        unknown = set(prompt) - _PROMPT_KEYS
        if unknown:
            raise ValueError(f"unsupported synthesize inputs: {sorted(unknown)}")
        return cls(
            phonemes=prompt.get("phonemes"),  # type: ignore[arg-type]
            voice=prompt.get("voice", "af_heart"),  # type: ignore[arg-type]
            speed=prompt.get("speed", 1.0),  # type: ignore[arg-type]
            stream=prompt.get("stream", False),  # type: ignore[arg-type]
        )

    def __post_init__(self) -> None:
        if not isinstance(self.phonemes, str) or not self.phonemes.strip():
            raise ValueError("phonemes must be a non-empty string")
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ValueError("voice must be a non-empty string")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise TypeError("speed must be a number")
        speed = float(self.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        object.__setattr__(self, "speed", speed)
        if type(self.stream) is not bool:
            raise TypeError("stream must be a boolean")


__all__ = ["KokoroSynthesisRequest"]
