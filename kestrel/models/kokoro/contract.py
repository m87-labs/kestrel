"""Kokoro synthesis request validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


_PROMPT_KEYS = frozenset({"text", "language", "voice", "speed", "stream"})


@dataclass(frozen=True, slots=True)
class KokoroSynthesisRequest:
    text: str
    language: str = "en-us"
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
        unknown = set(prompt) - _PROMPT_KEYS
        if unknown:
            raise ValueError(f"unsupported synthesize inputs: {sorted(unknown)}")
        return cls(
            text=prompt.get("text"),  # type: ignore[arg-type]
            language=prompt.get("language", "en-us"),  # type: ignore[arg-type]
            voice=prompt.get("voice", "af_heart"),  # type: ignore[arg-type]
            speed=prompt.get("speed", 1.0),  # type: ignore[arg-type]
            stream=prompt.get("stream", False),  # type: ignore[arg-type]
        )

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        for name in ("language", "voice"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise TypeError("speed must be a number")
        speed = float(self.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        object.__setattr__(self, "speed", speed)
        if type(self.stream) is not bool:
            raise TypeError("stream must be a boolean")


@dataclass(frozen=True, slots=True)
class PreparedKokoroSynthesis:
    """Validated leaf payload produced before scheduler admission."""

    request: KokoroSynthesisRequest
    phonemes: str

    def __post_init__(self) -> None:
        if not isinstance(self.phonemes, str) or not self.phonemes:
            raise ValueError("prepared Kokoro phonemes must be a non-empty string")


__all__ = ["KokoroSynthesisRequest"]
