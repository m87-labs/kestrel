"""Transcript quality metrics and fallback policy."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence


DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MAX_BEST_OF = 8


@dataclass(frozen=True, slots=True)
class TranscriptionQualityPolicy:
    """Validated decode retries and absolute transcript-quality thresholds."""

    temperatures: tuple[float, ...]
    best_of: int
    compression_ratio_threshold: float | None
    logprob_threshold: float | None
    no_speech_threshold: float | None

    def candidate_count(self, temperature: float) -> int:
        """Return the bounded candidate count for one fallback temperature."""

        return self.best_of if temperature > 0.0 else 1

    def is_silence(self, *, avg_logprob: float, no_speech_prob: float) -> bool:
        return (
            self.no_speech_threshold is not None
            and self.logprob_threshold is not None
            and no_speech_prob > self.no_speech_threshold
            and avg_logprob < self.logprob_threshold
        )

    def needs_fallback(
        self,
        *,
        avg_logprob: float,
        compression_ratio: float,
    ) -> bool:
        return (
            self.compression_ratio_threshold is not None
            and compression_ratio > self.compression_ratio_threshold
        ) or (
            self.logprob_threshold is not None and avg_logprob < self.logprob_threshold
        )


def _optional_finite_setting(
    settings: Mapping[str, object],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = settings.get(name, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"settings.{name} must be a finite real number or None")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"settings.{name} is outside its supported range")
    return result


def parse_quality_policy(
    settings: Mapping[str, object] | None,
) -> TranscriptionQualityPolicy:
    """Parse the public Whisper retry policy without mutating caller settings."""

    if settings is not None and not isinstance(settings, Mapping):
        raise TypeError("transcribe settings must be a mapping or None")
    if settings is not None and any(not isinstance(name, str) for name in settings):
        raise TypeError("transcribe setting names must be strings")
    values = {} if settings is None else settings
    raw_temperatures = values.get("temperature", DEFAULT_TEMPERATURES)
    if isinstance(raw_temperatures, bool):
        raise TypeError("settings.temperature must be a real number or sequence")
    if isinstance(raw_temperatures, Real):
        candidates: Sequence[object] = (raw_temperatures,)
    elif isinstance(raw_temperatures, Sequence) and not isinstance(
        raw_temperatures, (str, bytes, bytearray)
    ):
        candidates = raw_temperatures
    else:
        raise TypeError("settings.temperature must be a real number or sequence")
    if not candidates:
        raise ValueError("settings.temperature schedule must not be empty")
    temperatures = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("settings.temperature values must be finite real numbers")
        temperature = float(value)
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 1.0:
            raise ValueError("settings.temperature values must lie in [0, 1]")
        if temperatures and temperature <= temperatures[-1]:
            raise ValueError(
                "settings.temperature schedule must be strictly increasing"
            )
        temperatures.append(temperature)
    raw_best_of = values.get(
        "best_of",
        5 if any(temperature > 0.0 for temperature in temperatures) else 1,
    )
    if isinstance(raw_best_of, bool) or not isinstance(raw_best_of, Integral):
        raise TypeError("settings.best_of must be a positive integer")
    best_of = int(raw_best_of)
    if not 1 <= best_of <= MAX_BEST_OF:
        raise ValueError(f"settings.best_of must lie in [1, {MAX_BEST_OF}]")
    if best_of > 1 and not any(temperature > 0.0 for temperature in temperatures):
        raise ValueError("settings.best_of greater than one requires sampling")
    return TranscriptionQualityPolicy(
        temperatures=tuple(temperatures),
        best_of=best_of,
        compression_ratio_threshold=_optional_finite_setting(
            values,
            "compression_ratio_threshold",
            2.4,
            minimum=0.0,
        ),
        logprob_threshold=_optional_finite_setting(
            values,
            "logprob_threshold",
            -1.0,
            maximum=0.0,
        ),
        no_speech_threshold=_optional_finite_setting(
            values,
            "no_speech_threshold",
            0.6,
            minimum=0.0,
            maximum=1.0,
        ),
    )


def compression_ratio(text: str) -> float:
    """Return UTF-8 size divided by its zlib-compressed size."""

    if not isinstance(text, str):
        raise TypeError("transcript text must be a string")
    encoded = text.encode("utf-8")
    return len(encoded) / len(zlib.compress(encoded)) if encoded else 0.0


__all__ = [
    "DEFAULT_TEMPERATURES",
    "MAX_BEST_OF",
    "TranscriptionQualityPolicy",
    "compression_ratio",
    "parse_quality_policy",
]
