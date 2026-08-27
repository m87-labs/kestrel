"""The common ``transcribe`` contract used by Kestrel ASR models.

Model-specific tokenization and decoding deliberately do not live here. Qwen3-ASR
is a causal language model while Parakeet is a duration transducer; sharing either
decode loop would obscure rather than reduce complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Mapping


TimestampMode = Literal["none", "segment", "word", "character"]
TranscriptionTask = Literal["transcribe", "translate"]


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "word": self.text,
            "start": self.start,
            "end": self.end,
        }
        if self.probability is not None:
            value["probability"] = self.probability
        return value


@dataclass(frozen=True, slots=True)
class Character:
    text: str
    start: float
    end: float

    def as_dict(self) -> dict[str, object]:
        return {
            "character": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()
    characters: tuple[Character, ...] = ()

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
        if self.words:
            value["words"] = [word.as_dict() for word in self.words]
        if self.characters:
            value["characters"] = [item.as_dict() for item in self.characters]
        return value


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float
    source_duration_seconds: float
    clip_start_seconds: float
    segments: tuple[Segment, ...] = ()
    task: TranscriptionTask = "transcribe"
    language_probability: float | None = None

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "text": self.text,
            "language": self.language,
            "task": self.task,
            "duration_seconds": self.duration_seconds,
            "source_duration_seconds": self.source_duration_seconds,
            "clip_start_seconds": self.clip_start_seconds,
            "clip_end_seconds": self.clip_start_seconds + self.duration_seconds,
            "segments": [segment.as_dict() for segment in self.segments],
        }
        if self.language_probability is not None:
            value["language_probability"] = self.language_probability
        return value


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: object
    sample_rate: int | None = None
    language: str | None = None
    initial_prompt: str | None = None
    condition_on_previous_text: bool = True
    stream: bool = False
    task: TranscriptionTask = "transcribe"
    timestamps: TimestampMode = "segment"
    clip_start_seconds: float = 0.0
    clip_end_seconds: float | None = None

    @classmethod
    def from_prompt(cls, prompt: Mapping[str, object]) -> "TranscriptionRequest":
        if not isinstance(prompt, Mapping):
            raise TypeError("transcribe prompt must be a mapping")
        if any(not isinstance(name, str) for name in prompt):
            raise TypeError("transcribe prompt field names must be strings")
        fields = {
            "audio",
            "sample_rate",
            "language",
            "initial_prompt",
            "condition_on_previous_text",
            "stream",
            "task",
            "timestamps",
            "clip_start_seconds",
            "clip_end_seconds",
        }
        unknown = sorted(set(prompt) - fields)
        if unknown:
            raise ValueError(f"Unsupported transcribe option(s): {', '.join(unknown)}")
        if "audio" not in prompt:
            raise ValueError("audio must be provided for transcription")

        language = prompt.get("language")
        if language is not None and (
            not isinstance(language, str) or not language.strip()
        ):
            raise TypeError("language must be a non-empty string or None")
        initial_prompt = prompt.get("initial_prompt")
        if initial_prompt is not None and (
            not isinstance(initial_prompt, str) or not initial_prompt.strip()
        ):
            raise TypeError("initial_prompt must be a non-empty string or None")
        condition = prompt.get("condition_on_previous_text", True)
        stream = prompt.get("stream", False)
        if not isinstance(condition, bool):
            raise TypeError("condition_on_previous_text must be a boolean")
        if not isinstance(stream, bool):
            raise TypeError("stream must be a boolean")
        task = prompt.get("task", "transcribe")
        if task not in ("transcribe", "translate"):
            raise ValueError("task must be 'transcribe' or 'translate'")
        timestamps = prompt.get("timestamps", "segment")
        if timestamps not in ("none", "segment", "word", "character"):
            raise ValueError(
                "timestamps must be 'none', 'segment', 'word', or 'character'"
            )
        start = _seconds(prompt.get("clip_start_seconds", 0.0), "clip_start_seconds")
        end_value = prompt.get("clip_end_seconds")
        end = None if end_value is None else _seconds(end_value, "clip_end_seconds")
        if start < 0.0:
            raise ValueError("clip_start_seconds must be non-negative")
        if end is not None and end <= start:
            raise ValueError("clip_end_seconds must be greater than clip_start_seconds")

        sample_rate = prompt.get("sample_rate")
        if sample_rate is not None:
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral):
                raise TypeError("sample_rate must be a positive integer or None")
            if sample_rate <= 0:
                raise ValueError("sample_rate must be a positive integer or None")
            if sample_rate > 2**32 - 1:
                raise ValueError("sample_rate must fit the unsigned 32-bit contract")
        return cls(
            audio=prompt["audio"],
            sample_rate=int(sample_rate) if sample_rate is not None else None,
            language=language.strip() if isinstance(language, str) else None,
            initial_prompt=initial_prompt.strip()
            if isinstance(initial_prompt, str)
            else None,
            condition_on_previous_text=condition,
            stream=stream,
            task=task,  # type: ignore[arg-type]
            timestamps=timestamps,  # type: ignore[arg-type]
            clip_start_seconds=start,
            clip_end_seconds=end,
        )


@dataclass(frozen=True, slots=True)
class DecodeSettings:
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0

    @classmethod
    def from_mapping(cls, settings: object) -> "DecodeSettings":
        if settings is None:
            return cls()
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping or None")
        unknown = sorted(set(settings) - {"max_tokens", "temperature", "top_p"})
        if unknown:
            raise ValueError(
                f"Unsupported transcription setting(s): {', '.join(unknown)}"
            )
        max_tokens = settings.get("max_tokens", 4096)
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise TypeError("settings.max_tokens must be a positive integer")
        temperature = _number(settings.get("temperature", 0.0), "settings.temperature")
        top_p = _number(settings.get("top_p", 1.0), "settings.top_p")
        if temperature < 0:
            raise ValueError("settings.temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("settings.top_p must lie in (0, 1]")
        return cls(max_tokens, temperature, top_p)


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be a finite number")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be a finite number")
    return result


__all__ = [
    "Character",
    "Segment",
    "DecodeSettings",
    "TimestampMode",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionTask",
    "Word",
]
