"""Bounded, timestamp-driven orchestration for Whisper files and live PCM."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence, cast

import kestrel_native
import numpy as np
from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.models.asr.audio import snapshot_file_like
from kestrel.models.asr.contract import TranscriptionRequest
from kestrel.models.asr.longform import settled_to_thread
from kestrel.runtime.tokens import TextToken
from kestrel.skills import CapabilityInvoker, CapabilityOrchestrator
from torch import Tensor

from .audio import (
    _MAX_NATIVE_AUDIO_VALUES,
    _pcm_to_float32,
    _raw_shape_and_size,
    validate_audio_source,
)
from .quality import (
    TranscriptionQualityPolicy,
    compression_ratio,
    parse_quality_policy,
)
from .tokenizer import _LANGUAGES_WITHOUT_SPACES


_WINDOW_SECONDS = 30
_MAX_FILE_DURATION_SECONDS = 24 * 60 * 60
_MAX_NATIVE_READ_FRAMES = 1024 * 1024
_LIVE_UPDATE_SECONDS = 5
_LIVE_STABILITY_SECONDS = 1
_TIMESTAMP_BEGIN_ID = 50365
_TIMESTAMP_VOCAB_END = 51866
_TIMESTAMP_SECONDS = 0.02


def _checked_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"incremental audio reader returned invalid {name}")
    return value


class _NativeFileWindows:
    """One bounded source-rate ring over a native incremental reader."""

    def __init__(
        self,
        reader: object,
        *,
        clip_start_seconds: float = 0.0,
        clip_end_seconds: float | None = None,
    ) -> None:
        self.reader = reader
        self.sample_rate = _checked_positive_int(
            getattr(reader, "sample_rate", None), "sample_rate"
        )
        self.source_total_frames = _checked_positive_int(
            getattr(reader, "total_frames", None), "total_frames"
        )
        self.clip_start_frame = int(round(clip_start_seconds * self.sample_rate))
        if self.clip_start_frame >= self.source_total_frames:
            raise ValueError(
                "clip_start_seconds lies at or beyond the end of the audio"
            )
        self.clip_end_frame = (
            self.source_total_frames
            if clip_end_seconds is None
            else min(
                self.source_total_frames,
                int(round(clip_end_seconds * self.sample_rate)),
            )
        )
        if self.clip_end_frame <= self.clip_start_frame:
            raise ValueError("the selected audio clip contains no samples")
        self.total_frames = self.clip_end_frame - self.clip_start_frame
        self.window_frames = min(
            self.sample_rate * _WINDOW_SECONDS,
            _MAX_NATIVE_AUDIO_VALUES,
        )
        self.offset_frames = self.clip_start_frame
        self._decoded_frames = 0
        self._buffer = np.empty(0, dtype=np.float32)
        self._eof = False
        self._seek_pending = self.clip_start_frame > 0

    @property
    def source_duration_seconds(self) -> float:
        return self.source_total_frames / self.sample_rate

    @property
    def clip_start_seconds(self) -> float:
        return self.clip_start_frame / self.sample_rate

    @property
    def clip_end_seconds(self) -> float:
        return self.clip_end_frame / self.sample_rate

    @property
    def completed_frames(self) -> int:
        return self.offset_frames - self.clip_start_frame

    def _read(self, wanted: int) -> np.ndarray | None:
        chunk = self.reader.read(wanted)
        if chunk is None:
            self._eof = True
            if self._decoded_frames != self.source_total_frames:
                raise ValueError(
                    "incremental audio reader ended before its declared frame count"
                )
            return None
        if (
            not isinstance(chunk, np.ndarray)
            or chunk.dtype != np.float32
            or chunk.ndim != 1
            or chunk.size == 0
            or chunk.size > wanted
            or not chunk.flags.c_contiguous
        ):
            raise ValueError("incremental audio reader returned an invalid chunk")
        next_decoded = self._decoded_frames + int(chunk.size)
        if next_decoded > self.source_total_frames:
            raise ValueError(
                "incremental audio reader exceeded its declared frame count"
            )
        if not np.isfinite(chunk).all() or np.max(np.abs(chunk)) > 1.000001:
            raise ValueError("incremental audio reader returned invalid samples")
        self._decoded_frames = next_decoded
        return chunk

    def current(self) -> np.ndarray | None:
        if self._seek_pending:
            seek = getattr(self.reader, "seek", None)
            if callable(seek):
                position = seek(self.clip_start_frame)
                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position != self.clip_start_frame
                ):
                    raise ValueError(
                        "incremental audio reader returned an invalid seek position"
                    )
                self._decoded_frames = position
            self._seek_pending = False
        while self._decoded_frames < self.clip_start_frame and not self._eof:
            wanted = min(
                self.clip_start_frame - self._decoded_frames,
                _MAX_NATIVE_READ_FRAMES,
            )
            self._read(wanted)
        buffered_frames = int(self._buffer.size)
        parts = [self._buffer] if buffered_frames else []
        while (
            buffered_frames < self.window_frames
            and self._decoded_frames < self.clip_end_frame
            and not self._eof
        ):
            wanted = min(
                self.window_frames - buffered_frames,
                self.clip_end_frame - self._decoded_frames,
                _MAX_NATIVE_READ_FRAMES,
            )
            chunk = self._read(wanted)
            if chunk is None:
                break
            parts.append(chunk)
            buffered_frames += int(chunk.size)
        if not parts:
            return None
        self._buffer = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return self._buffer

    def advance(self, frames: int) -> None:
        if not 0 < frames <= self._buffer.size:
            raise ValueError("long-form window advance is outside the buffered audio")
        self.offset_frames += frames
        self._buffer = self._buffer[frames:]


class _LivePcmBuffer:
    """Bounded snapshots over caller-owned live mono PCM chunks."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.window_frames = min(
            sample_rate * _WINDOW_SECONDS,
            _MAX_NATIVE_AUDIO_VALUES,
        )
        self.update_frames = min(
            sample_rate * _LIVE_UPDATE_SECONDS,
            self.window_frames,
        )
        self.max_total_frames = sample_rate * _MAX_FILE_DURATION_SECONDS
        self.received_frames = 0
        self.consumed_frames = 0
        self.buffered_frames = 0
        self._start = 0
        self._storage = np.empty(
            self.window_frames + _MAX_NATIVE_READ_FRAMES,
            dtype=np.float32,
        )

    def append(self, chunk: object) -> None:
        if not isinstance(chunk, (np.ndarray, Tensor)):
            raise TypeError(
                "live audio chunks must be one-dimensional NumPy or CPU Torch PCM"
            )
        shape, size = _raw_shape_and_size(chunk)  # type: ignore[arg-type]
        if len(shape) != 1:
            raise ValueError(
                f"live audio chunks must be one-dimensional mono PCM, got shape {shape}"
            )
        if size == 0:
            raise ValueError("live audio chunks must not be empty")
        if size > _MAX_NATIVE_READ_FRAMES:
            raise ValueError(
                f"live audio chunk exceeds the {_MAX_NATIVE_READ_FRAMES}-value limit"
            )
        next_received = self.received_frames + size
        if next_received > self.max_total_frames:
            raise ValueError(
                f"live audio exceeds the {_MAX_FILE_DURATION_SECONDS}-second limit"
            )
        # The producer may immediately reuse its capture buffer after yielding.
        # Copy into one bounded store before suspension; keeping one array also
        # prevents tiny chunks from turning a sample bound into unbounded object
        # overhead.
        waveform = _pcm_to_float32(chunk)  # type: ignore[arg-type]
        end = self._start + self.buffered_frames
        if end + size > self._storage.size:
            self._storage[: self.buffered_frames] = self._storage[
                self._start : end
            ].copy()
            self._start = 0
            end = self.buffered_frames
        self._storage[end : end + size] = waveform
        self.received_frames = next_received
        self.buffered_frames += size

    def current(self) -> np.ndarray:
        wanted = min(self.buffered_frames, self.window_frames)
        if wanted <= 0:
            raise RuntimeError("live audio buffer is empty")
        return self._storage[self._start : self._start + wanted]

    def advance(self, frames: int) -> None:
        if not 0 < frames <= self.buffered_frames:
            raise ValueError("live audio advance is outside the buffered PCM")
        self._start += frames
        self.buffered_frames -= frames
        self.consumed_frames += frames
        if self.buffered_frames == 0:
            self._start = 0


@dataclass(frozen=True, slots=True)
class _CheckedLeaf:
    result: EngineResult
    language: str
    language_probability: float | None
    text: str
    segments: tuple[dict[str, object], ...]
    token_ids: tuple[int, ...]
    text_token_ids: tuple[int, ...]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass(slots=True)
class _LeafAccumulator:
    """Aggregate leaf metrics and selected transcript diagnostics once."""

    count: int = 0
    request_id: object | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    ttft_ms: float = 0.0
    cached_tokens: int = 0
    last_finish_reason: str = "stop"
    first_non_stop_reason: str | None = None
    score_weight: int = 0
    weighted_logprob: float = 0.0
    temperature: float = 0.0
    no_speech_prob: float = 0.0
    language_probability: float | None = None

    def add(self, leaf: _CheckedLeaf) -> None:
        metrics = leaf.result.metrics
        if self.count == 0:
            self.request_id = leaf.result.request_id
            self.ttft_ms = metrics.ttft_ms
        self.count += 1
        self.input_tokens += metrics.input_tokens
        self.output_tokens += metrics.output_tokens
        self.prefill_time_ms += metrics.prefill_time_ms
        self.decode_time_ms += metrics.decode_time_ms
        self.cached_tokens += metrics.cached_tokens
        self.last_finish_reason = leaf.result.finish_reason
        if self.first_non_stop_reason is None and leaf.result.finish_reason != "stop":
            self.first_non_stop_reason = leaf.result.finish_reason

        weight = max(1, len(leaf.text_token_ids))
        self.score_weight += weight
        self.weighted_logprob += leaf.avg_logprob * weight
        self.temperature = max(self.temperature, leaf.temperature)
        self.no_speech_prob = max(self.no_speech_prob, leaf.no_speech_prob)
        if self.language_probability is None:
            self.language_probability = leaf.language_probability

    def metrics(self) -> EngineMetrics:
        return EngineMetrics(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            prefill_time_ms=self.prefill_time_ms,
            decode_time_ms=self.decode_time_ms,
            ttft_ms=self.ttft_ms,
            cached_tokens=self.cached_tokens,
        )

    def output(
        self,
        text_parts: list[str],
        *,
        language: str,
        task: str,
        provisional_leaf: _CheckedLeaf | None = None,
    ) -> dict[str, float | None]:
        if self.count == 0 and provisional_leaf is None:
            raise RuntimeError("transcript diagnostics require at least one leaf")
        weight = self.score_weight
        weighted_logprob = self.weighted_logprob
        temperature = self.temperature
        no_speech_prob = self.no_speech_prob
        language_probability = self.language_probability
        if provisional_leaf is not None:
            provisional_weight = max(1, len(provisional_leaf.text_token_ids))
            weight += provisional_weight
            weighted_logprob += provisional_leaf.avg_logprob * provisional_weight
            temperature = max(temperature, provisional_leaf.temperature)
            no_speech_prob = max(no_speech_prob, provisional_leaf.no_speech_prob)
            if language_probability is None:
                language_probability = provisional_leaf.language_probability
        text = _join_transcript_text(text_parts, language=language, task=task)
        return {
            "language_probability": language_probability,
            "temperature": temperature,
            "avg_logprob": weighted_logprob / weight,
            "compression_ratio": compression_ratio(text),
            "no_speech_prob": no_speech_prob,
        }


def _checked_diagnostic(
    value: object,
    name: str,
    *,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"transcribe leaf returned invalid {name}")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < 0.0
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"transcribe leaf returned invalid {name}")
    return result


def _checked_optional_probability(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _checked_diagnostic(value, name, maximum=1.0)


def _checked_leaf_result(
    value: object,
    *,
    window_duration: float,
    require_words: bool,
) -> _CheckedLeaf:
    if not isinstance(value, EngineResult):
        raise TypeError("transcribe leaf returned a non-EngineResult value")
    output = value.output
    text = output.get("text")
    language = output.get("language")
    raw_segments = output.get("segments")
    if not isinstance(text, str) or not isinstance(language, str):
        raise ValueError("transcribe leaf returned invalid text or language")
    if not isinstance(raw_segments, list):
        raise ValueError("transcribe leaf returned invalid segments")
    temperature, avg_logprob, ratio, no_speech_prob = _quality_diagnostics(value)
    language_probability = _checked_optional_probability(
        output.get("language_probability"),
        "language_probability",
    )
    checked: list[dict[str, object]] = []
    previous_end = 0.0
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ValueError("transcribe leaf returned a non-mapping segment")
        start = raw.get("start")
        end = raw.get("end")
        segment_text = raw.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not isinstance(segment_text, str)
        ):
            raise ValueError("transcribe leaf returned malformed segment fields")
        start_value = float(start)
        end_value = float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < previous_end
            or end_value < start_value
            or end_value > window_duration + 1e-6
        ):
            raise ValueError("transcribe leaf returned invalid segment timestamps")
        checked_segment: dict[str, object] = {
            "start": start_value,
            "end": end_value,
            "text": segment_text,
            "temperature": temperature,
            "avg_logprob": avg_logprob,
            "compression_ratio": ratio,
            "no_speech_prob": no_speech_prob,
        }
        for name, expected in (
            ("temperature", temperature),
            ("avg_logprob", avg_logprob),
            ("compression_ratio", ratio),
            ("no_speech_prob", no_speech_prob),
        ):
            raw_value = raw.get(name)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or float(raw_value) != expected
            ):
                raise ValueError(
                    f"transcribe leaf segment {name} disagrees with its window"
                )
        if require_words and "words" not in raw:
            raise ValueError("word-timestamp leaf omitted segment words")
        if "words" in raw:
            raw_words = raw.get("words")
            if not isinstance(raw_words, list) or (require_words and not raw_words):
                raise ValueError("transcribe leaf returned invalid words")
            checked_words = []
            previous_word_end = start_value
            for word in raw_words:
                if not isinstance(word, Mapping):
                    raise ValueError("transcribe leaf returned a non-mapping word")
                word_start = word.get("start")
                word_end = word.get("end")
                word_text = word.get("word")
                probability = word.get("probability")
                if (
                    isinstance(word_start, bool)
                    or not isinstance(word_start, (int, float))
                    or isinstance(word_end, bool)
                    or not isinstance(word_end, (int, float))
                    or not isinstance(word_text, str)
                    or not word_text
                    or isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                ):
                    raise ValueError("transcribe leaf returned malformed word fields")
                word_start_value = float(word_start)
                word_end_value = float(word_end)
                probability_value = float(probability)
                if (
                    not math.isfinite(word_start_value)
                    or not math.isfinite(word_end_value)
                    or word_start_value < previous_word_end
                    or word_end_value < word_start_value
                    or word_end_value > end_value + 1e-6
                    or not math.isfinite(probability_value)
                    or not 0.0 <= probability_value <= 1.0
                ):
                    raise ValueError("transcribe leaf returned invalid word timing")
                checked_words.append(
                    {
                        "start": word_start_value,
                        "end": word_end_value,
                        "word": word_text,
                        "probability": probability_value,
                    }
                )
                previous_word_end = word_end_value
            checked_segment["words"] = checked_words
        checked.append(checked_segment)
        previous_end = end_value
    token_ids: list[int] = []
    for token in value.tokens:
        if (
            not isinstance(token, TextToken)
            or isinstance(token.token_id, bool)
            or not isinstance(token.token_id, int)
            or not 0 <= token.token_id < _TIMESTAMP_VOCAB_END
        ):
            raise ValueError("transcribe leaf returned invalid tokens")
        token_ids.append(token.token_id)
    checked_token_ids = tuple(token_ids)
    return _CheckedLeaf(
        value,
        language,
        language_probability,
        text,
        tuple(checked),
        checked_token_ids,
        tuple(token_id for token_id in checked_token_ids if token_id < 50257),
        temperature,
        avg_logprob,
        ratio,
        no_speech_prob,
    )


def _timestamp_advance_frames(
    leaf: _CheckedLeaf,
    *,
    sample_rate: int,
    available_frames: int,
) -> int:
    token_ids = leaf.token_ids
    is_timestamp = [
        _TIMESTAMP_BEGIN_ID <= token_id < _TIMESTAMP_VOCAB_END for token_id in token_ids
    ]
    has_pair = any(
        left and right for left, right in zip(is_timestamp, is_timestamp[1:])
    )
    if has_pair:
        last_timestamp = next(
            token_id
            for token_id in reversed(token_ids)
            if _TIMESTAMP_BEGIN_ID <= token_id < _TIMESTAMP_VOCAB_END
        )
        seconds = (last_timestamp - _TIMESTAMP_BEGIN_ID) * _TIMESTAMP_SECONDS
        frames = int(round(seconds * sample_rate))
        if 0 < frames <= available_frames:
            return frames
    return available_frames


def _committed_text_token_ids(
    leaf: _CheckedLeaf,
    *,
    sample_rate: int,
    advance_frames: int,
    all_segments_committed: bool,
) -> tuple[int, ...]:
    """Return only text context made durable by this window advance."""

    if all_segments_committed:
        return leaf.text_token_ids

    committed: list[int] = []
    pending: list[int] = []
    segment_open = False
    for token_id in leaf.token_ids:
        if _TIMESTAMP_BEGIN_ID <= token_id < _TIMESTAMP_VOCAB_END:
            if not segment_open:
                segment_open = True
                pending = []
                continue
            boundary_frames = int(
                round(
                    (token_id - _TIMESTAMP_BEGIN_ID) * _TIMESTAMP_SECONDS * sample_rate
                )
            )
            if boundary_frames <= advance_frames:
                committed.extend(pending)
            else:
                break
            segment_open = False
            pending = []
        elif segment_open and 0 <= token_id < 50257:
            pending.append(token_id)
    return tuple(committed)


def _join_transcript_text(
    text_parts: Sequence[str],
    *,
    language: str,
    task: str,
) -> str:
    parts = [part.strip() for part in text_parts if part.strip()]
    separator = (
        "" if task == "transcribe" and language in _LANGUAGES_WITHOUT_SPACES else " "
    )
    return separator.join(parts)


def _transcript_output(
    *,
    text_parts: Sequence[str],
    language: str,
    task: str,
    duration_seconds: float,
    source_duration_seconds: float,
    clip_start_seconds: float,
    clip_end_seconds: float,
    diagnostics: Mapping[str, object],
    segments: Sequence[dict[str, object]],
    expose_segments: bool,
    progress: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the one public transcript schema used by every source mode."""

    return {
        "text": _join_transcript_text(text_parts, language=language, task=task),
        "language": language,
        "task": task,
        "duration_seconds": duration_seconds,
        "source_duration_seconds": source_duration_seconds,
        "clip_start_seconds": clip_start_seconds,
        "clip_end_seconds": clip_end_seconds,
        **diagnostics,
        "segments": list(segments) if expose_segments else [],
        **({} if progress is None else progress),
    }


def _leaf_prompt(
    prompt: Mapping[str, object],
    *,
    audio: np.ndarray,
    sample_rate: int,
    language: object,
    timestamps: str,
    previous_text_token_ids: tuple[int, ...],
) -> dict[str, object]:
    leaf = {
        **prompt,
        "audio": audio,
        "sample_rate": sample_rate,
        "language": language,
        "timestamps": "word" if timestamps == "word" else "segment",
        "stream": False,
    }
    leaf.pop("clip_start_seconds", None)
    leaf.pop("clip_end_seconds", None)
    if previous_text_token_ids:
        leaf["_previous_text_token_ids"] = previous_text_token_ids
    return leaf


def _source_seconds(
    offset_frames: int,
    local_seconds: object,
    sample_rate: int,
) -> float:
    """Map a window timestamp onto the exact source PCM frame grid."""

    return (
        offset_frames + int(round(float(local_seconds) * sample_rate))
    ) / sample_rate


def _shift_segment(
    segment: Mapping[str, object],
    *,
    offset_frames: int,
    sample_rate: int,
    snap_to_frames: bool,
) -> dict[str, object]:
    offset_seconds = offset_frames / sample_rate

    def shift(value: object) -> float:
        if snap_to_frames:
            return _source_seconds(offset_frames, value, sample_rate)
        return offset_seconds + float(value)

    shifted = {
        **segment,
        "start": shift(segment["start"]),
        "end": shift(segment["end"]),
    }
    if "words" in segment:
        words = segment["words"]
        if not isinstance(words, list):
            raise RuntimeError("validated live word list changed type")
        shifted["words"] = [
            {
                **word,
                "start": shift(word["start"]),
                "end": shift(word["end"]),
            }
            for word in words
        ]
    return shifted


async def _run_live_pcm(
    invoke: CapabilityInvoker,
    chunks: object,
    *,
    sample_rate: int,
    image: object | None,
    prompt: Mapping[str, object],
    settings: Mapping[str, object] | None,
    policy: TranscriptionQualityPolicy,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> EngineResult:
    """Transcribe a bounded async PCM source with stable timestamp commits."""

    buffer = _LivePcmBuffer(sample_rate)
    iterator_factory = getattr(chunks, "__aiter__", None)
    if not callable(iterator_factory):
        raise TypeError("live audio must implement the asynchronous iterator protocol")
    iterator = iterator_factory()
    if not callable(getattr(iterator, "__anext__", None)):
        raise TypeError("live audio __aiter__ must return an asynchronous iterator")

    # WhisperLongFormOrchestrator validates these shared options once before
    # opening a file or taking ownership of a live iterator.
    requested_timestamps = cast(str, prompt.get("timestamps", "segment"))
    task = cast(str, prompt.get("task", "transcribe"))
    condition_on_previous_text = cast(
        bool, prompt.get("condition_on_previous_text", True)
    )

    invocations = _LeafAccumulator()
    selected = _LeafAccumulator()
    segments: list[dict[str, object]] = []
    text_parts: list[str] = []
    language: str | None = None
    last_detected_language: str | None = None
    last_detected_language_probability: float | None = None
    previous_text_token_ids: tuple[int, ...] = ()
    last_evaluated_received = 0

    async def evaluate(*, final: bool) -> None:
        nonlocal language, last_detected_language
        nonlocal last_detected_language_probability
        nonlocal previous_text_token_ids, last_evaluated_received
        window = buffer.current()
        window_offset_frames = buffer.consumed_frames
        window_duration = int(window.size) / sample_rate
        leaf_prompt = _leaf_prompt(
            prompt,
            audio=window,
            sample_rate=sample_rate,
            language=language if language is not None else prompt.get("language"),
            timestamps=requested_timestamps,
            previous_text_token_ids=(
                previous_text_token_ids if condition_on_previous_text else ()
            ),
        )
        raw = await _invoke_with_fallback(
            invoke,
            leaf_prompt,
            image=image,
            settings=settings,
            policy=policy,
        )
        leaf = _checked_leaf_result(
            raw,
            window_duration=window_duration,
            require_words=requested_timestamps == "word",
        )
        invocations.add(leaf)
        last_evaluated_received = buffer.received_frames
        last_detected_language = leaf.language
        last_detected_language_probability = leaf.language_probability
        if language is not None and leaf.language != language:
            raise ValueError("live transcription changed language")
        if leaf.text.strip() and not leaf.segments:
            raise ValueError(
                "live transcription returned text without timestamp segments"
            )

        if final:
            committed = leaf.segments
        else:
            stable_before = max(0.0, window_duration - _LIVE_STABILITY_SECONDS)
            committed_list = []
            for segment in leaf.segments:
                if float(segment["end"]) > stable_before + 1e-9:
                    break
                committed_list.append(segment)
            committed = tuple(committed_list)
            # A hard-window decode must advance by at least one ingestion
            # quantum. If its stable prefix ends earlier, finalize the decoded
            # window rather than rejecting valid audio or repeatedly decoding
            # the same nearly-full buffer. Silence advances below without
            # manufacturing output.
            if (
                window.size == buffer.window_frames
                and leaf.segments
                and (
                    not committed
                    or int(round(float(committed[-1]["end"]) * sample_rate))
                    < buffer.update_frames
                )
            ):
                committed = leaf.segments

        if committed:
            for segment in committed:
                segments.append(
                    _shift_segment(
                        segment,
                        offset_frames=window_offset_frames,
                        sample_rate=sample_rate,
                        snap_to_frames=True,
                    )
                )
                text_parts.append(str(segment["text"]))
            selected.add(leaf)
            leaf_selected = True
            # If a full window has no provisional segments, its unreported
            # suffix is silence and can be retired without another decode.
            advance = (
                int(window.size)
                if final
                or (
                    window.size == buffer.window_frames
                    and len(committed) == len(leaf.segments)
                )
                else int(round(float(committed[-1]["end"]) * sample_rate))
            )
        elif window.size == buffer.window_frames or final:
            # Empty timestamp output is an admitted no-speech window.
            selected.add(leaf)
            leaf_selected = True
            advance = int(window.size)
        else:
            advance = 0
            leaf_selected = False

        # Do not pin automatic language detection from a provisional or
        # no-speech prefix. Leading silence is common in live capture and can
        # otherwise force a spurious language for the entire session.
        if language is None and (committed or final):
            language = leaf.language
            selected.language_probability = leaf.language_probability

        if advance:
            if advance <= 0 or (
                window.size == buffer.window_frames and advance < buffer.update_frames
            ):
                raise ValueError(
                    "live transcription timestamps made insufficient bounded progress"
                )
            advance = min(advance, int(window.size))
            buffer.advance(advance)
            if condition_on_previous_text and leaf.temperature <= 0.5:
                committed_text_token_ids = _committed_text_token_ids(
                    leaf,
                    sample_rate=sample_rate,
                    advance_frames=advance,
                    all_segments_committed=len(committed) == len(leaf.segments),
                )
                previous_text_token_ids = (
                    *previous_text_token_ids,
                    *committed_text_token_ids,
                )[-128:]
            else:
                previous_text_token_ids = ()
        if emit is not None and not final:
            output_language = leaf.language if language is None else language
            duration = buffer.received_frames / sample_rate
            provisional = leaf.segments[len(committed) :]
            # Committed segment dictionaries are immutable after append; only
            # copy the list needed to isolate this replaceable snapshot.
            preview_segments = list(segments)
            preview_parts = list(text_parts)
            for segment in provisional:
                preview_segments.append(
                    _shift_segment(
                        segment,
                        offset_frames=window_offset_frames,
                        sample_rate=sample_rate,
                        snap_to_frames=True,
                    )
                )
                preview_parts.append(str(segment["text"]))
            diagnostics = selected.output(
                preview_parts,
                language=output_language,
                task=task,
                provisional_leaf=None if leaf_selected else leaf,
            )
            if language is None:
                diagnostics["language_probability"] = leaf.language_probability
            emit(
                _transcript_output(
                    text_parts=preview_parts,
                    language=output_language,
                    task=task,
                    duration_seconds=duration,
                    source_duration_seconds=duration,
                    clip_start_seconds=0.0,
                    clip_end_seconds=duration,
                    diagnostics=diagnostics,
                    segments=preview_segments,
                    expose_segments=requested_timestamps in {"segment", "word"},
                    progress={
                        "completed_seconds": buffer.consumed_frames / sample_rate,
                        "total_seconds": duration,
                        "provisional": True,
                    },
                )
            )

    saw_chunk = False
    try:
        async for chunk in iterator:  # type: ignore[attr-defined]
            saw_chunk = True
            buffer.append(chunk)
            while buffer.buffered_frames >= buffer.window_frames:
                await evaluate(final=False)
            if (
                buffer.buffered_frames > 0
                and buffer.received_frames - last_evaluated_received
                >= buffer.update_frames
            ):
                await evaluate(final=False)
        if not saw_chunk:
            raise ValueError("live audio must yield at least one PCM chunk")
        if buffer.buffered_frames:
            await evaluate(final=True)
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()

    if language is None:
        if last_detected_language is None:
            raise RuntimeError("live transcription did not resolve a language")
        language = last_detected_language
        selected.language_probability = last_detected_language_probability
    if invocations.count == 0 or selected.count == 0:
        raise RuntimeError("live transcription produced no leaf requests")
    duration = buffer.received_frames / sample_rate
    final_output = _transcript_output(
        text_parts=text_parts,
        language=language,
        task=task,
        duration_seconds=duration,
        source_duration_seconds=duration,
        clip_start_seconds=0.0,
        clip_end_seconds=duration,
        diagnostics=selected.output(text_parts, language=language, task=task),
        segments=segments,
        expose_segments=requested_timestamps in {"segment", "word"},
    )
    if emit is not None:
        emit(
            {
                **final_output,
                "completed_seconds": duration,
                "total_seconds": duration,
                "provisional": False,
            }
        )
    return EngineResult(
        request_id=invocations.request_id,
        # Live windows can be decoded repeatedly before their timestamped
        # prefix stabilizes, so returning leaf tokens would duplicate them.
        tokens=[],
        finish_reason=(
            invocations.first_non_stop_reason or invocations.last_finish_reason
        ),
        metrics=invocations.metrics(),
        output=final_output,
    )


def _quality_diagnostics(
    result: object,
    *,
    expected_temperature: float | None = None,
) -> tuple[float, float, float, float]:
    if not isinstance(result, EngineResult):
        raise TypeError("transcribe leaf returned a non-EngineResult value")
    output = result.output
    temperature = _checked_diagnostic(output.get("temperature"), "temperature")
    if expected_temperature is not None and temperature != expected_temperature:
        raise ValueError("transcribe leaf returned an unexpected temperature")
    raw_avg = output.get("avg_logprob")
    if (
        isinstance(raw_avg, bool)
        or not isinstance(raw_avg, (int, float))
        or not math.isfinite(float(raw_avg))
        or float(raw_avg) > 1e-6
    ):
        raise ValueError("transcribe leaf returned invalid avg_logprob")
    ratio = _checked_diagnostic(output.get("compression_ratio"), "compression_ratio")
    no_speech = _checked_diagnostic(
        output.get("no_speech_prob"),
        "no_speech_prob",
        maximum=1.0,
    )
    return temperature, float(raw_avg), ratio, no_speech


def _select_best_candidate(
    values: Sequence[object],
    *,
    expected_temperature: float,
) -> tuple[EngineResult, str, float, float, float]:
    """Validate one bounded candidate set and retain its strongest score."""

    candidates: list[tuple[EngineResult, float, float, float]] = []
    language: str | None = None
    for value in values:
        if not isinstance(value, EngineResult):
            raise TypeError("transcribe leaf returned a non-EngineResult value")
        result_language = value.output.get("language")
        if not isinstance(result_language, str) or not result_language:
            raise ValueError("transcribe leaf returned an invalid language")
        if language is None:
            language = result_language
        elif result_language != language:
            raise ValueError("best-of candidates disagreed on detected language")
        _temperature, avg_logprob, ratio, no_speech_prob = _quality_diagnostics(
            value,
            expected_temperature=expected_temperature,
        )
        candidates.append((value, avg_logprob, ratio, no_speech_prob))
    if language is None or not candidates:
        raise ValueError("best-of candidate set must not be empty")
    selected, avg_logprob, ratio, no_speech_prob = max(
        candidates,
        key=lambda candidate: candidate[1],
    )
    return selected, language, avg_logprob, ratio, no_speech_prob


@dataclass(slots=True)
class _FallbackState:
    automatic_language: bool
    language: str | None = None
    language_probability: float | None = None

    def select(
        self,
        values: Sequence[object],
        *,
        temperature: float,
    ) -> tuple[EngineResult, float, float, float]:
        result, language, avg_logprob, ratio, no_speech_prob = _select_best_candidate(
            values, expected_temperature=temperature
        )
        if self.automatic_language:
            if self.language is None:
                self.language = language
                self.language_probability = _checked_optional_probability(
                    result.output.get("language_probability"),
                    "language_probability",
                )
            elif language != self.language:
                raise ValueError("transcribe fallback changed the detected language")
            if self.language_probability is not None:
                result = replace(
                    result,
                    output={
                        **result.output,
                        "language_probability": self.language_probability,
                    },
                )
        return result, avg_logprob, ratio, no_speech_prob


async def _gather_candidates(*awaitables: Awaitable[object]) -> list[object]:
    gathering = asyncio.gather(*awaitables, return_exceptions=True)
    try:
        values = await asyncio.shield(gathering)
    except BaseException:
        # These are submitted engine requests, not disposable coroutine work.
        # Cancelling their result waiters loses the only handle while the GPU
        # keeps running, so settle the bounded set before returning ownership.
        try:
            await gathering
        except BaseException:
            pass
        raise
    errors = [value for value in values if isinstance(value, BaseException)]
    if errors:
        raise errors[0]
    return values


def _accept_attempt(
    result: EngineResult,
    *,
    avg_logprob: float,
    compression: float,
    no_speech_prob: float,
    final_attempt: bool,
    policy: TranscriptionQualityPolicy,
) -> EngineResult | None:
    if policy.is_silence(
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    ):
        return _silence_result(result)
    if final_attempt or not policy.needs_fallback(
        avg_logprob=avg_logprob,
        compression_ratio=compression,
    ):
        return result
    return None


async def _close_candidate_streams(values: Sequence[object]) -> None:
    async def close_or_settle(value: object) -> None:
        close = getattr(value, "aclose", None)
        if callable(close):
            await close()
            return
        # Custom stream implementations may expose only terminal settlement.
        result = getattr(value, "result", None)
        if callable(result):
            await result()

    await asyncio.gather(
        *(close_or_settle(value) for value in values),
        return_exceptions=True,
    )


async def _start_capability_stream(
    produce: Callable[
        [Callable[[dict[str, object]], None]],
        Awaitable[EngineResult],
    ],
) -> CapabilityStream:
    """Transfer producer ownership without leaking it during cancellation."""

    stream = CapabilityStream("transcribe", produce)
    try:
        await asyncio.sleep(0)
    except BaseException:
        try:
            await stream.aclose()
        except BaseException:
            pass
        raise
    return stream


async def _open_native_reader(audio: bytes | str | Path) -> object:
    """Open one native reader without losing its handle to task cancellation."""

    if isinstance(audio, bytes):
        call = kestrel_native.open_audio_mono
        value: bytes | Path = audio
    else:
        call = kestrel_native.open_audio_file_mono
        value = Path(audio)
    opening = asyncio.create_task(
        asyncio.to_thread(
            call,
            value,
            max_duration_seconds=_MAX_FILE_DURATION_SECONDS,
        )
    )
    try:
        return await asyncio.shield(opening)
    except asyncio.CancelledError:
        try:
            reader = await opening
        except BaseException:
            pass
        else:
            try:
                await asyncio.to_thread(reader.close)
            except BaseException:
                pass
        raise


def _snapshot_short_pcm(
    audio: np.ndarray | Tensor,
    sample_rate: object,
    clip_start_seconds: object,
    clip_end_seconds: object,
) -> np.ndarray:
    """Validate and own raw PCM before a progressive call returns."""

    source = validate_audio_source(
        audio,
        sample_rate=sample_rate,
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
    )
    if source.kind != "pcm":  # pragma: no cover - narrowed by argument type
        raise AssertionError("raw PCM snapshot produced a non-PCM source")
    return _pcm_to_float32(audio).copy()


def _silence_result(result: EngineResult) -> EngineResult:
    output = dict(result.output)
    output.update(text="", segments=[])
    return replace(
        result,
        tokens=[],
        output=output,
        logprobs=[] if result.logprobs is not None else None,
    )


async def _invoke_with_fallback(
    invoke: CapabilityInvoker,
    prompt: Mapping[str, object],
    *,
    image: object | None,
    settings: Mapping[str, object] | None,
    policy: TranscriptionQualityPolicy,
) -> EngineResult:
    """Run the fixed retry schedule and return the first admissible leaf."""

    base_settings = {} if settings is None else dict(settings)
    attempt_prompt = dict(prompt)
    fallback = _FallbackState(automatic_language=prompt.get("language") is None)
    for attempt, temperature in enumerate(policy.temperatures):
        if fallback.language is not None:
            attempt_prompt["language"] = fallback.language
        attempt_settings = {**base_settings, "temperature": temperature}
        raw_candidates = await _gather_candidates(
            *(
                invoke(
                    attempt_prompt,
                    image=image,
                    settings=attempt_settings,
                )
                for _ in range(policy.candidate_count(temperature))
            ),
        )
        raw_result, avg_logprob, ratio, no_speech_prob = fallback.select(
            raw_candidates, temperature=temperature
        )
        selected = _accept_attempt(
            raw_result,
            avg_logprob=avg_logprob,
            compression=ratio,
            no_speech_prob=no_speech_prob,
            final_attempt=attempt + 1 == len(policy.temperatures),
            policy=policy,
        )
        if selected is not None:
            return selected
    raise AssertionError("non-empty temperature schedule did not return")


async def _stream_short_with_fallback(
    invoke: CapabilityInvoker,
    prompt: Mapping[str, object],
    *,
    image: object | None,
    settings: Mapping[str, object] | None,
    policy: TranscriptionQualityPolicy,
    emit: Callable[[dict[str, object]], None],
) -> EngineResult:
    """Stream provisional text while keeping rejected attempts retractable."""

    base_settings = {} if settings is None else dict(settings)
    attempt_prompt = {**prompt, "stream": True}
    fallback = _FallbackState(automatic_language=prompt.get("language") is None)
    for attempt, temperature in enumerate(policy.temperatures):
        if fallback.language is not None:
            attempt_prompt["language"] = fallback.language
        stream_tasks = tuple(
            asyncio.ensure_future(
                invoke(
                    attempt_prompt,
                    image=image,
                    settings={**base_settings, "temperature": temperature},
                )
            )
            for _ in range(policy.candidate_count(temperature))
        )
        try:
            raw_stream_values = await asyncio.gather(
                *stream_tasks,
                return_exceptions=True,
            )
        except BaseException:
            for task in stream_tasks:
                task.cancel()
            await asyncio.gather(*stream_tasks, return_exceptions=True)
            opened = [
                task.result()
                for task in stream_tasks
                if not task.cancelled() and task.exception() is None
            ]
            await _close_candidate_streams(opened)
            raise
        errors = [
            value for value in raw_stream_values if isinstance(value, BaseException)
        ]
        if errors:
            await _close_candidate_streams(raw_stream_values)
            raise errors[0]
        raw_streams = raw_stream_values
        if any(
            not callable(getattr(raw_stream, "__aiter__", None))
            or not callable(getattr(raw_stream, "result", None))
            for raw_stream in raw_streams
        ):
            await _close_candidate_streams(raw_streams)
            raise TypeError("streaming transcribe leaf returned a non-stream value")
        raw_stream = raw_streams[0]
        text = ""
        try:
            async for update in raw_stream:  # type: ignore[attr-defined]
                delta = getattr(update, "text", None)
                if not isinstance(delta, str):
                    raise TypeError("streaming transcribe leaf returned invalid text")
                text += delta
                emit(
                    {
                        "text": text,
                        "temperature": temperature,
                        "attempt": attempt,
                        "provisional": True,
                    }
                )
            raw_candidates = await _gather_candidates(
                *(candidate.result() for candidate in raw_streams)
            )
        except BaseException:
            await _close_candidate_streams(raw_streams)
            raise
        raw_result, avg_logprob, ratio, no_speech_prob = fallback.select(
            raw_candidates, temperature=temperature
        )
        selected = _accept_attempt(
            raw_result,
            avg_logprob=avg_logprob,
            compression=ratio,
            no_speech_prob=no_speech_prob,
            final_attempt=attempt + 1 == len(policy.temperatures),
            policy=policy,
        )
        if selected is None:
            # Capability updates are replaceable snapshots, so the next
            # attempt can explicitly retract a rejected transcript.
            emit(
                {
                    "text": "",
                    "temperature": temperature,
                    "attempt": attempt,
                    "provisional": True,
                    "retrying": True,
                }
            )
            continue
        emit({**selected.output, "provisional": False})
        return selected
    raise AssertionError("non-empty temperature schedule did not return")


async def _invoke_short(
    invoke: CapabilityInvoker,
    prompt: Mapping[str, object],
    *,
    image: object | None,
    settings: Mapping[str, object] | None,
    policy: TranscriptionQualityPolicy,
) -> object:
    if prompt.get("stream", False) is not True:
        return await _invoke_with_fallback(
            invoke,
            prompt,
            image=image,
            settings=settings,
            policy=policy,
        )

    async def produce(emit: Callable[[dict[str, object]], None]) -> EngineResult:
        return await _stream_short_with_fallback(
            invoke,
            prompt,
            image=image,
            settings=settings,
            policy=policy,
            emit=emit,
        )

    return await _start_capability_stream(produce)


async def _run_long_file(
    invoke: CapabilityInvoker,
    windows: _NativeFileWindows,
    *,
    image: object | None,
    prompt: Mapping[str, object],
    settings: Mapping[str, object] | None,
    policy: TranscriptionQualityPolicy,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> EngineResult:
    # Shared prompt options were validated before the native reader was opened.
    requested_timestamps = cast(str, prompt.get("timestamps", "segment"))
    requested_language = prompt.get("language")
    task = cast(str, prompt.get("task", "transcribe"))
    accumulator = _LeafAccumulator()
    global_segments: list[dict[str, object]] = []
    text_parts: list[str] = []
    language: str | None = None
    last_detected_language: str | None = None
    last_detected_language_probability: float | None = None
    previous_text_token_ids: tuple[int, ...] = ()
    condition_on_previous_text = cast(
        bool, prompt.get("condition_on_previous_text", True)
    )

    while True:
        window = await settled_to_thread(windows.current)
        if window is None:
            break
        window_duration = int(window.size) / windows.sample_rate
        leaf_prompt = _leaf_prompt(
            prompt,
            audio=window,
            sample_rate=windows.sample_rate,
            language=language if language is not None else requested_language,
            timestamps=requested_timestamps,
            previous_text_token_ids=(
                previous_text_token_ids if condition_on_previous_text else ()
            ),
        )
        raw_result = await _invoke_with_fallback(
            invoke,
            leaf_prompt,
            image=image,
            settings=settings,
            policy=policy,
        )
        leaf = _checked_leaf_result(
            raw_result,
            window_duration=window_duration,
            require_words=requested_timestamps == "word",
        )
        last_detected_language = leaf.language
        last_detected_language_probability = leaf.language_probability
        if language is not None and leaf.language != language:
            raise ValueError("transcribe leaf changed language within one file")

        advance = _timestamp_advance_frames(
            leaf,
            sample_rate=windows.sample_rate,
            available_frames=int(window.size),
        )
        # A decode can contain complete timestamp pairs followed by an open
        # trailing segment that the short-form parser closes at the window end.
        # When the paired timestamps move the next window to an earlier frame,
        # publishing that open suffix would overlap and duplicate the next
        # decode. Commit only segments ending at or before the exact seek frame.
        committed_segments = tuple(
            segment
            for segment in leaf.segments
            if int(round(float(segment["end"]) * windows.sample_rate)) <= advance
        )
        if committed_segments:
            for segment in committed_segments:
                global_segments.append(
                    _shift_segment(
                        segment,
                        offset_frames=windows.offset_frames,
                        sample_rate=windows.sample_rate,
                        snap_to_frames=False,
                    )
                )
                text_parts.append(str(segment["text"]))
        elif advance == window.size and leaf.text.strip():
            text_parts.append(leaf.text)
        accumulator.add(leaf)
        if language is None and (
            committed_segments or (advance == window.size and leaf.text.strip())
        ):
            language = leaf.language
            accumulator.language_probability = leaf.language_probability
        if leaf.temperature > 0.5:
            previous_text_token_ids = ()
        elif condition_on_previous_text:
            if advance == window.size:
                committed_text_token_ids = leaf.text_token_ids
            else:
                final_timestamp_index = max(
                    index
                    for index, token_id in enumerate(leaf.token_ids)
                    if _TIMESTAMP_BEGIN_ID <= token_id < _TIMESTAMP_VOCAB_END
                )
                committed_text_token_ids = tuple(
                    token_id
                    for token_id in leaf.token_ids[:final_timestamp_index]
                    if token_id < 50257
                )
            previous_text_token_ids = (
                *previous_text_token_ids,
                *committed_text_token_ids,
            )[-128:]

        await settled_to_thread(windows.advance, advance)
        if emit is not None:
            output_language = leaf.language if language is None else language
            completed_seconds = min(
                windows.completed_frames / windows.sample_rate,
                windows.total_frames / windows.sample_rate,
            )
            total_seconds = windows.total_frames / windows.sample_rate
            diagnostics = accumulator.output(
                text_parts,
                language=output_language,
                task=task,
            )
            if language is None:
                diagnostics["language_probability"] = leaf.language_probability
            emit(
                _transcript_output(
                    text_parts=text_parts,
                    language=output_language,
                    task=task,
                    duration_seconds=windows.total_frames / windows.sample_rate,
                    source_duration_seconds=windows.source_duration_seconds,
                    clip_start_seconds=windows.clip_start_seconds,
                    clip_end_seconds=windows.clip_end_seconds,
                    diagnostics=diagnostics,
                    segments=global_segments,
                    expose_segments=requested_timestamps in {"segment", "word"},
                    progress={
                        "completed_seconds": completed_seconds,
                        "total_seconds": total_seconds,
                        "provisional": completed_seconds < total_seconds,
                    },
                )
            )

    if language is None:
        if last_detected_language is None:
            raise RuntimeError("long-form transcription did not resolve a language")
        language = last_detected_language
        accumulator.language_probability = last_detected_language_probability
    if accumulator.count == 0:
        raise RuntimeError("long-form transcription produced no leaf requests")
    return EngineResult(
        request_id=accumulator.request_id,
        # Timestamp-driven file windows can overlap, so concatenating their
        # generated tokens would expose duplicate or later-retracted text.
        # The structured transcript and segments above are the canonical result.
        tokens=[],
        finish_reason=(
            accumulator.first_non_stop_reason or accumulator.last_finish_reason
        ),
        metrics=accumulator.metrics(),
        output=_transcript_output(
            text_parts=text_parts,
            language=language,
            task=task,
            duration_seconds=windows.total_frames / windows.sample_rate,
            source_duration_seconds=windows.source_duration_seconds,
            clip_start_seconds=windows.clip_start_seconds,
            clip_end_seconds=windows.clip_end_seconds,
            diagnostics=accumulator.output(
                text_parts,
                language=language,
                task=task,
            ),
            segments=global_segments,
            expose_segments=requested_timestamps in {"segment", "word"},
        ),
    )


class WhisperLongFormOrchestrator(CapabilityOrchestrator):
    """Run timestamp-driven 30-second leaves for admitted Whisper audio."""

    async def run(
        self,
        invoke: CapabilityInvoker,
        *,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Mapping[str, object] | None,
    ) -> object:
        if image is not None:
            raise ValueError("transcribe does not accept an image")
        transcription = TranscriptionRequest.from_prompt(prompt)
        if transcription.timestamps == "character":
            raise ValueError("timestamps must be 'none', 'segment', or 'word'")
        policy = parse_quality_policy(settings)
        stream_requested = transcription.stream
        clip_start = transcription.clip_start_seconds
        clip_end = transcription.clip_end_seconds
        audio = transcription.audio
        if callable(getattr(audio, "__aiter__", None)):
            if clip_start != 0.0 or clip_end is not None:
                raise ValueError("clip ranges are not supported for live PCM")
            if transcription.sample_rate is None:
                raise ValueError("sample_rate is required for live PCM")
            rate = transcription.sample_rate
            if stream_requested:

                async def produce(
                    emit: Callable[[dict[str, object]], None],
                ) -> EngineResult:
                    return await _run_live_pcm(
                        invoke,
                        audio,
                        sample_rate=rate,
                        image=image,
                        prompt=prompt,
                        settings=settings,
                        policy=policy,
                        emit=emit,
                    )

                return await _start_capability_stream(produce)
            return await _run_live_pcm(
                invoke,
                audio,
                sample_rate=rate,
                image=image,
                prompt=prompt,
                settings=settings,
                policy=policy,
            )
        if transcription.sample_rate is not None:
            if stream_requested and isinstance(audio, (np.ndarray, Tensor)):
                owned_audio = await settled_to_thread(
                    _snapshot_short_pcm,
                    audio,
                    transcription.sample_rate,
                    clip_start,
                    clip_end,
                )
                prompt = {**prompt, "audio": owned_audio}
            return await _invoke_short(
                invoke,
                prompt,
                image=image,
                settings=settings,
                policy=policy,
            )
        if callable(getattr(audio, "read", None)):
            audio = await settled_to_thread(snapshot_file_like, audio)
            prompt = {**prompt, "audio": audio}
        if isinstance(audio, (bytes, str, Path)):
            reader = await _open_native_reader(audio)
        else:
            return await _invoke_short(
                invoke,
                prompt,
                image=image,
                settings=settings,
                policy=policy,
            )
        try:
            windows = _NativeFileWindows(
                reader,
                clip_start_seconds=clip_start,
                clip_end_seconds=clip_end,
            )
        except BaseException:
            reader.close()
            raise
        has_clip = clip_start != 0.0 or clip_end is not None
        if not has_clip and windows.total_frames <= windows.window_frames:
            reader.close()
            return await _invoke_short(
                invoke,
                prompt,
                image=image,
                settings=settings,
                policy=policy,
            )

        if stream_requested:

            async def produce(
                emit: Callable[[dict[str, object]], None],
            ) -> EngineResult:
                try:
                    return await _run_long_file(
                        invoke,
                        windows,
                        image=image,
                        prompt=prompt,
                        settings=settings,
                        policy=policy,
                        emit=emit,
                    )
                finally:
                    await asyncio.to_thread(reader.close)

            return await _start_capability_stream(produce)

        try:
            return await _run_long_file(
                invoke,
                windows,
                image=image,
                prompt=prompt,
                settings=settings,
                policy=policy,
            )
        finally:
            await asyncio.to_thread(reader.close)


__all__ = ["WhisperLongFormOrchestrator"]
