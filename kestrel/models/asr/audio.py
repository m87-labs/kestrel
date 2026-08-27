"""Audio decoding shared by Kestrel speech-to-text models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from torch import Tensor


AudioInput = str | Path | bytes | BinaryIO | np.ndarray | Tensor
MAX_SHORT_AUDIO_SECONDS = 30.0
MAX_AUDIO_SECONDS = 24 * 60 * 60
_MAX_READ_FRAMES = 1_048_576
_MAX_ENCODED_BYTES = 64 * 1024 * 1024
_ENCODED_READ_BYTES = 1024 * 1024
_MAX_ENCODED_READS = 65_536


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    waveform: np.ndarray
    duration_seconds: float
    source_duration_seconds: float
    clip_start_seconds: float


def snapshot_file_like(audio: object) -> object:
    """Own a binary stream from its current position under a hard byte limit."""

    read = getattr(audio, "read", None)
    if not callable(read):
        return audio
    encoded = bytearray()
    for _ in range(_MAX_ENCODED_READS):
        wanted = min(_ENCODED_READ_BYTES, _MAX_ENCODED_BYTES + 1 - len(encoded))
        chunk = read(wanted)
        if not isinstance(chunk, bytes):
            raise TypeError(
                "encoded audio file-like objects must return bytes from read(size)"
            )
        if len(chunk) > wanted:
            raise ValueError(
                "encoded audio file-like read exceeded its requested byte count"
            )
        if not chunk:
            break
        encoded.extend(chunk)
        if len(encoded) > _MAX_ENCODED_BYTES:
            raise ValueError(
                f"encoded audio file-like object exceeds the {_MAX_ENCODED_BYTES}-byte input limit"
            )
    else:
        raise ValueError("encoded audio file-like object made insufficient read progress")
    if not encoded:
        raise ValueError("encoded audio file-like object must not be empty")
    return bytes(encoded)


class AudioChunks:
    """Incrementally decode and split one clip without retaining the whole file."""

    def __init__(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None,
        clip_start_seconds: float,
        clip_end_seconds: float | None,
        target_sample_rate: int,
        max_duration_seconds: float,
    ) -> None:
        import kestrel_native

        if clip_start_seconds < 0 or (
            clip_end_seconds is not None and clip_end_seconds <= clip_start_seconds
        ):
            raise ValueError("invalid audio clip range")
        self._native = None
        self._waveform = None
        if isinstance(audio, (np.ndarray, Tensor)):
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                raise ValueError("sample_rate is required for raw PCM")
            self._waveform = _float_pcm(audio)
            self.sample_rate = sample_rate
            total_frames = int(self._waveform.size)
        else:
            if sample_rate is not None:
                raise ValueError("sample_rate must be omitted for encoded audio")
            audio = snapshot_file_like(audio)
            if not isinstance(audio, (str, Path, bytes)):
                raise TypeError(
                    "audio must be encoded bytes/path or raw NumPy/CPU Torch PCM"
                )
            self._native = (
                kestrel_native.open_audio_mono(
                    audio, max_duration_seconds=MAX_AUDIO_SECONDS
                )
                if isinstance(audio, bytes)
                else kestrel_native.open_audio_file_mono(
                    Path(audio), max_duration_seconds=MAX_AUDIO_SECONDS
                )
            )
            self.sample_rate = int(self._native.sample_rate)
            total_frames = int(self._native.total_frames)

        self.source_duration_seconds = total_frames / self.sample_rate
        self._clip_start_frame = round(clip_start_seconds * self.sample_rate)
        self._clip_end_frame = (
            total_frames
            if clip_end_seconds is None
            else min(total_frames, round(clip_end_seconds * self.sample_rate))
        )
        if self._clip_start_frame >= self._clip_end_frame:
            self.close()
            raise ValueError("the selected audio clip contains no samples")
        self.duration_seconds = (
            self._clip_end_frame - self._clip_start_frame
        ) / self.sample_rate
        if self.duration_seconds > max_duration_seconds:
            self.close()
            raise ValueError(
                f"audio clip must be at most {max_duration_seconds:g} seconds"
            )
        self.clip_start_seconds = self._clip_start_frame / self.sample_rate
        self.target_sample_rate = target_sample_rate
        self._position = self._clip_start_frame
        self._buffer = np.empty(0, dtype=np.float32)
        if self._native is not None and self._clip_start_frame:
            self._native.seek(self._clip_start_frame)

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None

    def __enter__(self) -> "AudioChunks":
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def _read(self, frames: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        received = 0
        while received < frames:
            wanted = min(frames - received, _MAX_READ_FRAMES)
            if self._waveform is not None:
                start = self._position + received
                chunk = self._waveform[start : start + wanted]
                if chunk.size == 0:
                    break
            else:
                if self._native is None:
                    raise ValueError("audio reader is closed")
                chunk = self._native.read(wanted)
                if chunk is None:
                    break
                chunk = _float_pcm(chunk)
                if chunk.size > wanted:
                    raise ValueError(
                        "incremental audio reader returned too many samples"
                    )
            parts.append(chunk)
            received += int(chunk.size)
        if received != frames:
            raise ValueError(
                "incremental audio reader ended before its declared frame count"
            )
        self._position += received
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def chunks(
        self,
        max_chunk_seconds: float,
        *,
        boundary_search_seconds: float = 5.0,
        boundary_window_seconds: float = 0.1,
    ) -> Iterator[DecodedAudio]:
        """Yield contiguous chunks, preferring quiet boundaries near the target."""

        import kestrel_native

        if max_chunk_seconds <= 0:
            raise ValueError("max_chunk_seconds must be positive")
        rate = self.sample_rate
        max_frames = round(max_chunk_seconds * rate)
        search_frames = round(boundary_search_seconds * rate)
        window_frames = max(4, round(boundary_window_seconds * rate))
        min_tail_frames = max(window_frames, round(0.5 * rate))
        emitted = 0
        total = self._clip_end_frame - self._clip_start_frame

        while emitted < total:
            remaining = total - emitted
            wanted = min(remaining, max_frames + search_frames)
            missing = wanted - int(self._buffer.size)
            if missing > 0:
                incoming = self._read(missing)
                self._buffer = (
                    incoming
                    if self._buffer.size == 0
                    else np.concatenate((self._buffer, incoming))
                )
            if remaining <= max_frames:
                boundary = remaining
            else:
                left = max(1, max_frames - search_frames)
                right = min(
                    int(self._buffer.size),
                    max_frames + search_frames,
                    remaining - min_tail_frames,
                )
                if right - left <= window_frames:
                    boundary = min(max_frames, remaining - min_tail_frames)
                else:
                    absolute = np.abs(self._buffer[left:right])
                    cumulative = np.pad(np.cumsum(absolute, dtype=np.float64), (1, 0))
                    energy = cumulative[window_frames:] - cumulative[:-window_frames]
                    quiet = int(energy.argmin())
                    local = int(absolute[quiet : quiet + window_frames].argmin())
                    boundary = min(
                        max_frames,
                        remaining - min_tail_frames,
                        left + quiet + local,
                    )

            source = np.ascontiguousarray(self._buffer[:boundary])
            self._buffer = self._buffer[boundary:]
            offset = self.clip_start_seconds + emitted / rate
            emitted += boundary
            duration = boundary / rate
            if rate == self.target_sample_rate:
                waveform = source.copy()
            else:
                waveform = _float_pcm(
                    kestrel_native.resample_audio_mono(
                        source,
                        rate,
                        self.target_sample_rate,
                        max_output_values=(
                            int(np.ceil(duration * self.target_sample_rate)) + 16
                        ),
                    )
                )
            yield DecodedAudio(
                waveform=waveform,
                duration_seconds=duration,
                source_duration_seconds=self.source_duration_seconds,
                clip_start_seconds=offset,
            )


def _float_pcm(audio: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(audio, Tensor):
        if audio.device.type != "cpu":
            raise ValueError("raw PCM tensors must be on CPU")
        audio = audio.detach().contiguous()
        if audio.dtype.is_floating_point:
            audio = audio.float()
        audio = audio.numpy()
    value = np.asarray(audio)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("raw PCM must be non-empty mono audio")
    if np.issubdtype(value.dtype, np.signedinteger):
        limit = float(max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max))
        value = value.astype(np.float32) / limit
    elif np.issubdtype(value.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(value.dtype).max + 1) / 2
        value = (value.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float32, copy=False)
    else:
        raise TypeError("raw PCM must have a real numeric dtype")
    value = np.ascontiguousarray(value)
    if not np.isfinite(value).all() or np.abs(value).max() > 1.000001:
        raise ValueError("PCM must contain finite samples in [-1, 1]")
    return value


def decode_audio(
    audio: AudioInput,
    *,
    sample_rate: int | None = None,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    target_sample_rate: int = 16_000,
    max_duration_seconds: float = MAX_SHORT_AUDIO_SECONDS,
) -> DecodedAudio:
    """Decode, clip, and resample one bounded request."""

    with AudioChunks(
        audio,
        sample_rate=sample_rate,
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
    ) as source:
        chunks = tuple(source.chunks(max_duration_seconds, boundary_search_seconds=0))
    if len(chunks) != 1:
        raise RuntimeError("bounded audio unexpectedly produced multiple chunks")
    return chunks[0]


__all__ = [
    "AudioChunks",
    "AudioInput",
    "DecodedAudio",
    "MAX_AUDIO_SECONDS",
    "MAX_SHORT_AUDIO_SECONDS",
    "decode_audio",
    "snapshot_file_like",
]
