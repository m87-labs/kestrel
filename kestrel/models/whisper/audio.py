"""CPU audio validation, decoding, resampling, and Whisper log-Mel features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import kestrel_native
import numpy as np
import torch
from kestrel.models.asr.audio import snapshot_file_like
from kestrel.models.asr.features import mel_filters
from torch import Tensor

from .config import WhisperPreprocessorConfig


AudioValue = str | Path | bytes | np.ndarray | Tensor
_TORCH_INTEGER_DTYPES = frozenset(
    getattr(torch, name)
    for name in (
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
    )
    if hasattr(torch, name)
)
# Float32 decode allocation budget, not a source sample-rate ceiling. This
# admits 30 seconds of 48 kHz eight-channel audio, and proportionally shorter
# higher-rate inputs. The native decoder downmixes an admitted speaker layout
# to mono before this reaches Whisper.
_MAX_ENCODED_DECODE_SAMPLE_VALUES = 30 * 48_000 * 8
_MAX_NATIVE_AUDIO_VALUES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Validated but not yet decoded request input.

    Validation is deliberately cheap enough for request admission. Encoded data
    is opened and featurized only by the preprocessing worker.
    """

    value: AudioValue
    kind: Literal["encoded", "pcm"]
    sample_rate: int | None
    clip_start_seconds: float = 0.0
    clip_end_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """The only audio object the GPU runtime should receive."""

    input_features: Tensor
    duration_seconds: float
    original_num_samples: int
    original_sample_rate: int
    resampled_num_samples: int
    clip_start_seconds: float = 0.0
    source_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        features = self.input_features
        if not isinstance(features, Tensor):
            raise TypeError("Prepared Whisper features must be a Torch tensor")
        if features.device.type != "cpu":
            raise ValueError("Prepared Whisper features must remain on CPU")
        if features.dtype is not torch.float32:
            raise ValueError("Prepared Whisper features must be float32")
        if tuple(features.shape) != (128, 3000):
            raise ValueError(
                f"Prepared Whisper features must have shape (128, 3000), got {tuple(features.shape)}"
            )
        if not features.is_contiguous():
            raise ValueError("Prepared Whisper features must be contiguous")
        if features.requires_grad:
            raise ValueError("Prepared Whisper features must not require gradients")
        if not torch.isfinite(features).all().item():
            raise ValueError("Prepared Whisper features contain NaN or infinity")

        integer_fields = {
            "original_num_samples": self.original_num_samples,
            "original_sample_rate": self.original_sample_rate,
            "resampled_num_samples": self.resampled_num_samples,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"PreparedAudio.{name} must be a positive integer")
            if int(value) <= 0:
                raise ValueError(f"PreparedAudio.{name} must be a positive integer")
        if self.resampled_num_samples > 480000:
            raise ValueError("PreparedAudio.resampled_num_samples exceeds 480000")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float, np.integer, np.floating)
        ):
            raise TypeError("PreparedAudio.duration_seconds must be a finite number")
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or not 0.0 < duration <= 30.0:
            raise ValueError("PreparedAudio.duration_seconds must be in (0, 30]")
        expected_duration = self.original_num_samples / self.original_sample_rate
        if not math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "PreparedAudio.duration_seconds is inconsistent with original samples"
            )
        if isinstance(self.clip_start_seconds, bool) or not isinstance(
            self.clip_start_seconds, (int, float, np.integer, np.floating)
        ):
            raise TypeError("PreparedAudio.clip_start_seconds must be a finite number")
        clip_start = float(self.clip_start_seconds)
        if not math.isfinite(clip_start) or clip_start < 0.0:
            raise ValueError(
                "PreparedAudio.clip_start_seconds must be finite and non-negative"
            )
        if self.source_duration_seconds is None:
            source_duration = duration
        else:
            if isinstance(self.source_duration_seconds, bool) or not isinstance(
                self.source_duration_seconds, (int, float, np.integer, np.floating)
            ):
                raise TypeError("PreparedAudio.source_duration_seconds must be finite")
            source_duration = float(self.source_duration_seconds)
        if (
            not math.isfinite(source_duration)
            or source_duration <= 0.0
            or clip_start + duration > source_duration + 1e-9
        ):
            raise ValueError("PreparedAudio clip lies outside its source duration")


def _validate_sample_rate(sample_rate: object) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be a positive integer for raw PCM")
    value = int(sample_rate)
    if value <= 0:
        raise ValueError("sample_rate must be a positive integer for raw PCM")
    if value > np.iinfo(np.uint32).max:
        raise ValueError("sample_rate must fit the native unsigned 32-bit contract")
    return value


def validate_clip_range(
    clip_start_seconds: object = 0.0,
    clip_end_seconds: object = None,
) -> tuple[float, float | None]:
    """Validate a source-relative half-open clip range in seconds."""

    if isinstance(clip_start_seconds, bool) or not isinstance(
        clip_start_seconds, (int, float, np.integer, np.floating)
    ):
        raise TypeError("clip_start_seconds must be a finite non-negative number")
    start = float(clip_start_seconds)
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("clip_start_seconds must be a finite non-negative number")
    if clip_end_seconds is None:
        return start, None
    if isinstance(clip_end_seconds, bool) or not isinstance(
        clip_end_seconds, (int, float, np.integer, np.floating)
    ):
        raise TypeError("clip_end_seconds must be a finite number or None")
    end = float(clip_end_seconds)
    if not math.isfinite(end) or end <= start:
        raise ValueError(
            "clip_end_seconds must be finite and greater than clip_start_seconds"
        )
    return start, end


def _clip_frame_bounds(
    total_frames: int,
    sample_rate: int,
    clip_start_seconds: float,
    clip_end_seconds: float | None,
) -> tuple[int, int]:
    start_frame = int(round(clip_start_seconds * sample_rate))
    if start_frame >= total_frames:
        raise ValueError("clip_start_seconds lies at or beyond the end of the audio")
    end_frame = (
        total_frames
        if clip_end_seconds is None
        else min(total_frames, int(round(clip_end_seconds * sample_rate)))
    )
    if end_frame <= start_frame:
        raise ValueError("the selected audio clip contains no samples")
    return start_frame, end_frame


def _raw_shape_and_size(audio: np.ndarray | Tensor) -> tuple[tuple[int, ...], int]:
    if isinstance(audio, Tensor):
        if audio.device.type != "cpu":
            raise ValueError("raw Torch PCM must be a CPU tensor")
        if audio.layout is not torch.strided:
            raise ValueError("raw Torch PCM must use strided layout")
        if (
            audio.dtype is torch.bool
            or audio.dtype.is_complex
            or not (
                audio.dtype.is_floating_point or audio.dtype in _TORCH_INTEGER_DTYPES
            )
        ):
            raise TypeError("raw Torch PCM must have a real numeric dtype")
        return tuple(audio.shape), int(audio.numel())
    if not isinstance(audio, np.ndarray):  # pragma: no cover - caller narrows
        raise TypeError("raw PCM must be a NumPy array or CPU Torch tensor")
    if audio.dtype == np.bool_ or np.issubdtype(audio.dtype, np.complexfloating):
        raise TypeError("raw NumPy PCM must have a real numeric dtype")
    if not (
        np.issubdtype(audio.dtype, np.integer)
        or np.issubdtype(audio.dtype, np.floating)
    ):
        raise TypeError("raw NumPy PCM must have a real numeric dtype")
    return tuple(audio.shape), int(audio.size)


def validate_audio_source(
    audio: object,
    *,
    sample_rate: object = None,
    max_duration_seconds: float = 30.0,
    clip_start_seconds: object = 0.0,
    clip_end_seconds: object = None,
) -> AudioSource:
    """Validate the initial one-item audio contract without decoding files."""

    clip_start, clip_end = validate_clip_range(
        clip_start_seconds,
        clip_end_seconds,
    )

    if isinstance(audio, (str, Path)):
        if sample_rate is not None:
            raise ValueError("sample_rate must be omitted for encoded audio")
        # File existence and type are properties of the handle opened by the
        # native decoder. A Python stat precheck would be both racy and unsafe
        # for Windows device namespaces.
        return AudioSource(
            value=Path(audio),
            kind="encoded",
            sample_rate=None,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
        )

    if isinstance(audio, bytes):
        if sample_rate is not None:
            raise ValueError("sample_rate must be omitted for encoded audio")
        if not audio:
            raise ValueError("encoded audio bytes must not be empty")
        return AudioSource(
            value=audio,
            kind="encoded",
            sample_rate=None,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
        )

    if callable(getattr(audio, "read", None)):
        if sample_rate is not None:
            raise ValueError("sample_rate must be omitted for encoded audio")
        encoded_file = snapshot_file_like(audio)
        assert isinstance(encoded_file, bytes)
        return AudioSource(
            value=encoded_file,
            kind="encoded",
            sample_rate=None,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
        )

    if isinstance(audio, (np.ndarray, Tensor)):
        rate = _validate_sample_rate(sample_rate)
        shape, size = _raw_shape_and_size(audio)
        if len(shape) != 1:
            raise ValueError(
                f"raw PCM must be one-dimensional mono audio, got shape {shape}"
            )
        if size == 0:
            raise ValueError("raw PCM must not be empty")
        # Reject before dtype normalization can allocate a float32 copy. This
        # mirrors the native binding's hard input bound at request admission.
        if size > _MAX_NATIVE_AUDIO_VALUES:
            raise ValueError(
                "raw PCM exceeds the "
                f"{_MAX_NATIVE_AUDIO_VALUES}-value native input limit"
            )
        start_frame, end_frame = _clip_frame_bounds(
            size,
            rate,
            clip_start,
            clip_end,
        )
        duration = (end_frame - start_frame) / rate
        if duration > max_duration_seconds:
            raise ValueError(
                f"audio duration {duration:.6g}s exceeds the {max_duration_seconds:g}s short-form limit"
            )
        return AudioSource(
            value=audio,
            kind="pcm",
            sample_rate=rate,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
        )

    raise TypeError(
        "audio must be a WAV/FLAC/MP3/OGG/Opus/WebM/M4A/MP4/MOV path, bytes, or binary file-like object, "
        "or one-dimensional NumPy/CPU Torch PCM"
    )


def _pcm_to_float32(audio: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(audio, Tensor):
        if audio.device.type != "cpu":
            raise ValueError("raw Torch PCM must be a CPU tensor")
        value = audio.detach()
        if value.dtype.is_floating_point:
            value = value.to(dtype=torch.float32)
        value = value.contiguous().numpy()
    else:
        value = np.asarray(audio)

    if np.issubdtype(value.dtype, np.signedinteger):
        info = np.iinfo(value.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform = value.astype(np.float32) / scale
    elif np.issubdtype(value.dtype, np.unsignedinteger):
        info = np.iinfo(value.dtype)
        midpoint = float(info.max + 1) / 2.0
        waveform = (value.astype(np.float32) - midpoint) / midpoint
    else:
        waveform = value.astype(np.float32, copy=False)

    waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("audio PCM contains NaN or infinity")
    if float(np.abs(waveform).max(initial=0.0)) > 1.000001:
        raise ValueError("floating-point PCM values must lie in [-1, 1]")
    return waveform


def _validate_decoded_audio_result(
    result: object,
) -> tuple[np.ndarray, int, str]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("native audio decoding produced an invalid result")
    waveform, sample_rate, audio_format = result
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.dtype != np.float32
        or waveform.ndim != 1
        or waveform.size == 0
        or not waveform.flags.c_contiguous
    ):
        raise ValueError("native audio decoding produced an invalid waveform")
    # Check the declared native allocation contract before finite/range scans;
    # malformed extension output must not force work over an oversized array.
    if waveform.size > _MAX_ENCODED_DECODE_SAMPLE_VALUES:
        raise ValueError(
            "native audio decoding exceeded the "
            f"{_MAX_ENCODED_DECODE_SAMPLE_VALUES}-value decoded limit"
        )
    if not np.isfinite(waveform).all():
        raise ValueError("native audio decoding produced NaN or infinity")
    if float(np.abs(waveform).max(initial=0.0)) > 1.000001:
        raise ValueError("native audio decoding produced PCM outside [-1, 1]")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) <= 0
    ):
        raise ValueError("native audio decoding produced an invalid sample rate")
    if not isinstance(audio_format, str) or audio_format not in {
        "WAV",
        "WAVEX",
        "FLAC",
        "MP3",
        "OGG",
        "OPUS",
        "WEBM",
        "M4A",
        "MP4",
    }:
        raise ValueError("native audio decoding produced an invalid format")
    return waveform, int(sample_rate), audio_format


def log_mel_spectrogram(
    waveform: np.ndarray | Tensor,
    *,
    config: WhisperPreprocessorConfig = WhisperPreprocessorConfig(),
) -> Tensor:
    """Return HF-compatible fixed-shape float32 features on CPU."""

    shape, _size = _raw_shape_and_size(waveform)
    if len(shape) != 1:
        raise ValueError(f"audio waveform must be one-dimensional, got shape {shape}")
    samples = torch.from_numpy(_pcm_to_float32(waveform))
    if samples.numel() == 0:
        raise ValueError("audio waveform must not be empty")
    if samples.numel() > config.n_samples:
        raise ValueError(
            f"audio has {samples.numel()} samples, maximum is {config.n_samples}"
        )
    if not torch.isfinite(samples).all().item():
        raise ValueError("audio waveform contains NaN or infinity")

    padded = torch.zeros(config.n_samples, dtype=torch.float32)
    padded[: samples.numel()].copy_(samples)
    window = torch.hann_window(config.n_fft, dtype=torch.float32)
    stft = torch.stft(
        padded,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().square()
    mel_spec = (
        mel_filters(
            config.n_fft,
            config.feature_size,
            config.sampling_rate,
        )
        @ magnitudes
    )
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).contiguous()


def prepare_audio(
    audio: AudioSource | object,
    *,
    sample_rate: object = None,
    config: WhisperPreprocessorConfig = WhisperPreprocessorConfig(),
) -> PreparedAudio:
    """Fully prepare one <=30-second source for encoder prefill."""

    if isinstance(audio, AudioSource):
        if sample_rate is not None:
            raise ValueError(
                "sample_rate must be omitted when audio is already an AudioSource"
            )
        source = validate_audio_source(
            audio.value,
            sample_rate=audio.sample_rate,
            max_duration_seconds=float(config.chunk_length),
            clip_start_seconds=audio.clip_start_seconds,
            clip_end_seconds=audio.clip_end_seconds,
        )
        if source.kind != audio.kind:
            raise ValueError(
                f"AudioSource kind {audio.kind!r} does not match its value"
            )
    else:
        source = validate_audio_source(
            audio,
            sample_rate=sample_rate,
            max_duration_seconds=float(config.chunk_length),
        )
    if source.kind == "encoded":
        if isinstance(source.value, bytes):
            decoded = kestrel_native.decode_audio_mono(
                source.value,
                max_duration_seconds=float(config.chunk_length),
                max_decoded_values=_MAX_ENCODED_DECODE_SAMPLE_VALUES,
            )
        elif isinstance(source.value, Path):
            decoded = kestrel_native.decode_audio_file(
                source.value,
                max_duration_seconds=float(config.chunk_length),
                max_decoded_values=_MAX_ENCODED_DECODE_SAMPLE_VALUES,
            )
        else:  # pragma: no cover - validated encoded AudioSource invariant
            raise TypeError(
                "encoded AudioSource must contain bytes or a filesystem path"
            )
        waveform, original_rate, _format = _validate_decoded_audio_result(decoded)
    else:
        if source.sample_rate is None:  # pragma: no cover - AudioSource invariant
            raise ValueError("raw PCM AudioSource is missing sample_rate")
        original_rate = source.sample_rate
        raw_size = _raw_shape_and_size(source.value)[1]  # type: ignore[arg-type]
        start_frame, end_frame = _clip_frame_bounds(
            raw_size,
            original_rate,
            source.clip_start_seconds,
            source.clip_end_seconds,
        )
        waveform = _pcm_to_float32(source.value[start_frame:end_frame])  # type: ignore[index]

    source_duration = (
        raw_size / original_rate
        if source.kind == "pcm"
        else int(waveform.size) / original_rate
    )
    if source.kind == "encoded":
        start_frame, end_frame = _clip_frame_bounds(
            int(waveform.size),
            original_rate,
            source.clip_start_seconds,
            source.clip_end_seconds,
        )
        waveform = np.ascontiguousarray(waveform[start_frame:end_frame])

    original_num_samples = int(waveform.size)
    if original_num_samples == 0:
        raise ValueError("audio must not be empty")
    duration = original_num_samples / original_rate
    if duration > config.chunk_length:
        raise ValueError(
            f"audio duration {duration:.6g}s exceeds the {config.chunk_length}s short-form limit"
        )

    resampled = kestrel_native.resample_audio_mono(
        waveform,
        original_rate,
        config.sampling_rate,
        max_output_values=config.n_samples,
    )
    if (
        not isinstance(resampled, np.ndarray)
        or resampled.dtype != np.float32
        or resampled.ndim != 1
        or resampled.size == 0
        or not resampled.flags.c_contiguous
    ):
        raise ValueError("audio resampling produced an invalid waveform")
    if resampled.size > config.n_samples:
        raise ValueError(
            f"resampled audio has {resampled.size} samples, maximum is {config.n_samples}"
        )
    if not np.isfinite(resampled).all():
        raise ValueError("audio resampling produced NaN or infinity")
    features = log_mel_spectrogram(resampled, config=config)
    return PreparedAudio(
        input_features=features,
        duration_seconds=float(duration),
        original_num_samples=original_num_samples,
        original_sample_rate=original_rate,
        resampled_num_samples=int(resampled.size),
        clip_start_seconds=start_frame / original_rate,
        source_duration_seconds=source_duration,
    )


__all__ = [
    "AudioSource",
    "PreparedAudio",
    "log_mel_spectrogram",
    "prepare_audio",
    "validate_clip_range",
    "validate_audio_source",
]
