from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch

import kestrel.models.asr.audio as asr_audio_module
import kestrel.models.whisper.audio as audio_module
from kestrel.models.whisper.audio import (
    AudioSource,
    PreparedAudio,
    log_mel_spectrogram,
    prepare_audio,
    validate_audio_source,
)


def test_public_package_pins_native_without_codec_or_backend_dependencies() -> None:
    project = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"kestrel-native==0.1.8"' in project
    assert "soundfile" not in project.lower()
    assert "soxr" not in project.lower()
    assert "kestrel-megakernel" not in project.lower()


def test_raw_pcm_contract_rejects_ambiguous_inputs() -> None:
    mono = np.zeros(1600, dtype=np.float32)
    with pytest.raises(TypeError, match="sample_rate"):
        validate_audio_source(mono)
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_audio_source(np.zeros((2, 1600), dtype=np.float32), sample_rate=16000)
    with pytest.raises(ValueError, match="30s"):
        validate_audio_source(np.zeros(480001, dtype=np.int16), sample_rate=16000)
    with pytest.raises(ValueError, match="CPU tensor"):
        validate_audio_source(torch.empty(1, device="meta"), sample_rate=16000)
    with pytest.raises(ValueError, match="32-bit"):
        validate_audio_source(mono, sample_rate=1 << 32)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    (
        (True, None, "clip_start_seconds"),
        (-1.0, None, "clip_start_seconds"),
        (float("nan"), None, "clip_start_seconds"),
        (0.0, False, "clip_end_seconds"),
        (1.0, 1.0, "greater than"),
        (2.0, 1.0, "greater than"),
    ),
)
def test_clip_range_validation_fails_closed(start, end, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_audio_source(
            np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            clip_start_seconds=start,
            clip_end_seconds=end,
        )


def test_raw_clip_is_sliced_before_normalization_and_short_form_limit() -> None:
    waveform = np.linspace(-0.5, 0.5, 40 * 16_000, dtype=np.float32)
    source = validate_audio_source(
        waveform,
        sample_rate=16_000,
        clip_start_seconds=1.0,
        clip_end_seconds=1.1,
    )

    prepared = prepare_audio(source)

    assert prepared.original_num_samples == 1600
    assert prepared.duration_seconds == pytest.approx(0.1)
    assert prepared.clip_start_seconds == pytest.approx(1.0)
    assert prepared.source_duration_seconds == pytest.approx(40.0)


def test_raw_pcm_native_value_limit_precedes_normalization_allocation() -> None:
    oversized = np.broadcast_to(
        np.zeros(1, dtype=np.float64),
        (32 * 1024 * 1024 + 1,),
    )

    with pytest.raises(ValueError, match="33554432-value native input limit"):
        validate_audio_source(oversized, sample_rate=2_000_000)


def test_encoded_audio_rejects_explicit_sample_rate(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"not decoded during validation")
    with pytest.raises(ValueError, match="must be omitted"):
        validate_audio_source(path, sample_rate=16000)
    assert validate_audio_source(path).kind == "encoded"


def test_binary_file_like_is_snapshotted_from_its_current_position() -> None:
    class PartialReader:
        def __init__(self, value: bytes) -> None:
            self.value = value
            self.offset = 2

        def read(self, size: int) -> bytes:
            end = min(len(self.value), self.offset + min(size, 3))
            result = self.value[self.offset : end]
            self.offset = end
            return result

    source = validate_audio_source(PartialReader(b"xxencoded"))

    assert source == AudioSource(value=b"encoded", kind="encoded", sample_rate=None)


def test_binary_file_like_fails_closed_on_type_and_byte_limit(monkeypatch) -> None:
    with pytest.raises(TypeError, match="must return bytes"):
        validate_audio_source(io.StringIO("encoded"))

    monkeypatch.setattr(asr_audio_module, "_MAX_ENCODED_BYTES", 8)
    with pytest.raises(ValueError, match="8-byte input limit"):
        validate_audio_source(io.BytesIO(b"123456789"))

    with pytest.raises(ValueError, match="sample_rate must be omitted"):
        validate_audio_source(io.BytesIO(b"encoded"), sample_rate=16_000)


def test_encoded_path_admission_does_not_touch_the_filesystem(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "missing.flac"

    def refuse_filesystem_touch(*_args, **_kwargs):
        pytest.fail("encoded path admission must not inspect the filesystem")

    with monkeypatch.context() as path_guard:
        path_guard.setattr(type(path), "is_file", refuse_filesystem_touch)
        path_guard.setattr(type(path), "stat", refuse_filesystem_touch)
        path_guard.setattr(type(path), "read_bytes", refuse_filesystem_touch)
        source = validate_audio_source(path)

    assert source == AudioSource(value=path, kind="encoded", sample_rate=None)


@pytest.mark.parametrize(
    "audio_format",
    ("WAV", "WAVEX", "FLAC", "MP3", "OGG", "OPUS", "WEBM", "M4A", "MP4"),
)
def test_encoded_bytes_use_bounded_native_decode(
    monkeypatch,
    audio_format: str,
) -> None:
    waveform = np.linspace(-0.5, 0.5, 300, dtype=np.float32)
    calls: list[tuple[bytes, float, int]] = []

    def decode(
        encoded: bytes,
        *,
        max_duration_seconds: float,
        max_decoded_values: int,
    ) -> tuple[np.ndarray, int, str]:
        calls.append((encoded, max_duration_seconds, max_decoded_values))
        return waveform, 100, audio_format

    monkeypatch.setattr(audio_module.kestrel_native, "decode_audio_mono", decode)
    prepared = prepare_audio(b"encoded")

    assert prepared.original_num_samples == 300
    assert prepared.original_sample_rate == 100
    assert calls == [(b"encoded", 30.0, 30 * 48_000 * 8)]


def test_encoded_path_uses_native_file_api_without_python_read(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "audio.flac"
    path.write_bytes(b"native owns the opened file")
    calls: list[tuple[object, float, int]] = []

    def decode_file(
        source_path: object,
        *,
        max_duration_seconds: float,
        max_decoded_values: int,
    ) -> tuple[np.ndarray, int, str]:
        calls.append((source_path, max_duration_seconds, max_decoded_values))
        return np.zeros(160, dtype=np.float32), 16_000, "FLAC"

    monkeypatch.setattr(audio_module.kestrel_native, "decode_audio_file", decode_file)
    with monkeypatch.context() as path_guard:
        path_guard.setattr(
            type(path),
            "read_bytes",
            lambda _path: pytest.fail("model code must not buffer an encoded path"),
        )
        path_guard.setattr(
            type(path),
            "is_file",
            lambda _path: pytest.fail("model code must not precheck an encoded path"),
        )
        path_guard.setattr(
            type(path),
            "stat",
            lambda _path, **_kwargs: pytest.fail(
                "model code must defer encoded path metadata to native admission"
            ),
        )
        prepared = prepare_audio(path)

    assert prepared.original_num_samples == 160
    assert prepared.original_sample_rate == 16_000
    assert calls == [(path, 30.0, 30 * 48_000 * 8)]


def test_native_decode_refusal_propagates_without_fallback(monkeypatch) -> None:
    calls = 0

    def refuse(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("native decoder refused input")

    monkeypatch.setattr(audio_module.kestrel_native, "decode_audio_mono", refuse)
    with pytest.raises(ValueError, match="native decoder refused input"):
        prepare_audio(b"encoded")
    assert calls == 1


@pytest.mark.parametrize(
    ("decoded", "message"),
    (
        ((np.zeros(0, dtype=np.float32), 16_000, "WAV"), "invalid waveform"),
        ((np.zeros(16, dtype=np.float64), 16_000, "WAV"), "invalid waveform"),
        ((np.zeros((16, 1), dtype=np.float32), 16_000, "WAV"), "invalid waveform"),
        ((np.zeros(32, dtype=np.float32)[::2], 16_000, "WAV"), "invalid waveform"),
        ((np.array([np.nan], dtype=np.float32), 16_000, "WAV"), "NaN"),
        ((np.array([np.inf], dtype=np.float32), 16_000, "WAV"), "NaN"),
        ((np.array([1.1], dtype=np.float32), 16_000, "WAV"), r"\[-1, 1\]"),
        ((np.zeros(16, dtype=np.float32), True, "WAV"), "sample rate"),
        ((np.zeros(16, dtype=np.float32), 0, "WAV"), "sample rate"),
        ((np.zeros(16, dtype=np.float32), 16_000, "AAC"), "format"),
        ((np.zeros(16, dtype=np.float32), 16_000, None), "format"),
        ((np.zeros(16, dtype=np.float32), 16_000), "invalid result"),
    ),
    ids=(
        "empty",
        "float64",
        "two_dimensional",
        "noncontiguous",
        "nan",
        "infinity",
        "out_of_range",
        "boolean_rate",
        "zero_rate",
        "unknown_format",
        "non_string_format",
        "wrong_tuple_length",
    ),
)
def test_invalid_native_decode_result_fails_closed(
    monkeypatch,
    decoded: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        audio_module.kestrel_native,
        "decode_audio_mono",
        lambda *_args, **_kwargs: decoded,
    )

    with pytest.raises(ValueError, match=message):
        prepare_audio(b"encoded")


def test_oversized_native_decode_result_is_rejected_before_value_scans(
    monkeypatch,
) -> None:
    monkeypatch.setattr(audio_module, "_MAX_ENCODED_DECODE_SAMPLE_VALUES", 8)
    monkeypatch.setattr(
        audio_module.kestrel_native,
        "decode_audio_mono",
        lambda *_args, **_kwargs: (np.zeros(9, dtype=np.float32), 16_000, "WAV"),
    )
    monkeypatch.setattr(
        audio_module.np,
        "isfinite",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized decoded output must be rejected before finite scans"
        ),
    )

    with pytest.raises(ValueError, match="8-value decoded limit"):
        prepare_audio(b"encoded")


def test_float_pcm_range_and_finiteness_fail_before_stft() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        prepare_audio(np.array([2.0], dtype=np.float32), sample_rate=16000)
    with pytest.raises(ValueError, match="NaN"):
        prepare_audio(np.array([np.nan], dtype=np.float32), sample_rate=16000)


def test_prepared_audio_and_source_carriers_fail_closed() -> None:
    features = torch.zeros((128, 3000), dtype=torch.float32)
    with pytest.raises(ValueError, match="inconsistent"):
        PreparedAudio(
            input_features=features,
            duration_seconds=0.2,
            original_num_samples=1600,
            original_sample_rate=16000,
            resampled_num_samples=1600,
        )
    with pytest.raises(ValueError, match="kind"):
        prepare_audio(
            AudioSource(
                value=np.zeros(1600, dtype=np.float32),
                kind="encoded",  # type: ignore[arg-type]
                sample_rate=16000,
            )
        )
    with pytest.raises(ValueError, match="outside its source duration"):
        PreparedAudio(
            input_features=features,
            duration_seconds=0.1,
            original_num_samples=1600,
            original_sample_rate=16000,
            resampled_num_samples=1600,
            clip_start_seconds=1.0,
            source_duration_seconds=1.05,
        )


def test_log_mel_matches_transformers_oracle() -> None:
    transformers = pytest.importorskip("transformers", minversion="4.56.0")
    rng = np.random.default_rng(7)
    waveform = (0.05 * rng.standard_normal(32000)).astype(np.float32)

    ours = log_mel_spectrogram(waveform)
    oracle = transformers.WhisperFeatureExtractor(feature_size=128)(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features[0]

    assert ours.shape == (128, 3000)
    assert ours.dtype is torch.float32
    torch.testing.assert_close(ours, oracle, rtol=0.0, atol=0.0)


def test_resampling_returns_fixed_features() -> None:
    waveform = np.linspace(-0.25, 0.25, 800, dtype=np.float32)
    prepared = prepare_audio(waveform, sample_rate=8000)
    assert prepared.input_features.shape == (128, 3000)
    assert prepared.original_sample_rate == 8000
    assert prepared.original_num_samples == 800
    assert prepared.resampled_num_samples == 1600
    assert prepared.duration_seconds == pytest.approx(0.1)


def test_resampling_uses_bounded_native_api(monkeypatch) -> None:
    waveform = np.linspace(-0.25, 0.25, 800, dtype=np.float32)
    calls: list[tuple[np.ndarray, int, int, int]] = []

    def resample(
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
        *,
        max_output_values: int,
    ) -> np.ndarray:
        calls.append((samples, source_rate, target_rate, max_output_values))
        return np.repeat(samples, 2)

    monkeypatch.setattr(audio_module.kestrel_native, "resample_audio_mono", resample)
    prepared = prepare_audio(waveform, sample_rate=8000)

    assert prepared.resampled_num_samples == 1600
    assert len(calls) == 1
    samples, source_rate, target_rate, max_output_values = calls[0]
    np.testing.assert_array_equal(samples, waveform)
    assert source_rate == 8000
    assert target_rate == 16000
    assert max_output_values == 480000


def test_native_resample_refusal_propagates_without_fallback(monkeypatch) -> None:
    calls = 0

    def refuse(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("native resampler refused ratio")

    monkeypatch.setattr(audio_module.kestrel_native, "resample_audio_mono", refuse)
    with pytest.raises(ValueError, match="native resampler refused ratio"):
        prepare_audio(np.zeros(800, dtype=np.float32), sample_rate=8000)
    assert calls == 1


@pytest.mark.parametrize(
    "invalid",
    (
        [0.0],
        np.zeros(1600, dtype=np.float64),
        np.zeros((1600, 1), dtype=np.float32),
        np.zeros(3200, dtype=np.float32)[::2],
        np.array([np.nan], dtype=np.float32),
    ),
    ids=("non_array", "float64", "two_dimensional", "noncontiguous", "nan"),
)
def test_invalid_native_resample_output_fails_closed(monkeypatch, invalid) -> None:
    monkeypatch.setattr(
        audio_module.kestrel_native,
        "resample_audio_mono",
        lambda *_args, **_kwargs: invalid,
    )

    with pytest.raises(ValueError, match="invalid waveform|NaN"):
        prepare_audio(np.zeros(800, dtype=np.float32), sample_rate=8000)


def test_oversized_native_resample_output_is_rejected_before_value_scans(
    monkeypatch,
) -> None:
    oversized = np.zeros(9, dtype=np.float32)
    original_isfinite = np.isfinite
    monkeypatch.setattr(
        audio_module.kestrel_native,
        "resample_audio_mono",
        lambda *_args, **_kwargs: oversized,
    )

    def refuse_oversized_scan(values, *args, **kwargs):
        if values is oversized:
            pytest.fail(
                "oversized resample output must be rejected before finite scans"
            )
        return original_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(audio_module.np, "isfinite", refuse_oversized_scan)

    with pytest.raises(ValueError, match="9 samples, maximum is 8"):
        prepare_audio(
            np.zeros(8, dtype=np.float32),
            sample_rate=16_000,
            config=audio_module.WhisperPreprocessorConfig(n_samples=8),
        )


# Frozen libFLAC 1.5.0 fixture: exactly 1,600 frames of mono PCM16 zeroes at
# 16 kHz. Decoded PCM-byte SHA-256: 5a312281df4bd8dfbb4d4a94ad0bf44d01bb8cfced1206b90e21b4ca0568cdb1.
# Encoded-byte SHA-256: 8239dd5608b00c9b3be8eb055452ea00437fb04e977ee5b27915714d4c74ac8d.
_FLAC_PCM16_MONO_16KHZ = base64.b64decode(
    "ZkxhQwAAACIQABAAAAANAAANA+gA8AAABkCfCt/b50DLnNgYijV2HiR2hAAAKCAAAAByZWZl"
    "cmVuY2UgbGliRkxBQyAxLjUuMCAyMDI1MDIxMQAAAAD/+HUIAAY/NAAAAGe/"
)


def test_flac_bytes_use_native_decode_and_prepare_audio() -> None:
    waveform, sample_rate, detected_format = (
        audio_module.kestrel_native.decode_audio_mono(
            _FLAC_PCM16_MONO_16KHZ,
            max_duration_seconds=30.0,
            max_decoded_values=30 * 48_000 * 8,
        )
    )
    prepared = prepare_audio(_FLAC_PCM16_MONO_16KHZ)

    assert waveform.shape == (1600,)
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    np.testing.assert_array_equal(waveform, np.zeros(1600, dtype=np.float32))
    assert sample_rate == 16000
    assert detected_format == "FLAC"
    assert prepared.original_num_samples == 1600
    assert prepared.original_sample_rate == 16000
    assert prepared.resampled_num_samples == 1600
    assert prepared.duration_seconds == pytest.approx(0.1)


def test_flac_path_uses_native_decode_and_prepare_audio(tmp_path) -> None:
    path = tmp_path / "silence.flac"
    path.write_bytes(_FLAC_PCM16_MONO_16KHZ)

    prepared = prepare_audio(path)

    assert prepared.original_num_samples == 1600
    assert prepared.original_sample_rate == 16000
    assert prepared.resampled_num_samples == 1600
    assert prepared.duration_seconds == pytest.approx(0.1)
