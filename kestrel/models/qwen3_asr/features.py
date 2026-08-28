"""Qwen3-ASR audio feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from kestrel.models.asr.features import mel_filters
from torch import Tensor


def _feature_lengths(num_samples: int) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("Qwen3-ASR expects non-empty audio")
    valid_frames = num_samples // 160
    feature_frames = max(num_samples, 8_000) // 160
    return valid_frames, ((feature_frames + 99) // 100) * 100


def qwen3_asr_features(waveform: np.ndarray | Tensor) -> tuple[Tensor, Tensor]:
    samples = torch.as_tensor(waveform, dtype=torch.float32)
    if samples.ndim not in (1, 2):
        raise ValueError(
            "Qwen3-ASR audio must be one waveform or an equal-length batch"
        )
    single = samples.ndim == 1
    if not single and samples.shape[0] == 0:
        raise ValueError("Qwen3-ASR expects a non-empty audio batch")
    valid_samples = samples.shape[-1]
    valid_frames, padded_frames = _feature_lengths(valid_samples)
    if valid_samples < 8_000:
        samples = torch.nn.functional.pad(samples, (0, 8_000 - valid_samples))
    spectrum = (
        torch.stft(
            samples,
            n_fft=400,
            hop_length=160,
            window=torch.hann_window(400, device=samples.device),
            return_complex=True,
        )[..., :-1]
        .abs()
        .square()
    )
    mel = mel_filters(400, 128, 16_000).to(samples.device) @ spectrum
    log_mel = mel.clamp_min(1e-10).log10()
    floor = (
        log_mel.max() - 8 if single else log_mel.amax(dim=(-2, -1), keepdim=True) - 8
    )
    features = (torch.maximum(log_mel, floor) + 4) / 4
    if features.shape[-1] < padded_frames:
        features = torch.nn.functional.pad(
            features, (0, padded_frames - features.shape[-1])
        )
    if single:
        features = features.unsqueeze(0)
    mask = torch.arange(padded_frames, device=samples.device)[None] < valid_frames
    if not single:
        mask = mask.expand(samples.shape[0], -1)
    return features.contiguous(), mask.contiguous()


__all__ = ["qwen3_asr_features"]
