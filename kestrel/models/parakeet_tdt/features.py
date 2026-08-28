"""Parakeet TDT audio feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from kestrel.models.asr.features import mel_filters
from torch import Tensor


def parakeet_features(waveform: np.ndarray | Tensor) -> tuple[Tensor, Tensor]:
    samples = torch.as_tensor(waveform, dtype=torch.float32)
    if samples.ndim not in (1, 2):
        raise ValueError("Parakeet audio must be one waveform or an equal-length batch")
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    elif samples.shape[0] == 0:
        raise ValueError("Parakeet expects a non-empty audio batch")
    if samples.shape[-1] < 320:
        raise ValueError("Parakeet audio is too short to normalize")
    samples = torch.cat(
        (samples[:, :1], samples[:, 1:] - 0.97 * samples[:, :-1]), dim=1
    )
    spectrum = (
        torch.stft(
            samples,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=torch.hann_window(400, periodic=False, device=samples.device),
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .square()
    )
    mel = mel_filters(512, 128, 16_000).to(samples.device) @ spectrum
    features = torch.log(mel + 2**-24).transpose(1, 2)
    valid_frames = samples.shape[1] // 160
    mask = torch.arange(features.shape[1], device=samples.device)[None] < valid_frames
    mask = mask.expand(samples.shape[0], -1)
    valid = features[:, :valid_frames]
    mean = valid.mean(1, keepdim=True)
    std = valid.std(1, keepdim=True)
    features = ((features - mean) / (std + 1e-5)) * mask.unsqueeze(-1)
    return features.contiguous(), mask.contiguous()


__all__ = ["parakeet_features"]
