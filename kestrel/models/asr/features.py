"""Shared Slaney-normalized Mel filter bank."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    mel = 3 * frequency / 200
    if isinstance(frequency, np.ndarray):
        logarithmic = frequency >= 1_000
        mel[logarithmic] = 15 + np.log(frequency[logarithmic] / 1_000) * (
            27 / np.log(6.4)
        )
    elif frequency >= 1_000:
        mel = 15 + np.log(frequency / 1_000) * (27 / np.log(6.4))
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    frequency = 200 * mel / 3
    logarithmic = mel >= 15
    frequency[logarithmic] = 1_000 * np.exp(
        (mel[logarithmic] - 15) * (np.log(6.4) / 27)
    )
    return frequency


@lru_cache(maxsize=8)
def mel_filters(n_fft: int, n_mels: int, sample_rate: int) -> Tensor:
    bins = n_fft // 2 + 1
    edges = _mel_to_hz(
        np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2)
    )
    fft = np.linspace(0, sample_rate / 2, bins)
    delta = np.diff(edges)
    slopes = edges[None] - fft[:, None]
    filters = np.maximum(
        0, np.minimum(-slopes[:, :-2] / delta[:-1], slopes[:, 2:] / delta[1:])
    )
    filters *= 2 / (edges[2:] - edges[:-2])[None]
    return torch.from_numpy(filters.astype(np.float32).T.copy())


__all__ = ["mel_filters"]
