"""Parakeet TDT audio feature extraction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from kestrel.models.asr.features import mel_filters
from torch import Tensor


def _emphasise(samples: Tensor) -> Tensor:
    return torch.cat(
        (samples[:, :1], samples[:, 1:] - 0.97 * samples[:, :-1]), dim=1
    )


def _log_mel(emphasised: Tensor) -> Tensor:
    """Log-mel energies of a pre-emphasised batch, ``[rows, mel, frames]``."""

    spectrum = (
        torch.stft(
            emphasised,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=torch.hann_window(400, periodic=False, device=emphasised.device),
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .square()
    )
    mel = mel_filters(512, 128, 16_000).to(emphasised.device) @ spectrum
    return torch.log(mel + 2**-24)


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
    features = _log_mel(_emphasise(samples)).transpose(1, 2)
    valid_frames = samples.shape[1] // 160
    mask = torch.arange(features.shape[1], device=samples.device)[None] < valid_frames
    mask = mask.expand(samples.shape[0], -1)
    valid = features[:, :valid_frames]
    mean = valid.mean(1, keepdim=True)
    std = valid.std(1, keepdim=True)
    features = ((features - mean) / (std + 1e-5)) * mask.unsqueeze(-1)
    return features.contiguous(), mask.contiguous()


def parakeet_cohort_features(
    waveforms: Tensor, sizes: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Features for a cohort of unequal waveforms, out of one spectrogram.

    ``waveforms`` is ``[rows, longest]``, each row zero-filled past its own
    ``sizes[row]`` samples. One spectrogram over that batch computes exactly
    what a spectrogram per distinct length computed -- 53 of them for one
    LibriSpeech cohort of 128, and ``torch.stft`` costs a quarter of a
    millisecond a call before a kernel runs. Pre-emphasis is per row and the
    padding is zeroed after it, so no frame of a row reads a sample the
    shorter input would not have had (its own frames end at ``size // 160``,
    and a window reaching past that read the zeros ``torch.stft`` pads with
    anyway); and a 512-point transform does not depend on how many frames are
    batched with it. Bitwise on CUDA, checked over 11 M values on both sets.
    (CPU takes a different path through its transform for a clip only a few
    frames long, and differs there in the last ulp.)

    A row's mean and standard deviation do not carry over, because Torch picks
    its reduction by the shape and layout it is handed. Those are still taken
    a length group at a time, out of a block laid out the way the per-length
    call laid it out: bitwise what it was, for one copy and two reductions per
    group instead of a whole spectrogram.
    """

    if waveforms.ndim != 2 or waveforms.shape[0] != len(sizes):
        raise ValueError("Parakeet cohort features need one size per waveform row")
    if not sizes:
        raise ValueError("Parakeet expects a non-empty audio batch")
    if min(sizes) < 320:
        raise ValueError("Parakeet audio is too short to normalize")

    device = waveforms.device
    lengths = torch.tensor(sizes, device=device)
    inside = torch.arange(waveforms.shape[1], device=device)[None] < lengths[:, None]
    energies = _log_mel(torch.where(inside, _emphasise(waveforms), 0.0))

    groups: dict[int, list[int]] = {}
    for row, size in enumerate(sizes):
        groups.setdefault(size, []).append(row)
    mean = torch.empty(
        (len(sizes), 1, energies.shape[1]), dtype=energies.dtype, device=device
    )
    std = torch.empty_like(mean)
    for size, rows in groups.items():
        where = torch.tensor(rows, device=device)
        block = energies[:, :, : 1 + size // 160][where].transpose(1, 2)
        block = block[:, : size // 160]
        mean[where] = block.mean(1, keepdim=True)
        std[where] = block.std(1, keepdim=True)

    mask = torch.arange(energies.shape[-1], device=device)[None] < (
        lengths // 160
    )[:, None]
    features = energies.transpose(1, 2)
    features = ((features - mean) / (std + 1e-5)) * mask.unsqueeze(-1)
    return features.contiguous(), mask.contiguous()


__all__ = ["parakeet_cohort_features", "parakeet_features"]
