"""Pinned waveform staging: one packed async copy per cohort.

The rows a cohort uploads used to travel as one pageable ``.to(device)`` per
distinct waveform length, and Torch finishes a pageable copy with a stream
synchronize -- so the upload drained whatever the compute stream still held.
These pin the packing contract (order, values, buffer reuse) the async copy
replaces it with.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kestrel.models.asr.audio import DecodedAudio
from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime, _WaveformStaging


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _audio(waveform: np.ndarray) -> DecodedAudio:
    return DecodedAudio(waveform, waveform.size / 16_000, waveform.size / 16_000, 0.0)


@requires_cuda
def test_stage_packs_rows_end_to_end_in_order() -> None:
    staging = _WaveformStaging(torch.device("cuda"))
    blocks = [
        np.arange(4, dtype=np.float32),
        np.arange(10, 13, dtype=np.float32),
        np.arange(20, 27, dtype=np.float32),
    ]

    staged = staging.stage(blocks)

    assert staged.shape == (14,)
    assert staged.tolist() == [float(v) for block in blocks for v in block]


@requires_cuda
def test_stage_reuses_its_buffers_without_corrupting_a_live_copy() -> None:
    """Rotating slots: an earlier stage's values survive later stages."""
    staging = _WaveformStaging(torch.device("cuda"), slots=2)
    first = staging.stage([np.full(2048, 1.0, dtype=np.float32)])
    second = staging.stage([np.full(2048, 2.0, dtype=np.float32)])
    third = staging.stage([np.full(2048, 3.0, dtype=np.float32)])

    assert first.max().item() == 1.0
    assert second.max().item() == 2.0
    assert third.max().item() == 3.0


@requires_cuda
def test_batch_features_match_the_pageable_upload() -> None:
    """Staged uploads must not move a single feature value."""
    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = torch.device("cuda")
    runtime.dtype = torch.float32
    rng = np.random.default_rng(7)
    rows = [
        (index, _audio(rng.standard_normal(size).astype(np.float32) * 0.1))
        for index, size in enumerate((16_000, 16_000, 9_600, 24_321))
    ]

    runtime._staging = None
    pageable_features, pageable_mask = runtime._batch_audio_features(rows)
    runtime._staging = _WaveformStaging(runtime.device)
    staged_features, staged_mask = runtime._batch_audio_features(rows)

    assert torch.equal(staged_features, pageable_features)
    assert torch.equal(staged_mask, pageable_mask)
