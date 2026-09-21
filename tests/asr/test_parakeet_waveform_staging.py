"""Pinned waveform staging: one packed async copy per cohort.

The rows a cohort uploads used to travel as one pageable ``.to(device)`` per
distinct waveform length, and Torch finishes a pageable copy with a stream
synchronize -- so the upload drained whatever the compute stream still held.
These pin the packing contract (order, values) the async copy replaces it with.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kestrel.models.asr.audio import DecodedAudio
from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime, _stage_waveforms


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _audio(waveform: np.ndarray) -> DecodedAudio:
    return DecodedAudio(waveform, waveform.size / 16_000, waveform.size / 16_000, 0.0)


@requires_cuda
def test_stage_packs_rows_end_to_end_in_order() -> None:
    blocks = [
        np.arange(4, dtype=np.float32),
        np.arange(10, 13, dtype=np.float32),
        np.arange(20, 27, dtype=np.float32),
    ]

    staged = _stage_waveforms(blocks, torch.device("cuda"))

    assert staged.shape == (14,)
    assert staged.tolist() == [float(v) for block in blocks for v in block]


@requires_cuda
def test_a_staged_copy_is_not_overtaken_by_the_next_one() -> None:
    """Each cohort gets its own pinned buffer and drops it on the way out.

    Nothing here waits on the copy: Torch's caching host allocator records it
    on the pinned block and will not hand that block back until it has
    completed, so a buffer freed mid-flight cannot be repacked underneath a
    copy still reading it. Stage several cohorts behind queued device work and
    every one of them must still read back the values it was given.
    """

    device = torch.device("cuda")
    ballast = torch.randn(4096, 4096, device=device)
    staged = []
    for value in range(1, 6):
        ballast @ ballast  # keep the copies queued behind real work
        staged.append(
            _stage_waveforms([np.full(2**20, value, dtype=np.float32)], device)
        )
    torch.cuda.synchronize()

    assert [int(item.min().item()) for item in staged] == [1, 2, 3, 4, 5]
    assert [int(item.max().item()) for item in staged] == [1, 2, 3, 4, 5]


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

    runtime._pin_waveforms = False
    pageable_features, pageable_mask = runtime._batch_audio_features(rows)
    runtime._pin_waveforms = True
    staged_features, staged_mask = runtime._batch_audio_features(rows)

    assert torch.equal(staged_features, pageable_features)
    assert torch.equal(staged_mask, pageable_mask)
