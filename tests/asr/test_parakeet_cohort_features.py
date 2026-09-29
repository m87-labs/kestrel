"""One spectrogram per cohort must equal one per distinct waveform length.

``parakeet_cohort_features`` replaces a ``parakeet_features`` call per
distinct length -- around 80 of them for a LibriSpeech cohort of 128 -- with
one spectrogram over the padded batch, and claims the identical tensor. These
hold it to that against the per-length path it replaced: bitwise on CUDA,
where the claim is made and where the engine runs it, and to the last ulp on
CPU, whose transform takes a different path for a clip only a few frames long.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kestrel.models.parakeet_tdt.features import (
    parakeet_cohort_features,
    parakeet_features,
)


def _per_length(
    waveforms: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The path this replaces: a call per distinct length, padded and stacked."""

    groups: dict[int, list[int]] = {}
    for row, waveform in enumerate(waveforms):
        groups.setdefault(waveform.size, []).append(row)
    features: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for _size, indices in groups.items():
        batch = np.stack([waveforms[index] for index in indices])
        rows, row_masks = parakeet_features(torch.from_numpy(batch).to(device))
        for at, index in enumerate(indices):
            features[index] = rows[at]
            masks[index] = row_masks[at]
    width = max(row.shape[0] for row in features.values())
    order = range(len(waveforms))
    return (
        torch.stack(
            [F.pad(features[i], (0, 0, 0, width - features[i].shape[0])) for i in order]
        ),
        torch.stack(
            [F.pad(masks[i], (0, width - masks[i].shape[0])) for i in order]
        ),
    )


def _padded(waveforms: list[np.ndarray]) -> torch.Tensor:
    width = max(waveform.size for waveform in waveforms)
    batch = np.zeros((len(waveforms), width), dtype=np.float32)
    for row, waveform in enumerate(waveforms):
        batch[row, : waveform.size] = waveform
    return torch.from_numpy(batch)


def _noise(sizes: tuple[int, ...]) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [(rng.standard_normal(size) * 0.1).astype(np.float32) for size in sizes]


MIXED = (16_000, 16_000, 9_600, 24_321, 320, 481)


def _both_ways(
    sizes: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Mixed lengths, repeats, the shortest clip allowed, and an odd width."""

    waveforms = _noise(sizes)
    features, mask = parakeet_cohort_features(
        _padded(waveforms).to(device), list(sizes)
    )
    expected_features, expected_mask = _per_length(waveforms, device)
    assert torch.equal(mask, expected_mask)
    # What a row does not own is zero either way -- the per-length path padded
    # its stack with a literal zero, the cohort multiplies by a false mask --
    # so only the frames a row owns are compared.
    assert not features[~mask].any()
    return features[mask], expected_features[expected_mask]


def test_one_spectrogram_matches_one_per_length() -> None:
    features, expected = _both_ways(MIXED, torch.device("cpu"))

    torch.testing.assert_close(features, expected, rtol=0, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_one_spectrogram_is_bit_for_bit_on_cuda() -> None:
    features, expected = _both_ways(MIXED, torch.device("cuda"))

    assert torch.equal(features, expected)


def test_a_one_row_cohort_matches_the_single_waveform_path() -> None:
    waveforms = _noise((5_003,))

    features, mask = parakeet_cohort_features(_padded(waveforms), [5_003])
    expected_features, expected_mask = parakeet_features(
        torch.from_numpy(waveforms[0])
    )

    assert torch.equal(features, expected_features)
    assert torch.equal(mask, expected_mask)


def test_a_cohort_of_equal_rows_is_untouched_by_padding() -> None:
    waveforms = _noise((8_000, 8_000, 8_000))

    features, mask = parakeet_cohort_features(_padded(waveforms), [8_000] * 3)
    expected_features, expected_mask = parakeet_features(
        torch.from_numpy(np.stack(waveforms))
    )

    assert torch.equal(features, expected_features)
    assert torch.equal(mask, expected_mask)
