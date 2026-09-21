"""Cut long Parakeet audio at pauses into contiguous segments of at most 30 s.

Speech boundaries come from frame energy or a checkpoint's VAD head. A cut is
the midpoint of the last usable pause before the cap; no overlap or context
margins are added because both measured worse for the transducer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np

from kestrel.models.asr.audio import AudioChunks, DecodedAudio


SEGMENT_SECONDS = 30.0
MIN_PAUSE_SECONDS = 0.2
MIN_SEGMENT_SECONDS = 1.0
# The pause source runs over one block at a time so a 24 h file never lands in
# memory whole. Two minutes of 16 kHz mono is 3.8 MiB of PCM, and a subsampler
# pass over it costs about what one batch of eight 30 s segments already costs.
BLOCK_SECONDS = 120.0
# `parakeet_features` refuses to normalize anything shorter.
_MIN_SEGMENT_SAMPLES = 320
# `AudioChunks.chunks` bounds a resampled block at `ceil(seconds * rate) + 16`
# output samples, so a clip whose duration is exactly the cap can still decode
# to a few samples past it. `fits_one_segment` allows for that; anything it
# turns down goes through the walk, which measures the samples it actually got.
_RESAMPLE_SLACK_SAMPLES = 16
_ENERGY_FRAME_SECONDS = 0.02
_ENERGY_FLOOR_PERCENTILE = 2.0
_ENERGY_LOUD_PERCENTILE = 90.0
_ENERGY_SPEECH_DB = 8.0
# A pause source reports region edges on its own frame grid, so a gap of
# exactly `min_pause` comes out as a difference of two multiples of that frame
# -- and 281.0 - 280.8 is 0.19999999999998863, not 0.2. Ties go to the pause.
_PAUSE_EPSILON = 1e-6

# A pause source: speech regions in seconds, relative to the waveform's start.
SpeechRegions = Callable[[np.ndarray, int], list[tuple[float, float]]]


def energy_speech(
    waveform: np.ndarray, sample_rate: int
) -> list[tuple[float, float]]:
    """Speech by frame energy, for checkpoints that carry no VAD head.

    Frame energies in dB against a noise floor taken as a low percentile of the
    block: anything `_ENERGY_SPEECH_DB` above the floor is speech. When the
    block has no dynamic range to speak of -- the loud percentile sits within
    that margin of the floor -- there is nothing to measure a pause against, so
    the block counts as speech rather than being dropped on a guess, and the
    walk falls back to cutting at the cap.
    """

    duration = waveform.size / sample_rate
    frame = max(int(_ENERGY_FRAME_SECONDS * sample_rate), 1)
    count = waveform.size // frame
    if count == 0:
        return [(0.0, duration)]
    blocks = waveform[: count * frame].reshape(count, frame).astype(np.float32)
    energy = 10 * np.log10(np.square(blocks).mean(1) + 1e-10)
    floor = float(np.percentile(energy, _ENERGY_FLOOR_PERCENTILE))
    loud = float(np.percentile(energy, _ENERGY_LOUD_PERCENTILE))
    if loud - floor < _ENERGY_SPEECH_DB:
        return [(0.0, duration)]
    speaking = energy > floor + _ENERGY_SPEECH_DB
    edges = np.flatnonzero(np.diff(np.r_[0, speaking.astype(np.int8), 0]))
    return [
        (start * _ENERGY_FRAME_SECONDS, min(end * _ENERGY_FRAME_SECONDS, duration))
        for start, end in edges.reshape(-1, 2)
    ]


def pauses_from_speech(
    speech: list[tuple[float, float]],
    duration: float,
    *,
    min_pause: float = MIN_PAUSE_SECONDS,
) -> list[tuple[float, float]]:
    """Gaps of at least `min_pause` between (and around) the speech regions."""

    least = min_pause - _PAUSE_EPSILON
    pauses: list[tuple[float, float]] = []
    previous = 0.0
    for start, end in sorted(speech):
        if start - previous >= least:
            pauses.append((previous, start))
        previous = max(previous, end)
    if duration - previous >= least:
        pauses.append((previous, duration))
    return pauses


def next_cut(pauses: list[tuple[float, float]]) -> float:
    """Where a segment starting at zero ends, given the pauses ahead of it.

    The last pause lying fully inside `(MIN_SEGMENT_SECONDS, SEGMENT_SECONDS)`
    wins, which makes segments as long as the cap allows while keeping every
    cut in silence. Failing that, any pause whose midpoint lands in that window
    is used, and failing that the cut falls at the cap.
    """

    inside = [
        item
        for item in pauses
        if item[0] >= MIN_SEGMENT_SECONDS and item[1] <= SEGMENT_SECONDS
    ]
    if not inside:
        inside = [
            item
            for item in pauses
            if MIN_SEGMENT_SECONDS <= (item[0] + item[1]) / 2 <= SEGMENT_SECONDS
        ]
    if not inside:
        return SEGMENT_SECONDS
    first, last = inside[-1]
    return (first + last) / 2


def fits_one_segment(source: AudioChunks) -> bool:
    """Whether `source` is short enough that `pause_segments` emits it whole.

    Decided from the clip's duration, before a sample is read, so a caller can
    know the shape of a batch without cutting it. The answer is the one the
    walk over the decoded samples would give, and conservative where it cannot
    be: a resampled clip is measured against the most samples the resampler may
    return, so a clip this accepts is never one the walk would cut.
    """

    rate = source.target_sample_rate
    samples = source.duration_seconds * rate
    most = (
        round(samples)
        if source.sample_rate == rate
        else math.ceil(samples) + _RESAMPLE_SLACK_SAMPLES
    )
    return most <= round(SEGMENT_SECONDS * rate)


def pause_segments(
    source: AudioChunks, speech: SpeechRegions
) -> Iterator[DecodedAudio]:
    """Cut `source` at pauses into contiguous segments of at most 30 s.

    The source decodes forwards only, so the walk runs against a carry buffer:
    blocks arrive, `speech` marks them, and a cut is emitted once more than a
    segment's worth is in hand. That cut is the one a walk over the finished
    file would make -- every candidate pause lies inside the first 30 s of the
    carry, which is fully read by then. A pause still running at the end of the
    carry is usable too: whatever its true end, the midpoint measured so far is
    inside it, so the cut still lands in silence.
    """

    rate = source.target_sample_rate
    cap_samples = round(SEGMENT_SECONDS * rate)
    if fits_one_segment(source):
        # A clip that already fits in one segment is emitted whole, and the
        # pause marks are only ever read to choose a cut -- so there is no
        # cut to choose and no reason to mark anything. Skipping the pause
        # source here is what most requests do: every LibriSpeech and AMI
        # utterance, and any clip a caller already segmented.
        for block in source.chunks(BLOCK_SECONDS, boundary_search_seconds=0.0):
            yield DecodedAudio(
                block.waveform,
                block.waveform.size / rate,
                block.source_duration_seconds,
                block.clip_start_seconds,
            )
        return
    carry = np.empty(0, dtype=np.float32)
    regions: list[tuple[float, float]] = []
    origin = source_duration = 0.0
    consumed = cuts = 0
    started = False
    for block in source.chunks(BLOCK_SECONDS, boundary_search_seconds=0.0):
        if not started:
            origin, started = block.clip_start_seconds, True
        source_duration = block.source_duration_seconds
        offset = carry.size / rate
        regions.extend(
            (start + offset, end + offset)
            for start, end in speech(block.waveform, rate)
        )
        carry = (
            block.waveform
            if carry.size == 0
            else np.concatenate((carry, block.waveform))
        )
        while carry.size > cap_samples:
            cut = next_cut(pauses_from_speech(regions, carry.size / rate))
            index = round(cut * rate)
            seconds = index / rate
            # `.copy()` so a yielded segment does not pin the whole carry.
            piece, carry = carry[:index].copy(), carry[index:]
            if _has_speech(regions, seconds):
                yield DecodedAudio(
                    piece, seconds, source_duration, origin + consumed / rate
                )
            consumed += index
            cuts += 1
            regions = [
                (max(start - seconds, 0.0), end - seconds)
                for start, end in regions
                if end > seconds
            ]
    if not started:
        return
    seconds = carry.size / rate
    # A recording that fits in one segment goes through whole, so audio too
    # short for the recogniser still fails the way it always has. A remainder
    # that small left behind by a cut carries nothing and is dropped.
    if cuts and (
        carry.size < _MIN_SEGMENT_SAMPLES or not _has_speech(regions, seconds)
    ):
        return
    yield DecodedAudio(
        carry.copy(), seconds, source_duration, origin + consumed / rate
    )


def _has_speech(regions: list[tuple[float, float]], seconds: float) -> bool:
    """Whether any speech falls in `[0, seconds)` -- silence is never decoded."""

    return any(end > 0 and start < seconds for start, end in regions)


__all__ = [
    "BLOCK_SECONDS",
    "MIN_PAUSE_SECONDS",
    "MIN_SEGMENT_SECONDS",
    "SEGMENT_SECONDS",
    "SpeechRegions",
    "energy_speech",
    "fits_one_segment",
    "next_cut",
    "pause_segments",
    "pauses_from_speech",
]
