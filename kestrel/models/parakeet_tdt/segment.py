"""Pause-aligned segmentation of long audio for Parakeet TDT.

Measured with ``nvidia/parakeet-tdt-0.6b-v3`` over six Earnings-22 calls of
15-22 minutes:

=========================================================  =====
protocol                                                     WER
=========================================================  =====
pause-aligned 30 s segments, no overlap (VAD-head pauses)   6.37
pause-aligned 30 s segments, no overlap (energy pauses)     6.81
pause-aligned 30 s segments, +/- 4 s context margins        6.60
fixed 30 s grid, no overlap                                 7.97
whole recording, full attention                            10.42
fixed 180 s windows, no overlap (what this used to do)     10.82
whole recording, local attention 512 / 256 / 128 / 64      10.39-12.70
=========================================================  =====

So: cut the recording at pauses, never mid-word, cap every segment at 30 s,
run full attention inside the segment, and join the text in order. Overlap and
context margins are deliberately absent -- they make a transducer *worse*. It
emits a window's last words only when that window's audio ends, so a margin
that hides the true end loses them, and a window that opens mid-speech deletes
its first words. Contiguous, pause-aligned cuts avoid both.

The cut policy is one walk from the start of the recording: the next cut is the
midpoint of the last pause that ends before ``start + cap`` and begins after
``start + min_segment``; with no such pause the cut falls at the cap. A pause is
a gap of at least ``MIN_PAUSE_SECONDS`` between speech regions, and speech comes
from one of two sources behind the ``SpeechDetector`` interface -- the loaded
checkpoint's own head (``vad.VadHeadSpeech``) or ``energy_speech`` below. The
walk, the per-segment run, the text join and the timestamps are shared; only the
detector differs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np

from kestrel.models.asr.audio import DecodedAudio


SEGMENT_SECONDS = 30.0
MIN_PAUSE_SECONDS = 0.2
MIN_SEGMENT_SECONDS = 1.0
# The detector runs over one block at a time so a 24 h file never lands in
# memory whole. Two minutes of 16 kHz mono is 3.8 MiB of PCM, and a subsampler
# pass over it costs about what one batch of eight 30 s segments already costs.
BLOCK_SECONDS = 120.0
# ``parakeet_features`` refuses anything shorter; a cut can leave a remainder
# this small and it carries no transcribable audio.
_MIN_SEGMENT_SAMPLES = 320
_ENERGY_FRAME_SECONDS = 0.02
_ENERGY_FLOOR_PERCENTILE = 2.0
_ENERGY_SPEECH_DB = 8.0
_ENERGY_LOUD_PERCENTILE = 90.0
# A detector reports region edges on its own frame grid, so a gap of exactly
# `min_pause` comes out as a difference of two multiples of the frame -- and
# 281.0 - 280.8 is 0.19999999999998863, not 0.2. Ties go to the pause.
_PAUSE_EPSILON = 1e-6


class SpeechDetector(Protocol):
    """Speech regions in seconds, relative to the start of ``waveform``."""

    def __call__(
        self, waveform: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]: ...


def energy_speech(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = _ENERGY_FRAME_SECONDS,
    floor_percentile: float = _ENERGY_FLOOR_PERCENTILE,
    speech_db: float = _ENERGY_SPEECH_DB,
) -> list[tuple[float, float]]:
    """Pause source (b) -- frame energy, for checkpoints with no VAD head.

    Frame energies in dB against a noise floor taken as a low percentile of the
    block: anything ``speech_db`` above the floor is speech. When the block has
    no dynamic range to speak of -- the loud percentile sits within ``speech_db``
    of the floor -- there is nothing to call a pause, so the whole block counts
    as speech and the walk falls back to cutting at the cap.
    """

    duration = waveform.size / sample_rate
    frame = max(int(frame_seconds * sample_rate), 1)
    count = waveform.size // frame
    if count == 0:
        return [(0.0, duration)]
    blocks = waveform[: count * frame].reshape(count, frame).astype(np.float32)
    energy = 10 * np.log10(np.square(blocks).mean(1) + 1e-10)
    floor = float(np.percentile(energy, floor_percentile))
    loud = float(np.percentile(energy, _ENERGY_LOUD_PERCENTILE))
    if loud - floor < speech_db:
        return [(0.0, duration)]
    return _runs(energy > floor + speech_db, frame_seconds, duration)


def _runs(
    speaking: np.ndarray, frame_seconds: float, duration: float
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    index = 0
    while index < speaking.size:
        if not speaking[index]:
            index += 1
            continue
        end = index
        while end < speaking.size and speaking[end]:
            end += 1
        regions.append((index * frame_seconds, min(end * frame_seconds, duration)))
        index = end
    return regions


def pauses_from_speech(
    speech: list[tuple[float, float]],
    duration: float,
    *,
    min_pause: float = MIN_PAUSE_SECONDS,
) -> list[tuple[float, float]]:
    """Gaps of at least ``min_pause`` between (and around) the speech regions."""

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


def next_cut(
    pauses: list[tuple[float, float]],
    start: float,
    *,
    cap: float = SEGMENT_SECONDS,
    min_segment: float = MIN_SEGMENT_SECONDS,
) -> float:
    """Where a segment opening at ``start`` should end.

    The last pause fully inside ``(start + min_segment, start + cap)`` wins,
    which makes segments as long as the cap allows while keeping every cut in
    silence. Failing that, any pause whose midpoint lands in the window is used,
    and failing that the cut falls at the cap.
    """

    low, high = start + min_segment, start + cap
    inside = [item for item in pauses if item[0] >= low and item[1] <= high]
    if not inside:
        inside = [
            item for item in pauses if low <= (item[0] + item[1]) / 2 <= high
        ]
    if not inside:
        return high
    first, last = inside[-1]
    return (first + last) / 2


def segment_edges(
    pauses: list[tuple[float, float]],
    duration: float,
    *,
    cap: float = SEGMENT_SECONDS,
    min_segment: float = MIN_SEGMENT_SECONDS,
) -> list[float]:
    """The whole walk over a recording whose duration is already known."""

    edges = [0.0]
    while duration - edges[-1] > cap:
        edges.append(next_cut(pauses, edges[-1], cap=cap, min_segment=min_segment))
    edges.append(duration)
    return edges


class PauseSegmenter:
    """Turn an ``AudioChunks`` source into pause-aligned segments to transcribe.

    The source decodes forwards only, so the walk runs against a carry buffer:
    blocks arrive, the detector marks their speech, and a cut is emitted as soon
    as more than ``cap`` seconds are in hand. That decision is the one the walk
    over the finished file would make -- the candidate pauses all lie inside the
    first ``cap`` seconds of the carry, which are fully read by then. A pause
    still running at the end of the carry is usable too: whatever its true end,
    the midpoint measured so far is inside it, so the cut is still in silence.
    """

    def __init__(
        self,
        speech: SpeechDetector,
        *,
        cap: float = SEGMENT_SECONDS,
        min_pause: float = MIN_PAUSE_SECONDS,
        min_segment: float = MIN_SEGMENT_SECONDS,
        block_seconds: float = BLOCK_SECONDS,
    ) -> None:
        if not cap > min_segment > 0:
            raise ValueError("a segment cap must exceed its positive minimum")
        if block_seconds < cap:
            raise ValueError("a detector block must hold at least one full segment")
        self._speech = speech
        self._cap = cap
        self._min_pause = min_pause
        self._min_segment = min_segment
        self._block_seconds = block_seconds

    def segments(self, source: object) -> Iterator[DecodedAudio]:
        rate = int(getattr(source, "target_sample_rate", 16_000))
        carry = np.empty(0, dtype=np.float32)
        speech: list[tuple[float, float]] = []
        start_seconds = 0.0
        source_duration = 0.0
        started = False
        blocks = source.chunks(  # type: ignore[attr-defined]
            self._block_seconds, boundary_search_seconds=0.0
        )
        for block in blocks:
            if not started:
                start_seconds = block.clip_start_seconds
                started = True
            source_duration = block.source_duration_seconds
            offset = carry.size / rate
            speech.extend(
                (first + offset, last + offset)
                for first, last in self._speech(block.waveform, rate)
            )
            carry = (
                block.waveform
                if carry.size == 0
                else np.concatenate((carry, block.waveform))
            )
            while carry.size / rate > self._cap:
                cut = next_cut(
                    pauses_from_speech(
                        speech, carry.size / rate, min_pause=self._min_pause
                    ),
                    0.0,
                    cap=self._cap,
                    min_segment=self._min_segment,
                )
                index = min(carry.size, max(1, round(cut * rate)))
                seconds = index / rate
                piece, carry = carry[:index], carry[index:]
                if _transcribable(piece, speech, seconds):
                    yield _decoded(piece, rate, start_seconds, source_duration)
                start_seconds += seconds
                speech = [
                    (max(first - seconds, 0.0), last - seconds)
                    for first, last in speech
                    if last > seconds
                ]
        if started and _transcribable(carry, speech, carry.size / rate):
            yield _decoded(carry, rate, start_seconds, source_duration)


def _transcribable(
    waveform: np.ndarray, speech: list[tuple[float, float]], duration: float
) -> bool:
    """Silence never reaches the recogniser, and neither does a stray remainder."""

    if waveform.size < _MIN_SEGMENT_SAMPLES:
        return False
    return any(last > 0 and first < duration for first, last in speech)


def _decoded(
    waveform: np.ndarray,
    rate: int,
    start_seconds: float,
    source_duration: float,
) -> DecodedAudio:
    return DecodedAudio(
        waveform=np.ascontiguousarray(waveform),
        duration_seconds=waveform.size / rate,
        source_duration_seconds=source_duration,
        clip_start_seconds=start_seconds,
    )


__all__ = [
    "BLOCK_SECONDS",
    "MIN_PAUSE_SECONDS",
    "MIN_SEGMENT_SECONDS",
    "PauseSegmenter",
    "SEGMENT_SECONDS",
    "SpeechDetector",
    "energy_speech",
    "next_cut",
    "pauses_from_speech",
    "segment_edges",
]
