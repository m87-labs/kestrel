from __future__ import annotations

import numpy as np
import pytest
import torch

from kestrel.models.asr.audio import AudioChunks
from kestrel.models.parakeet_tdt.config import (
    ParakeetEncoderConfig,
    ParakeetTdtConfig,
)
from kestrel.models.parakeet_tdt.model import ParakeetTdt
from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime
from kestrel.models.parakeet_tdt.segment import (
    SEGMENT_SECONDS,
    energy_speech,
    next_cut,
    pause_segments,
    pauses_from_speech,
)
from kestrel.models.parakeet_tdt.vad import VadHead, head_speech, speech_regions


SAMPLE_RATE = 16_000
BURST, GAP = 4.0, 0.5


def _tiny_config() -> ParakeetTdtConfig:
    config = ParakeetTdtConfig(
        encoder=ParakeetEncoderConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_mel_bins=128,
            conv_kernel_size=9,
            subsampling_conv_channels=8,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            subsampling_factor=8,
            max_position_embeddings=1024,
            hidden_act="silu",
        ),
        blank_token_id=3,
        decoder_hidden_size=16,
        durations=(0, 1, 2, 3, 4),
        max_symbols_per_step=10,
        num_decoder_layers=1,
        pad_token_id=3,
        vocab_size=8,
        hidden_act="relu",
    )
    config.validate()
    return config


def _bursts(count: int, *, seed: int = 0) -> np.ndarray:
    """Loud bursts separated by exact silences, so every real pause is known."""

    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _ in range(count):
        samples = round(BURST * SAMPLE_RATE)
        parts.append(
            np.clip(rng.standard_normal(samples) * 0.1, -1, 1).astype(np.float32)
        )
        parts.append(np.zeros(round(GAP * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(parts)


def _source(
    waveform: np.ndarray, *, rate: int = SAMPLE_RATE, start: float = 0.0
) -> AudioChunks:
    return AudioChunks(
        waveform,
        sample_rate=rate,
        clip_start_seconds=start,
        clip_end_seconds=None,
        target_sample_rate=SAMPLE_RATE,
        max_duration_seconds=24 * 60 * 60,
    )


def _whole_file_walk(
    pauses: list[tuple[float, float]], duration: float
) -> list[float]:
    """The cut policy as a walk over a recording whose length is known."""

    edges = [0.0]
    while duration - edges[-1] > SEGMENT_SECONDS:
        shifted = [(a - edges[-1], b - edges[-1]) for a, b in pauses]
        edges.append(edges[-1] + next_cut(shifted))
    return edges + [duration]


def test_next_cut_follows_the_measured_policy() -> None:
    # The last pause that ends before the cap, taken at its midpoint.
    assert next_cut([(4.0, 4.5), (12.0, 12.4), (26.0, 26.6), (31.0, 31.5)]) == 26.3
    # Pauses inside the minimum segment are not cut points.
    assert next_cut([(0.2, 0.6)]) == SEGMENT_SECONDS
    assert next_cut([(1.2, 1.6)]) == pytest.approx(1.4)
    # A pause straddling the cap still gives a cut inside silence.
    assert next_cut([(28.0, 31.0)]) == 29.5
    # Nothing usable: cut at the cap.
    assert next_cut([]) == SEGMENT_SECONDS
    assert next_cut([(40.0, 41.0)]) == SEGMENT_SECONDS


def test_pauses_from_speech_keeps_a_gap_that_lands_exactly_on_the_minimum() -> None:
    # 281.0 - 280.8 evaluates to 0.19999999999998863, and a pause source's
    # region edges are always multiples of its frame, so these ties are routine.
    start, end = 14_040 * 0.02, 14_050 * 0.02
    assert end - start < 0.2

    assert pauses_from_speech([(0.0, start), (end, 300.0)], 300.0) == [(start, end)]


def test_pauses_from_speech_brackets_and_merges() -> None:
    # A 0.1 s gap is below the minimum; a nested region opens no pause behind
    # the one containing it; the head and tail of the recording count.
    assert pauses_from_speech([(1.0, 4.0), (4.1, 8.0)], 10.0) == [
        (0.0, 1.0),
        (8.0, 10.0),
    ]
    assert pauses_from_speech([(0.0, 5.0), (1.0, 3.0)], 6.0) == [(5.0, 6.0)]


def test_energy_speech_finds_silences_and_declines_to_guess() -> None:
    pauses = pauses_from_speech(
        energy_speech(_bursts(3), SAMPLE_RATE), 3 * (BURST + GAP)
    )
    assert len(pauses) == 3
    for index, (start, end) in enumerate(pauses):
        assert start == pytest.approx(index * (BURST + GAP) + BURST, abs=0.05)
        assert end == pytest.approx((index + 1) * (BURST + GAP), abs=0.05)

    # No dynamic range: nothing to measure a pause against, so the block is
    # claimed rather than dropped on a guess. Digital silence lands here too.
    rng = np.random.default_rng(1)
    flat = np.clip(rng.standard_normal(SAMPLE_RATE * 5) * 0.1, -1, 1).astype(np.float32)
    assert energy_speech(flat, SAMPLE_RATE) == [(0.0, 5.0)]
    assert energy_speech(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE) == [
        (0.0, 5.0)
    ]
    # Speech in a small minority is the same case: the loud percentile sits in
    # the noise. Only a VAD head skips long silences reliably.
    mostly_silent = np.concatenate(
        (_bursts(2), np.zeros(120 * SAMPLE_RATE, dtype=np.float32))
    )
    assert energy_speech(mostly_silent, SAMPLE_RATE) == [
        (0.0, pytest.approx(mostly_silent.size / SAMPLE_RATE))
    ]


def test_speech_regions_bridges_short_gaps_and_drops_short_runs() -> None:
    probabilities = np.zeros(40)
    probabilities[0:3] = 0.9  # 0.24 s of speech
    probabilities[4:6] = 0.9  # one 0.08 s frame later: bridged
    probabilities[20:21] = 0.9  # 0.08 s alone: dropped
    assert speech_regions(probabilities, frame_seconds=0.08) == [
        (0.0, pytest.approx(0.48))
    ]

    probabilities = np.zeros(40)
    probabilities[0:5] = 0.9
    probabilities[10:15] = 0.9  # a 0.4 s gap stays a gap
    assert speech_regions(probabilities, frame_seconds=0.08) == [
        (0.0, pytest.approx(0.4)),
        (pytest.approx(0.8), pytest.approx(1.2)),
    ]

    assert speech_regions(np.zeros(40), frame_seconds=0.08) == []


def test_pause_segments_matches_the_whole_file_walk() -> None:
    # Longer than two pause-source blocks, so the carry buffer crosses them.
    waveform = _bursts(60)
    duration = waveform.size / SAMPLE_RATE
    assert duration > 2 * 120.0

    segments = list(pause_segments(_source(waveform), energy_speech))
    reference = _whole_file_walk(
        pauses_from_speech(energy_speech(waveform, SAMPLE_RATE), duration), duration
    )

    starts = [item.clip_start_seconds for item in segments]
    ends = [item.clip_start_seconds + item.duration_seconds for item in segments]
    assert starts == pytest.approx(reference[:-1])
    assert ends == pytest.approx(reference[1:])
    # Contiguous, never overlapping, capped, and covering the recording.
    assert ends[:-1] == pytest.approx(starts[1:])
    assert max(item.duration_seconds for item in segments) <= SEGMENT_SECONDS
    assert ends[-1] == pytest.approx(duration)
    for start in starts[1:]:
        assert BURST <= start % (BURST + GAP) <= BURST + GAP


def test_pause_segments_skips_silence_without_losing_the_clock() -> None:
    silence = 120
    speech_seconds = _bursts(2).size / SAMPLE_RATE
    waveform = np.concatenate(
        (
            _bursts(2),
            np.zeros(silence * SAMPLE_RATE, dtype=np.float32),
            _bursts(2, seed=3),
        )
    )

    def oracle(block: np.ndarray, rate: int) -> list[tuple[float, float]]:
        loud = np.abs(block[: block.size // 320 * 320].reshape(-1, 320)).max(1) > 1e-6
        edges = np.flatnonzero(np.diff(np.r_[0, loud.astype(np.int8), 0]))
        return [(a * 0.02, b * 0.02) for a, b in edges.reshape(-1, 2)]

    duration = waveform.size / SAMPLE_RATE
    segments = list(pause_segments(_source(waveform), oracle))
    starts = [item.clip_start_seconds for item in segments]

    # Segments with no speech at all never reach the recogniser...
    assert sum(item.duration_seconds for item in segments) < duration - 60
    for item in segments:
        assert np.abs(item.waveform).max() > 0
    # ...and skipping them still advances the clock, so the segments after the
    # gap keep their true place and word timestamps stay right across it.
    assert starts == sorted(starts)
    assert starts[-1] > speech_seconds + 60
    assert starts[-1] + segments[-1].duration_seconds == pytest.approx(duration)


def test_pause_segments_handles_short_and_offset_recordings() -> None:
    # Shorter than one segment, and shorter than the minimum segment: one
    # segment either way, positioned by the clip offset.
    for seconds, start in ((25.0, 0.0), (25.0, 7.5), (0.5, 0.0)):
        waveform = np.clip(
            np.random.default_rng(2).standard_normal(round(seconds * SAMPLE_RATE))
            * 0.1,
            -1,
            1,
        ).astype(np.float32)
        segments = list(pause_segments(_source(waveform, start=start), energy_speech))
        assert len(segments) == 1
        assert segments[0].duration_seconds == pytest.approx(seconds - start)
        assert segments[0].clip_start_seconds == pytest.approx(start)

    # Audio too short for the recogniser passes through rather than being
    # swallowed, so the request fails the way it always has.
    tiny = np.full(64, 0.1, dtype=np.float32)
    assert [
        item.waveform.size for item in pause_segments(_source(tiny), energy_speech)
    ] == [64]


def test_pause_segments_reads_a_resampled_source() -> None:
    waveform = _bursts(20)[::2].copy()  # the same audio at 8 kHz
    duration = waveform.size / 8_000

    segments = list(pause_segments(_source(waveform, rate=8_000), energy_speech))

    ends = [item.clip_start_seconds + item.duration_seconds for item in segments]
    assert len(segments) > 1
    assert max(item.duration_seconds for item in segments) <= SEGMENT_SECONDS
    assert ends[-1] == pytest.approx(duration, abs=0.01)
    # Everything downstream is 16 kHz, whatever the source was.
    assert sum(item.waveform.size for item in segments) == pytest.approx(
        duration * SAMPLE_RATE, rel=1e-3
    )


def test_vad_head_is_a_capability_of_the_weights() -> None:
    model = ParakeetTdt(_tiny_config())
    assert model.vad_head is None
    assert not any("vad_head" in key for key in model.state_dict())
    assert model.encoder_frame_seconds == pytest.approx(0.08)
    with pytest.raises(ValueError, match="no VAD head"):
        model.speech_probabilities(
            torch.zeros((1, 16, 128)), torch.ones((1, 16), dtype=torch.bool)
        )

    source = ParakeetTdt(_tiny_config())
    source.attach_vad_head()
    state = source.state_dict()
    assert {key for key in state if key.startswith("vad_head.")} == {
        f"vad_head.{name}.{kind}"
        for name in ("proj", "ctx", "out")
        for kind in ("weight", "bias")
    }

    # A head the model has no room for, a partial head and a mis-shaped head
    # all fail loudly rather than falling back to the energy pauses.
    target = ParakeetTdt(_tiny_config())
    with pytest.raises(RuntimeError, match="vad_head"):
        target.load_state_dict(state, strict=True)
    target.attach_vad_head()
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(
            {k: v for k, v in state.items() if k != "vad_head.ctx.weight"}, strict=True
        )
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(
            {**state, "vad_head.proj.weight": torch.zeros(7, 3, 1)}, strict=True
        )
    target.load_state_dict(state, strict=True)
    assert isinstance(target.vad_head, VadHead)


def test_head_speech_reads_the_head_through_a_local_subsampler() -> None:
    from kestrel.models.parakeet_tdt.features import parakeet_features

    torch.manual_seed(0)
    model = ParakeetTdt(_tiny_config()).eval()
    model.attach_vad_head()
    waveform = _bursts(2)

    # The subsampler is purely local, so a block gives the frames the whole
    # file would have given -- the premise of scanning in 120 s pieces.
    with torch.inference_mode():
        whole, _ = model.encoder.subsampling(
            *parakeet_features(torch.from_numpy(waveform))
        )
        half, valid = model.encoder.subsampling(
            *parakeet_features(torch.from_numpy(waveform[: waveform.size // 2]))
        )
    frames = int(valid[0].sum()) - 4  # drop the block's own padded edge
    torch.testing.assert_close(
        whole[0, :frames], half[0, :frames], rtol=1e-3, atol=1e-3
    )

    with torch.no_grad():
        model.vad_head.out.bias.fill_(20.0)
    assert head_speech(model, waveform, SAMPLE_RATE) == [
        (0.0, pytest.approx(waveform.size / SAMPLE_RATE, abs=0.1))
    ]
    with torch.no_grad():
        model.vad_head.out.bias.fill_(-20.0)
    assert head_speech(model, waveform, SAMPLE_RATE) == []


def test_runtime_picks_the_pause_source_from_the_loaded_weights() -> None:
    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.model = ParakeetTdt(_tiny_config()).eval()

    assert runtime._speech_regions() is energy_speech

    runtime.model.attach_vad_head()
    chosen = runtime._speech_regions()

    assert chosen.func is head_speech
    assert chosen.args == (runtime.model,)
