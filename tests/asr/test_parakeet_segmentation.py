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
    PauseSegmenter,
    energy_speech,
    next_cut,
    pauses_from_speech,
    segment_edges,
)
from kestrel.models.parakeet_tdt.vad import (
    VadHead,
    VadHeadSpeech,
    speech_regions,
)


SAMPLE_RATE = 16_000


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


def _speech_and_silence(
    bursts: int,
    *,
    speech_seconds: float = 4.0,
    silence_seconds: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Loud bursts separated by exact silences, so every real pause is known."""

    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _ in range(bursts):
        samples = round(speech_seconds * SAMPLE_RATE)
        parts.append(
            np.clip(rng.standard_normal(samples) * 0.1, -1, 1).astype(np.float32)
        )
        parts.append(np.zeros(round(silence_seconds * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(parts)


def _oracle_speech(
    waveform: np.ndarray, sample_rate: int
) -> list[tuple[float, float]]:
    """A detector that knows exactly where `_speech_and_silence` put its bursts."""

    frame = sample_rate // 50
    count = waveform.size // frame
    if count == 0:
        return [(0.0, waveform.size / sample_rate)]
    loud = np.abs(waveform[: count * frame].reshape(count, frame)).max(1) > 1e-6
    regions: list[tuple[float, float]] = []
    index = 0
    while index < count:
        if not loud[index]:
            index += 1
            continue
        end = index
        while end < count and loud[end]:
            end += 1
        regions.append((index * 0.02, end * 0.02))
        index = end
    return regions


def _source(waveform: np.ndarray) -> AudioChunks:
    return AudioChunks(
        waveform,
        sample_rate=SAMPLE_RATE,
        clip_start_seconds=0.0,
        clip_end_seconds=None,
        target_sample_rate=SAMPLE_RATE,
        max_duration_seconds=24 * 60 * 60,
    )


def test_next_cut_takes_the_last_pause_that_ends_before_the_cap() -> None:
    pauses = [(4.0, 4.5), (12.0, 12.4), (26.0, 26.6), (31.0, 31.5)]

    assert next_cut(pauses, 0.0) == pytest.approx(26.3)


def test_next_cut_skips_pauses_inside_the_minimum_segment() -> None:
    assert next_cut([(0.2, 0.6)], 0.0) == pytest.approx(30.0)
    assert next_cut([(1.2, 1.6)], 0.0) == pytest.approx(1.4)


def test_next_cut_falls_back_to_the_cap_without_a_usable_pause() -> None:
    assert next_cut([], 0.0) == pytest.approx(30.0)
    assert next_cut([(40.0, 41.0)], 0.0) == pytest.approx(30.0)


def test_next_cut_accepts_a_straddling_pause_by_its_midpoint() -> None:
    # Ends past the cap, but its middle is inside the window: still silence.
    assert next_cut([(28.0, 31.0)], 0.0) == pytest.approx(29.5)


def test_next_cut_walks_from_the_current_segment_start() -> None:
    pauses = [(4.0, 4.5), (50.0, 50.4)]

    assert next_cut(pauses, 26.3) == pytest.approx(50.2)


def test_segment_edges_covers_the_recording_without_gaps_or_overlap() -> None:
    pauses = [(float(index) * 4.5 + 4.0, float(index) * 4.5 + 4.5) for index in range(20)]

    edges = segment_edges(pauses, 90.0)

    assert edges[0] == 0.0
    assert edges[-1] == 90.0
    assert all(
        right - left <= 30.0 + 1e-9 for left, right in zip(edges, edges[1:])
    )
    assert edges == sorted(edges)


def test_pauses_from_speech_brackets_the_speech_regions() -> None:
    pauses = pauses_from_speech([(1.0, 4.0), (4.1, 8.0)], 10.0)

    # The 0.1 s gap between the regions is below the 0.2 s minimum.
    assert pauses == [(0.0, 1.0), (8.0, 10.0)]


def test_pauses_from_speech_ignores_overlapping_regions() -> None:
    # The nested region must not open a pause behind the one that contains it.
    assert pauses_from_speech([(0.0, 5.0), (1.0, 3.0)], 6.0) == [(5.0, 6.0)]


def test_pauses_from_speech_keeps_a_gap_that_lands_exactly_on_the_minimum() -> None:
    # 281.0 - 280.8 evaluates to 0.19999999999998863, and a detector's region
    # edges are always multiples of its frame, so these ties are routine.
    start, end = 14_040 * 0.02, 14_050 * 0.02
    assert end - start < 0.2

    pauses = pauses_from_speech([(0.0, start), (end, 300.0)], 300.0)

    assert pauses == [(start, end)]


def test_energy_speech_finds_the_silences() -> None:
    waveform = _speech_and_silence(3)

    regions = energy_speech(waveform, SAMPLE_RATE)
    pauses = pauses_from_speech(regions, waveform.size / SAMPLE_RATE)

    assert len(pauses) == 3
    for index, (start, end) in enumerate(pauses):
        assert start == pytest.approx(index * 4.5 + 4.0, abs=0.05)
        assert end == pytest.approx(index * 4.5 + 4.5, abs=0.05)


def test_energy_speech_calls_a_block_without_dynamic_range_all_speech() -> None:
    rng = np.random.default_rng(1)
    waveform = np.clip(rng.standard_normal(SAMPLE_RATE * 5) * 0.1, -1, 1).astype(
        np.float32
    )

    regions = energy_speech(waveform, SAMPLE_RATE)

    assert regions == [(0.0, 5.0)]
    assert pauses_from_speech(regions, 5.0) == []


def test_speech_regions_bridges_short_gaps_and_drops_short_runs() -> None:
    probabilities = np.zeros(40)
    probabilities[0:3] = 0.9  # 0.24 s of speech
    probabilities[4:6] = 0.9  # after a single 0.08 s frame: bridged
    probabilities[20:21] = 0.9  # 0.08 s alone: dropped

    regions = speech_regions(probabilities, frame_seconds=0.08)

    assert regions == [(0.0, pytest.approx(0.48))]


def test_speech_regions_keeps_a_wide_gap_apart() -> None:
    probabilities = np.zeros(40)
    probabilities[0:5] = 0.9
    probabilities[10:15] = 0.9

    regions = speech_regions(probabilities, frame_seconds=0.08)

    assert regions == [
        (0.0, pytest.approx(0.4)),
        (pytest.approx(0.8), pytest.approx(1.2)),
    ]


def test_segmenter_cuts_in_silence_and_caps_every_segment() -> None:
    waveform = _speech_and_silence(50)

    segments = list(PauseSegmenter(energy_speech).segments(_source(waveform)))

    assert len(segments) > 1
    assert max(item.duration_seconds for item in segments) <= 30.0
    starts = [item.clip_start_seconds for item in segments]
    ends = [
        item.clip_start_seconds + item.duration_seconds for item in segments
    ]
    # Contiguous and non-overlapping: a transducer loses words to both.
    assert ends[:-1] == pytest.approx(starts[1:])
    assert ends[-1] == pytest.approx(waveform.size / SAMPLE_RATE)
    for start in starts[1:]:
        assert 4.0 <= start % 4.5 <= 4.5


def test_segmenter_skips_audio_without_speech() -> None:
    waveform = np.concatenate(
        (
            _speech_and_silence(2),
            np.zeros(120 * SAMPLE_RATE, dtype=np.float32),
            _speech_and_silence(2, seed=3),
        )
    )

    segments = list(PauseSegmenter(_oracle_speech).segments(_source(waveform)))
    transcribed = sum(item.duration_seconds for item in segments)

    assert transcribed < waveform.size / SAMPLE_RATE - 60.0
    assert all(np.abs(item.waveform).max() > 0 for item in segments)


def test_energy_speech_keeps_a_mostly_silent_block_rather_than_guessing() -> None:
    # With speech in a small minority the loud percentile sits in the noise, so
    # there is no floor to measure a pause against: the detector claims the
    # block instead of dropping audio it cannot vouch for. Only a checkpoint
    # with a VAD head skips long silences reliably.
    waveform = np.concatenate(
        (_speech_and_silence(2), np.zeros(120 * SAMPLE_RATE, dtype=np.float32))
    )

    assert energy_speech(waveform, SAMPLE_RATE) == [
        (0.0, pytest.approx(waveform.size / SAMPLE_RATE))
    ]


def test_segmenter_matches_the_whole_file_walk_across_block_boundaries() -> None:
    # Longer than one detector block, so the carry buffer has to cross it.
    waveform = _speech_and_silence(60)
    duration = waveform.size / SAMPLE_RATE
    assert duration > 200.0

    segments = list(PauseSegmenter(energy_speech).segments(_source(waveform)))
    reference = segment_edges(
        pauses_from_speech(energy_speech(waveform, SAMPLE_RATE), duration), duration
    )

    starts = [item.clip_start_seconds for item in segments]
    assert starts == pytest.approx(reference[:-1], abs=0.05)


def test_segmenter_preserves_a_clip_offset() -> None:
    waveform = _speech_and_silence(20)
    source = AudioChunks(
        waveform,
        sample_rate=SAMPLE_RATE,
        clip_start_seconds=5.0,
        clip_end_seconds=None,
        target_sample_rate=SAMPLE_RATE,
        max_duration_seconds=24 * 60 * 60,
    )

    segments = list(PauseSegmenter(energy_speech).segments(source))

    assert segments[0].clip_start_seconds == pytest.approx(5.0)


def test_vad_head_is_absent_until_the_weights_carry_one() -> None:
    model = ParakeetTdt(_tiny_config())

    assert model.vad_head is None
    assert not any("vad_head" in key for key in model.state_dict())
    with pytest.raises(ValueError, match="no VAD head"):
        model.speech_probabilities(
            torch.zeros((1, 16, 128)), torch.ones((1, 16), dtype=torch.bool)
        )


def test_attached_vad_head_round_trips_through_a_strict_state_dict() -> None:
    source = ParakeetTdt(_tiny_config())
    source.attach_vad_head()
    state = source.state_dict()
    assert {key for key in state if key.startswith("vad_head.")} == {
        "vad_head.proj.weight",
        "vad_head.proj.bias",
        "vad_head.ctx.weight",
        "vad_head.ctx.bias",
        "vad_head.out.weight",
        "vad_head.out.bias",
    }

    target = ParakeetTdt(_tiny_config())
    with pytest.raises(RuntimeError, match="vad_head"):
        target.load_state_dict(state, strict=True)
    target.attach_vad_head()
    target.load_state_dict(state, strict=True)

    assert isinstance(target.vad_head, VadHead)


def test_encoder_frame_seconds_follows_the_subsampling_factor() -> None:
    model = ParakeetTdt(_tiny_config())

    assert model.encoder_frame_seconds == pytest.approx(0.08)


def test_subsampler_output_is_local_so_blocks_glue() -> None:
    from kestrel.models.parakeet_tdt.features import parakeet_features

    torch.manual_seed(0)
    model = ParakeetTdt(_tiny_config()).eval()
    model.attach_vad_head()
    waveform = _speech_and_silence(2)

    with torch.inference_mode():
        whole, _ = model.subsample(*parakeet_features(torch.from_numpy(waveform)))
        half, valid = model.subsample(
            *parakeet_features(torch.from_numpy(waveform[: waveform.size // 2]))
        )
    frames = int(valid[0].sum()) - 4  # drop the block's own padded edge

    torch.testing.assert_close(
        whole[0, :frames], half[0, :frames], rtol=1e-3, atol=1e-3
    )


def test_vad_head_speech_reads_the_head_through_the_subsampler() -> None:
    model = ParakeetTdt(_tiny_config()).eval()
    model.attach_vad_head()
    # Force the head to call everything speech, then nothing.
    with torch.no_grad():
        model.vad_head.out.bias.fill_(20.0)
    detector = VadHeadSpeech(model, device=torch.device("cpu"), dtype=torch.float32)
    waveform = _speech_and_silence(2)

    regions = detector(waveform, SAMPLE_RATE)
    assert regions == [(0.0, pytest.approx(waveform.size / SAMPLE_RATE, abs=0.1))]

    with torch.no_grad():
        model.vad_head.out.bias.fill_(-20.0)
    assert detector(waveform, SAMPLE_RATE) == []


def test_runtime_picks_the_pause_source_from_the_loaded_weights() -> None:
    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.model = ParakeetTdt(_tiny_config()).eval()

    assert runtime._segmenter()._speech is energy_speech

    runtime.model.attach_vad_head()

    assert isinstance(runtime._segmenter()._speech, VadHeadSpeech)


def test_segmenter_rejects_a_block_smaller_than_one_segment() -> None:
    with pytest.raises(ValueError, match="at least one full segment"):
        PauseSegmenter(energy_speech, cap=30.0, block_seconds=10.0)
