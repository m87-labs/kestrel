import numpy as np

from kestrel.audio import SpeechOnsetTrimmer


def test_speech_onset_trimmer_spans_chunks() -> None:
    sample_rate = 24_000
    block = sample_rate * 10 // 1_000
    silence = np.zeros(2 * block, dtype=np.float32)
    speech = np.tile(np.array((-0.1, 0.1), dtype=np.float32), 3 * block)
    waveform = np.concatenate((silence, speech))
    split = silence.size + block // 2
    trimmer = SpeechOnsetTrimmer(sample_rate)

    assert trimmer.push(waveform[:split]).size == 0
    np.testing.assert_array_equal(
        trimmer.push(waveform[split:]), waveform[silence.size :]
    )
    assert trimmer.finish().size == 0


def test_speech_onset_trimmer_preserves_audio_without_detected_speech() -> None:
    silence = np.zeros(240, dtype=np.float32)
    trimmer = SpeechOnsetTrimmer(24_000)

    assert trimmer.push(silence).size == 0
    np.testing.assert_array_equal(trimmer.finish(), silence)


def test_speech_onset_trimmer_discards_a_startup_click_before_speech() -> None:
    block = 240
    waveform = np.zeros(8 * block, dtype=np.float32)
    click = np.tile(np.array((-0.1, 0.1), dtype=np.float32), 121)
    waveform[block - 1 : 2 * block + 1] = click
    speech_start = 4 * block
    waveform[speech_start:] = np.tile(
        np.array((-0.1, 0.1), dtype=np.float32), 2 * block
    )
    trimmer = SpeechOnsetTrimmer(24_000)

    np.testing.assert_array_equal(trimmer.push(waveform), waveform[speech_start:])
    assert trimmer.finish().size == 0


def test_speech_onset_trimmer_rejects_sparse_same_polarity_spikes() -> None:
    block = 240
    waveform = np.zeros(3 * block, dtype=np.float32)
    for start in range(0, waveform.size, block):
        waveform[start : start + 2] = 1.0
    trimmer = SpeechOnsetTrimmer(24_000)

    assert trimmer.push(waveform).size == 0
    np.testing.assert_array_equal(trimmer.finish(), waveform)
