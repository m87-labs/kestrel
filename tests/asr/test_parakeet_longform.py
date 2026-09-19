from __future__ import annotations

import asyncio

import numpy as np
import pytest

from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.models.asr.audio import DecodedAudio
from kestrel.models.parakeet_tdt import MODEL_ID
from kestrel.models.parakeet_tdt.longform import ParakeetLongFormOrchestrator


def test_parakeet_stream_aggregates_chunk_progress(monkeypatch) -> None:
    class Source:
        duration_seconds = 360.0
        source_duration_seconds = 400.0
        clip_start_seconds = 20.0
        closed = False
        chunk_seconds = None

        def chunks(self, seconds):
            self.chunk_seconds = seconds
            yield DecodedAudio(np.zeros(16), 180.0, 400.0, 20.0)
            yield DecodedAudio(np.zeros(16), 180.0, 400.0, 200.0)

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(
        "kestrel.models.parakeet_tdt.longform.open_audio_source",
        lambda _audio, _request: source,
    )
    calls = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append((prompt, image, settings))
        index = len(calls)
        await asyncio.sleep(0)
        return EngineResult(
            request_id=index,
            tokens=[],
            finish_reason="stop",
            metrics=EngineMetrics(0, 0, 0.0, 0.0, 0.0),
            output={
                "text": f"part {index}",
                "segments": [
                    {
                        "text": f"part {index}",
                        "start": 0.0,
                        "end": 1.0,
                        "words": [{"word": "part", "start": 0.0, "end": 0.5}],
                    }
                ],
            },
        )

    async def run():
        value = await ParakeetLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={
                "audio": np.zeros(16, dtype=np.float32),
                "sample_rate": 16_000,
                "stream": True,
                "timestamps": "word",
            },
            settings={"max_tokens": 32},
        )
        assert isinstance(value, CapabilityStream)
        updates = [update.output async for update in value]
        return updates, await value.result()

    updates, result = asyncio.run(run())

    assert result.output["text"] == "part 1 part 2"
    assert result.output["segments"][0]["start"] == 20.0
    assert result.output["segments"][1]["start"] == 200.0
    assert result.output["segments"][1]["words"][0]["start"] == 200.0
    assert updates[-1]["provisional"] is False
    assert source.closed is True
    assert source.chunk_seconds == 180
    assert all(call[0]["stream"] is False for call in calls)
    assert all(call[2] == {"max_tokens": 32} for call in calls)

    from kestrel.models import get_spec

    assert isinstance(
        get_spec(MODEL_ID).orchestrators()["transcribe"],
        ParakeetLongFormOrchestrator,
    )


def test_runtime_transcribes_pause_aligned_segments() -> None:
    """The measured long-audio protocol, end to end through the runtime.

    Pause-aligned segments of at most 30 s, contiguous and never overlapping,
    text joined in order, word times offset by the segment that produced them.
    """

    from types import SimpleNamespace

    import torch

    from kestrel.models.parakeet_tdt.model import TdtOutput
    from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime
    from kestrel.models.asr.contract import Word

    sample_rate = 16_000
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(50):
        parts.append(
            np.clip(rng.standard_normal(4 * sample_rate) * 0.1, -1, 1).astype(
                np.float32
            )
        )
        parts.append(np.zeros(sample_rate // 2, dtype=np.float32))
    waveform = np.concatenate(parts)
    duration = waveform.size / sample_rate

    class _Model:
        config = SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8))
        vad_head = None
        calls = 0

        def generate(self, features, mask, *, max_tokens=None):
            del mask, max_tokens
            ids = []
            for _ in range(features.shape[0]):
                type(self).calls += 1
                ids.append(type(self).calls)
            return TdtOutput(
                sequences=torch.tensor([[0, value] for value in ids]),
                durations=torch.tensor([[0, 1] for _ in ids]),
                lengths=torch.tensor([2 for _ in ids]),
                encoder_frame_seconds=0.08,
            )

    class _Tokenizer:
        blank_token_id = 0

        def decode(self, token_ids):
            return f"part {token_ids[1]}."

        def words(self, token_ids, durations, frame_seconds):
            del durations, frame_seconds
            return (Word(f"part{token_ids[1]}", 0.0, 1.0),)

    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.model = _Model()
    runtime.tokenizer = _Tokenizer()
    runtime._batch_decoder = None
    runtime._encoder_graph = None

    seen: list[object] = []
    batch_features = ParakeetTdtRuntime._batch_audio_features

    def capture(rows):
        seen.extend(audio for _index, audio in rows)
        return batch_features(runtime, rows)

    runtime._batch_audio_features = capture

    result = runtime.forward(
        "transcribe",
        (
            {
                "audio": waveform,
                "sample_rate": sample_rate,
                "timestamps": "word",
            },
        ),
    )[0]

    assert len(seen) > 1
    assert max(item.duration_seconds for item in seen) <= 30.0
    starts = [item.clip_start_seconds for item in seen]
    ends = [item.clip_start_seconds + item.duration_seconds for item in seen]
    assert starts[0] == 0.0
    assert ends[:-1] == pytest.approx(starts[1:])  # contiguous, never overlapping
    assert ends[-1] == pytest.approx(duration)
    for start in starts[1:]:
        assert 4.0 <= start % 4.5 <= 4.5  # every cut landed in a silence

    assert result["text"] == " ".join(
        f"part {index + 1}." for index in range(len(seen))
    )
    assert result["duration_seconds"] == pytest.approx(duration)
    word_starts = [
        segment["words"][0]["start"] for segment in result["segments"]
    ]
    assert word_starts == pytest.approx(starts)
