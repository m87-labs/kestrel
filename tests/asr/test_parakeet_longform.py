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


def _runtime(vad_head=None):
    from types import SimpleNamespace

    import torch

    from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime

    class _Tokenizer:
        blank_token_id = 0

        def decode(self, token_ids):
            return f"part {token_ids[1]}."

        def words(self, token_ids, durations, frame_seconds):
            from kestrel.models.asr.contract import Word

            del durations, frame_seconds
            return (Word(f"part{token_ids[1]}", 0.0, 1.0),)

    runtime = ParakeetTdtRuntime.__new__(ParakeetTdtRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8)),
        vad_head=vad_head,
    )
    runtime.tokenizer = _Tokenizer()
    runtime._batch_decoder = None
    runtime._encoder_graph = None
    return runtime


def test_runtime_transcribes_pause_aligned_segments() -> None:
    """The measured protocol end to end: pause-aligned segments of at most
    30 s, contiguous and never overlapping, text joined in order, word times
    offset by the segment that produced them."""

    import torch

    from kestrel.models.parakeet_tdt.model import TdtOutput
    from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime

    rng = np.random.default_rng(0)
    parts = []
    for _ in range(50):
        parts.append(
            np.clip(rng.standard_normal(4 * 16_000) * 0.1, -1, 1).astype(np.float32)
        )
        parts.append(np.zeros(8_000, dtype=np.float32))
    waveform = np.concatenate(parts)
    duration = waveform.size / 16_000

    runtime = _runtime()
    calls = 0

    def generate(features, mask, *, max_tokens=None):
        nonlocal calls
        del mask, max_tokens
        ids = []
        for _ in range(features.shape[0]):
            calls += 1
            ids.append(calls)
        return TdtOutput(
            sequences=torch.tensor([[0, value] for value in ids]),
            durations=torch.tensor([[0, 1] for _ in ids]),
            lengths=torch.tensor([2 for _ in ids]),
            encoder_frame_seconds=0.08,
        )

    runtime.model.generate = generate
    seen: list[object] = []
    batch_features = ParakeetTdtRuntime._batch_audio_features
    runtime._batch_audio_features = lambda rows: (
        seen.extend(audio for _index, audio in rows),
        batch_features(runtime, rows),
    )[1]

    result = runtime.forward(
        "transcribe",
        ({"audio": waveform, "sample_rate": 16_000, "timestamps": "word"},),
    )[0]

    starts = [item.clip_start_seconds for item in seen]
    ends = [item.clip_start_seconds + item.duration_seconds for item in seen]
    assert len(seen) > 1
    assert max(item.duration_seconds for item in seen) <= 30.0
    assert starts[0] == 0.0
    assert ends[:-1] == pytest.approx(starts[1:])  # contiguous, never overlapping
    assert ends[-1] == pytest.approx(duration)
    for start in starts[1:]:
        assert 4.0 <= start % 4.5 <= 4.5  # every cut landed in a silence

    assert result["text"] == " ".join(f"part {i + 1}." for i in range(len(seen)))
    assert result["duration_seconds"] == pytest.approx(duration)
    word_starts = [segment["words"][0]["start"] for segment in result["segments"]]
    assert word_starts == pytest.approx(starts)


def test_runtime_passes_a_live_stream_window_through_whole(monkeypatch) -> None:
    """A live window arrives already cut, carrying decoder state and sample
    offsets into itself, so the segmenter must not touch it."""

    from contextlib import contextmanager
    from types import SimpleNamespace

    import torch

    from kestrel.models.parakeet_tdt.model import TdtOutput, TdtState
    from kestrel.models.parakeet_tdt.runtime import _StreamWindow

    waveform = np.clip(
        np.random.default_rng(0).standard_normal(90 * 16_000) * 0.1, -1, 1
    ).astype(np.float32)
    requested: list[tuple] = []
    decoded: list[int] = []

    class Source:
        target_sample_rate = 16_000
        duration_seconds = source_duration_seconds = 90.0
        clip_start_seconds = 0.0

        def chunks(self, seconds, **kwargs):
            requested.append((seconds, kwargs))
            return iter((DecodedAudio(waveform, 90.0, 90.0, 0.0),))

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_error):
            pass

    @contextmanager
    def launch(features, mask):
        yield features, mask

    def generate_encoded(encoded, valid, **kwargs):
        del encoded, kwargs
        decoded.append(int(valid.sum()))
        return TdtOutput(
            sequences=torch.tensor([[0, 1]]),
            durations=torch.tensor([[0, 1]]),
            lengths=torch.tensor([2]),
            state=TdtState(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
            encoder_frame_seconds=0.08,
        )

    runtime = _runtime()
    runtime._encoder_graph = SimpleNamespace(launch=launch)
    runtime.model.generate_encoded = generate_encoded
    monkeypatch.setattr(
        "kestrel.models.parakeet_tdt.runtime.AudioChunks",
        lambda *_a, **_k: Source(),
    )

    runtime.forward(
        "transcribe",
        (
            {
                "audio": waveform,
                "sample_rate": 16_000,
                "timestamps": "none",
                "_stream_window": _StreamWindow(None, 0, None, 90.0),
            },
        ),
    )

    # The whole window in one piece, at the streaming block size -- not the
    # segmenter, which would have asked for 120 s blocks with no boundary search.
    assert requested == [(180, {})]
    assert len(decoded) == 1


def test_runtime_batches_segments_of_different_lengths() -> None:
    """Segments now vary in length within a batch, which fixed chunks never
    did: each request must still get its own text and its own clock."""

    import torch

    from kestrel.models.parakeet_tdt.model import TdtOutput

    def audio(seed, bursts, burst, gap):
        rng = np.random.default_rng(seed)
        parts = []
        for _ in range(bursts):
            parts.append(
                np.clip(rng.standard_normal(int(burst * 16_000)) * 0.1, -1, 1).astype(
                    np.float32
                )
            )
            parts.append(np.zeros(int(gap * 16_000), dtype=np.float32))
        return np.concatenate(parts)

    runtime = _runtime()
    batches = []

    def generate(features, mask, *, max_tokens=None):
        del max_tokens
        batches.append(features.shape[0])
        rows = features.shape[0]
        return TdtOutput(
            sequences=torch.tensor([[0, index + 1] for index in range(rows)]),
            durations=torch.tensor([[0, 1]] * rows),
            lengths=torch.tensor([2] * rows),
            encoder_frame_seconds=0.08,
        )

    runtime.model.generate = generate
    prompts = tuple(
        {"audio": audio(*spec), "sample_rate": 16_000, "timestamps": "word"}
        for spec in ((0, 20, 4.0, 0.5), (1, 9, 9.0, 0.3), (2, 40, 2.0, 0.7))
    )

    results = runtime.forward("transcribe", prompts)

    assert max(batches) > 1  # the requests really did share a batch
    for prompt, result in zip(prompts, results, strict=True):
        starts = [segment["start"] for segment in result["segments"]]
        assert starts == sorted(starts)
        assert result["duration_seconds"] == pytest.approx(
            prompt["audio"].size / 16_000
        )
        assert result["text"]
