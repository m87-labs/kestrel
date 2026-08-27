from __future__ import annotations

import asyncio

import numpy as np

from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.models.asr.audio import DecodedAudio
from kestrel.models.parakeet_tdt.longform import ParakeetLongFormOrchestrator
from kestrel.models.qwen3_asr.longform import Qwen3AsrLongFormOrchestrator


def _result(request_id: int, text: str) -> EngineResult:
    return EngineResult(
        request_id=request_id,
        tokens=[],
        finish_reason="stop",
        metrics=EngineMetrics(1, 1, 1.0, 1.0, 1.0),
        output={
            "text": text,
            "language": "en",
            "segments": [{"text": text, "start": 0.0, "end": 1.0}],
        },
    )


def test_qwen_live_pcm_revises_then_commits() -> None:
    calls = []
    closed = False

    async def audio():
        nonlocal closed
        try:
            yield np.zeros(31 * 16_000, dtype=np.float32)
            yield np.zeros(4 * 16_000, dtype=np.float32)
        finally:
            closed = True

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(prompt)
        await asyncio.sleep(0)
        return _result(len(calls), f"version {len(calls)}")

    async def run():
        stream = await Qwen3AsrLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={
                "audio": audio(),
                "sample_rate": 16_000,
                "stream": True,
                "timestamps": "segment",
            },
            settings=None,
        )
        assert isinstance(stream, CapabilityStream)
        updates = [update.output async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(run())
    assert len(calls) == 3
    assert all(call["stream"] is False for call in calls)
    assert result.output["text"] == "version 1 version 3"
    assert result.output["duration_seconds"] == 35.0
    assert result.output["segments"][1]["start"] == 30.0
    assert result.metrics.input_tokens == 3
    assert updates[-1]["provisional"] is False
    assert closed is True


def test_qwen_short_file_stream_uses_capability_updates(monkeypatch) -> None:
    class Source:
        duration_seconds = 10.0
        source_duration_seconds = 10.0
        clip_start_seconds = 0.0
        closed = False

        def chunks(self, _seconds):
            yield DecodedAudio(np.zeros(16), 10.0, 10.0, 0.0)

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(
        "kestrel.models.qwen3_asr.longform.open_audio_source",
        lambda _audio, _request: source,
    )

    async def invoke(prompt, *, image=None, settings=None):
        assert prompt["stream"] is True
        return _result(1, "short")

    async def run():
        stream = await Qwen3AsrLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={
                "audio": np.zeros(16, dtype=np.float32),
                "sample_rate": 16_000,
                "stream": True,
            },
            settings=None,
        )
        assert isinstance(stream, CapabilityStream)
        updates = [update.output async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(run())
    assert result.output["text"] == "short"
    assert updates[-1]["provisional"] is False
    assert source.closed is True


def test_parakeet_live_pcm_previews_then_commits_exact_blocks(monkeypatch) -> None:
    monkeypatch.setattr("kestrel.models.parakeet_tdt.longform._STREAM_CHUNK_SECONDS", 6)
    calls = []

    async def audio():
        for _ in range(8):
            yield np.zeros(16_000, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(prompt)
        await asyncio.sleep(0)
        result = _result(len(calls), f"version {len(calls)}")
        window = prompt.get("_stream_window")
        if window is not None:
            result.output["duration_seconds"] = window.duration_seconds
            result.output["_stream_state"] = object()
        return result

    async def run():
        stream = await ParakeetLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={
                "audio": audio(),
                "sample_rate": 16_000,
                "stream": True,
                "timestamps": "word",
            },
            settings=None,
        )
        assert isinstance(stream, CapabilityStream)
        updates = [update.output async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(run())
    assert len(calls) == 4
    assert result.output["text"] == "version 3 version 4"
    assert result.output["duration_seconds"] == 8.0
    assert result.output["segments"][1]["start"] == 6.0
    assert updates[-1]["provisional"] is False


def test_parakeet_live_pcm_does_not_strand_a_tiny_final_tail(monkeypatch) -> None:
    monkeypatch.setattr("kestrel.models.parakeet_tdt.longform._STREAM_CHUNK_SECONDS", 6)
    exact_sizes = []

    async def audio():
        yield np.zeros(6 * 16_000, dtype=np.float32)
        yield np.zeros(160, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        exact_sizes.append(prompt["audio"].size)
        return _result(1, "final")

    result = asyncio.run(
        ParakeetLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={
                "audio": audio(),
                "sample_rate": 16_000,
            },
            settings=None,
        )
    )
    assert exact_sizes == [6 * 16_000 + 160]
    assert result.output["duration_seconds"] == 6.01


def test_live_pcm_without_stream_skips_provisional_decodes() -> None:
    calls = 0

    async def audio():
        for _ in range(3):
            yield np.zeros(16_000, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal calls
        calls += 1
        return _result(calls, "final")

    result = asyncio.run(
        ParakeetLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": audio(), "sample_rate": 16_000},
            settings=None,
        )
    )
    assert calls == 1
    assert result.output["text"] == "final"
