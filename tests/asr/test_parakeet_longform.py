from __future__ import annotations

import asyncio

import numpy as np

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
