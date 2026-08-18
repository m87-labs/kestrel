from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest
from kestrel.engine import EngineMetrics, EngineResult
from kestrel.runtime.tokens import TextToken

from kestrel.models.whisper.longform import WhisperLongFormOrchestrator
from kestrel.models.whisper.quality import compression_ratio


def _silent_pcm16_wav(sample_rate: int, frames: int) -> bytes:
    payload = bytes(frames * 2)
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


@pytest.mark.parametrize("source_kind", ("path", "bytes"))
def test_real_incremental_reader_feeds_bounded_longform_windows(
    tmp_path: Path,
    source_kind: str,
) -> None:
    sample_rate = 8_000
    frames = 31 * sample_rate
    path = tmp_path / "long.wav"
    encoded = _silent_pcm16_wav(sample_rate, frames)
    path.write_bytes(encoded)
    audio = path if source_kind == "path" else encoded
    window_sizes: list[int] = []

    async def invoke(prompt, *, image=None, settings=None):
        window_sizes.append(int(prompt["audio"].size))
        duration = prompt["audio"].size / prompt["sample_rate"]
        index = len(window_sizes)
        text = f"window {index}"
        diagnostics = {
            "temperature": float(settings["temperature"]),
            "avg_logprob": -0.25,
            "compression_ratio": compression_ratio(text),
            "no_speech_prob": 0.0,
        }
        return EngineResult(
            request_id=index,
            tokens=[TextToken(10)],
            finish_reason="stop",
            metrics=EngineMetrics(3, 1, 1.0, 1.0, 1.0),
            output={
                "text": text,
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": duration, "text": text, **diagnostics}
                ],
                **diagnostics,
            },
        )

    result = asyncio.run(
        WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": audio, "timestamps": "segment"},
            settings=None,
        )
    )

    assert window_sizes == [30 * sample_rate, sample_rate]
    assert result.output == {
        "text": "window 1 window 2",
        "language": "en",
        "language_probability": None,
        "task": "transcribe",
        "duration_seconds": 31.0,
        "source_duration_seconds": 31.0,
        "clip_start_seconds": 0.0,
        "clip_end_seconds": 31.0,
        "temperature": 0.0,
        "avg_logprob": -0.25,
        "compression_ratio": compression_ratio("window 1 window 2"),
        "no_speech_prob": 0.0,
        "segments": [
            {
                "start": 0.0,
                "end": 30.0,
                "text": "window 1",
                "temperature": 0.0,
                "avg_logprob": -0.25,
                "compression_ratio": compression_ratio("window 1"),
                "no_speech_prob": 0.0,
            },
            {
                "start": 30.0,
                "end": 31.0,
                "text": "window 2",
                "temperature": 0.0,
                "avg_logprob": -0.25,
                "compression_ratio": compression_ratio("window 2"),
                "no_speech_prob": 0.0,
            },
        ],
    }
