from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from kestrel.engine import EngineMetrics, EngineResult
from kestrel.runtime import ExecutionShape
from kestrel.models.kokoro.contract import (
    KokoroSynthesisRequest,
    PreparedKokoroSynthesis,
)
from kestrel.models.kokoro.model import KokoroOutput
from kestrel.models.kokoro.orchestrator import (
    KokoroSynthesisOrchestrator,
    build_orchestrators,
)
from kestrel.models.kokoro.runtime import KokoroRuntime, SAMPLE_RATE


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor, float]] = []

    def eval(self) -> "_Model":
        return self

    def __call__(
        self, phonemes: str, reference: torch.Tensor, speed: float
    ) -> KokoroOutput:
        self.calls.append((phonemes, reference, speed))
        return KokoroOutput(
            audio=torch.tensor([0.25, -0.5], dtype=torch.float32),
            durations=torch.ones(3, dtype=torch.long),
        )


class _Voices:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def style(self, voice: str, phoneme_length: int) -> torch.Tensor:
        self.calls.append((voice, phoneme_length))
        return torch.ones(1, 256)


class _G2P:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.threads: list[str] = []

    def phonemize_segments(self, text: str, language: str) -> tuple[str, ...]:
        self.calls.append((text, language))
        self.threads.append(threading.current_thread().name)
        return ("həlˈO", "wɜrld")


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model="hexgrad/Kokoro-82M",
        resolved_device=lambda: torch.device("cpu"),
    )


def test_runtime_serves_validated_single_pass_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    voices = _Voices()
    monkeypatch.setattr("kestrel.models.kokoro.runtime.empty_cache", lambda _device: None)
    runtime = KokoroRuntime(_cfg(), model=model, voices=voices)
    assert runtime.execution_shape is ExecutionShape.SINGLE_PASS
    assert runtime.tasks() == ("synthesize",)

    (result,) = runtime.forward(
        "synthesize",
        (
            {
                "_prepared": PreparedKokoroSynthesis(
                    KokoroSynthesisRequest(text="Hello", speed=1.25),
                    "həlˈO",
                ),
            },
        ),
    )
    torch.testing.assert_close(
        result["audio"], torch.tensor([0.25, -0.5], dtype=torch.float32)
    )
    assert result["sample_rate"] == SAMPLE_RATE
    assert result["duration_seconds"] == 2 / SAMPLE_RATE
    assert voices.calls == [("af_heart", 5)]
    assert result["language"] == "en-us"
    assert model.calls[0][0] == "həlˈO"
    assert model.calls[0][2] == 1.25
    runtime.shutdown()


def _result(request_id: int, samples: int) -> EngineResult:
    audio = torch.empty(samples, dtype=torch.float32)
    audio[0::2] = request_id / 10
    audio[1::2] = -request_id / 10
    return EngineResult(
        request_id=request_id,
        tokens=[],
        finish_reason="stop",
        metrics=EngineMetrics(0, 0, 0.0, 0.0, 0.0),
        output={
            "audio": audio,
            "sample_rate": SAMPLE_RATE,
            "voice": "af_heart",
            "language": "a",
        },
    )


def test_long_form_streaming_preserves_synthesis_and_is_lossless() -> None:
    async def scenario() -> None:
        calls: list[dict[str, object]] = []
        g2p = _G2P()

        async def invoke(prompt: Any, **_kwargs: Any) -> EngineResult:
            calls.append(dict(prompt))
            return _result(len(calls), SAMPLE_RATE // 25)

        text = "First sentence. " + "word " * 80 + "Last sentence."
        orchestrator = KokoroSynthesisOrchestrator(g2p)
        result = await orchestrator.run(
            invoke,
            image=None,
            prompt={"text": text},
            settings=None,
        )
        assert isinstance(result, EngineResult)
        assert g2p.calls == [(text, "a")]
        assert all(thread.startswith("kokoro-g2p") for thread in g2p.threads)
        assert len(calls) == 2
        nonstream_audio = result.output["audio"].copy()
        nonstream_phonemes = tuple(
            call["_prepared"].phonemes for call in calls
        )
        prepared = calls[0]["_prepared"]
        assert isinstance(prepared, PreparedKokoroSynthesis)
        assert prepared.request.text == text
        assert prepared.phonemes == "həlˈO"

        calls.clear()
        g2p.calls.clear()
        g2p.threads.clear()
        stream = await orchestrator.run(
            invoke,
            image=None,
            prompt={
                "text": text,
                "stream": True,
            },
            settings=None,
        )
        updates = []
        async for update in stream:
            updates.append(update.output["audio"])
        result = await stream.result()
        assert len(calls) == len(updates) > 1
        np.testing.assert_array_equal(
            result.output["audio"], np.concatenate(updates)
        )
        np.testing.assert_array_equal(result.output["audio"], nonstream_audio)
        assert g2p.calls == [(text, "a")]
        assert len(calls) == 2
        assert tuple(call["_prepared"].phonemes for call in calls) == nonstream_phonemes
        assert all(language == "a" for _text, language in g2p.calls)
        assert all(thread.startswith("kokoro-g2p") for thread in g2p.threads)
        assert all(set(call) == {"_prepared"} for call in calls)
        assert build_orchestrators()["synthesize"] is build_orchestrators()[
            "synthesize"
        ]

    asyncio.run(scenario())
