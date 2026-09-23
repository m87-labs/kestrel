"""Model-owned long-form and append-only streaming composition for Kokoro."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
import torch

from kestrel.audio import SpeechOnsetTrimmer
from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.skills import CapabilityInvoker, CapabilityOrchestrator

from .contract import KokoroSynthesisRequest, PreparedKokoroSynthesis
from .g2p import KokoroG2P, normalize_language
from .runtime import SAMPLE_RATE


class Phonemizer(Protocol):
    def phonemize_segments(self, text: str, language: str) -> tuple[str, ...]: ...


def _pcm(result: EngineResult) -> np.ndarray:
    audio = result.output.get("audio")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
        raise TypeError("Kokoro leaf output audio must contain float32 PCM")
    return np.ascontiguousarray(audio).reshape(-1)


class KokoroSynthesisOrchestrator(CapabilityOrchestrator):
    """Run G2P on one model-owned worker before submitting inference leaves."""

    def __init__(self, phonemizer: Phonemizer | None = None) -> None:
        self._phonemizer = phonemizer or KokoroG2P()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kokoro-g2p",
        )

    async def _phonemize(
        self,
        request: KokoroSynthesisRequest,
        text: str,
    ) -> tuple[str, ...]:
        loop = asyncio.get_running_loop()
        phonemes = await loop.run_in_executor(
            self._executor,
            self._phonemizer.phonemize_segments,
            text,
            normalize_language(request.language),
        )
        if not phonemes:
            raise ValueError("G2P produced no Kokoro phonemes")
        return phonemes

    async def _synthesize(
        self,
        invoke: CapabilityInvoker,
        request: KokoroSynthesisRequest,
        *,
        emit: Callable[[dict[str, object]], None] | None = None,
    ) -> EngineResult:
        results: list[EngineResult] = []
        audio: list[np.ndarray] = []
        onset = SpeechOnsetTrimmer(SAMPLE_RATE)
        for phonemes in await self._phonemize(request, request.text):
            result = await invoke(
                {"_prepared": PreparedKokoroSynthesis(request, phonemes)}
            )
            if not isinstance(result, EngineResult):
                raise TypeError("Kokoro leaf invocation must return EngineResult")
            chunk = onset.push(_pcm(result))
            results.append(result)
            if chunk.size:
                audio.append(chunk)
                if emit is not None:
                    emit({"audio": chunk, "sample_rate": SAMPLE_RATE})

        pending = onset.finish()
        if pending.size:
            audio.append(pending)
            if emit is not None:
                emit({"audio": pending, "sample_rate": SAMPLE_RATE})

        if not audio:
            waveform = np.empty(0, dtype=np.float32)
        elif len(audio) == 1:
            waveform = audio[0]
        else:
            waveform = np.concatenate(audio)
        metrics = EngineMetrics(
            input_tokens=sum(result.metrics.input_tokens for result in results),
            output_tokens=sum(result.metrics.output_tokens for result in results),
            prefill_time_ms=sum(result.metrics.prefill_time_ms for result in results),
            decode_time_ms=sum(result.metrics.decode_time_ms for result in results),
            ttft_ms=results[0].metrics.ttft_ms,
            cached_tokens=sum(result.metrics.cached_tokens for result in results),
        )
        return EngineResult(
            request_id=results[-1].request_id,
            tokens=[token for result in results for token in result.tokens],
            finish_reason=results[-1].finish_reason,
            metrics=metrics,
            output={
                "audio": waveform,
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": waveform.size / SAMPLE_RATE,
                "voice": results[-1].output.get("voice", request.voice),
                "language": results[-1].output.get("language", request.language),
            },
        )

    async def run(
        self,
        invoke: CapabilityInvoker,
        *,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Mapping[str, object] | None,
    ) -> object:
        if image is not None:
            raise ValueError("synthesize does not accept an image")
        if settings:
            raise ValueError("Kokoro synthesize does not accept sampling settings")
        request = KokoroSynthesisRequest.from_mapping(prompt)
        if not request.stream:
            return await self._synthesize(invoke, request)

        async def produce(
            emit: Callable[[dict[str, object]], None],
        ) -> EngineResult:
            return await self._synthesize(invoke, request, emit=emit)

        return CapabilityStream("synthesize", produce, coalesce=False)


_ORCHESTRATOR = KokoroSynthesisOrchestrator()


def build_orchestrators() -> Mapping[str, CapabilityOrchestrator]:
    return {"synthesize": _ORCHESTRATOR}


__all__ = [
    "KokoroSynthesisOrchestrator",
    "build_orchestrators",
]
