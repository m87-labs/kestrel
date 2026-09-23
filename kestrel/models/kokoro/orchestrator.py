"""Model-owned long-form and append-only streaming composition for Kokoro."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import torch

from kestrel.audio import SpeechOnsetTrimmer
from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.skills import CapabilityInvoker, CapabilityOrchestrator

from .contract import KokoroSynthesisRequest
from .runtime import SAMPLE_RATE


_MAX_PHONEMES = 510
_PHONEME_BREAKS = frozenset(" \t\n.!?…:;,—。！？、，；：")


def _split_phonemes(phonemes: str) -> tuple[str, ...]:
    chunks = []
    remaining = phonemes.strip()
    while len(remaining) > _MAX_PHONEMES:
        window = remaining[:_MAX_PHONEMES]
        boundary = max((window.rfind(char) + 1 for char in _PHONEME_BREAKS), default=0)
        if boundary == 0:
            boundary = _MAX_PHONEMES
        chunk = remaining[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def _pcm(result: EngineResult) -> np.ndarray:
    audio = result.output.get("audio")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
        raise TypeError("Kokoro leaf output audio must contain float32 PCM")
    return np.ascontiguousarray(audio).reshape(-1)


class KokoroSynthesisOrchestrator(CapabilityOrchestrator):
    """Split long phoneme inputs and stream completed audio segments."""

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
        for phonemes in _split_phonemes(request.phonemes):
            result = await invoke({"_prepared": replace(request, phonemes=phonemes)})
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
