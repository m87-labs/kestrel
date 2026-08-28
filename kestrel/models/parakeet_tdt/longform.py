"""Progressive long-file orchestration for Parakeet TDT."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from kestrel.engine import CapabilityStream, EngineResult
from kestrel.skills import CapabilityInvoker, CapabilityOrchestrator

from kestrel.models.asr.audio import (
    AudioChunks,
    DecodedAudio,
    snapshot_file_like,
)
from kestrel.models.asr.contract import TranscriptionRequest
from kestrel.models.asr.live import LiveAudioBuffer
from kestrel.models.asr.longform import (
    END_OF_AUDIO,
    aggregate_metrics,
    open_audio_source,
    settled_to_thread,
    shift_segments,
)

from .contract import parse_request
from .runtime import _StreamWindow


# Tried 30s: 39.1% live/file WER vs 0.0% at 180s on a 40s boundary gate.
_STREAM_CHUNK_SECONDS = 180
# Tried 30s/60s left context: better on repeated audio but worse across natural
# utterance boundaries (51.2%/72.0% vs 40.8% reference WER); keeping 10s.
_LIVE_LEFT_SECONDS = 10
_LIVE_CHUNK_SECONDS = 2
_LIVE_RIGHT_SECONDS = 2
# H100: bounded previews plus the exact commit process 180s in 2.57s versus
# ~24s for repeatedly decoding the growing full window.


@dataclass(frozen=True, slots=True)
class _LiveWindow:
    audio: DecodedAudio
    start_sample: int
    sample_count: int | None
    duration_seconds: float
    block_start_seconds: float


async def _live_windows(
    chunks: object, source: LiveAudioBuffer, *, previews: bool
) -> AsyncIterator[_LiveWindow]:
    iterator_factory = getattr(chunks, "__aiter__", None)
    if not callable(iterator_factory):
        raise TypeError("live audio must implement the asynchronous iterator protocol")
    iterator = iterator_factory()
    if not callable(getattr(iterator, "__anext__", None)):
        raise TypeError("live audio __aiter__ must return an asynchronous iterator")
    previewed_frames = 0
    chunk_frames = _LIVE_CHUNK_SECONDS * source.sample_rate
    right_frames = _LIVE_RIGHT_SECONDS * source.sample_rate
    left_frames = _LIVE_LEFT_SECONDS * source.sample_rate

    def ready() -> bool:
        block_frames = min(source.buffered_frames, source.window_frames)
        return block_frames - previewed_frames >= chunk_frames + right_frames

    def preview() -> _LiveWindow:
        nonlocal previewed_frames
        offset = max(0, previewed_frames - left_frames)
        audio = source.snapshot(
            offset_frames=offset,
            frame_count=previewed_frames - offset + chunk_frames + right_frames,
        )
        start_sample = round((previewed_frames - offset) * 16_000 / source.sample_rate)
        previewed_frames += chunk_frames
        return _LiveWindow(
            audio,
            start_sample,
            round(chunk_frames * 16_000 / source.sample_rate),
            previewed_frames / source.sample_rate,
            source.consumed_frames / source.sample_rate,
        )

    def exact(frames: int) -> _LiveWindow:
        nonlocal previewed_frames
        audio = source.snapshot(frame_count=frames)
        source.advance(frames)
        previewed_frames = 0
        return _LiveWindow(
            audio,
            0,
            None,
            audio.duration_seconds,
            audio.clip_start_seconds,
        )

    saw_chunk = False
    try:
        async for chunk in iterator:
            saw_chunk = True
            source.append(chunk)
            while (
                source.buffered_frames >= source.window_frames + source.min_tail_frames
            ):
                if previews:
                    while ready():
                        yield await asyncio.to_thread(preview)
                yield await asyncio.to_thread(exact, source.window_frames)
            if previews:
                while ready():
                    yield await asyncio.to_thread(preview)
        if not saw_chunk:
            raise ValueError("live audio must yield at least one PCM chunk")
        if source.buffered_frames:
            yield await asyncio.to_thread(exact, source.buffered_frames)
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()


class _Accumulator:
    def __init__(self, source: Any, timestamps: str) -> None:
        self.source = source
        self.timestamps = timestamps
        self.results: list[EngineResult] = []
        self.text_parts: list[str] = []
        self.segments: list[dict[str, object]] = []

    def add(self, result: EngineResult, chunk: DecodedAudio) -> None:
        self.results.append(result)
        text = result.output.get("text")
        if isinstance(text, str) and text.strip():
            self.text_parts.append(text.strip())
        self.segments.extend(
            shift_segments(result.output.get("segments"), chunk.clip_start_seconds)
        )

    def output(self, *, provisional: bool | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "text": " ".join(self.text_parts),
            "language": None,
            "task": "transcribe",
            "duration_seconds": self.source.duration_seconds,
            "source_duration_seconds": self.source.source_duration_seconds,
            "clip_start_seconds": self.source.clip_start_seconds,
            "clip_end_seconds": (
                self.source.clip_start_seconds + self.source.duration_seconds
            ),
            "segments": list(self.segments) if self.timestamps != "none" else [],
        }
        if provisional is not None:
            value["provisional"] = provisional
        return value

    def result(
        self,
        *,
        metrics_results: list[EngineResult] | None = None,
    ) -> EngineResult:
        if not self.results:
            raise RuntimeError("Parakeet transcription produced no chunks")
        measured = self.results if metrics_results is None else metrics_results
        metrics = aggregate_metrics(measured)
        return EngineResult(
            request_id=measured[0].request_id,
            tokens=[token for item in self.results for token in item.tokens],
            finish_reason=self.results[-1].finish_reason,
            metrics=metrics,
            output=self.output(),
        )


def _leaf_prompt(
    prompt: Mapping[str, object],
    chunk: DecodedAudio,
) -> dict[str, object]:
    return {
        **prompt,
        "audio": chunk.waveform,
        "sample_rate": 16_000,
        "clip_start_seconds": 0.0,
        "clip_end_seconds": None,
        "stream": False,
    }


async def _run_chunks(
    invoke: CapabilityInvoker,
    source: AudioChunks,
    *,
    image: object | None,
    prompt: Mapping[str, object],
    settings: Mapping[str, object] | None,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> EngineResult:
    accumulator = _Accumulator(source, str(prompt.get("timestamps", "segment")))
    iterator = source.chunks(_STREAM_CHUNK_SECONDS)
    try:
        while True:
            chunk = await settled_to_thread(next, iterator, END_OF_AUDIO)
            if chunk is END_OF_AUDIO:
                break
            assert isinstance(chunk, DecodedAudio)
            value = await invoke(
                _leaf_prompt(prompt, chunk),
                image=image,
                settings=settings,
            )
            if not isinstance(value, EngineResult):
                raise TypeError("Parakeet leaf returned a non-EngineResult value")
            accumulator.add(value, chunk)
            if emit is not None:
                emit(accumulator.output(provisional=True))
        result = accumulator.result()
        if emit is not None:
            emit({**result.output, "provisional": False})
        return result
    finally:
        await settled_to_thread(source.close)


async def _run_live_pcm(
    invoke: CapabilityInvoker,
    request: TranscriptionRequest,
    *,
    image: object | None,
    prompt: Mapping[str, object],
    settings: Mapping[str, object] | None,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> EngineResult:
    if request.sample_rate is None:
        raise ValueError("sample_rate is required for live PCM")
    if request.clip_start_seconds != 0 or request.clip_end_seconds is not None:
        raise ValueError("clip ranges are not supported for live PCM")
    source = LiveAudioBuffer(
        request.sample_rate,
        window_seconds=_STREAM_CHUNK_SECONDS,
        update_seconds=_LIVE_CHUNK_SECONDS,
    )
    accumulator = _Accumulator(source, request.timestamps)
    invocations: list[EngineResult] = []
    state = None
    windows = _live_windows(request.audio, source, previews=emit is not None)
    try:
        async for window in windows:
            leaf = _leaf_prompt(prompt, window.audio)
            if window.sample_count is not None:
                leaf["_stream_window"] = _StreamWindow(
                    state,
                    window.start_sample,
                    window.sample_count,
                    window.duration_seconds,
                )
            value = await invoke(
                leaf,
                image=image,
                settings=settings,
            )
            if not isinstance(value, EngineResult):
                raise TypeError("Parakeet leaf returned a non-EngineResult value")
            invocations.append(value)
            if window.sample_count is None:
                state = None
                accumulator.add(value, window.audio)
                if emit is not None:
                    emit(
                        {
                            **accumulator.output(provisional=True),
                            "completed_seconds": window.block_start_seconds
                            + window.duration_seconds,
                            "total_seconds": source.duration_seconds,
                        }
                    )
                continue
            state = value.output.pop("_stream_state", None)
            if state is None:
                raise RuntimeError("Parakeet leaf returned no streaming state")
            if emit is not None:
                text_parts = list(accumulator.text_parts)
                text = value.output.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
                preview = accumulator.output(provisional=True)
                preview.update(
                    text=" ".join(text_parts),
                    segments=[
                        *accumulator.segments,
                        *shift_segments(
                            value.output.get("segments"), window.block_start_seconds
                        ),
                    ]
                    if request.timestamps != "none"
                    else [],
                    completed_seconds=window.block_start_seconds
                    + window.duration_seconds,
                    total_seconds=source.duration_seconds,
                )
                emit(preview)
    finally:
        await windows.aclose()

    result = accumulator.result(metrics_results=invocations)
    if emit is not None:
        emit(
            {
                **result.output,
                "completed_seconds": source.duration_seconds,
                "total_seconds": source.duration_seconds,
                "provisional": False,
            }
        )
    return result


class ParakeetLongFormOrchestrator(CapabilityOrchestrator):
    async def run(
        self,
        invoke: CapabilityInvoker,
        *,
        image: object | None,
        prompt: Mapping[str, object],
        settings: Mapping[str, object] | None,
    ) -> object:
        if image is not None:
            raise ValueError("transcribe does not accept an image")
        request, _ = parse_request(prompt, settings)
        audio = request.audio
        if callable(getattr(audio, "__aiter__", None)):
            if request.stream:

                async def produce(
                    emit: Callable[[dict[str, object]], None],
                ) -> EngineResult:
                    return await _run_live_pcm(
                        invoke,
                        request,
                        image=image,
                        prompt=prompt,
                        settings=settings,
                        emit=emit,
                    )

                return CapabilityStream("transcribe", produce)
            return await _run_live_pcm(
                invoke,
                request,
                image=image,
                prompt=prompt,
                settings=settings,
            )
        if callable(getattr(audio, "read", None)):
            audio = await settled_to_thread(snapshot_file_like, audio)
        owned_prompt = {**prompt, "audio": audio}
        if not request.stream:
            return await invoke(owned_prompt, image=image, settings=settings)
        source = await settled_to_thread(open_audio_source, audio, request)

        async def produce(
            emit: Callable[[dict[str, object]], None],
        ) -> EngineResult:
            return await _run_chunks(
                invoke,
                source,
                image=image,
                prompt=owned_prompt,
                settings=settings,
                emit=emit,
            )

        return CapabilityStream("transcribe", produce)


__all__ = ["ParakeetLongFormOrchestrator"]
