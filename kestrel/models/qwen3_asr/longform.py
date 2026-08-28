"""Bounded long-file orchestration for Qwen3-ASR."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from kestrel.engine import CapabilityStream, EngineResult
from kestrel.skills.base import CapabilityInvoker, CapabilityOrchestrator

from kestrel.models.asr.audio import AudioChunks, DecodedAudio, snapshot_file_like
from kestrel.models.asr.contract import DecodeSettings, TranscriptionRequest
from kestrel.models.asr.live import LiveAudioBuffer, live_audio_windows
from kestrel.models.asr.longform import (
    END_OF_AUDIO,
    aggregate_metrics,
    open_audio_source,
    settled_to_thread,
    shift_segments,
)


_CHUNK_SECONDS = {"none": 1_200, "segment": 30, "word": 180}
_NO_SPACE_LANGUAGES = frozenset({"ja", "th", "yue", "zh"})


def _join_text(parts: list[str], languages: list[str]) -> str:
    separator = "" if languages and set(languages) <= _NO_SPACE_LANGUAGES else " "
    return separator.join(part.strip() for part in parts if part.strip())


class _Accumulator:
    def __init__(self, source: Any, timestamps: str) -> None:
        self.source = source
        self.timestamps = timestamps
        self.results: list[EngineResult] = []
        self.text_parts: list[str] = []
        self.languages: list[str] = []
        self.segments: list[dict[str, object]] = []

    @property
    def text(self) -> str:
        return _join_text(self.text_parts, self.languages)

    def add(self, result: EngineResult, chunk: DecodedAudio) -> None:
        self.results.append(result)
        text = result.output.get("text")
        if isinstance(text, str) and text.strip():
            self.text_parts.append(text)
        language = result.output.get("language")
        if isinstance(language, str) and language:
            for code in language.split(","):
                if code and code not in self.languages:
                    self.languages.append(code)
        self.segments.extend(
            shift_segments(result.output.get("segments"), chunk.clip_start_seconds)
        )

    def output(self, *, provisional: bool | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "text": self.text,
            "language": ",".join(self.languages) or None,
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
            raise RuntimeError("long-form transcription produced no chunks")
        measured = self.results if metrics_results is None else metrics_results
        metrics = aggregate_metrics(measured)
        logprobs = None
        if all(item.logprobs is not None for item in self.results):
            logprobs = [
                value for item in self.results for value in (item.logprobs or ())
            ]
        return EngineResult(
            request_id=measured[0].request_id,
            tokens=[token for item in self.results for token in item.tokens],
            finish_reason=next(
                (
                    item.finish_reason
                    for item in self.results
                    if item.finish_reason != "stop"
                ),
                self.results[-1].finish_reason,
            ),
            metrics=metrics,
            output=self.output(),
            logprobs=logprobs,
        )


def _leaf_prompt(
    prompt: Mapping[str, object],
    chunk: DecodedAudio,
    previous_text: str | None,
    *,
    stream: bool,
) -> dict[str, object]:
    leaf = dict(prompt)
    leaf.update(
        audio=chunk.waveform,
        sample_rate=16_000,
        clip_start_seconds=0.0,
        clip_end_seconds=None,
        stream=stream,
    )
    initial = prompt.get("initial_prompt")
    context = [value for value in (initial, previous_text) if isinstance(value, str)]
    leaf["initial_prompt"] = " ".join(context) or None
    return leaf


async def _invoke_leaf(
    invoke: CapabilityInvoker,
    prompt: Mapping[str, object],
    *,
    image: object | None,
    settings: Mapping[str, object] | None,
    emit_delta: Callable[[str], None] | None,
) -> EngineResult:
    value = await invoke(prompt, image=image, settings=settings)
    if emit_delta is not None and hasattr(value, "__aiter__"):
        partial = ""
        async for update in value:  # type: ignore[union-attr]
            delta = getattr(update, "text", "")
            if isinstance(delta, str) and delta:
                partial += delta
                emit_delta(partial)
        value = await value.result()  # type: ignore[union-attr]
    if not isinstance(value, EngineResult):
        raise TypeError("Qwen3-ASR leaf returned a non-EngineResult value")
    return value


async def _run_chunks(
    invoke: CapabilityInvoker,
    source: AudioChunks,
    *,
    image: object | None,
    prompt: Mapping[str, object],
    settings: Mapping[str, object] | None,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> EngineResult:
    timestamps = str(prompt.get("timestamps", "segment"))
    accumulator = _Accumulator(source, timestamps)
    iterator = source.chunks(_CHUNK_SECONDS[timestamps])
    previous_text = None
    try:
        while True:
            chunk = await settled_to_thread(next, iterator, END_OF_AUDIO)
            if chunk is END_OF_AUDIO:
                break
            assert isinstance(chunk, DecodedAudio)

            def emit_delta(partial: str) -> None:
                if emit is None:
                    return
                prefix = accumulator.text
                text = _join_text([prefix, partial], accumulator.languages)
                emit({**accumulator.output(provisional=True), "text": text})

            leaf = await _invoke_leaf(
                invoke,
                _leaf_prompt(
                    prompt,
                    chunk,
                    previous_text,
                    stream=emit is not None,
                ),
                image=image,
                settings=settings,
                emit_delta=emit_delta if emit is not None else None,
            )
            accumulator.add(leaf, chunk)
            if prompt.get("condition_on_previous_text", True):
                previous_text = str(leaf.output.get("text", "")) or None
            if emit is not None:
                emit(accumulator.output(provisional=True))
        result = accumulator.result()
        if emit is not None:
            emit({**result.output, "provisional": False})
        return result
    finally:
        await settled_to_thread(source.close)


def _live_preview(
    accumulator: _Accumulator,
    result: EngineResult,
    chunk: DecodedAudio,
    source: LiveAudioBuffer,
) -> dict[str, object]:
    text_parts = list(accumulator.text_parts)
    text = result.output.get("text")
    if isinstance(text, str) and text.strip():
        text_parts.append(text)
    languages = list(accumulator.languages)
    language = result.output.get("language")
    if isinstance(language, str):
        for code in language.split(","):
            if code and code not in languages:
                languages.append(code)
    output = accumulator.output(provisional=True)
    output.update(
        text=_join_text(text_parts, languages),
        language=",".join(languages) or None,
        segments=[
            *accumulator.segments,
            *shift_segments(result.output.get("segments"), chunk.clip_start_seconds),
        ]
        if accumulator.timestamps != "none"
        else [],
        completed_seconds=source.consumed_frames / source.sample_rate,
        total_seconds=source.duration_seconds,
    )
    return output


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
        window_seconds=30,
        update_seconds=5 if emit is not None else 30,
    )
    accumulator = _Accumulator(source, request.timestamps)
    invocations: list[EngineResult] = []
    previous_text = None
    windows = live_audio_windows(request.audio, source, previews=emit is not None)
    try:
        async for chunk, commit in windows:
            result = await _invoke_leaf(
                invoke,
                _leaf_prompt(prompt, chunk, previous_text, stream=False),
                image=image,
                settings=settings,
                emit_delta=None,
            )
            invocations.append(result)
            if not commit:
                if emit is not None:
                    emit(_live_preview(accumulator, result, chunk, source))
                continue
            accumulator.add(result, chunk)
            if request.condition_on_previous_text:
                previous_text = str(result.output.get("text", "")) or None
            if emit is not None:
                emit(
                    {
                        **accumulator.output(provisional=True),
                        "completed_seconds": source.consumed_frames
                        / source.sample_rate,
                        "total_seconds": source.duration_seconds,
                    }
                )
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


class Qwen3AsrLongFormOrchestrator(CapabilityOrchestrator):
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
        request = TranscriptionRequest.from_prompt(prompt)
        if request.task != "transcribe":
            raise ValueError("Qwen3-ASR does not support speech translation")
        if request.timestamps == "character":
            raise ValueError("Qwen3-ASR does not support character timestamps")
        DecodeSettings.from_mapping(settings)
        if callable(getattr(request.audio, "__aiter__", None)):
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
        timestamps = request.timestamps
        audio = await settled_to_thread(snapshot_file_like, request.audio)
        owned_prompt = {**prompt, "audio": audio}
        source = await settled_to_thread(open_audio_source, audio, request)
        chunk_seconds = _CHUNK_SECONDS[timestamps]
        if source.duration_seconds <= chunk_seconds and not request.stream:
            await settled_to_thread(source.close)
            return await invoke(owned_prompt, image=image, settings=settings)
        if not request.stream:
            return await _run_chunks(
                invoke,
                source,
                image=image,
                prompt=owned_prompt,
                settings=settings,
            )

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


__all__ = ["Qwen3AsrLongFormOrchestrator"]
