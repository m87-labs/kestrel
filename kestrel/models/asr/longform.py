"""Shared mechanics for bounded long-form transcription."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from kestrel.engine import EngineMetrics, EngineResult

from .audio import AudioChunks, MAX_AUDIO_SECONDS
from .contract import TranscriptionRequest


END_OF_AUDIO = object()


async def settled_to_thread(function: Callable[..., Any], *args: object) -> Any:
    """Finish native audio work before propagating cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except BaseException:
        try:
            await task
        except BaseException:
            pass
        raise


def open_audio_source(audio: object, request: TranscriptionRequest) -> AudioChunks:
    return AudioChunks(
        audio,  # type: ignore[arg-type]
        sample_rate=request.sample_rate,
        clip_start_seconds=request.clip_start_seconds,
        clip_end_seconds=request.clip_end_seconds,
        target_sample_rate=16_000,
        max_duration_seconds=MAX_AUDIO_SECONDS,
    )


def shift_segments(
    segments: object,
    offset_seconds: float,
) -> list[dict[str, object]]:
    shifted = []
    for raw in segments if isinstance(segments, list) else ():
        if not isinstance(raw, Mapping):
            continue
        segment = dict(raw)
        segment["start"] = float(segment["start"]) + offset_seconds
        segment["end"] = float(segment["end"]) + offset_seconds
        for key in ("words", "characters"):
            items = segment.get(key)
            if isinstance(items, list):
                segment[key] = [
                    {
                        **item,
                        "start": float(item["start"]) + offset_seconds,
                        "end": float(item["end"]) + offset_seconds,
                    }
                    for item in items
                    if isinstance(item, Mapping)
                ]
        shifted.append(segment)
    return shifted


def aggregate_metrics(results: list[EngineResult]) -> EngineMetrics:
    return EngineMetrics(
        input_tokens=sum(item.metrics.input_tokens for item in results),
        output_tokens=sum(item.metrics.output_tokens for item in results),
        prefill_time_ms=sum(item.metrics.prefill_time_ms for item in results),
        decode_time_ms=sum(item.metrics.decode_time_ms for item in results),
        ttft_ms=results[0].metrics.ttft_ms,
        cached_tokens=sum(item.metrics.cached_tokens for item in results),
    )


__all__ = [
    "END_OF_AUDIO",
    "aggregate_metrics",
    "open_audio_source",
    "settled_to_thread",
    "shift_segments",
]
