"""Bounded buffering for live mono PCM transcription."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from torch import Tensor

from .audio import DecodedAudio, MAX_AUDIO_SECONDS, _float_pcm


MAX_LIVE_CHUNK_VALUES = 1_048_576


class LiveAudioBuffer:
    """Own live PCM until a model-sized window is ready."""

    def __init__(
        self,
        sample_rate: int,
        *,
        window_seconds: float,
        update_seconds: float,
        target_sample_rate: int = 16_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.window_frames = round(window_seconds * sample_rate)
        self.update_frames = round(update_seconds * sample_rate)
        if not 0 < self.update_frames <= self.window_frames:
            raise ValueError("live update interval must fit within its window")
        self.max_total_frames = sample_rate * MAX_AUDIO_SECONDS
        self.received_frames = 0
        self.consumed_frames = 0
        self.buffered_frames = 0
        self._last_evaluated_frames = 0
        self._start = 0
        self._storage = np.empty(
            self.window_frames + MAX_LIVE_CHUNK_VALUES,
            dtype=np.float32,
        )

    @property
    def duration_seconds(self) -> float:
        return self.received_frames / self.sample_rate

    @property
    def source_duration_seconds(self) -> float:
        return self.duration_seconds

    @property
    def clip_start_seconds(self) -> float:
        return 0.0

    @property
    def full(self) -> bool:
        return self.buffered_frames >= self.window_frames

    @property
    def ready_for_update(self) -> bool:
        return (
            self.buffered_frames > 0
            and self.received_frames - self._last_evaluated_frames >= self.update_frames
        )

    def append(self, chunk: object) -> None:
        if not isinstance(chunk, (np.ndarray, Tensor)):
            raise TypeError(
                "live audio chunks must be one-dimensional NumPy or CPU Torch PCM"
            )
        waveform = _float_pcm(chunk)
        size = int(waveform.size)
        if size > MAX_LIVE_CHUNK_VALUES:
            raise ValueError(
                f"live audio chunk exceeds the {MAX_LIVE_CHUNK_VALUES}-value limit"
            )
        if self.received_frames + size > self.max_total_frames:
            raise ValueError(f"live audio exceeds the {MAX_AUDIO_SECONDS}-second limit")

        end = self._start + self.buffered_frames
        if end + size > self._storage.size:
            self._storage[: self.buffered_frames] = self._storage[
                self._start : end
            ].copy()
            self._start = 0
            end = self.buffered_frames
        self._storage[end : end + size] = waveform
        self.received_frames += size
        self.buffered_frames += size

    def snapshot(
        self,
        *,
        offset_frames: int = 0,
        frame_count: int | None = None,
    ) -> DecodedAudio:
        if self.buffered_frames == 0:
            raise RuntimeError("live audio buffer is empty")
        available = self.buffered_frames - offset_frames
        frames = (
            min(available, self.window_frames) if frame_count is None else frame_count
        )
        if offset_frames < 0 or frames <= 0 or frames > available:
            raise ValueError("live audio snapshot is outside buffered PCM")
        start = self._start + offset_frames
        source = self._storage[start : start + frames].copy()
        duration = frames / self.sample_rate
        if self.sample_rate == self.target_sample_rate:
            waveform = source
        else:
            import kestrel_native

            waveform = _float_pcm(
                kestrel_native.resample_audio_mono(
                    source,
                    self.sample_rate,
                    self.target_sample_rate,
                    max_output_values=(
                        int(np.ceil(duration * self.target_sample_rate)) + 16
                    ),
                )
            )
        self._last_evaluated_frames = self.consumed_frames + offset_frames + frames
        return DecodedAudio(
            waveform=waveform,
            duration_seconds=duration,
            source_duration_seconds=self.duration_seconds,
            clip_start_seconds=(self.consumed_frames + offset_frames)
            / self.sample_rate,
        )

    def advance(self, frames: int) -> None:
        if not 0 < frames <= self.buffered_frames:
            raise ValueError("live audio advance is outside buffered PCM")
        self._start += frames
        self.buffered_frames -= frames
        self.consumed_frames += frames
        if self.buffered_frames == 0:
            self._start = 0


async def live_audio_windows(
    chunks: object,
    buffer: LiveAudioBuffer,
) -> AsyncIterator[tuple[DecodedAudio, bool]]:
    """Yield provisional and committed windows from one owned async source."""

    iterator_factory = getattr(chunks, "__aiter__", None)
    if not callable(iterator_factory):
        raise TypeError("live audio must implement the asynchronous iterator protocol")
    iterator = iterator_factory()
    if not callable(getattr(iterator, "__anext__", None)):
        raise TypeError("live audio __aiter__ must return an asynchronous iterator")

    saw_chunk = False
    try:
        async for chunk in iterator:
            saw_chunk = True
            buffer.append(chunk)
            while buffer.full:
                frames = min(buffer.buffered_frames, buffer.window_frames)
                window = await asyncio.to_thread(buffer.snapshot)
                buffer.advance(frames)
                yield window, True
            if buffer.ready_for_update:
                yield await asyncio.to_thread(buffer.snapshot), False
        if not saw_chunk:
            raise ValueError("live audio must yield at least one PCM chunk")
        if buffer.buffered_frames:
            frames = buffer.buffered_frames
            window = await asyncio.to_thread(buffer.snapshot)
            buffer.advance(frames)
            yield window, True
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
