from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult
from kestrel.runtime.tokens import TextToken

from kestrel.models.whisper.longform import (
    _MAX_NATIVE_AUDIO_VALUES,
    _LivePcmBuffer,
    _NativeFileWindows,
    _checked_leaf_result,
    _committed_text_token_ids,
    WhisperLongFormOrchestrator,
)
from kestrel.models.whisper.quality import compression_ratio


class _Reader:
    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self._samples = samples
        self._offset = 0
        self.sample_rate = sample_rate
        self.total_frames = int(samples.size)
        self.closed = False
        self.read_calls = 0
        self.read_sizes: list[int] = []

    def read(self, max_frames: int) -> np.ndarray | None:
        self.read_calls += 1
        self.read_sizes.append(max_frames)
        if self._offset == self._samples.size:
            return None
        end = min(self._samples.size, self._offset + max_frames)
        result = self._samples[self._offset : end]
        self._offset = end
        return result

    def close(self) -> None:
        self.closed = True


class _SeekReader(_Reader):
    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        super().__init__(samples, sample_rate)
        self.seek_calls: list[int] = []

    def seek(self, frame: int) -> int:
        self.seek_calls.append(frame)
        self._offset = frame
        return frame


class _LeafStream:
    def __init__(self, deltas: list[str], result: EngineResult) -> None:
        self._deltas = iter(deltas)
        self._result = result
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0.01)
        try:
            text = next(self._deltas)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return type("Update", (), {"text": text})()

    async def result(self) -> EngineResult:
        return self._result

    async def aclose(self) -> None:
        self.closed = True


def _result(
    request_id: int,
    text: str,
    segments: list[dict[str, object]],
    token_ids: list[int],
    *,
    language: str = "en",
    temperature: float = 0.0,
    avg_logprob: float = -0.25,
    no_speech_prob: float = 0.01,
    language_probability: float | None = None,
) -> EngineResult:
    diagnostics = {
        "temperature": temperature,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio(text),
        "no_speech_prob": no_speech_prob,
    }
    return EngineResult(
        request_id=request_id,
        tokens=[TextToken(token_id) for token_id in token_ids],
        finish_reason="stop",
        metrics=EngineMetrics(
            input_tokens=3,
            output_tokens=len(token_ids),
            prefill_time_ms=1.0,
            decode_time_ms=2.0,
            ttft_ms=3.0,
            cached_tokens=0,
        ),
        output={
            "text": text,
            "language": language,
            "language_probability": language_probability,
            "segments": [{**segment, **diagnostics} for segment in segments],
            **diagnostics,
        },
    )


def _segment(start: float, end: float, text: str) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "temperature": 0.0,
        "avg_logprob": -0.25,
        "compression_ratio": compression_ratio(text),
        "no_speech_prob": 0.01,
    }


def test_live_pcm_tiny_chunks_use_one_bounded_store() -> None:
    buffer = _LivePcmBuffer(sample_rate=100)
    storage = buffer._storage
    for value in range(1_000):
        buffer.append(np.array([value / 1_000], dtype=np.float32))

    assert buffer._storage is storage
    assert buffer._storage.size == buffer.window_frames + 1024 * 1024
    np.testing.assert_allclose(buffer.current(), np.arange(1_000) / 1_000)


def _run(invoke, prompt, settings=None):
    return asyncio.run(
        WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt=prompt,
            settings=settings,
        )
    )


def test_short_file_preserves_the_single_leaf_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(dict(prompt))
        return _result(1, "short", [], [10])

    result = _run(invoke, {"audio": Path("short.wav"), "timestamps": "none"})

    assert isinstance(result, EngineResult)
    assert calls == [{"audio": Path("short.wav"), "timestamps": "none"}]
    assert reader.read_calls == 0
    assert reader.closed is True


def test_best_of_failure_waits_for_every_candidate() -> None:
    started = 0
    finished: list[int] = []

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal started
        candidate = started
        started += 1
        if candidate == 0:
            await asyncio.sleep(0)
            raise RuntimeError("candidate failed")
        await asyncio.sleep(0.01)
        finished.append(candidate)
        return _result(candidate, "unused", [], [10], temperature=0.2)

    with pytest.raises(RuntimeError, match="candidate failed"):
        _run(
            invoke,
            {"audio": np.zeros(1, dtype=np.float32), "sample_rate": 16_000},
            {"temperature": 0.2, "best_of": 3},
        )

    assert started == 3
    assert finished == [1, 2]


@pytest.mark.parametrize(
    ("case", "require_words", "message"),
    (
        ("container", False, "segments"),
        ("timestamp", False, "timestamps"),
        ("diagnostic", False, "disagrees"),
        ("words", True, "omitted"),
        ("tokens", False, "tokens"),
    ),
)
def test_leaf_boundary_rejects_malformed_results(
    case: str,
    require_words: bool,
    message: str,
) -> None:
    result = _result(1, "text", [_segment(0.0, 1.0, "text")], [10])
    segments = result.output["segments"]
    assert isinstance(segments, list)
    if case == "container":
        result.output["segments"] = "not a list"
    elif case == "timestamp":
        segments[0]["end"] = 31.0
    elif case == "diagnostic":
        segments[0]["avg_logprob"] = -0.5
    elif case == "tokens":
        result.tokens = [object()]  # type: ignore[list-item]

    with pytest.raises(ValueError, match=message):
        _checked_leaf_result(result, window_duration=30.0, require_words=require_words)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("stream", "yes", "stream"),
        ("timestamps", "token", "timestamps"),
        ("task", "summarize", "task"),
        ("condition_on_previous_text", 1, "condition_on_previous_text"),
    ),
)
def test_shared_options_fail_before_source_open(
    option: str,
    value: object,
    message: str,
) -> None:
    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("invalid options must fail before inference")

    with pytest.raises((TypeError, ValueError), match=message):
        _run(invoke, {"audio": Path("must-not-open.wav"), option: value})


def test_short_file_retries_on_low_selected_token_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    attempts: list[tuple[float, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        attempts.append((settings["temperature"], prompt.get("language")))
        return _result(
            len(attempts),
            "retry" if len(attempts) == 1 else "accepted",
            [],
            [10],
            temperature=settings["temperature"],
            avg_logprob=-1.1 if len(attempts) == 1 else -0.5,
            language_probability=0.75 if len(attempts) == 1 else None,
        )

    result = _run(invoke, {"audio": Path("short.wav"), "timestamps": "none"})

    assert result.output["text"] == "accepted"
    assert result.output["temperature"] == 0.2
    assert result.output["language_probability"] == 0.75
    assert attempts == [(0.0, None), *[(0.2, "en")] * 5]
    assert reader.closed is True


def test_short_stream_retracts_rejected_attempt_before_selected_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    attempts: list[float] = []

    async def invoke(prompt, *, image=None, settings=None):
        attempts.append(settings["temperature"])
        accepted = len(attempts) == 2
        result = _result(
            len(attempts),
            "accepted" if accepted else "rejected",
            [],
            [10],
            temperature=settings["temperature"],
            avg_logprob=-0.5 if accepted else -1.1,
        )
        return _LeafStream(
            ["accept", "ed"] if accepted else ["reject", "ed"],
            result,
        )

    async def scenario():
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": Path("short.wav"), "timestamps": "none", "stream": True},
            settings={"temperature": [0.0, 0.2]},
        )
        assert isinstance(stream, CapabilityStream)
        updates = [update async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(scenario())

    assert attempts == [0.0, *[0.2] * 5]
    assert any(update.output.get("retrying") is True for update in updates)
    assert updates[-1].output["provisional"] is False
    assert updates[-1].text == "accepted"
    assert result.output["text"] == "accepted"
    assert reader.closed is True


def test_short_stream_owns_raw_pcm_before_returning() -> None:
    audio = np.arange(100, dtype=np.float32) / 1000
    expected = audio.copy()
    started = asyncio.Event()
    release = asyncio.Event()
    observed: np.ndarray | None = None

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal observed
        started.set()
        await release.wait()
        observed = np.asarray(prompt["audio"]).copy()
        return _LeafStream(
            [],
            _result(1, "", [], [10], temperature=settings["temperature"]),
        )

    async def scenario() -> None:
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": audio, "sample_rate": 10, "stream": True},
            settings={"temperature": 0.0},
        )
        assert isinstance(stream, CapabilityStream)
        await asyncio.wait_for(started.wait(), timeout=2.0)
        audio.fill(1.0)
        release.set()
        await asyncio.wait_for(stream.result(), timeout=2.0)

    asyncio.run(scenario())

    assert observed is not None
    np.testing.assert_array_equal(observed, expected)


def test_short_stream_cancellation_closes_candidates_that_already_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    opened = asyncio.Event()
    candidate = _LeafStream([], _result(1, "", [], [10], temperature=0.2))
    calls = 0

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            opened.set()
            return candidate
        await asyncio.Future()

    async def scenario() -> None:
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": Path("short.wav"), "stream": True},
            settings={"temperature": 0.2, "best_of": 2},
        )
        assert isinstance(stream, CapabilityStream)
        await asyncio.wait_for(opened.wait(), timeout=2.0)
        await stream.aclose()

    asyncio.run(scenario())

    assert calls == 2
    assert candidate.closed is True


def test_short_stream_cancellation_settles_real_engine_streams_without_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    settling = asyncio.Event()
    release = asyncio.Event()
    settling_count = 0
    settled_count = 0
    calls = 0

    class ResultOnlyStream:
        def __init__(self) -> None:
            self._settled = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def result(self) -> EngineResult:
            nonlocal settled_count, settling_count
            if self._settled:
                return _result(1, "", [], [10], temperature=0.2)
            settling_count += 1
            if settling_count == 2:
                settling.set()
            await release.wait()
            self._settled = True
            settled_count += 1
            return _result(1, "", [], [10], temperature=0.2)

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal calls
        calls += 1
        return ResultOnlyStream()

    async def scenario() -> None:
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": Path("short.wav"), "stream": True},
            settings={"temperature": 0.2, "best_of": 2},
        )
        assert isinstance(stream, CapabilityStream)
        await asyncio.wait_for(settling.wait(), timeout=2.0)
        closing = asyncio.create_task(stream.aclose())
        try:
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        await asyncio.wait_for(closing, timeout=2.0)

    asyncio.run(scenario())

    assert calls == 2
    assert settled_count == 2


def test_cancelling_native_reader_open_closes_the_late_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    started = threading.Event()
    release = threading.Event()

    def open_reader(*args, **kwargs):
        started.set()
        release.wait()
        return reader

    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        open_reader,
        raising=False,
    )

    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("cancelled open must not reach inference")

    async def scenario() -> None:
        task = asyncio.create_task(
            WhisperLongFormOrchestrator().run(
                invoke,
                image=None,
                prompt={"audio": Path("short.wav")},
                settings=None,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 2.0)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert reader.closed is True


def test_cancelling_file_like_snapshot_settles_the_reader_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingFile:
        def read(self, size: int) -> bytes:
            started.set()
            release.wait()
            finished.set()
            return b""

    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("cancelled snapshot must not reach inference")

    async def scenario() -> None:
        task = asyncio.create_task(
            WhisperLongFormOrchestrator().run(
                invoke,
                image=None,
                prompt={"audio": BlockingFile()},
                settings=None,
            )
        )
        did_start = await asyncio.to_thread(started.wait, 2.0)
        task.cancel()
        try:
            assert did_start
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert finished.is_set()


def test_cancelling_long_file_read_settles_before_reader_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingReader(_Reader):
        def __init__(self) -> None:
            super().__init__(np.zeros(301, dtype=np.float32), sample_rate=10)
            self.closed_while_reading = False

        def read(self, max_frames: int) -> np.ndarray | None:
            started.set()
            release.wait()
            result = super().read(max_frames)
            finished.set()
            return result

        def close(self) -> None:
            self.closed_while_reading = not finished.is_set()
            super().close()

    reader = BlockingReader()
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )

    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("cancelled read must not reach inference")

    async def scenario() -> None:
        task = asyncio.create_task(
            WhisperLongFormOrchestrator().run(
                invoke,
                image=None,
                prompt={"audio": Path("long.wav")},
                settings=None,
            )
        )
        did_start = await asyncio.to_thread(started.wait, 2.0)
        task.cancel()
        try:
            assert did_start
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert finished.is_set()
    assert reader.closed is True
    assert reader.closed_while_reading is False


def test_high_no_speech_probability_skips_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.arange(100, dtype=np.float32) / 1000, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls = 0

    async def invoke(prompt, *, image=None, settings=None):
        nonlocal calls
        calls += 1
        return _result(
            calls,
            "hallucinated speech",
            [],
            [10],
            avg_logprob=-1.1,
            no_speech_prob=0.9,
        )

    result = _run(invoke, {"audio": Path("short.wav"), "timestamps": "none"})

    assert result.output["text"] == ""
    assert result.output["segments"] == []
    assert result.tokens == []
    assert calls == 1
    assert reader.closed is True


def test_long_encoded_bytes_use_incremental_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.zeros(301, dtype=np.float32), sample_rate=10)
    opened: list[bytes] = []

    def open_bytes(data, *, max_duration_seconds):
        opened.append(data)
        return reader

    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_mono",
        open_bytes,
        raising=False,
    )

    async def invoke(prompt, *, image=None, settings=None):
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            1,
            "bytes",
            [{"start": 0.0, "end": duration, "text": "bytes"}],
            [10],
        )

    result = _run(invoke, {"audio": b"encoded"})

    assert opened == [b"encoded"]
    assert result.output["duration_seconds"] == 30.1
    assert reader.closed is True


def test_clip_range_uses_exact_native_seek_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.arange(1000, dtype=np.float32) / 1000
    reader = _SeekReader(samples, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls: list[dict[str, Any]] = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(dict(prompt))
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            len(calls),
            "clip",
            [{"start": 0.0, "end": duration, "text": "clip"}],
            [10],
        )

    _run(
        invoke,
        {
            "audio": Path("source.wav"),
            "clip_start_seconds": 25.0,
            "clip_end_seconds": 35.0,
        },
    )

    assert reader.seek_calls == [250]
    assert reader.read_sizes == [100]
    np.testing.assert_array_equal(calls[0]["audio"], samples[250:350])
    assert reader.closed is True


@pytest.mark.parametrize("timestamps", ("segment", "word", "none"))
def test_long_file_uses_timestamp_driven_bounded_windows(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: str,
) -> None:
    samples = np.arange(650, dtype=np.float32) / 1000
    reader = _Reader(samples, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls: list[dict[str, Any]] = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(dict(prompt))
        index = len(calls)
        if index == 1:
            segment = {"start": 0.0, "end": 20.0, "text": "a"}
            if timestamps == "word":
                segment["words"] = [
                    {"start": 1.0, "end": 2.0, "word": " a", "probability": 0.9}
                ]
            return _result(
                index,
                "a",
                [segment],
                [50365, 50365 + 1000],
            )
        duration = prompt["audio"].size / prompt["sample_rate"]
        text = "b" if index == 2 else "c"
        segment = {"start": 0.0, "end": duration, "text": text}
        if timestamps == "word":
            segment["words"] = [
                {
                    "start": 0.0,
                    "end": min(1.0, duration),
                    "word": f" {text}",
                    "probability": 0.8,
                }
            ]
        return _result(
            index,
            text,
            [segment],
            [10],
        )

    result = _run(
        invoke,
        {"audio": "long.wav", "language": None, "timestamps": timestamps},
        {"max_tokens": 200},
    )

    assert isinstance(result, EngineResult)
    assert len(calls) == 3
    assert [call["audio"].size for call in calls] == [300, 300, 150]
    np.testing.assert_array_equal(calls[0]["audio"], samples[:300])
    np.testing.assert_array_equal(calls[1]["audio"], samples[200:500])
    np.testing.assert_array_equal(calls[2]["audio"], samples[500:650])
    assert [call["language"] for call in calls] == [None, "en", "en"]
    expected_leaf_timestamps = "word" if timestamps == "word" else "segment"
    assert all(call["timestamps"] == expected_leaf_timestamps for call in calls)
    assert "_previous_text_token_ids" not in calls[0]
    assert "_previous_text_token_ids" not in calls[1]
    assert calls[2]["_previous_text_token_ids"] == (10,)
    assert result.output["text"] == "a b c"
    assert result.output["language"] == "en"
    assert result.output["temperature"] == 0.0
    assert result.output["avg_logprob"] == -0.25
    assert result.output["compression_ratio"] == compression_ratio("a b c")
    assert result.output["no_speech_prob"] == 0.01
    expected_segments = [
        _segment(0.0, 20.0, "a"),
        _segment(20.0, 50.0, "b"),
        _segment(50.0, 65.0, "c"),
    ]
    if timestamps == "word":
        expected_segments[0]["words"] = [
            {"start": 1.0, "end": 2.0, "word": " a", "probability": 0.9}
        ]
        expected_segments[1]["words"] = [
            {"start": 20.0, "end": 21.0, "word": " b", "probability": 0.8}
        ]
        expected_segments[2]["words"] = [
            {"start": 50.0, "end": 51.0, "word": " c", "probability": 0.8}
        ]
    assert result.output["segments"] == (
        expected_segments if timestamps in {"segment", "word"} else []
    )
    assert result.metrics.input_tokens == 9
    assert result.metrics.output_tokens == 4
    assert reader.closed is True


def test_long_file_does_not_publish_open_segment_past_timestamp_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.zeros(400, dtype=np.float32)
    reader = _Reader(samples, sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls: list[dict[str, Any]] = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(dict(prompt))
        if len(calls) == 1:
            # The duplicate 10-second timestamp completes the first segment and
            # starts a second one, which ends with EOS rather than a timestamp.
            # That open suffix must be decoded again from the new window, not
            # published against the old window's 30-second fallback end.
            return _result(
                1,
                "kept provisional",
                [
                    {"start": 0.0, "end": 10.0, "text": "kept"},
                    {"start": 10.0, "end": 30.0, "text": "provisional"},
                ],
                [50365, 10, 50865, 50865, 11, 50257],
            )
        return _result(
            2,
            "final",
            [{"start": 0.0, "end": 30.0, "text": "final"}],
            [10],
        )

    result = _run(invoke, {"audio": Path("long.wav")})

    assert [call["audio"].size for call in calls] == [300, 300]
    np.testing.assert_array_equal(calls[1]["audio"], samples[100:400])
    assert calls[1]["_previous_text_token_ids"] == (10,)
    assert result.output["text"] == "kept final"
    assert [
        (segment["start"], segment["end"], segment["text"])
        for segment in result.output["segments"]
    ] == [
        (0.0, 10.0, "kept"),
        (10.0, 40.0, "final"),
    ]
    assert result.tokens == []


def test_long_file_does_not_lock_language_from_leading_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.zeros(400, dtype=np.float32), sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    prompts: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        prompts.append(dict(prompt))
        if len(prompts) == 1:
            return _result(
                1,
                "",
                [],
                [],
                language="fr",
                language_probability=0.8,
            )
        return _result(
            2,
            "hello",
            [{"start": 0.0, "end": 10.0, "text": "hello"}],
            [11],
            language="en",
            language_probability=0.9,
        )

    result = _run(invoke, {"audio": Path("long.wav")})

    assert [prompt["language"] for prompt in prompts] == [None, None]
    assert result.output["language"] == "en"
    assert result.output["language_probability"] == 0.9
    assert result.output["text"] == "hello"


@pytest.mark.parametrize(
    ("task", "parts", "expected"),
    (
        ("transcribe", ("你好", "世界"), "你好世界"),
        ("translate", ("hello", "world"), "hello world"),
    ),
)
def test_long_file_joins_segments_in_the_emitted_text_language(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    parts: tuple[str, str],
    expected: str,
) -> None:
    reader = _Reader(np.zeros(400, dtype=np.float32), sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    texts = iter(parts)

    async def invoke(prompt, *, image=None, settings=None):
        text = next(texts)
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            1,
            text,
            [{"start": 0.0, "end": duration, "text": text}],
            [10],
            language="zh",
        )

    result = _run(
        invoke,
        {"audio": Path("long.wav"), "language": "zh", "task": task},
    )

    assert result.output["text"] == expected
    assert result.output["compression_ratio"] == compression_ratio(expected)


def test_high_temperature_fallback_resets_previous_text_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.zeros(601, dtype=np.float32), sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    prompts: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        prompts.append(dict(prompt))
        leaf = len(prompts)
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            len(prompts),
            f"part {leaf}",
            [{"start": 0.0, "end": duration, "text": f"part {leaf}"}],
            [10 + leaf],
            temperature=settings["temperature"],
            avg_logprob=-0.2,
        )

    _run(
        invoke,
        {"audio": Path("long.wav")},
        {"temperature": 0.6, "best_of": 1},
    )

    assert len(prompts) == 3
    assert all("_previous_text_token_ids" not in prompt for prompt in prompts)
    assert reader.closed is True


def test_long_file_closes_reader_when_a_leaf_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.zeros(301, dtype=np.float32), sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )

    async def invoke(prompt, *, image=None, settings=None):
        raise RuntimeError("leaf failed")

    with pytest.raises(RuntimeError, match="leaf failed"):
        _run(invoke, {"audio": Path("long.wav")})
    assert reader.closed is True


def test_long_file_stream_coalesces_bounded_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(np.zeros(610, dtype=np.float32), sample_rate=10)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )
    calls: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        calls.append(dict(prompt))
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            len(calls),
            f"part {len(calls)}",
            [{"start": 0.0, "end": duration, "text": f"part {len(calls)}"}],
            [10],
        )

    async def scenario() -> None:
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": Path("long.wav"), "stream": True},
            settings=None,
        )
        assert isinstance(stream, CapabilityStream)
        result = await stream.result()
        assert result.output["text"] == "part 1 part 2 part 3"
        update = await anext(stream)
        assert update.output["completed_seconds"] == 61.0
        assert update.output["total_seconds"] == 61.0
        assert update.output["provisional"] is False
        assert update.output["text"] == result.output["text"]
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(scenario())
    assert all(call["stream"] is False for call in calls)
    assert reader.closed is True


def test_high_rate_file_never_exceeds_native_read_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extreme_reader = _Reader(np.zeros(1, dtype=np.float32), sample_rate=2_000_000)
    assert _NativeFileWindows(extreme_reader).window_frames == _MAX_NATIVE_AUDIO_VALUES

    sample_rate = 40_000
    reader = _Reader(np.zeros(30 * sample_rate + 1, dtype=np.float32), sample_rate)
    monkeypatch.setattr(
        "kestrel.models.whisper.longform.kestrel_native.open_audio_file_mono",
        lambda *args, **kwargs: reader,
        raising=False,
    )

    async def invoke(prompt, *, image=None, settings=None):
        duration = prompt["audio"].size / prompt["sample_rate"]
        return _result(
            1,
            "audio",
            [{"start": 0.0, "end": duration, "text": "audio"}],
            [10],
        )

    _run(invoke, {"audio": "high-rate.wav"})
    assert max(reader.read_sizes) <= 1024 * 1024


def test_live_pcm_stream_commits_only_stable_timestamped_prefixes() -> None:
    closed = False

    async def chunks():
        nonlocal closed
        try:
            yield np.full(50, 0.1, dtype=np.float32)
            yield np.full(50, 0.2, dtype=np.float32)
        finally:
            closed = True

    prompts: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        prompts.append(dict(prompt))
        index = len(prompts)
        # The first model timestamp is deliberately between source frames.
        # Advancing and reporting must use the same rounded frame boundary or
        # the next committed segment can overlap it.
        # The final segment ends one second before the source. The final
        # snapshot must still report the trailing silence as completed.
        end = (3.04, 4.0, 2.0)[index - 1]
        return _result(
            index,
            ("one", "two", "three")[index - 1],
            [
                {
                    "start": 0.0,
                    "end": end,
                    "text": ("one", "two", "three")[index - 1],
                }
            ],
            [9 + index],
            temperature=settings["temperature"],
        )

    async def scenario():
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": chunks(), "sample_rate": 10, "stream": True},
            settings={"temperature": 0.0},
        )
        assert isinstance(stream, CapabilityStream)
        updates = [update async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(scenario())

    assert [prompt["audio"].size for prompt in prompts] == [50, 70, 30]
    assert "_previous_text_token_ids" not in prompts[0]
    assert prompts[1]["_previous_text_token_ids"] == (10,)
    assert prompts[2]["_previous_text_token_ids"] == (10, 11)
    # Progress delivery is scheduler-dependent: a fast consumer may observe
    # every snapshot, while a slow one observes only the newest pending value.
    # Either way, delivered snapshots must be ordered stable prefixes and end
    # at the exact final transcript.
    delivered_text = [update.text for update in updates]
    stable_prefixes = ["one", "one two", "one two three"]
    assert delivered_text
    assert delivered_text == sorted(
        delivered_text,
        key=stable_prefixes.index,
    )
    assert len(delivered_text) == len(set(delivered_text))
    assert set(delivered_text) <= set(stable_prefixes)
    assert delivered_text[-1] == "one two three"
    assert updates[-1].output["completed_seconds"] == 10.0
    assert result.output["text"] == "one two three"
    assert result.output["duration_seconds"] == 10.0
    assert result.output["segments"] == [
        _segment(0.0, 3.0, "one"),
        _segment(3.0, 7.0, "two"),
        _segment(7.0, 9.0, "three"),
    ]
    assert result.tokens == []
    assert closed is True


def test_live_context_excludes_uncommitted_trailing_segment() -> None:
    leaf = _checked_leaf_result(
        _result(
            1,
            "one two",
            [
                {"start": 0.0, "end": 2.0, "text": "one"},
                {"start": 2.0, "end": 5.0, "text": "two"},
            ],
            [50365, 10, 50465, 50465, 11, 50615, 50257],
        ),
        window_duration=5.0,
        require_words=False,
    )

    assert _committed_text_token_ids(
        leaf,
        sample_rate=10,
        advance_frames=20,
        all_segments_committed=False,
    ) == (10,)


def test_live_pcm_can_return_one_final_result_without_progress_stream() -> None:
    async def chunks():
        yield np.arange(20, dtype=np.int16)

    async def invoke(prompt, *, image=None, settings=None):
        np.testing.assert_allclose(
            prompt["audio"],
            np.arange(20, dtype=np.float32) / 32768.0,
        )
        return _result(
            1,
            "done",
            [{"start": 0.0, "end": 2.0, "text": "done"}],
            [10],
            temperature=settings["temperature"],
        )

    result = _run(
        invoke,
        {"audio": chunks(), "sample_rate": 10},
        {"temperature": 0.0},
    )

    assert result.output["text"] == "done"
    assert result.output["duration_seconds"] == 2.0


def test_live_pcm_does_not_lock_language_from_leading_silence() -> None:
    async def chunks():
        yield np.zeros(50, dtype=np.float32)

    prompts: list[dict[str, object]] = []

    async def invoke(prompt, *, image=None, settings=None):
        prompts.append(dict(prompt))
        if len(prompts) == 1:
            return _result(
                1,
                "",
                [],
                [],
                language="fr",
                language_probability=0.8,
                temperature=settings["temperature"],
            )
        return _result(
            2,
            "hello",
            [{"start": 0.0, "end": 5.0, "text": "hello"}],
            [11],
            language="en",
            language_probability=0.9,
            temperature=settings["temperature"],
        )

    result = _run(
        invoke,
        {"audio": chunks(), "sample_rate": 10},
        {"temperature": 0.0},
    )

    assert [prompt["language"] for prompt in prompts] == [None, None]
    assert result.output["language"] == "en"
    assert result.output["language_probability"] == 0.9
    assert result.output["text"] == "hello"


def test_live_pcm_stream_finishes_an_exact_window_with_trailing_silence() -> None:
    async def chunks():
        yield np.zeros(300, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        return _result(
            1,
            "done",
            [{"start": 0.0, "end": 2.0, "text": "done"}],
            [10],
            temperature=settings["temperature"],
        )

    async def scenario():
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": chunks(), "sample_rate": 10, "stream": True},
            settings={"temperature": 0.0},
        )
        updates = [update async for update in stream]
        return updates, await stream.result()

    updates, result = asyncio.run(scenario())

    assert updates[-1].output["provisional"] is False
    assert updates[-1].output["completed_seconds"] == 30.0
    assert updates[-1].output["total_seconds"] == 30.0
    assert result.output["text"] == "done"


def test_live_pcm_finalizes_full_window_when_stable_prefix_is_too_short() -> None:
    async def chunks():
        yield np.zeros(300, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        return _result(
            1,
            "one two",
            [
                {"start": 0.0, "end": 1.0, "text": "one"},
                {"start": 1.0, "end": 30.0, "text": "two"},
            ],
            [10, 11],
            temperature=settings["temperature"],
        )

    result = _run(
        invoke,
        {"audio": chunks(), "sample_rate": 10},
        {"temperature": 0.0},
    )

    assert result.output["text"] == "one two"
    assert [
        (segment["start"], segment["end"], segment["text"])
        for segment in result.output["segments"]
    ] == [
        (0.0, 1.0, "one"),
        (1.0, 30.0, "two"),
    ]
    assert result.output["duration_seconds"] == 30.0


@pytest.mark.parametrize(
    ("chunk", "message"),
    (
        (np.zeros((2, 3), dtype=np.float32), "one-dimensional"),
        (np.empty(0, dtype=np.float32), "must not be empty"),
        (np.array([float("nan")], dtype=np.float32), "NaN or infinity"),
        (b"raw bytes", "NumPy or CPU Torch"),
    ),
)
def test_live_pcm_rejects_malformed_chunks(chunk: object, message: str) -> None:
    async def chunks():
        yield chunk

    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("malformed live chunks must fail before inference")

    with pytest.raises((TypeError, ValueError), match=message):
        _run(invoke, {"audio": chunks(), "sample_rate": 16_000})


def test_live_pcm_enforces_total_duration_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kestrel.models.whisper.longform._MAX_FILE_DURATION_SECONDS", 1)

    async def chunks():
        yield np.zeros(10, dtype=np.float32)
        yield np.zeros(1, dtype=np.float32)

    async def invoke(prompt, *, image=None, settings=None):
        raise AssertionError("overlong live input must fail before inference")

    with pytest.raises(ValueError, match="1-second limit"):
        _run(invoke, {"audio": chunks(), "sample_rate": 10})


def test_closing_live_stream_closes_the_input_iterator() -> None:
    closed = False

    async def chunks():
        nonlocal closed
        try:
            yield np.zeros(50, dtype=np.float32)
            await asyncio.Future()
        finally:
            closed = True

    async def invoke(prompt, *, image=None, settings=None):
        return _result(
            1,
            "provisional",
            [{"start": 0.0, "end": 5.0, "text": "provisional"}],
            [10],
            temperature=settings["temperature"],
        )

    async def scenario() -> None:
        stream = await WhisperLongFormOrchestrator().run(
            invoke,
            image=None,
            prompt={"audio": chunks(), "sample_rate": 10, "stream": True},
            settings={"temperature": 0.0},
        )
        assert isinstance(stream, CapabilityStream)
        update = await anext(stream)
        assert update.text == "provisional"
        assert update.output["completed_seconds"] == 0.0
        await stream.aclose()

    asyncio.run(scenario())
    assert closed is True
