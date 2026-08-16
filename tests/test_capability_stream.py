"""Bounded progress delivery for compound capability orchestration."""

from __future__ import annotations

import asyncio

import pytest

from kestrel.engine import CapabilityStream, EngineMetrics, EngineResult


def _result(text: str) -> EngineResult:
    return EngineResult(
        request_id=7,
        tokens=[],
        finish_reason="stop",
        metrics=EngineMetrics(0, 0, 0.0, 0.0, 0.0),
        output={"text": text},
    )


def test_progress_is_coalesced_and_result_does_not_require_iteration() -> None:
    async def scenario() -> None:
        async def produce(emit):
            emit({"text": "first"})
            emit({"text": "latest"})
            return _result("done")

        stream = CapabilityStream("transcribe", produce)
        assert (await stream.result()).output == {"text": "done"}
        update = await anext(stream)
        assert update.task == "transcribe"
        assert update.index == 1
        assert update.output == {"text": "latest"}
        assert update.text == "latest"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(scenario())


def test_close_cancels_compound_producer() -> None:
    async def scenario() -> None:
        cancelled = asyncio.Event()

        async def produce(emit):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        stream = CapabilityStream("transcribe", produce)
        await asyncio.sleep(0)
        await stream.aclose()
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_producer_failure_reaches_iterator_and_result() -> None:
    async def scenario() -> None:
        async def produce(emit):
            raise ValueError("bad compound request")

        stream = CapabilityStream("transcribe", produce)
        with pytest.raises(ValueError, match="bad compound"):
            await anext(stream)
        with pytest.raises(ValueError, match="bad compound"):
            await stream.result()

    asyncio.run(scenario())
