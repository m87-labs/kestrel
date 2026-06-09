"""ModelStream session object behavior."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kestrel.engine import EngineMetrics, EngineResult, ModelStream, ModelStreamUpdate
from kestrel.engine._types import _ModelStreamCompletion, _ModelStreamQueue


def _result(request_id: int) -> EngineResult:
    return EngineResult(
        request_id=request_id,
        tokens=[],
        finish_reason="stop",
        metrics=EngineMetrics(
            input_tokens=0,
            output_tokens=0,
            prefill_time_ms=0.0,
            decode_time_ms=0.0,
            ttft_ms=0.0,
        ),
        output={"done": True},
    )


def test_model_stream_sends_chunks_and_yields_updates() -> None:
    async def go() -> None:
        queue: _ModelStreamQueue = asyncio.Queue()
        future: asyncio.Future[EngineResult] = asyncio.get_running_loop().create_future()
        sent: list[tuple[int, dict[str, Any]]] = []
        closed: list[int] = []

        async def send_chunk(session_id: int, chunk: dict[str, Any]) -> None:
            sent.append((session_id, chunk))

        async def close_session(session_id: int) -> None:
            closed.append(session_id)
            result = _result(session_id)
            future.set_result(result)
            queue.put_nowait(_ModelStreamCompletion(result=result))

        stream = ModelStream(
            session_id=7,
            task="point",
            queue=queue,
            result_future=future,
            send_chunk=send_chunk,
            close_session=close_session,
        )

        await stream.send(frame="f0")
        queue.put_nowait(
            ModelStreamUpdate(
                session_id=7,
                task="point",
                output={"points": [[0.25, 0.75]]},
            )
        )
        update = await stream.__anext__()
        result = await stream.close()

        assert sent == [(7, {"frame": "f0"})]
        assert closed == [7]
        assert update.output == {"points": [[0.25, 0.75]]}
        assert result.output == {"done": True}
        with pytest.raises(RuntimeError, match="closed"):
            await stream.send(frame="f1")

    asyncio.run(go())


def test_model_stream_surfaces_completion_errors() -> None:
    async def go() -> None:
        queue: _ModelStreamQueue = asyncio.Queue()
        future: asyncio.Future[EngineResult] = asyncio.get_running_loop().create_future()

        async def send_chunk(session_id: int, chunk: dict[str, Any]) -> None:
            raise AssertionError("not used")

        async def close_session(session_id: int) -> None:
            raise AssertionError("not used")

        stream = ModelStream(
            session_id=9,
            task="point",
            queue=queue,
            result_future=future,
            send_chunk=send_chunk,
            close_session=close_session,
        )
        queue.put_nowait(_ModelStreamCompletion(error=ValueError("bad frame")))

        with pytest.raises(ValueError, match="bad frame"):
            await stream.__anext__()

    asyncio.run(go())
