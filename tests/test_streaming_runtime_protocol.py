"""Streaming runtime protocol shape."""

from __future__ import annotations

from typing import Any

from kestrel.runtime import ExecutionShape, StreamingRuntime


class _StreamingStub:
    model_name = "streaming-stub"
    device = None
    execution_shape = ExecutionShape.STREAMING
    primary_stream = object()

    def tasks(self) -> tuple[str, ...]:
        return ("point",)

    def start(self, task: str, inputs: Any) -> dict[str, Any]:
        return {"task": task, "inputs": inputs}

    def step(self, session: Any, inputs: Any) -> tuple[Any, Any]:
        return session, inputs

    def finish(self, session: Any) -> None:
        self.finished = session

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_streaming_execution_shape_is_public() -> None:
    assert ExecutionShape.STREAMING.value == "streaming"


def test_streaming_runtime_protocol_is_exported() -> None:
    runtime: StreamingRuntime = _StreamingStub()
    session = runtime.start("point", {"points": [[0.5, 0.5]]})
    output = runtime.step(session, {"frame": object()})
    runtime.finish(session)

    assert runtime.execution_shape is ExecutionShape.STREAMING
    assert runtime.tasks() == ("point",)
    assert output[0] == session
