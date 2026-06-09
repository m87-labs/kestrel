"""engine.stream() routes stateful streaming work through the kernel loop."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

import torch

from kestrel.engine import InferenceEngine
from kestrel.runtime import ExecutionShape

from tests.scheduler._fake_runtime import FakeRuntime


class _StubStreamingRuntime:
    def __init__(self, model_name: str = "tracker") -> None:
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.STREAMING
        self.calls: list[tuple[str, Any]] = []
        self.finished: list[Any] = []

    def tasks(self) -> tuple[str, ...]:
        return ("point",)

    def start(self, task: str, inputs: Any) -> dict[str, Any]:
        self.calls.append(("start", task, inputs))
        return {"step": 0, "seed": inputs}

    def step(self, session: Any, inputs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append(("step", session, inputs))
        next_session = dict(session)
        next_session["step"] += 1
        return {"points": inputs["frame"], "step": next_session["step"]}, next_session

    def finish(self, session: Any) -> None:
        self.finished.append(session)

    def shutdown(self) -> None:
        pass


def _engine_with(ar: FakeRuntime, streaming: _StubStreamingRuntime) -> InferenceEngine:
    engine = object.__new__(InferenceEngine)
    engine._default_model = ar.model_name
    engine._runtimes = {ar.model_name: ar, streaming.model_name: streaming}
    engine._scheduler_queue = queue.Queue()
    engine._single_pass_queue = queue.Queue()
    engine._streaming_start_queue = queue.Queue()
    engine._streaming_chunk_queue = queue.Queue()
    engine._model_stream_models = {}
    engine._model_stream_queues = {}
    engine._scheduler_event = threading.Event()
    engine._run_gate = threading.Event()
    engine._run_gate.set()
    engine._paused_flag = threading.Event()
    engine._paused_event = threading.Event()
    engine._shutdown = False
    engine._scheduler_error = None
    engine._photon_reporter = None
    engine._adapter_provider = None
    engine._default_temperature = 0.2
    engine._default_top_p = 0.9
    engine._compute_stream = None
    engine._skills_override = None
    engine._request_ids = iter(range(1, 1_000_000))
    return engine


def test_stream_routes_updates_and_close_through_kernel_loop() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        tracker = _StubStreamingRuntime()
        engine = _engine_with(ar, tracker)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None

        thread = threading.Thread(
            target=engine._scheduler_loop,
            name="kernel",
            daemon=True,
        )
        thread.start()
        try:
            stream = await engine.stream(
                tracker.model_name,
                "point",
                {"points": [[0.5, 0.5]]},
            )
            await stream.send(frame=[[0.25, 0.75]])
            update = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
            result = await asyncio.wait_for(stream.close(), timeout=5.0)

            assert update.output == {"points": [[0.25, 0.75]], "step": 1}
            assert result.output == {"closed": True}
            assert tracker.finished == [
                {"step": 1, "seed": {"points": [[0.5, 0.5]]}}
            ]
        finally:
            engine._shutdown = True
            engine._scheduler_queue.put(None)
            engine._scheduler_event.set()
            thread.join(timeout=5.0)

    asyncio.run(go())
