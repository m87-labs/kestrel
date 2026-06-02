"""engine.run() routes single-pass work through the real kernel loop.

Drives the actual _scheduler_loop on CPU with an autoregressive
FakeRuntime as the default lane plus a stub single-pass runtime
registered alongside it, and checks that engine.run(model, task, inputs)
round-trips through the single-pass lane. Complements
test_single_pass_executor.py (which unit-tests the lane in isolation) by
exercising the kernel's multi-lane fold + ingress routing.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
import torch

from kestrel.engine import InferenceEngine
from kestrel.runtime import ExecutionShape

from tests.scheduler._fake_runtime import FakeRuntime


class _StubSinglePass:
    """Single-pass runtime: forward() echoes the task + inputs."""

    def __init__(self, model_name: str = "stub-sp") -> None:
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self.primary_stream = None
        self.calls: list[tuple[str, Any]] = []

    def forward(self, task: str, inputs: Any) -> Any:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("forward failed")
        return {"task": task, "inputs": inputs}


def _engine_with(ar: FakeRuntime, sp: _StubSinglePass) -> InferenceEngine:
    """Wire an engine for the loop without create() (no Moondream/warmup)."""
    import queue as _queue

    engine = object.__new__(InferenceEngine)
    engine._default_model = ar.model_name
    engine._runtimes = {ar.model_name: ar, sp.model_name: sp}
    engine._scheduler_queue = _queue.Queue()
    engine._single_pass_queue = _queue.Queue()
    engine._scheduler_event = threading.Event()
    engine._run_gate = threading.Event()
    engine._run_gate.set()
    engine._paused_flag = threading.Event()
    engine._paused_event = threading.Event()
    engine._shutdown = False
    engine._photon_reporter = None
    engine._adapter_provider = None
    engine._default_temperature = 0.2
    engine._default_top_p = 0.9
    engine._skills = None
    engine._request_ids = iter(range(1, 1_000_000))
    return engine


async def _run(task: str, inputs: Any) -> Any:
    ar = FakeRuntime(model_name="ar-default", device="cpu")
    sp = _StubSinglePass()
    engine = _engine_with(ar, sp)
    engine._loop = asyncio.get_running_loop()
    engine._initialized = True
    engine._init_task = None

    thread = threading.Thread(target=engine._scheduler_loop, name="kernel", daemon=True)
    thread.start()
    try:
        return await asyncio.wait_for(engine.run(sp.model_name, task, inputs), timeout=5.0)
    finally:
        engine._shutdown = True
        engine._scheduler_queue.put(None)
        engine._scheduler_event.set()
        thread.join(timeout=5.0)


def test_run_routes_single_pass_through_kernel_loop() -> None:
    result = asyncio.run(_run("segment", {"points": [[1, 2]]}))
    assert result.output == {"task": "segment", "inputs": {"points": [[1, 2]]}}


def test_run_propagates_forward_error() -> None:
    with pytest.raises(ValueError, match="forward failed"):
        asyncio.run(_run("boom", {}))


def test_run_rejects_unknown_model() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        with pytest.raises(ValueError, match="Unknown model"):
            await engine.run("nope", "segment", {})

    asyncio.run(go())


def test_run_rejects_autoregressive_model() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        with pytest.raises(ValueError, match="not a single-pass model"):
            await engine.run("ar-default", "segment", {})

    asyncio.run(go())
