"""The owner thread services single-pass jobs alongside the AR loop.

A single-pass runtime fulfils a request with one ``run(task, inputs)``
call and no decode loop. These tests drive the real ``_scheduler_loop``
on CPU with a ``FakeRuntime`` as the autoregressive default plus a stub
single-pass runtime registered alongside it, and check that a job
submitted through ``_submit_single_pass`` round-trips through the loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from kestrel.engine import InferenceEngine
from kestrel.runtime import ExecutionShape

from tests.scheduler._fake_runtime import FakeRuntime


class _StubSinglePass:
    """Minimal single-pass runtime: echoes the task + inputs back."""

    def __init__(self, *, model_name: str = "stub-sp") -> None:
        self.model_name = model_name
        self.device = "cpu"
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self.calls: list[tuple[str, Any]] = []

    def run(self, task: str, inputs: Any) -> Any:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("task failed")
        return {"task": task, "inputs": inputs}


def _engine_with(default_ar: FakeRuntime, single_pass: _StubSinglePass) -> InferenceEngine:
    """Build an engine wired for the scheduler loop without ``create``.

    Bypasses Moondream construction / warmup / photon: registers the
    runtimes directly and lets the test start the loop thread itself.
    """
    engine = object.__new__(InferenceEngine)
    engine._default_model = default_ar.model_name
    engine._runtimes = {
        default_ar.model_name: default_ar,
        single_pass.model_name: single_pass,
    }
    # Loop plumbing normally set up in __init__ / _initialize.
    import queue as _queue

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
    engine._skills = None  # scheduler defaults to a QuerySkill registry
    return engine


async def _run_case(task: str, inputs: Any) -> Any:
    ar = FakeRuntime(model_name="ar-default", device="cpu")
    sp = _StubSinglePass()
    engine = _engine_with(ar, sp)
    engine._loop = asyncio.get_running_loop()
    # The loop is started by hand below, so skip _ensure_started's
    # Moondream build path.
    engine._initialized = True
    engine._init_task = None

    thread = threading.Thread(target=engine._scheduler_loop, name="t", daemon=True)
    thread.start()
    try:
        return await asyncio.wait_for(
            engine._submit_single_pass(sp.model_name, task, inputs), timeout=5.0
        )
    finally:
        # Signal shutdown so the loop's should_exit() trips and the
        # thread joins.
        engine._shutdown = True
        engine._scheduler_queue.put(None)
        engine._scheduler_event.set()
        thread.join(timeout=5.0)


def test_single_pass_job_round_trips_through_loop() -> None:
    result = asyncio.run(_run_case("segment", {"points": [[1, 2]]}))
    assert result == {"task": "segment", "inputs": {"points": [[1, 2]]}}


def test_single_pass_job_error_propagates() -> None:
    with pytest.raises(ValueError, match="task failed"):
        asyncio.run(_run_case("boom", {}))


def test_submit_rejects_unknown_model() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True  # skip _ensure_started's build path
        with pytest.raises(ValueError, match="Unknown model"):
            await engine._submit_single_pass("nope", "segment", {})

    asyncio.run(go())


def test_submit_rejects_autoregressive_model() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        with pytest.raises(ValueError, match="not a single-pass model"):
            await engine._submit_single_pass("ar-default", "segment", {})

    asyncio.run(go())
