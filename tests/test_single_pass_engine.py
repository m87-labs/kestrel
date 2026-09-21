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
import queue as _queue
import threading
from typing import Any

import pytest
import torch

from kestrel.engine import InferenceEngine
from kestrel.engine._types import _AutoregressiveRequest
from kestrel.engine.single_pass import _SinglePassRequest
from kestrel.runtime import ExecutionShape

from tests.scheduler._fake_runtime import FakeRuntime


class _StubSinglePass:
    """Single-pass runtime: forward() echoes the task + inputs."""

    def __init__(self, model_name: str = "stub-sp") -> None:
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self.batch_capacity = 2
        self.calls: list[tuple[str, Any]] = []

    def forward(self, task: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        self.calls.append((task, inputs))
        if task == "boom":
            raise ValueError("forward failed")
        return tuple({"task": task, "inputs": value} for value in inputs)

    def tasks(self) -> tuple[str, ...]:
        return ("segment", "boom")

    def shutdown(self) -> None:
        pass


def _engine_with(ar: FakeRuntime, sp: _StubSinglePass) -> InferenceEngine:
    """Wire an engine for the loop without create() (no Moondream/warmup)."""
    import queue as _queue

    engine = object.__new__(InferenceEngine)
    engine._default_model = ar.model_name
    engine._runtimes = {ar.model_name: ar, sp.model_name: sp}
    engine._scheduler_queue = _queue.Queue()
    engine._single_pass_queue = _queue.Queue()
    engine._streaming_start_queue = _queue.Queue()
    engine._streaming_chunk_queue = _queue.Queue()
    engine._model_stream_models = {}
    engine._model_stream_queues = {}
    engine._scheduler_event = threading.Event()
    engine._run_gate = threading.Event()
    engine._run_gate.set()
    engine._paused_flag = threading.Event()
    engine._paused_event = threading.Event()
    engine._scheduler_error = None
    engine._shutdown = False
    engine._photon_reporter = None
    engine._adapter_provider = None
    engine._default_temperature = 0.2
    engine._default_top_p = 0.9
    engine._compute_stream = None
    engine._skills_override = None
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
        return await asyncio.wait_for(
            engine.run(sp.model_name, task, inputs), timeout=5.0
        )
    finally:
        engine._shutdown = True
        engine._scheduler_queue.put(None)
        engine._scheduler_event.set()
        thread.join(timeout=5.0)


def test_run_routes_single_pass_through_kernel_loop() -> None:
    result = asyncio.run(_run("segment", {"points": [[1, 2]]}))
    assert result.output == {"task": "segment", "inputs": {"points": [[1, 2]]}}


def test_paused_loop_settles_cancelled_ingress_and_shutdown() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        single_pass = _StubSinglePass()
        engine = _engine_with(ar, single_pass)
        engine._loop = asyncio.get_running_loop()
        engine._paused_flag.set()
        engine._run_gate.clear()

        future = engine._loop.create_future()
        single_pass_future = engine._loop.create_future()
        cancelled = threading.Event()
        cancelled.set()
        engine._scheduler_queue.put(
            _AutoregressiveRequest(
                request_id=1,
                prompt="",
                prompt_tokens=(),
                image=None,
                image_hash=None,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                submitted_at=0.0,
                future=future,
                stream_queue=None,
                skill=ar.skills().resolve("query"),
                request_context=object(),
                cancel_event=cancelled,
            )
        )
        engine._single_pass_queue.put(
            (
                single_pass.model_name,
                _SinglePassRequest(
                    request_id=2,
                    future=single_pass_future,
                    task="segment",
                    inputs={},
                    submitted_at=0.0,
                    cancel_event=cancelled,
                ),
            )
        )

        thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
        thread.start()
        assert (await asyncio.wait_for(future, timeout=5.0)).finish_reason == "cancelled"
        assert (
            await asyncio.wait_for(single_pass_future, timeout=5.0)
        ).finish_reason == "cancelled"

        engine._scheduler_queue.put(None)
        engine._scheduler_event.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    asyncio.run(go())


def test_paused_shutdown_does_not_wait_for_executor_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BusyExecutor:
        has_work = True

        def __init__(self) -> None:
            self.drains = 0

        def drain(self):  # type: ignore[no-untyped-def]
            self.drains += 1
            return ()

        def shutdown(self):  # type: ignore[no-untyped-def]
            return ()

    ar = FakeRuntime(model_name="ar-default", device="cpu")
    engine = _engine_with(ar, _StubSinglePass())
    engine._loop = asyncio.new_event_loop()
    engine._paused_flag.set()
    engine._run_gate.clear()
    executor = BusyExecutor()
    monkeypatch.setattr(
        "kestrel.engine.core.AutoregressiveExecutor",
        lambda *args, **kwargs: executor,
    )

    class ShutdownAfterIdleWait(threading.Event):
        waits = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.waits += 1
            if self.waits == 2:
                engine._scheduler_queue.put(None)
                engine._scheduler_event.set()
            return False

    engine._run_gate = ShutdownAfterIdleWait()
    thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    stopped = not thread.is_alive()
    if not stopped:
        engine._paused_flag.clear()
        engine._run_gate.set()
        engine._scheduler_event.set()
        thread.join(timeout=5.0)

    engine._loop.close()
    assert stopped
    assert executor.drains == 2  # initial pause and shutdown, not the idle poll


def test_default_model_can_be_single_pass() -> None:
    async def go() -> None:
        ar = FakeRuntime(model_name="unused-ar", device="cpu")
        sp = _StubSinglePass("sp-default")
        engine = _engine_with(ar, sp)
        engine._default_model = sp.model_name
        engine._model_ids = [sp.model_name]
        engine._runtimes = {sp.model_name: sp}
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None

        thread = threading.Thread(
            target=engine._scheduler_loop, name="kernel", daemon=True
        )
        thread.start()
        try:
            result = await asyncio.wait_for(
                engine.model().segment(points=[[1, 2]]), timeout=5.0
            )
            assert result.output == {
                "task": "segment",
                "inputs": {"points": [[1, 2]]},
            }
        finally:
            engine._shutdown = True
            engine._scheduler_queue.put(None)
            engine._scheduler_event.set()
            thread.join(timeout=5.0)

    asyncio.run(go())


def test_run_propagates_forward_error() -> None:
    with pytest.raises(ValueError, match="forward failed"):
        asyncio.run(_run("boom", {}))


def test_run_fails_if_scheduler_dies_after_single_pass_enqueue() -> None:
    """Regression: run() must not hang if scheduler failure drains before put().

    The scheduler thread sets ``_scheduler_error`` and drains ingress once on
    fatal failure. If a single-pass caller enqueues immediately after that
    drain, there is no kernel left to consume the request. run() must detect
    the latched failure after enqueue and fail the request itself.
    """

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None

        class _QueueThatFailsAfterPut(_queue.Queue):
            def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
                super().put(item, *args, **kwargs)
                engine._scheduler_error = RuntimeError("scheduler crashed")

        engine._single_pass_queue = _QueueThatFailsAfterPut()

        with pytest.raises(RuntimeError, match="scheduler is not running") as exc:
            await asyncio.wait_for(
                engine.run(sp.model_name, "segment", {"points": []}), timeout=1.0
            )
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "scheduler crashed" in str(exc.value.__cause__)
        assert engine._single_pass_queue.empty()

    asyncio.run(go())


def test_worker_fails_ar_if_scheduler_dies_after_forwarding() -> None:
    """Regression: AR submit must not hang if scheduler drains before put().

    The worker checks the latched scheduler error before forwarding requests,
    but the scheduler can fail between that check and ``_scheduler_queue.put``.
    In that race, the request lands in a queue that no scheduler will consume.
    """

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._queue = asyncio.Queue()

        class _QueueThatFailsAfterPut(_queue.Queue):
            def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
                super().put(item, *args, **kwargs)
                if item is not None:
                    engine._scheduler_error = RuntimeError("scheduler crashed")

        engine._scheduler_queue = _QueueThatFailsAfterPut()
        future = asyncio.get_running_loop().create_future()
        request = _AutoregressiveRequest(
            request_id=1,
            prompt="",
            prompt_tokens=[],
            image=None,
            image_hash=None,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            submitted_at=0.0,
            future=future,
            stream_queue=None,
            skill=object(),  # type: ignore[arg-type]
            request_context=None,
        )

        await engine._queue.put(request)
        await engine._queue.put(None)
        await asyncio.wait_for(engine._worker_loop(), timeout=1.0)

        with pytest.raises(RuntimeError, match="scheduler is not running") as exc:
            await asyncio.wait_for(future, timeout=1.0)
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "scheduler crashed" in str(exc.value.__cause__)
        assert engine._scheduler_queue.get_nowait() is None
        assert engine._scheduler_queue.empty()

    asyncio.run(go())


def test_worker_forwards_one_queued_ar_batch_before_waking_scheduler() -> None:
    async def go() -> None:
        engine = _engine_with(
            FakeRuntime(model_name="ar-default", device="cpu", max_batch_size=2),
            _StubSinglePass(),
        )
        engine._queue = asyncio.Queue()
        requests = [object(), object(), object()]
        for request in requests:
            engine._queue.put_nowait(request)
        engine._queue.put_nowait(None)

        class _WakeEvent:
            queue_sizes = []

            def set(self) -> None:
                self.queue_sizes.append(engine._scheduler_queue.qsize())

        wake_event = _WakeEvent()
        engine._scheduler_event = wake_event

        await asyncio.wait_for(engine._worker_loop(), timeout=1.0)

        assert wake_event.queue_sizes == [2, 3, 4]
        assert [engine._scheduler_queue.get_nowait() for _ in range(4)] == [
            *requests,
            None,
        ]
        assert engine._scheduler_queue.empty()

    asyncio.run(go())


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


def test_run_resolves_when_event_pending_with_no_other_traffic() -> None:
    """Regression (P1): a pending GPU event must not hang the kernel.

    A launched single-pass forward signals completion via a CUDA event,
    which sets no host event. If the kernel blocks on wake_event.wait()
    while the only in-flight work is that pending event, it sleeps past
    GPU completion and engine.run() never resolves. With nothing else in
    flight to wake the loop, run() must still complete on its own.
    """
    import kestrel.engine.single_pass as sp_mod

    class _PendingEvent:
        def __init__(self) -> None:
            self._polls = 0

        def record(self, *a: Any, **k: Any) -> None:
            pass

        def query(self) -> bool:
            # Not done for the first few polls — forces the kernel to park
            # with a pending event and nothing else to wake it.
            self._polls += 1
            return self._polls > 3

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None

        orig_make_event = sp_mod.make_event
        sp_mod.make_event = lambda device: _PendingEvent()
        try:
            thread = threading.Thread(
                target=engine._scheduler_loop, name="kernel", daemon=True
            )
            thread.start()
            try:
                # No other traffic: only this single-pass request is in
                # flight. It must resolve via the kernel's poll, not hang.
                result = await asyncio.wait_for(
                    engine.run(sp.model_name, "segment", {"k": 1}), timeout=5.0
                )
                assert result.output == {"task": "segment", "inputs": {"k": 1}}
            finally:
                engine._shutdown = True
                engine._scheduler_queue.put(None)
                engine._scheduler_event.set()
                thread.join(timeout=5.0)
        finally:
            sp_mod.make_event = orig_make_event

    asyncio.run(go())


def test_run_cancellation_skips_queued_work_and_drains_launched_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.engine.single_pass as sp_mod

    recorded = threading.Event()
    ready = threading.Event()

    class _ControlledEvent:
        def record(self, *a: Any, **k: Any) -> None:
            recorded.set()

        def query(self) -> bool:
            return ready.is_set()

    monkeypatch.setattr(sp_mod, "make_event", lambda device: _ControlledEvent())

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _StubSinglePass()
        engine = _engine_with(ar, sp)
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None
        thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
        thread.start()
        requests: list[asyncio.Task[Any]] = []
        try:
            launched = asyncio.create_task(
                engine.run(sp.model_name, "segment", {"request": "launched"})
            )
            requests.append(launched)
            assert await asyncio.to_thread(recorded.wait, 1.0)

            queued = asyncio.create_task(
                engine.run(sp.model_name, "segment", {"request": "queued"})
            )
            requests.append(queued)
            await asyncio.sleep(0)
            launched.cancel()
            queued.cancel()
            await asyncio.sleep(0)

            assert not launched.done()
            assert not queued.done()
            ready.set()
            outcomes = await asyncio.wait_for(
                asyncio.gather(*requests, return_exceptions=True), timeout=5.0
            )
            assert all(
                isinstance(outcome, asyncio.CancelledError) for outcome in outcomes
            )
            assert sp.calls == [
                ("segment", ({"request": "launched"},)),
            ]
        finally:
            ready.set()
            await asyncio.gather(*requests, return_exceptions=True)
            engine._shutdown = True
            engine._scheduler_queue.put(None)
            engine._scheduler_event.set()
            thread.join(timeout=5.0)

    asyncio.run(go())


def test_engine_shutdown_tears_down_single_pass_runtime() -> None:
    """Regression (P2): engine.shutdown() must not AttributeError on a
    single-pass runtime.

    shutdown() iterates every registered runtime and calls runtime.shutdown();
    a conformant single-pass runtime implements that, so a co-hosted
    Moondream + single-pass engine tears down cleanly.
    """

    class _RecordingSinglePass(_StubSinglePass):
        def __init__(self) -> None:
            super().__init__("recording-sp")
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        sp = _RecordingSinglePass()
        engine = _engine_with(ar, sp)
        # Minimal lifecycle state shutdown() touches; no loop/worker started.
        engine._queue = asyncio.Queue()
        engine._worker_task = None
        engine._scheduler_thread = None

        await engine.shutdown()

        assert engine._shutdown is True
        assert sp.shutdown_called is True

    asyncio.run(go())


def test_single_pass_lane_crash_does_not_kill_the_kernel() -> None:
    """A single-pass lane whose advance() raises is isolated.

    The broken lane's in-flight request is failed, but the kernel keeps
    running and other lanes still serve — one bad single-pass request
    must not take down autoregressive decode or sibling lanes.
    """

    async def go() -> None:
        ar = FakeRuntime(model_name="ar-default", device="cpu")
        good = _StubSinglePass("good-sp")
        bad = _StubSinglePass("bad-sp")
        engine = _engine_with(ar, good)
        engine._runtimes[bad.model_name] = bad
        engine._loop = asyncio.get_running_loop()
        engine._initialized = True
        engine._init_task = None

        # Make the bad lane's executor.advance() raise (a bug escaping the
        # executor's own forward() try/except). Patch after the loop builds
        # its executors, so wrap _scheduler_loop's lane construction by
        # poisoning the runtime's forward to raise a BaseException-free
        # error path: simplest is to monkeypatch SinglePassExecutor.advance
        # for the bad runtime via a sentinel task.
        import kestrel.engine.single_pass as sp_mod

        orig_advance = sp_mod.SinglePassExecutor.advance

        def advance(self):  # type: ignore[no-untyped-def]
            # The bad lane explodes whenever it has a request to service,
            # before it can produce a completion — simulating a bug that
            # escapes the executor's own forward() try/except.
            if self._runtime.model_name == "bad-sp" and (
                self._in_flight or not self._queue.empty()
            ):
                raise RuntimeError("lane exploded")
            return orig_advance(self)

        sp_mod.SinglePassExecutor.advance = advance
        try:
            thread = threading.Thread(
                target=engine._scheduler_loop, name="kernel", daemon=True
            )
            thread.start()
            try:
                # The bad lane crashes on its request; the good lane still works.
                with pytest.raises(
                    RuntimeError, match="Engine shut down|lane exploded"
                ):
                    await asyncio.wait_for(
                        engine.run("bad-sp", "segment", {}), timeout=5.0
                    )
                good_result = await asyncio.wait_for(
                    engine.run("good-sp", "segment", {"ok": 1}), timeout=5.0
                )
                assert good_result.output == {"task": "segment", "inputs": {"ok": 1}}
            finally:
                engine._shutdown = True
                engine._scheduler_queue.put(None)
                engine._scheduler_event.set()
                thread.join(timeout=5.0)
        finally:
            sp_mod.SinglePassExecutor.advance = orig_advance

    asyncio.run(go())
