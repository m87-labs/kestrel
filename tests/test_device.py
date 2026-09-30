"""Tests for the device-agnostic primitive wrappers in ``kestrel.device``."""

from contextlib import contextmanager
import threading
import weakref

import pytest
import torch

from kestrel.device import (
    NoopEvent,
    empty_cache,
    get_device_capability,
    get_device_sm_count,
    make_event,
    make_stream,
    materialize_blas_runtime,
    set_device,
    stream_context,
    synchronize,
)


CPU = torch.device("cpu")
CUDA = torch.device("cuda")
MPS = torch.device("mps")


# --- CPU path: every primitive is a safe no-op or returns a sentinel -------


def test_set_device_cpu_is_noop() -> None:
    set_device(CPU)


def test_synchronize_cpu_is_noop() -> None:
    synchronize(CPU)


def test_empty_cache_cpu_is_noop() -> None:
    empty_cache(CPU)


def test_empty_cache_cuda_targets_supplied_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.device as device_module

    target = torch.device("cuda:3")
    events = []
    active_devices = []

    @contextmanager
    def cuda_device(device):
        events.append(("enter", device))
        active_devices.append(device)
        try:
            yield
        finally:
            assert active_devices.pop() == device
            events.append(("exit", device))

    def cuda_empty_cache() -> None:
        assert active_devices == [target]
        events.append(("empty_cache", target))

    monkeypatch.setattr(device_module.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(device_module.torch.cuda, "empty_cache", cuda_empty_cache)

    empty_cache(target)

    assert events == [
        ("enter", target),
        ("empty_cache", target),
        ("exit", target),
    ]


def test_get_device_capability_cpu_returns_zero_tuple() -> None:
    assert get_device_capability(CPU) == (0, 0)


def test_get_device_sm_count_cpu_returns_zero() -> None:
    assert get_device_sm_count(CPU) == 0


def test_make_stream_cpu_returns_none() -> None:
    assert make_stream(CPU) is None


def test_make_event_cpu_returns_noop() -> None:
    e = make_event(CPU)
    assert isinstance(e, NoopEvent)
    # All NoopEvent operations succeed silently.
    e.record()
    e.record(stream=None)
    e.wait()
    e.synchronize()
    assert e.query() is True
    assert e.elapsed_time(NoopEvent()) == 0.0


def test_stream_context_with_none_yields_inline() -> None:
    entered = False
    with stream_context(None):
        entered = True
    assert entered


def test_materialize_blas_runtime_runs_non_cuda_operation_inline() -> None:
    thread = threading.get_ident()
    observed = []

    materialize_blas_runtime(
        CPU,
        None,
        lambda: observed.append(
            (threading.get_ident(), torch.is_inference_mode_enabled())
        ),
    )

    assert observed == [(thread, True)]


def test_materialize_blas_runtime_returns_cuda_handle_from_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kestrel.device as device_module

    caller_thread = threading.get_ident()
    events = []
    result_ref = None

    class Result:
        pass

    class Stream:
        def synchronize(self):
            assert result_ref is not None and result_ref() is not None
            events.append(("synchronize", threading.get_ident(), self))

    compute_stream = Stream()

    @contextmanager
    def cuda_device(device):
        events.append(("device-enter", threading.get_ident(), device))
        try:
            yield
        finally:
            events.append(("device-exit", threading.get_ident(), device))

    @contextmanager
    def use_stream(stream):
        events.append(("stream-enter", threading.get_ident(), stream))
        try:
            yield
        finally:
            events.append(("stream-exit", threading.get_ident(), stream))

    monkeypatch.setattr(device_module.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(device_module, "stream_context", use_stream)

    def operation():
        nonlocal result_ref
        result = Result()
        result_ref = weakref.ref(result)
        events.append(
            (
                "operation",
                threading.get_ident(),
                torch.is_inference_mode_enabled(),
            )
        )
        return result

    materialize_blas_runtime(
        torch.device("cuda:0"),
        compute_stream,
        operation,
    )

    worker_threads = {event[1] for event in events}
    assert caller_thread not in worker_threads
    assert [event[0] for event in events] == [
        "device-enter",
        "stream-enter",
        "operation",
        "synchronize",
        "stream-exit",
        "device-exit",
    ]
    assert events[2][2] is True
    assert result_ref is not None and result_ref() is None


# --- CUDA path: thin wrappers, only run when present ------------------------


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_make_stream_cuda_returns_real_stream() -> None:
    stream = make_stream(CUDA)
    assert isinstance(stream, torch.cuda.Stream)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_make_event_cuda_returns_real_event() -> None:
    event = make_event(CUDA, enable_timing=False, blocking=False)
    assert isinstance(event, torch.cuda.Event)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_get_device_capability_cuda_matches_torch() -> None:
    assert get_device_capability(CUDA) == torch.cuda.get_device_capability()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_get_device_sm_count_cuda_matches_torch() -> None:
    assert (
        get_device_sm_count(CUDA)
        == torch.cuda.get_device_properties(CUDA).multi_processor_count
    )


# --- MPS path: stream is None, event is NoopEvent, sync uses torch.mps.* ---


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_make_stream_mps_returns_none() -> None:
    assert make_stream(MPS) is None


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_make_event_mps_returns_noop() -> None:
    assert isinstance(make_event(MPS), NoopEvent)


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_synchronize_mps_uses_torch_mps() -> None:
    # No exception → it dispatched. Hardware is always available so the
    # call is essentially free; we're testing the dispatch table, not perf.
    synchronize(MPS)


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_set_device_mps_is_noop() -> None:
    # MPS doesn't have a per-process device-set concept; we just ensure
    # the call doesn't raise.
    set_device(MPS)


def test_tensor_handoff_tracks_each_device_and_handles_nested_aliases(monkeypatch):
    from kestrel.device import InputStreamHandoff, record_tensor_streams

    events = []
    phase = ["caller"]

    class Tensor(torch.Tensor):
        @staticmethod
        def __new__(cls, index):
            value = torch.Tensor._make_subclass(cls, torch.empty(0))
            value.index = index
            return value

        @property
        def device(self):
            return torch.device("cuda", self.index)

        def record_stream(self, stream):
            events.append(("lifetime", stream))

    class Stream:
        def __init__(self, device):
            self.identity = (phase[0], device.index)

        def wait_event(self, event):
            events.append(("wait", self.identity, event.producer))

    class Event:
        def record(self, stream):
            self.producer = stream.identity
            events.append(("record", self.producer))

    monkeypatch.setattr(torch.cuda, "current_stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    a, b = Tensor(0), Tensor(1)
    values = {"nested": [a, b], "alias": a, "cpu": torch.empty(0)}
    values["cycle"] = values
    handoff = InputStreamHandoff(values)
    assert sorted(events) == [("record", ("caller", 0)), ("record", ("caller", 1))]

    phase[0] = "scheduler"
    events.clear()
    handoff.wait()
    assert sorted(events[:2]) == [
        ("wait", ("scheduler", 0), ("caller", 0)),
        ("wait", ("scheduler", 1), ("caller", 1)),
    ]
    assert sorted(stream.identity for kind, stream in events[2:]) == [
        ("scheduler", 0), ("scheduler", 1)
    ]

    phase[0] = "caller"
    events.clear()
    record_tensor_streams({"result": (a, b, a)})
    assert sorted(stream.identity for kind, stream in events) == [
        ("caller", 0), ("caller", 1)
    ]


def test_cpu_tensor_handoff_does_not_create_cuda_events(monkeypatch):
    from kestrel.device import InputStreamHandoff, record_tensor_streams

    def unexpected(*args, **kwargs):
        raise AssertionError("CPU request accessed CUDA")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    monkeypatch.setattr(torch.cuda, "Event", unexpected)
    inputs = {"pixels": torch.zeros(2), "metadata": [None, 1, "image"]}
    InputStreamHandoff(inputs).wait()
    record_tensor_streams(inputs)
