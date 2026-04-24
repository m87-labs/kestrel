"""Tests for the device-agnostic primitive wrappers in ``kestrel.device``."""

import pytest
import torch

from kestrel.device import (
    NoopEvent,
    empty_cache,
    get_device_capability,
    make_event,
    make_stream,
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


def test_get_device_capability_cpu_returns_zero_tuple() -> None:
    assert get_device_capability(CPU) == (0, 0)


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
