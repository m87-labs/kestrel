from __future__ import annotations

import pytest
import torch

from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph


def test_graph_input_copies_group_dtypes_and_preserve_views(monkeypatch):
    from kestrel.runtime.single_pass_graph import _GraphInputCopies

    sources = (
        torch.arange(12.0).reshape(3, 4).t(),
        torch.arange(5),
        torch.arange(6.0).reshape(2, 3),
        torch.empty(0),
    )
    destinations = tuple(
        torch.empty_strided(value.shape, value.stride(), dtype=value.dtype)
        for value in sources
    )
    calls = []
    original = torch._foreach_copy_

    def copy(dst, src):
        calls.append(tuple(value.dtype for value in dst))
        return original(dst, src)

    monkeypatch.setattr(torch, "_foreach_copy_", copy)
    _GraphInputCopies(destinations).copy(sources)
    for destination, source in zip(destinations, sources, strict=True):
        torch.testing.assert_close(destination, source, atol=0, rtol=0)
    # Empty storage aliases conservatively retain sequential copying.
    assert not calls
    plan = _GraphInputCopies(destinations[:-1])
    plan.copy(sources[:-1])
    assert calls == [(torch.float32, torch.float32), (torch.int64,)]
    assert all(
        all(a is b for a, b in zip(dst, src, strict=True))
        for _, dst, src in plan.groups
    )


def test_graph_input_aliases_keep_sequential_copy_semantics(monkeypatch):
    from kestrel.runtime.single_pass_graph import _GraphInputCopies

    first, second = torch.tensor([1.0, 3.0]), torch.tensor([2.0])

    def forbidden(*args, **kwargs):
        raise AssertionError("aliased input copies must remain ordered")

    monkeypatch.setattr(torch, "_foreach_copy_", forbidden)
    _GraphInputCopies((first, second)).copy((second.expand_as(first), first[1:]))
    assert first.tolist() == [2.0, 2.0]
    assert second.item() == 2.0


def test_graph_input_copy_failure_releases_caller_sources(monkeypatch):
    import gc
    import weakref
    from kestrel.runtime.single_pass_graph import _GraphInputCopies

    plan = _GraphInputCopies((torch.empty(3), torch.empty(4)))
    sources = (torch.ones(3), torch.ones(4))
    references = tuple(weakref.ref(value) for value in sources)

    def fail(*args):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch, "_foreach_copy_", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        plan.copy(sources)
    assert all(
        all(a is b for a, b in zip(dst, src, strict=True))
        for _, dst, src in plan.groups
    )
    del sources
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mutating_forward_restores_capture_inputs():
    from kestrel_kernels.cubin_runtime import _GRAPH_EXECUTION

    phases = []

    def forward(value):
        phases.append(
            (_GRAPH_EXECUTION.get(), torch.cuda.is_current_stream_capturing())
        )
        value.add_(1)
        return (value,)

    session = FixedShapeSinglePassGraph(
        enabled=True, device=torch.device("cuda"), stream=None, run_forward=forward
    )
    try:
        for initial in (2, 7, -3):
            value = torch.full((16,), initial, device="cuda")
            with session.launch(value) as (output,):
                torch.testing.assert_close(output, torch.full_like(output, initial + 1))
                torch.testing.assert_close(value, torch.full_like(value, initial))
    finally:
        session.shutdown()
    assert phases == [(True, False), (True, True)]
    assert not _GRAPH_EXECUTION.get()


def test_disabled_session_runs_eager_and_refuses_after_shutdown() -> None:
    calls = []

    def forward(value: torch.Tensor) -> tuple[torch.Tensor]:
        calls.append(value)
        return (value + 1,)

    session = FixedShapeSinglePassGraph(
        enabled=False,
        device=torch.device("cpu"),
        stream=None,
        run_forward=forward,
        max_entries=1,
    )
    with session.launch(torch.tensor([2])) as (output,):
        assert output.tolist() == [3]
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with session.launch(torch.tensor([4])):
                pass
        with pytest.raises(RuntimeError, match="during an active lease"):
            session.shutdown()
    assert len(calls) == 1
    session.shutdown()
    session.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        with session.launch(torch.tensor([2])):
            pass


def test_shutdown_releases_state_when_stream_synchronize_fails() -> None:
    session = FixedShapeSinglePassGraph(
        enabled=False,
        device=torch.device("cpu"),
        stream=None,
        run_forward=lambda value: (value,),
    )

    class _FailingStream:
        def synchronize(self) -> None:
            raise RuntimeError("sync failed")

    session.enabled = True
    session._stream = _FailingStream()  # type: ignore[assignment]
    session._entries[()] = object()  # type: ignore[index,assignment]
    with pytest.raises(RuntimeError, match="sync failed"):
        session.shutdown()
    assert not session._entries
    assert session._stream is None
    assert session._run_forward(torch.tensor([1])) == ()
    session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_session_owns_nondefault_stream_and_waits_for_producer() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    session = FixedShapeSinglePassGraph(
        enabled=True,
        device=device,
        stream=None,
        run_forward=lambda value: (value.square() + 1,),
    )
    assert session._stream is not None
    assert session._stream != torch.cuda.default_stream(device)

    value = torch.zeros(4096, device=device)
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(5_000_000)
        value.fill_(3)
        with session.launch(value) as (output,):
            assert session._stream is not None
            session._stream.synchronize()
            torch.testing.assert_close(output, value.square() + 1)
    session.shutdown()

    with pytest.raises(ValueError, match="non-default stream"):
        FixedShapeSinglePassGraph(
            enabled=True,
            device=device,
            stream=torch.cuda.default_stream(device),
            run_forward=lambda item: (item,),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_disabled_session_keeps_eager_forward_and_consumers_ordered() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    target = torch.cuda.Stream(device=device)
    producer = torch.cuda.Stream(device=device)
    observed_streams = []

    def forward(value: torch.Tensor) -> tuple[torch.Tensor]:
        observed_streams.append(torch.cuda.current_stream(device))
        return (value.square() + 1,)

    session = FixedShapeSinglePassGraph(
        enabled=False,
        device=device,
        stream=target,
        run_forward=forward,
    )
    value = torch.zeros(4096, device=device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(5_000_000)
        value.fill_(3)
        with session.launch(value) as (output,):
            assert torch.cuda.current_stream(device) == target
            consumed = output + 2
            target.synchronize()
            torch.testing.assert_close(consumed, value.square() + 3)
    assert observed_streams == [target]
    session.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_session_retains_four_shapes_and_evicts_the_fifth() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    calls = 0

    def forward(value: torch.Tensor) -> tuple[torch.Tensor]:
        nonlocal calls
        calls += 1
        return (value.square() + 1,)

    session = FixedShapeSinglePassGraph(
        enabled=True,
        device=device,
        stream=stream,
        run_forward=forward,
        max_entries=4,
    )
    first = torch.arange(4, device=device, dtype=torch.float32)
    with session.launch(first) as (output,):
        stream.synchronize()
        pointer = output.data_ptr()
        torch.testing.assert_close(output, first.square() + 1)
    assert calls == 2

    with session.launch(first + 1) as (output,):
        stream.synchronize()
        assert output.data_ptr() == pointer
        torch.testing.assert_close(output, (first + 1).square() + 1)
    assert calls == 2

    pointers = {4: pointer}
    for size in (1, 2, 8):
        value = torch.arange(size, device=device, dtype=torch.float32)
        with session.launch(value) as (output,):
            stream.synchronize()
            pointers[size] = output.data_ptr()
            torch.testing.assert_close(output, value.square() + 1)
    assert calls == 8

    for size in (4, 1, 2, 8):
        value = torch.arange(size, device=device, dtype=torch.float32) + 2
        with session.launch(value) as (output,):
            stream.synchronize()
            assert output.data_ptr() == pointers[size]
            torch.testing.assert_close(output, value.square() + 1)
    assert calls == 8

    fifth = torch.arange(6, device=device, dtype=torch.float32)
    with session.launch(fifth) as (output,):
        stream.synchronize()
        torch.testing.assert_close(output, fifth.square() + 1)
    assert calls == 10

    # Shape four was the least recently used after the replay sequence above.
    with session.launch(first) as (output,):
        stream.synchronize()
        torch.testing.assert_close(output, first.square() + 1)
    assert calls == 12
    session.shutdown()
