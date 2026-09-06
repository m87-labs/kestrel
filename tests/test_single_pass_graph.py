from __future__ import annotations

import pytest
import torch

from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph


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
    assert len(calls) == 1
    session.shutdown()
    session.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        with session.launch(torch.tensor([2])):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_session_replays_stable_outputs_and_bounds_shapes() -> None:
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
        max_entries=1,
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

    with session.launch(torch.arange(5, device=device)) as (output,):
        stream.synchronize()
        torch.testing.assert_close(output, torch.arange(5, device=device).square() + 1)
    assert calls == 4

    with session.launch(first) as (output,):
        stream.synchronize()
        torch.testing.assert_close(output, first.square() + 1)
    assert calls == 6
    session.shutdown()
