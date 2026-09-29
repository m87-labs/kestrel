"""Owned capture streams must never alias the framework's pooled streams."""

import pytest
import torch

from kestrel.runtime.cuda_stream import OwnedCudaStream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_owned_streams_are_unique_and_close_is_idempotent():
    device = torch.device("cuda", torch.cuda.current_device())
    streams = []
    try:
        streams.extend(OwnedCudaStream(device) for _ in range(40))
        handles = {owner.stream.cuda_stream for owner in streams}
        pooled = {torch.cuda.Stream(device=device).cuda_stream for _ in range(40)}
        assert len(handles) == 40
        assert not handles.intersection(pooled)
        for owner in streams:
            assert owner.stream.device == device
            with torch.cuda.stream(owner.stream):
                value = torch.ones(32, device=device)
                result = value.square() + 1
            owner.close()
            torch.testing.assert_close(result, torch.full_like(result, 2))
            assert owner._handle.value is None
            owner.close()
    finally:
        for owner in streams:
            owner.close()
