"""Tests for ``kestrel.utils.PackedBuffer``."""

import numpy as np
import pytest
import torch

from kestrel.utils import PackedBuffer

CPU = torch.device("cpu")


def _build(device: torch.device) -> PackedBuffer:
    return PackedBuffer(
        [
            ("input_ids", (1, 5), torch.long),
            ("seq_idx", (1, 5), torch.int32),
            ("position_ids", (3, 1, 5), torch.long),
            ("batch_indices", (4,), torch.int64),
            ("last_positions", (4,), torch.int32),
            ("cu_seq_lens_q", (5,), torch.int32),
            ("scales", (2,), torch.float32),
        ],
        device=device,
        pin_memory=False,
    )


def test_fields_expose_requested_shape_and_dtype():
    pb = _build(CPU)
    assert pb.input_ids.cpu.shape == (1, 5)
    assert pb.input_ids.cpu.dtype == torch.long
    assert pb.seq_idx.cpu.dtype == torch.int32
    assert pb.position_ids.cpu.shape == (3, 1, 5)
    assert pb.scales.cpu.dtype == torch.float32
    assert pb.cu_seq_lens_q.cpu.shape == (5,)


def test_numpy_view_shares_storage_with_cpu_tensor():
    pb = _build(CPU)
    pb.input_ids.np[0, :] = np.arange(5, dtype=np.int64)
    # Writing through the numpy view must be visible on the torch cpu view.
    assert torch.equal(pb.input_ids.cpu[0], torch.arange(5, dtype=torch.long))


def test_fields_are_disjoint_across_dtypes():
    pb = _build(CPU)
    pb.input_ids.cpu.fill_(7)
    pb.seq_idx.cpu.fill_(3)
    pb.scales.cpu.fill_(1.5)
    pb.batch_indices.cpu.copy_(torch.tensor([10, 11, 12, 13]))
    # Each field keeps its own values — no aliasing between packed regions.
    assert torch.equal(pb.input_ids.cpu, torch.full((1, 5), 7, dtype=torch.long))
    assert torch.equal(pb.seq_idx.cpu, torch.full((1, 5), 3, dtype=torch.int32))
    assert torch.allclose(pb.scales.cpu, torch.tensor([1.5, 1.5]))
    assert torch.equal(pb.batch_indices.cpu, torch.tensor([10, 11, 12, 13]))


def test_single_copy_round_trips_every_field():
    # On CPU the "gpu" tensor is just a second host tensor, so copy_to_gpu /
    # copy_to_cpu still exercise the one-shot whole-buffer transfer.
    pb = _build(CPU)
    pb.input_ids.cpu[0] = torch.arange(5, dtype=torch.long)
    pb.seq_idx.cpu[0] = torch.arange(5, dtype=torch.int32)
    pb.scales.cpu.copy_(torch.tensor([0.25, 0.75]))
    pb.copy_to_gpu()
    assert torch.equal(pb.input_ids.gpu[0], torch.arange(5, dtype=torch.long))
    assert torch.equal(pb.seq_idx.gpu[0], torch.arange(5, dtype=torch.int32))
    assert torch.allclose(pb.scales.gpu, torch.tensor([0.25, 0.75]))


def test_unknown_field_raises_attribute_error():
    pb = _build(CPU)
    with pytest.raises(AttributeError):
        _ = pb.does_not_exist


def test_bfloat16_field_has_no_numpy_view():
    pb = PackedBuffer(
        [("weights", (4,), torch.bfloat16)], device=CPU, pin_memory=False
    )
    assert pb.weights.np is None
    assert pb.weights.cpu.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_round_trip_matches_per_field_copies():
    dev = torch.device("cuda")
    pb = _build(dev)
    pb.input_ids.cpu[0] = torch.arange(5, dtype=torch.long)
    pb.batch_indices.cpu.copy_(torch.tensor([1, 2, 3, 4]))
    pb.copy_to_gpu()
    torch.cuda.synchronize()
    assert torch.equal(pb.input_ids.gpu[0].cpu(), torch.arange(5, dtype=torch.long))
    assert torch.equal(pb.batch_indices.gpu.cpu(), torch.tensor([1, 2, 3, 4]))
