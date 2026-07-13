from types import SimpleNamespace

import torch

from kestrel.models.moondream.runtime import MoondreamRuntime


def _field(size: int):
    return SimpleNamespace(
        cpu=torch.ones(size, dtype=torch.int64),
        gpu=torch.ones(size, dtype=torch.int64),
    )


def test_decode_padding_zeros_gpu_and_host_mirrors():
    """Eager megakernel composition checks consume host mirrors, so padding must zero both sides."""
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    slot = SimpleNamespace(
        decode_token_ids=torch.ones(8, dtype=torch.int64),
        decode_coord_values=torch.ones(8, 1),
        decode_size_values=torch.ones(8, 2),
        meta=SimpleNamespace(
            batch_idx=_field(8),
            input_pos=_field(8),
            lora_slot_ids=_field(8),
        ),
    )

    runtime._zero_decode_graph_padding(slot, batch_size=3, graph_batch_size=8)

    for tensor in (
        slot.meta.batch_idx.cpu,
        slot.meta.batch_idx.gpu,
        slot.meta.input_pos.cpu,
        slot.meta.input_pos.gpu,
        slot.meta.lora_slot_ids.cpu,
        slot.meta.lora_slot_ids.gpu,
    ):
        assert torch.equal(tensor[:3], torch.ones_like(tensor[:3]))
        assert torch.equal(tensor[3:], torch.zeros_like(tensor[3:]))
