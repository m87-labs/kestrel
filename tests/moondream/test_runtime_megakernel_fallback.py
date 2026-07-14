from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import runtime as runtime_mod
from kestrel.models.moondream.runtime import MoondreamRuntime


def _forward_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    runtime.model = SimpleNamespace(text=object())
    runtime._embed_packed_token_batch = lambda *args: torch.ones(1, 1, 2)
    slot = SimpleNamespace(
        decode_token_ids=torch.zeros(1, dtype=torch.int64),
        decode_coord_values=torch.zeros(1, 1),
        decode_size_values=torch.zeros(1, 2),
        logits=torch.zeros(1, 3),
        hidden_last=torch.zeros(1, 2),
    )
    runtime._decode_graphs = SimpleNamespace(
        enabled=True,
    )
    runtime._megakernel_buckets = frozenset({1})
    monkeypatch.setattr(runtime_mod, "lm_head", lambda hidden, text: torch.ones(1, 3))
    return runtime, slot


def test_graphs_on_capacity_ineligibility_falls_back_for_only_this_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, slot = _forward_runtime(monkeypatch)
    native_calls = []
    runtime._megakernel_decode_hidden = lambda *args: (_ for _ in ()).throw(
        runtime_mod.megakernel_decode.MegakernelNotEligible("position exceeds baked extent")
    )
    runtime._native_decode_hidden = lambda *args: (
        native_calls.append(args) or torch.full((1, 1, 2), 2.0)
    )

    runtime._run_decode_forward(slot, 1)

    assert len(native_calls) == 1
    assert torch.equal(slot.hidden_last, torch.full((1, 2), 2.0))


def test_graphs_on_genuine_megakernel_failure_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, slot = _forward_runtime(monkeypatch)
    runtime._megakernel_decode_hidden = lambda *args: (_ for _ in ()).throw(
        RuntimeError("kernel failed")
    )
    runtime._native_decode_hidden = lambda *args: pytest.fail(
        "genuine megakernel failures must not be hidden by native fallback"
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        runtime._run_decode_forward(slot, 1)


def test_padding_clears_host_and_device_megakernel_metadata() -> None:
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    slot = SimpleNamespace(
        decode_token_ids=torch.ones(4, dtype=torch.int64),
        decode_coord_values=torch.ones(4, 1),
        decode_size_values=torch.ones(4, 2),
        meta=SimpleNamespace(
            batch_idx=SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4)),
            input_pos=SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4)),
            lora_slot_ids=SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4)),
        ),
    )

    runtime._zero_decode_graph_padding(slot, batch_size=3, graph_batch_size=4)

    for mirrored in (slot.meta.batch_idx, slot.meta.input_pos, slot.meta.lora_slot_ids):
        assert mirrored.cpu.tolist() == [1, 1, 1, 0]
        assert mirrored.gpu.tolist() == [1, 1, 1, 0]


def test_capture_zero_clears_all_host_and_device_metadata() -> None:
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)

    def mirrored():
        return SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4))

    slot = SimpleNamespace(
        decode_token_ids=torch.ones(4, dtype=torch.int64),
        decode_coord_values=torch.ones(4, 1),
        decode_size_values=torch.ones(4, 2),
        meta=SimpleNamespace(
            batch_idx=mirrored(),
            input_pos=mirrored(),
            lora_slot_ids=mirrored(),
            active_token_ids=mirrored(),
            active_lora_ids=mirrored(),
            active_lora_meta=mirrored(),
        ),
        paged_kv_page_table=torch.ones(4, 1),
        paged_kv_seqlens_k=torch.ones(4),
    )

    runtime._zero_decode_graph_capture_buffers(slot)

    for mirrored_value in (
        slot.meta.batch_idx,
        slot.meta.input_pos,
        slot.meta.lora_slot_ids,
        slot.meta.active_token_ids,
        slot.meta.active_lora_ids,
        slot.meta.active_lora_meta,
    ):
        assert not mirrored_value.cpu.any()
        assert not mirrored_value.gpu.any()


def test_static_megakernel_bucket_applies_to_every_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, eager_slot = _forward_runtime(monkeypatch)
    native_slot = SimpleNamespace(
        decode_token_ids=torch.zeros(1, dtype=torch.int64),
        decode_coord_values=torch.zeros(1, 1),
        decode_size_values=torch.zeros(1, 2),
        logits=torch.zeros(1, 3),
        hidden_last=torch.zeros(1, 2),
    )
    megakernel_calls = []
    native_calls = []
    runtime._megakernel_decode_hidden = lambda *args: (
        megakernel_calls.append(args) or torch.ones(1, 1, 2)
    )
    runtime._native_decode_hidden = lambda *args: (
        native_calls.append(args) or torch.ones(1, 1, 2)
    )

    runtime._run_decode_forward(eager_slot, 1)
    runtime._run_decode_forward(native_slot, 1)

    assert len(megakernel_calls) == 2
    assert len(native_calls) == 0


def test_eager_capacity_ineligibility_does_not_disable_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, slot = _forward_runtime(monkeypatch)
    runtime._decode_graphs = SimpleNamespace(enabled=False)
    runtime._lora_workspace = None
    native_calls = []
    runtime._megakernel_decode_hidden = lambda *args: (_ for _ in ()).throw(
        runtime_mod.megakernel_decode.MegakernelNotEligible("position exceeds baked extent")
    )
    runtime._native_decode_hidden = lambda *args: (
        native_calls.append(args) or torch.full((1, 1, 2), 2.0)
    )
    runtime._run_decode_forward(slot, 1)

    assert len(native_calls) == 1
