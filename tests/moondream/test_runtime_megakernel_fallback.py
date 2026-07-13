from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import runtime as runtime_mod
from kestrel.models.moondream.runtime import MoondreamRuntime


def _forward_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    runtime._megakernel_served_buckets = {1}
    runtime.model = SimpleNamespace(text=object())
    runtime._embed_packed_token_batch = lambda *args: torch.ones(1, 1, 2)
    slot = SimpleNamespace(
        decode_token_ids=torch.zeros(1, dtype=torch.int64),
        decode_coord_values=torch.zeros(1, 1),
        decode_size_values=torch.zeros(1, 2),
        logits=torch.zeros(1, 3),
        hidden_last=torch.zeros(1, 2),
    )
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
    assert runtime._megakernel_served_buckets == {1}


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


def test_failed_eager_warmup_disables_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    runtime._megakernel_target = ("moondream3", 90, 132)
    runtime._megakernel_served_buckets = set()
    runtime._megakernel_failed_buckets = set()
    runtime._decode_graphs = SimpleNamespace(enabled=False)
    runtime._lora_workspace = None
    runtime.max_batch_size = 1
    runtime._decode_slots = [SimpleNamespace(
        decode_token_ids=torch.zeros(1, dtype=torch.int64),
        decode_coord_values=torch.zeros(1, 1),
        decode_size_values=torch.zeros(1, 2),
    )]
    runtime._zero_decode_graph_capture_buffers = lambda slot: None
    runtime._prepare_decode_graph_step = lambda slot, batch_size: None
    runtime._embed_packed_token_batch = lambda *args: torch.zeros(1, 1, 1)
    runtime._megakernel_decode_hidden = lambda *args: (_ for _ in ()).throw(
        RuntimeError("build failed")
    )
    monkeypatch.setattr(runtime_mod, "make_decode_graph_batch_sizes", lambda max_batch: [1])
    monkeypatch.setattr(runtime_mod.megakernel_decode, "has_megakernel", lambda *args: True)

    runtime._warm_megakernel_eager()

    assert runtime._megakernel_failed_buckets == {1}
    assert not runtime._megakernel_eager_unwarmed(1)
