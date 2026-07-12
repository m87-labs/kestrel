from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kestrel.models.moondream.moe import MoEConfig, MoEModule
from kestrel.models.moondream import runtime as runtime_mod
from kestrel.utils import CpuGpuBuffer


class _FakeMoeRuntime:
    def __init__(self) -> None:
        self.capacities = []
        self.handles = {}

    def prepare(self, spec, capacity, *, device):
        key = (spec, capacity, torch.device(device))
        handle = self.handles.get(key)
        if handle is not None:
            return handle
        self.capacities.append(capacity)
        handle = SimpleNamespace(
            spec=spec,
            capacity=capacity,
            device=torch.device(device),
            impl=None,
        )
        self.handles[key] = handle
        return handle


def _make_module(runtime: _FakeMoeRuntime) -> MoEModule:
    module = object.__new__(MoEModule)
    module.top_k = 8
    module.hidden_size = 1024
    module.input_size = 2048
    module.num_experts = 64
    module.config = MoEConfig()
    module._moe_runtime = runtime
    return module


def test_prefill_moe_handles_reuse_power_of_two_capacity_bucket() -> None:
    runtime = _FakeMoeRuntime()
    module = _make_module(runtime)

    first = module._get_moe_handle(
        hidden_states=torch.empty((129, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=0,
        mode="prefill",
    )
    second = module._get_moe_handle(
        hidden_states=torch.empty((200, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=0,
        mode="prefill",
    )

    assert first is second
    assert [capacity.max_tokens for capacity in runtime.capacities] == [256]


def test_decode_moe_handles_remain_exact_capacity_for_cuda_graph_buckets() -> None:
    runtime = _FakeMoeRuntime()
    module = _make_module(runtime)

    first = module._get_moe_handle(
        hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=0,
        mode="decode",
    )
    second = module._get_moe_handle(
        hidden_states=torch.empty((16, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=0,
        mode="decode",
    )

    assert first is not second
    assert [capacity.max_tokens for capacity in runtime.capacities] == [8, 16]


def test_lora_rank_is_part_of_moe_handle_capacity() -> None:
    runtime = _FakeMoeRuntime()
    module = _make_module(runtime)

    first = module._get_moe_handle(
        hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=1,
        max_lora_rank=4,
        mode="decode",
    )
    second = module._get_moe_handle(
        hidden_states=torch.empty((8, 2048), dtype=torch.bfloat16),
        weight_format="bf16",
        max_loras=1,
        max_lora_rank=8,
        mode="decode",
    )

    assert first is not second
    assert [capacity.max_lora_rank for capacity in runtime.capacities] == [4, 8]


class _FakeLoraWorkspace:
    max_slots = 4

    def __init__(self) -> None:
        self._ranks = [3, 7, 5]

    def moe_lora_rank_for_id(self, lora_id: int) -> int:
        return self._ranks[lora_id]


def _make_decode_meta(max_batch: int) -> SimpleNamespace:
    device = torch.device("cpu")
    return SimpleNamespace(
        lora_slot_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_token_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_lora_ids=CpuGpuBuffer(
            max_batch, dtype=torch.int32, device=device, pin_memory=False
        ),
        active_lora_meta=CpuGpuBuffer(
            max_batch + 3, dtype=torch.int32, device=device, pin_memory=False
        ),
    )


def test_decode_lora_metadata_uses_compact_graph_stable_buffers() -> None:
    runtime = runtime_mod.MoondreamRuntime.__new__(runtime_mod.MoondreamRuntime)
    runtime._lora_workspace = _FakeLoraWorkspace()
    slot = SimpleNamespace(meta=_make_decode_meta(8))

    slot.meta.lora_slot_ids.np[:6] = [0, 3, 1, 3, 2, 0]

    runtime._prepare_decode_lora_metadata(slot, 6)

    torch.testing.assert_close(
        slot.meta.active_lora_meta.gpu[:7],
        torch.tensor([3, 7, 4, 0, 2, 3, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        slot.meta.active_token_ids.gpu[:4],
        torch.tensor([1, 3, 2, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        slot.meta.active_lora_ids.gpu[:3],
        torch.tensor([2, 0, 1], dtype=torch.int32),
    )
    assert slot.meta.moe_lora_metadata.active_token_ids.numel() == 6
    assert slot.meta.moe_lora_metadata.active_lora_ids.numel() == 3
    assert slot.meta.moe_lora_metadata.active_lora_meta.numel() == 7

    slot.meta.lora_slot_ids.np[:6] = 0
    runtime._prepare_decode_lora_metadata(slot, 6)

    torch.testing.assert_close(
        slot.meta.active_lora_meta.gpu[:7],
        torch.zeros((7,), dtype=torch.int32),
    )


# ---------------------------------------------------------------------------
# Non-FP8 MoE checkpoints are unrepresentable (kestrel#121)
#
# The MD3 MoE up-projection base is stored in the 8-row gate/up interleave and
# every MoE LoRA up-B adapter is interleaved to match by construction. A non-FP8
# MoE checkpoint would land the up weight half-split under an interleaved
# adapter -> scrambled expand. The loader hard-fails instead of loading it, so
# the base-vs-adapter layout coupling can never be violated. This test pins that
# error path (delete-not-gate: the mismatch is unrepresentable).
# ---------------------------------------------------------------------------


class _Node(dict):
    """Dict that also exposes its keys as attributes (mirrors the model tree's
    dual item/attribute access used by the weight loader)."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - attribute miss path
            raise AttributeError(key) from exc


def _leaf(*, bias: bool = True) -> _Node:
    node = _Node(weight=torch.zeros(1))
    if bias:
        node["bias"] = torch.zeros(1)
    return node


def _moe_block() -> _Node:
    # is_moe is detected via ``hasattr(block.mlp, "router")``.
    mlp = _Node(router=_leaf())
    return _Node(
        ln=_leaf(),
        attn=_Node(qkv=_leaf(), proj=_leaf()),
        mlp=mlp,
    )


def _fake_md3_model_with_moe() -> _Node:
    text = _Node(
        blocks=[_moe_block()],
        wte=torch.zeros(1),
        post_ln=_leaf(),
        lm_head=_leaf(),
    )
    return _Node(text=text)


def test_non_fp8_moe_checkpoint_raises() -> None:
    from kestrel.models.moondream.weights import _assign_md3_text_weights

    model = _fake_md3_model_with_moe()

    with pytest.raises(ValueError, match="Non-FP8 MoE checkpoints are not supported"):
        # No captured moe_quant scales -> the (deleted) non-FP8 MoE branch.
        _assign_md3_text_weights(lambda name: torch.zeros(1), model, moe_scales=None)


def test_non_fp8_moe_error_names_offending_tensor() -> None:
    from kestrel.models.moondream.weights import _assign_md3_text_weights

    model = _fake_md3_model_with_moe()

    with pytest.raises(ValueError, match=r"mlp\.experts\.weight"):
        _assign_md3_text_weights(lambda name: torch.zeros(1), model, moe_scales=None)


# ---------------------------------------------------------------------------
# Engine-owned MoE up-slab survives module moves (kestrel#121 codex P2).
#
# The slab is a plain attribute (not a param/buffer) so it does not double the
# state_dict, but that means PyTorch's ``Module._apply`` (behind ``.to()`` /
# ``.cuda()``) never visits it and moves each per-layer view INDEPENDENTLY --
# tearing apart the cross-tensor storage aliasing the megakernel contract needs.
# ``MoEUpSlabModuleDict._apply`` moves the slab with the same fn and re-points the
# views. These tests pin: (a) the override restores aliasing after a move, and
# (b) a plain ``nn.ModuleDict`` does NOT -- proving the override is load-bearing.
# ---------------------------------------------------------------------------

_MOVE_E, _MOVE_INTER, _MOVE_HIDDEN, _MOVE_NL, _MOVE_START = 4, 16, 8, 3, 1


def _build_moe_slab_text(cls):
    from kestrel.models.moondream._moe_layout import (
        build_md3_moe_up_slab,
        set_md3_moe_up_layer,
    )
    from kestrel.models.moondream.layers import build_dense_mlp, build_moe_mlp

    blocks = nn.ModuleList()
    for i in range(_MOVE_NL):
        if i >= _MOVE_START:
            mlp = build_moe_mlp(_MOVE_HIDDEN, _MOVE_INTER, _MOVE_E, torch.float32, top_k=2)
        else:
            mlp = build_dense_mlp(_MOVE_HIDDEN, 4 * _MOVE_HIDDEN, torch.float32)
        blocks.append(nn.ModuleDict({"mlp": mlp}))
    text = cls({"blocks": blocks})

    two_inter = 2 * _MOVE_INTER
    up_w_slab, up_scale_slab = build_md3_moe_up_slab(
        text, num_experts=_MOVE_E, two_inter=two_inter, hidden=_MOVE_HIDDEN, device="cpu")

    torch.manual_seed(0)
    for li in range(_MOVE_START, _MOVE_NL):
        raw_up = torch.randint(0, 256, (_MOVE_E, two_inter, _MOVE_HIDDEN), dtype=torch.uint8)
        raw_scale = torch.rand(_MOVE_E, two_inter)
        set_md3_moe_up_layer(
            text.blocks[li].mlp["mlp"], up_w_slab, up_scale_slab, li, raw_up, raw_scale, _MOVE_INTER)
    return text


def _moe_up_views(text):
    return [
        (li, text.blocks[li].mlp["mlp"].up_experts)
        for li in range(_MOVE_START, _MOVE_NL)
    ]


def _assert_slab_aliased(text) -> None:
    """Mirror the megakernel session's contract check: every MoE layer's
    up_experts weight/scale aliases the ONE slab storage at its ``[NL]`` offset."""
    w_slab, s_slab = text.moe_up_w_slab, text.moe_up_scale_slab
    w_base = w_slab.untyped_storage().data_ptr()
    s_base = s_slab.untyped_storage().data_ptr()
    w_stride = _MOVE_E * 2 * _MOVE_INTER * _MOVE_HIDDEN
    s_stride = _MOVE_E * 2 * _MOVE_INTER
    for li, up in _moe_up_views(text):
        assert up.weight.untyped_storage().data_ptr() == w_base
        assert up.weight.storage_offset() == li * w_stride
        assert up.weight.device == w_slab.device
        assert up.scale.untyped_storage().data_ptr() == s_base
        assert up.scale.storage_offset() == li * s_stride


def test_moe_up_slab_survives_module_move() -> None:
    from kestrel.models.moondream._moe_layout import MoEUpSlabModuleDict

    text = _build_moe_slab_text(MoEUpSlabModuleDict)
    _assert_slab_aliased(text)
    before = {li: up.weight.clone() for li, up in _moe_up_views(text)}
    scale_before = {li: up.scale.clone() for li, up in _moe_up_views(text)}

    # ``_apply(clone)`` reproduces exactly what a real ``.to(device)`` does: it
    # allocates fresh storage per param/buffer (breaking naive aliasing), which is
    # the CPU-only stand-in for a cross-device move.
    text._apply(lambda t: t.clone())

    _assert_slab_aliased(text)  # override re-established the single-slab aliasing
    for li, up in _moe_up_views(text):
        assert torch.equal(up.weight, before[li])
        assert torch.equal(up.scale, scale_before[li])
        assert up.scale.dtype == torch.float32


def test_moe_up_slab_no_op_move_keeps_aliasing() -> None:
    from kestrel.models.moondream._moe_layout import MoEUpSlabModuleDict

    text = _build_moe_slab_text(MoEUpSlabModuleDict)
    # Same-device ``.to`` is a no-op: fn returns the same tensors, the rebind is
    # skipped, and the resident-slab aliasing is untouched.
    text.to("cpu")
    _assert_slab_aliased(text)


def test_plain_moduledict_loses_slab_aliasing_on_move() -> None:
    # Baseline (no override): a real move DOES tear the aliasing apart, so the
    # megakernel contract would fail -- this is the bug the override fixes.
    text = _build_moe_slab_text(nn.ModuleDict)
    _assert_slab_aliased(text)  # aliased at load, before any move
    text._apply(lambda t: t.clone())
    w_base = text.moe_up_w_slab.untyped_storage().data_ptr()
    aliased = [
        up.weight.untyped_storage().data_ptr() == w_base for _, up in _moe_up_views(text)
    ]
    assert not any(aliased)  # every view is now a private copy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for a real device move")
def test_moe_up_slab_survives_cuda_move() -> None:
    from kestrel.models.moondream._moe_layout import MoEUpSlabModuleDict

    text = _build_moe_slab_text(MoEUpSlabModuleDict)
    before = {li: up.weight.clone() for li, up in _moe_up_views(text)}

    text.to("cuda:0")

    assert text.moe_up_w_slab.is_cuda
    _assert_slab_aliased(text)
    for li, up in _moe_up_views(text):
        assert up.weight.is_cuda
        assert torch.equal(up.weight.cpu(), before[li])


# ---------------------------------------------------------------------------
# THE PERMUTATION ITSELF (kestrel #121).
#
# The 8-row gate/up interleave is SHAPE-PRESERVING: gate row n lands at physical
# row 16*(n//8) + n%8 and up row n at that + 8. No shape or dtype check can tell
# a correct interleave from an inverted, by-16, or dropped one -- only the index
# math can. Every downstream kernel (kestrel-kernels compact/warpgemv, the mkl
# megakernel MoE consumers) assumes exactly this map, so it is pinned here rather
# than left to whole-model cosine gates to notice.
# ---------------------------------------------------------------------------

def _expected_rows(inter: int) -> list[int]:
    """Physical row for each logical row of a [gate | up] slab, per the contract."""
    gate = [16 * (n // 8) + n % 8 for n in range(inter)]
    return gate + [r + 8 for r in gate]


def test_interleave_gate_up_rows8_matches_the_row_contract() -> None:
    from kestrel.models.moondream._moe_layout import _interleave_gate_up_rows8

    experts, inter, hidden = 2, 24, 3
    # Row-identifying values: element (e, j, :) == j, so a permuted row is
    # readable straight off the output.
    plain = (torch.arange(2 * inter, dtype=torch.float32)[None, :, None]
             .expand(experts, 2 * inter, hidden).contiguous())

    out = _interleave_gate_up_rows8(plain, inter)

    assert out.shape == plain.shape
    for logical, physical in enumerate(_expected_rows(inter)):
        assert torch.equal(out[:, physical], plain[:, logical]), (
            f"logical row {logical} should land at physical row {physical}")
    # Spot-check the contract literally: gate[8:16] occupies rows 16..23,
    # up[8:16] occupies rows 24..31.
    assert out[0, 16, 0] == 8 and out[0, 23, 0] == 15
    assert out[0, 24, 0] == inter + 8 and out[0, 31, 0] == inter + 15


def test_interleave_gate_up_rows8_permutes_scales_identically() -> None:
    """Per-channel scales are [E, 2I] (no hidden tail) and MUST carry the same
    permutation as the weight rows, or every neuron gets the wrong scale."""
    from kestrel.models.moondream._moe_layout import _interleave_gate_up_rows8

    experts, inter = 3, 16
    plain = (torch.arange(2 * inter, dtype=torch.float32)[None, :]
             .expand(experts, 2 * inter).contiguous())

    out = _interleave_gate_up_rows8(plain, inter)

    assert out.shape == plain.shape
    for logical, physical in enumerate(_expected_rows(inter)):
        assert torch.equal(out[:, physical], plain[:, logical])


def test_interleave_up_b_rows8_permutes_rows_and_carries_rank() -> None:
    """LoRA up-B is [*lead, 2I, rank]: rows follow the slab, rank is untouched
    (the expand is column-independent, so a wrong row order lands every delta on
    the wrong neuron while passing all shape checks)."""
    from kestrel.models.moondream._moe_layout import _interleave_up_b_rows8

    slots, experts, inter, rank = 2, 3, 16, 5
    plain = (torch.arange(2 * inter, dtype=torch.float32)[None, None, :, None]
             .expand(slots, experts, 2 * inter, rank).contiguous())
    # Make the rank axis distinguishable so a rank/row transpose cannot pass.
    plain = plain * 100 + torch.arange(rank, dtype=torch.float32)

    out = _interleave_up_b_rows8(plain, inter)

    assert out.shape == plain.shape
    for logical, physical in enumerate(_expected_rows(inter)):
        assert torch.equal(out[..., physical, :], plain[..., logical, :])


def test_set_md3_moe_up_layer_writes_the_interleaved_slab() -> None:
    """The loader entry point stores the PERMUTED bytes -- pinned end to end, so
    a regression in the loader (not just the helper) is caught."""
    from kestrel.models.moondream._moe_layout import (
        MoEUpSlabModuleDict,
        build_md3_moe_up_slab,
        set_md3_moe_up_layer,
    )

    text = _build_moe_slab_text(MoEUpSlabModuleDict)
    two_inter = 2 * _MOVE_INTER
    up_w_slab, up_scale_slab = build_md3_moe_up_slab(
        text, num_experts=_MOVE_E, two_inter=two_inter, hidden=_MOVE_HIDDEN, device="cpu")

    raw_up = (torch.arange(two_inter, dtype=torch.uint8)[None, :, None]
              .expand(_MOVE_E, two_inter, _MOVE_HIDDEN).contiguous())
    raw_scale = (torch.arange(two_inter, dtype=torch.float32)[None, :]
                 .expand(_MOVE_E, two_inter).contiguous())
    layer = _MOVE_START
    fused = text.blocks[layer].mlp["mlp"]
    set_md3_moe_up_layer(
        fused, up_w_slab, up_scale_slab, layer, raw_up, raw_scale, _MOVE_INTER)

    weight, scale = fused.up_experts.weight, fused.up_experts.scale
    for logical, physical in enumerate(_expected_rows(_MOVE_INTER)):
        assert torch.equal(weight[:, physical], raw_up[:, logical])
        assert torch.equal(scale[:, physical], raw_scale[:, logical])
