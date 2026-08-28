import torch
from torch import nn

from kestrel.runtime.bounded_projection import (
    PackedBoundedProjections,
    PackedLinear,
    bind_declared_packed_projections,
    declared_packed_projection_source_keys,
)


def test_packed_linear_binding_waits_for_all_streamed_parts() -> None:
    module = nn.Module()
    module.packed = PackedLinear(
        3,
        (2, 1),
        source_names=("q", "k"),
    )
    q = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = {"q.weight": q}

    bind_declared_packed_projections(
        module,
        state,
        require_complete=False,
    )
    assert state == {"q.weight": q}

    k = torch.arange(3, dtype=torch.float32).reshape(1, 3) + 10
    state["k.weight"] = k
    bind_declared_packed_projections(
        module,
        state,
        require_complete=False,
    )
    assert set(state) == {"packed.weight"}
    torch.testing.assert_close(state["packed.weight"], torch.cat((q, k)))


def test_packed_bounded_binding_waits_for_streamed_bounds() -> None:
    module = nn.Module()
    module.packed = PackedBoundedProjections(
        2,
        (1, 1),
        source_names=("left", "right"),
        use_bounds=True,
    )
    state = {
        "left.linear.weight": torch.ones((1, 2)),
        "right.linear.weight": torch.full((1, 2), 2.0),
    }

    bind_declared_packed_projections(
        module,
        state,
        require_complete=False,
    )
    assert "packed.linear.weight" not in state

    for source in ("left", "right"):
        state[f"{source}.input_min"] = torch.tensor(-1.0)
        state[f"{source}.input_max"] = torch.tensor(1.0)
        state[f"{source}.output_min"] = torch.tensor(-2.0)
        state[f"{source}.output_max"] = torch.tensor(2.0)
    bind_declared_packed_projections(
        module,
        state,
        require_complete=False,
    )
    assert set(state) == {
        "packed.input_max",
        "packed.input_min",
        "packed.linear.weight",
        "packed.output_max",
        "packed.output_min",
    }


def test_packed_binding_skips_already_loaded_target() -> None:
    module = nn.Module()
    module.packed = PackedLinear(
        3,
        (2, 1),
        source_names=("q", "k"),
    )
    state: dict[str, torch.Tensor] = {}

    bind_declared_packed_projections(
        module,
        state,
        already_bound_targets={"packed.weight"},
    )

    assert state == {}


def test_declared_source_keys_include_weights_and_bounds() -> None:
    module = nn.Module()
    module.attn = nn.Module()
    module.attn.qkv = PackedBoundedProjections(
        2,
        (1, 1),
        source_names=("q", "k"),
        use_bounds=True,
    )

    assert declared_packed_projection_source_keys(module) == {
        "attn.q.linear.weight",
        "attn.k.linear.weight",
        "attn.q.input_min",
        "attn.q.input_max",
        "attn.q.output_min",
        "attn.q.output_max",
        "attn.k.input_min",
        "attn.k.input_max",
        "attn.k.output_min",
        "attn.k.output_max",
    }
