"""8-row gate/up interleave for the MD3 FP8 MoE up-projection.

Leaf module (torch only, no moondream imports) so both the weight loader
(:mod:`.weights`) and the LoRA workspace (:mod:`.lora_workspace`) can share one
definition of the permutation without an import cycle. It is a pure row shuffle
of the ``2*inter`` gate/up axis -- ``gate[0:8], up[0:8], gate[8:16], ...`` -- and
the whole-model cos/argmax gates catch any drift from the megakernel's layout.
"""

from __future__ import annotations

import torch


def _interleave_gate_up_rows8(gate_up: torch.Tensor, inter: int) -> torch.Tensor:
    """8-row gate/up row interleave: gate[0:8], up[0:8], gate[8:16], up[8:16], ...

    Pure row permutation of the leading ``2*inter`` (gate/up output) axis of a
    ``[rows, 2*inter, *tail]`` tensor; trailing axes (hidden columns) are carried
    along untouched, so the per-neuron dot is bit-identical to the logical
    ``[gate[0:inter], up[0:inter]]`` half-split. THE layout MD3's FP8 MoE stores
    so the up weight is the same physical bytes the megakernel consumes.
    """
    if inter % 8 != 0:
        raise ValueError(f"gate/up interleave requires inter divisible by 8, got {inter}")
    if gate_up.shape[1] != 2 * inter:
        raise ValueError(f"expected gate_up second dim {2 * inter}, got {gate_up.shape[1]}")
    gate = gate_up[:, :inter, ...].reshape(gate_up.shape[0], inter // 8, 8, *gate_up.shape[2:])
    up = gate_up[:, inter:, ...].reshape(gate_up.shape[0], inter // 8, 8, *gate_up.shape[2:])
    return torch.stack((gate, up), dim=2).reshape(gate_up.shape).contiguous()


def _interleave_up_b_rows8(up_b: torch.Tensor, inter: int) -> torch.Tensor:
    """Interleave-by-8 the ``2*inter`` (up-proj row) axis of a LoRA up-B factor.

    ``up_b`` is ``[*lead, 2*inter, rank]``; the rank axis is the per-neuron LoRA
    column and is carried along, so the expand stays column-independent while the
    rows are permuted to match the base up-weights' physical layout (see
    :func:`_interleave_gate_up_rows8`).
    """
    two_i, rank = int(up_b.shape[-2]), int(up_b.shape[-1])
    lead = up_b.shape[:-2]
    flat = up_b.reshape(-1, two_i, rank)
    out = _interleave_gate_up_rows8(flat, inter)
    return out.reshape(*lead, two_i, rank).contiguous()
