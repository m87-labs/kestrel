"""8-row gate/up interleave for the MD3 FP8 MoE up-projection.

Leaf module (torch only, no moondream imports) so both the weight loader
(:mod:`.weights`) and the LoRA workspace (:mod:`.lora_workspace`) can share one
definition of the permutation without an import cycle. It is a pure row shuffle
of the ``2*inter`` gate/up axis -- ``gate[0:8], up[0:8], gate[8:16], ...`` -- and
the whole-model cos/argmax gates catch any drift from the megakernel's layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


# THE ENGINE-OWNED MoE up-weight slab (kestrel#121 completion / kestrel-proprietary#384).
#
# The MD3 FP8 MoE up-projection is the largest duplicated tensor in the running model: the native
# gated-GELU reads it per layer AND the whole-model megakernel reads it as one stacked
# ``[NL, E, 2I, H]`` descriptor indexed by the GLOBAL layer field. To store it exactly once, the
# engine allocates that stacked slab UP FRONT at weight load and points each MoE layer's
# ``up_experts.weight``/``scale`` at a layer-view of it; the megakernel then consumes the same slab
# byte-for-byte (see md3 session / bundle backend), no per-session stack + copy. ``NL`` is the FULL
# text-block count (global layer indexing) -- the dense pre-MoE layers keep zero rows they never
# read, which is the padding the megakernel's global ``FIELD_LAYER`` tile coordinate requires. Both
# the loader (:mod:`.weights`) and the megakernel bench/test builders share these two helpers so the
# slab-and-views contract is defined once.


def build_md3_moe_up_slab(
    text: nn.Module,
    *,
    num_experts: int,
    two_inter: int,
    hidden: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the engine-owned ``[NL, E, 2I, H]`` uint8 up-weight slab (fp8 bits) and the
    ``[NL, E, 2I]`` float32 up-scale slab, and stash them on ``text`` so the megakernel can consume
    them directly. ``NL == len(text.blocks)`` (global layer indexing). Rows for dense pre-MoE layers
    stay zero (the megakernel never tiles them). Returns ``(up_w_slab, up_scale_slab)``; the caller
    fills each MoE layer's view via :func:`set_md3_moe_up_layer`."""
    n_layers = len(text.blocks)
    up_w_slab = torch.zeros(
        n_layers, num_experts, two_inter, hidden, dtype=torch.uint8, device=device)
    up_scale_slab = torch.zeros(
        n_layers, num_experts, two_inter, dtype=torch.float32, device=device)
    # Stash the slabs on the text module: THE contract the megakernel session asserts against (each
    # MoE layer's up_experts view must alias these). Plain attributes -- not buffers -- so they do
    # not appear twice in the state_dict (the per-layer views already round-trip the bytes).
    text.moe_up_w_slab = up_w_slab
    text.moe_up_scale_slab = up_scale_slab
    return up_w_slab, up_scale_slab


def _bind_md3_moe_up_view(
    up_experts: nn.Module,
    up_w_slab: torch.Tensor,
    up_scale_slab: torch.Tensor,
    layer_idx: int,
) -> None:
    """Point ``up_experts.weight``/``scale`` at the zero-copy ``layer_idx`` view of the slabs.

    THE single source of truth for the slab-view aliasing (used by the initial load in
    :func:`set_md3_moe_up_layer` and by the post-move rebind in
    :func:`rebind_md3_moe_up_views`). After this the layer's up bytes live ONLY in the slab."""
    up_experts.weight = nn.Parameter(up_w_slab[layer_idx], requires_grad=False)
    # register_buffer overwrites an existing "scale" buffer in place; the view keeps the slab
    # resident, so the engine co-owns it regardless of any later megakernel session lifetime.
    up_experts.register_buffer("scale", up_scale_slab[layer_idx])


def set_md3_moe_up_layer(
    fused: nn.Module,
    up_w_slab: torch.Tensor,
    up_scale_slab: torch.Tensor,
    layer_idx: int,
    up_weight_uint8: torch.Tensor,
    up_scale: torch.Tensor,
    inter: int,
) -> None:
    """Interleave-by-8 a single MoE layer's raw up weight/scale into slab row ``layer_idx`` and
    re-point ``fused.up_experts.weight``/``scale`` at the (zero-copy) layer-view. After this the
    layer's up bytes live ONLY in the slab -- no private per-layer copy."""
    up_w_slab[layer_idx].copy_(_interleave_gate_up_rows8(up_weight_uint8, inter))
    up_scale_slab[layer_idx].copy_(_interleave_gate_up_rows8(up_scale, inter).float())
    _bind_md3_moe_up_view(fused.up_experts, up_w_slab, up_scale_slab, layer_idx)


def rebind_md3_moe_up_views(
    text: nn.Module,
    up_w_slab: torch.Tensor,
    up_scale_slab: torch.Tensor,
) -> None:
    """Re-point every MoE layer's ``up_experts`` view at the (already-filled) slabs.

    The slab data is untouched -- only the per-layer views are rebound, at their GLOBAL
    ``layer_idx`` offset. Used after a module move (:class:`MoEUpSlabModuleDict._apply`) once the
    slabs themselves have been moved, to restore the aliasing PyTorch tears apart when it moves the
    per-layer views independently."""
    for layer_idx, blk in enumerate(text.blocks):
        # MoE blocks carry a router (dense pre-MoE layers do not) -- same probe the megakernel
        # session uses to skip the zero-padded rows it never tiles.
        if not hasattr(blk.mlp, "router"):
            continue
        _bind_md3_moe_up_view(
            blk.mlp["mlp"].up_experts, up_w_slab, up_scale_slab, layer_idx)


class MoEUpSlabModuleDict(nn.ModuleDict):
    """``nn.ModuleDict`` that keeps the engine-owned MoE up-weight slab aliased through moves.

    The slab (stashed as a plain attribute in :func:`build_md3_moe_up_slab`) is deliberately NOT a
    parameter or buffer -- it must not double the state_dict. But that also means PyTorch's
    ``Module._apply`` (the engine behind ``.to()`` / ``.cuda()`` / ``.float()`` / ``to_empty()``)
    never visits it: it walks only registered params/buffers, moving each per-layer
    ``up_experts.weight``/``scale`` view INDEPENDENTLY and never preserving cross-tensor storage
    aliasing. So a post-load ``model.to(device)`` would leave the slab on the old device while the 20
    views become 20 private copies -- silently losing the native dedup, and tripping the megakernel
    session's alias assert. This override moves the slab with the SAME ``fn`` and re-points every MoE
    layer's view back at it. Idempotent: a device/dtype-preserving ``fn`` returns the same tensors,
    so the rebind is skipped (the normal no-move path is untouched)."""

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        slab_w = getattr(self, "moe_up_w_slab", None)
        slab_s = getattr(self, "moe_up_scale_slab", None)
        if slab_w is None or slab_s is None:
            return self  # dense / non-FP8 model: no slab to keep aliased
        new_w, new_s = fn(slab_w), fn(slab_s)
        if new_w is slab_w and new_s is slab_s:
            return self  # no-op move: views already alias the resident slab
        self.moe_up_w_slab = new_w
        self.moe_up_scale_slab = new_s
        rebind_md3_moe_up_views(self, new_w, new_s)
        return self
