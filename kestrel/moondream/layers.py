"""Layer building blocks for the Moondream text transformer.

Adapted from the Moondream project (Apache-2.0).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal

from ..ops.layernorm_cuda import layernorm_bias

# Re-export LoRA for convenience
from .lora import LoRA, MoEMLPLoRA, DenseMLPLoRA  # noqa: F401
from .lora_workspace import DenseLoRALayerWorkspace, MoELoRALayerWorkspace


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def layer_norm(x: torch.Tensor, w: LayerNormWeights) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16:
        try:
            return layernorm_bias(x, w.weight, w.bias)
        except Exception:
            pass
    return F.layer_norm(x, w.bias.shape, w.weight, w.bias)


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return F.linear(x, w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


@dataclass(frozen=True)
class _DenseLoRARouting:
    """Batched routing data for dense LoRA using moe_lora_align_block_size."""
    topk_weights: torch.Tensor
    sorted_token_ids: torch.Tensor  # [max_loras, EM]
    expert_ids: torch.Tensor  # [max_loras, num_blocks]
    num_tokens_post_padded: torch.Tensor  # [max_loras]
    block_size_m: int


def _prepare_dense_lora_routing(
    lora_slot_ids: torch.Tensor,
    *,
    max_slots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _DenseLoRARouting:
    """Prepare batched routing for dense LoRA.

    Uses moe_lora_align_block_size with num_experts=1 (each slot treated as
    one expert). Slot 0 (no LoRA) is filtered via token_lora_mapping = -1.
    """
    # slot 0 -> -1 (no LoRA), slot N -> N-1 (lora_id)
    token_lora_mapping = (lora_slot_ids - 1).to(torch.int32)

    # For dense LoRA, treat each slot as a separate "LoRA" with num_experts=1
    # max_loras = max_slots - 1 (slot 0 excluded from workspace)
    max_loras = max_slots - 1
    block_size_m = 16
    num_experts = 1  # Dense LoRA: each slot is one "expert"

    # Shape topk_ids as [num_tokens, 1] with all zeros (single expert per "LoRA")
    num_tokens = lora_slot_ids.shape[0]
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32, device=device)

    from ..fused_moe.routing import moe_lora_align_block_size

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        block_size_m,
        num_experts,
        max_loras,
    )

    # topk_weights shape [M, top_k] is required by the kernel for shape inference.
    # Values unused when mul_routed_weight=False but shape must be correct.
    topk_weights = torch.ones((num_tokens, 1), dtype=dtype, device=device)

    return _DenseLoRARouting(
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size_m=block_size_m,
    )


def _apply_dense_lora_with_routing(
    x: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    routing: _DenseLoRARouting,
) -> None:
    """Apply dense LoRA using batched kernel.

    Workspace layout: lora_a/lora_b have shape [max_slots, rank, hidden].
    We slice off slot 0 to get [max_loras, rank, hidden] where max_loras = max_slots - 1.
    """
    from ..fused_moe.lora_kernels import apply_moe_lora_batched

    num_tokens = x.shape[0]

    # Output needs shape [num_tokens, top_k, out_dim] for apply_moe_lora_batched.
    out_dim = output.shape[-1]
    output_3d = output.view(num_tokens, 1, out_dim)

    # Slice off slot 0 from workspace (slot 0 = no LoRA, always zeros)
    # This aligns with the lora_id mapping: slot N -> lora_id N-1
    lora_a_active = lora_a[1:]  # [max_loras, rank, hidden]
    lora_b_active = lora_b[1:]  # [max_loras, out_dim, rank]

    apply_moe_lora_batched(
        x=x,
        topk_weights=routing.topk_weights,
        output=output_3d,
        lora_a=lora_a_active,
        lora_b=lora_b_active,
        sorted_token_ids=routing.sorted_token_ids,
        expert_ids=routing.expert_ids,
        num_tokens_post_padded=routing.num_tokens_post_padded,
        top_k=1,
        num_experts=1,  # Dense LoRA: each slot is one "expert"
        block_size_m=routing.block_size_m,
        mul_routed_weight=False,
    )


def apply_dense_lora(
    x: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_slot_ids: torch.Tensor,
) -> None:
    """Apply mixed-slot dense LoRA using the batched kernel.

    Uses moe_lora_align_block_size to route tokens to their LoRA adapters,
    then applies the batched kernel with num_experts=1.

    Slot 0 tokens are filtered out at routing time (no LoRA applied).
    Slot N (N >= 1) maps to lora_id N-1 in the workspace.

    Args:
        x: Input activations, shape [num_tokens, hidden_dim].
        output: Output tensor to accumulate into, shape [num_tokens, out_dim].
        lora_a: LoRA A weights, shape [max_slots, rank, hidden_dim].
        lora_b: LoRA B weights, shape [max_slots, out_dim, rank].
        lora_slot_ids: Per-token slot indices, shape [num_tokens].
    """
    max_slots = lora_a.shape[0]
    routing = _prepare_dense_lora_routing(
        lora_slot_ids,
        max_slots=max_slots,
        device=x.device,
        dtype=x.dtype,
    )
    _apply_dense_lora_with_routing(x, output, lora_a, lora_b, routing)


def mlp(
    x: torch.Tensor,
    w: MLPWeights,
    *,
    lora_workspace: DenseLoRALayerWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense MLP with optional mixed-slot LoRA.

    Args:
        x: Input tensor, shape [batch, seq_len, dim].
        w: MLP weights (fc1, fc2).
        lora_workspace: Multi-slot LoRA workspace for this layer, or None.
        lora_slot_ids: Per-sequence slot indices, shape [batch], or None.
    """
    B, T, C = x.shape
    use_lora = lora_workspace is not None and lora_slot_ids is not None

    routing = None
    if use_lora:
        # Expand slot IDs for all tokens in each sequence and prepare routing once.
        slot_ids_expanded = lora_slot_ids.repeat_interleave(T)
        routing = _prepare_dense_lora_routing(
            slot_ids_expanded,
            max_slots=lora_workspace.up_a.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

    h = linear(x, w.fc1)
    if use_lora:
        # Flatten for LoRA kernel: [batch * seq_len, dim]
        x_flat = x.view(-1, C)
        # Use separate buffer for LoRA delta, then add to base output
        lora_delta = torch.zeros_like(h.view(-1, h.shape[-1]))
        assert routing is not None
        _apply_dense_lora_with_routing(
            x_flat, lora_delta, lora_workspace.up_a, lora_workspace.up_b, routing
        )
        h = h + lora_delta.view_as(h)

    h = gelu_approx(h)
    out = linear(h, w.fc2)
    if use_lora:
        h_flat = h.view(-1, h.shape[-1])
        # Use separate buffer for LoRA delta, then add to base output
        lora_delta = torch.zeros_like(out.view(-1, out.shape[-1]))
        assert routing is not None
        _apply_dense_lora_with_routing(
            h_flat, lora_delta, lora_workspace.down_a, lora_workspace.down_b, routing
        )
        out = out + lora_delta.view_as(out)

    return out


def build_dense_mlp(d_model: int, d_ffn: int, dtype: torch.dtype) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "fc1": nn.Linear(d_model, d_ffn, dtype=dtype),
            "fc2": nn.Linear(d_ffn, d_model, dtype=dtype),
        }
    )


def build_moe_mlp(
    d_model: int, d_ffn: int, n_experts: int, dtype: torch.dtype, *, top_k: int
) -> nn.ModuleDict:
    from ..fused_moe import ExpertWeights, FusedMoEModule

    router = nn.Linear(d_model, n_experts, dtype=dtype)
    up_experts = ExpertWeights(n_experts, d_model, d_ffn * 2, dtype=dtype)
    down_experts = ExpertWeights(n_experts, d_ffn, d_model, dtype=dtype)
    fused = FusedMoEModule(
        up_experts,
        down_experts,
        top_k=top_k,
        hidden_size=d_ffn,
        input_size=d_model,
        num_experts=n_experts,
    )
    return nn.ModuleDict({"router": router, "mlp": fused})


def moe_mlp(
    x: torch.Tensor,
    mlp_module: nn.Module,
    experts_per_token: int,
    *,
    mode: Literal["prefill", "decode"] = "decode",
    lora_workspace: MoELoRALayerWorkspace | None = None,
    lora_slot_ids: torch.Tensor | None = None,
    single_lora_id: int | None = None,
) -> torch.Tensor:
    B, T, C = x.shape
    x_flat = x.view(-1, C)

    router = mlp_module["router"]
    fused_mlp = mlp_module["mlp"]

    router_logits = router(x_flat)
    topk_weights, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1)
    topk_idxs = topk_idxs.to(torch.int32)

    # Expand slot IDs for all tokens if we have a sequence length > 1
    expanded_slot_ids = None
    if lora_workspace is not None and lora_slot_ids is not None:
        expanded_slot_ids = lora_slot_ids.repeat_interleave(T)

    mlp_out = fused_mlp(
        x_flat,
        topk_weights,
        topk_idxs,
        lora_workspace,
        expanded_slot_ids,
        single_lora_id,
    ).view(B, T, C)
    return mlp_out


__all__ = [
    "LayerNormWeights",
    "LinearWeights",
    "MLPWeights",
    "layer_norm",
    "mlp",
    "apply_dense_lora",
    "moe_mlp",
    "build_dense_mlp",
    "build_moe_mlp",
    "LoRA",
]
