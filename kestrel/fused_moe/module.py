from dataclasses import dataclass, field
from math import prod
from typing import Literal

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from kestrel.moondream.lora_workspace import MoELoRALayerWorkspace
from kestrel.utils.buffers import FixedBuffer
from kestrel_kernels import get_runtime

from .lora_kernels import apply_moe_lora_batched, apply_moe_lora_single
from .routing import moe_align_block_size, moe_lora_align_block_size


_KERNELS = get_runtime()
_MOE = _KERNELS.moe


def _to_power_of_2(x: int) -> int:
    """Round x down to the nearest power of 2.

    Triton's tl.arange requires the range to be a power of 2. The CuTe MoE
    configs can return non-power-of-2 block_m values (e.g., 192 for FP8 with
    large token counts). This helper ensures LoRA kernels receive valid values.

    We round DOWN (not up) to ensure the resulting block size has precompiled
    routing kernels available (precompiled for: 16, 32, 64, 128, 192).
    """
    if x <= 0:
        return 1
    if x & (x - 1) == 0:
        return x  # Already a power of 2
    # Round down: find the highest set bit
    return 1 << (x.bit_length() - 1)



class _MoEWorkspaces:
    def __init__(self) -> None:
        self.up = FixedBuffer("MoE up workspace")
        self.down = FixedBuffer("MoE down workspace")
        self.output = FixedBuffer("MoE output workspace")
        self.activation = FixedBuffer("MoE activation workspace")
        self.lora_up = FixedBuffer("MoE LoRA up workspace")
        self.lora_down = FixedBuffer("MoE LoRA down workspace")
        self.fp8_bits = FixedBuffer("MoE FP8 bits workspace")
        self.fp8_scale = FixedBuffer("MoE FP8 scale workspace")


# Shared workspace for all MoE layers. Since layers execute sequentially,
# we can safely reuse the same buffers across all layers, reducing memory
# from O(num_layers * workspace_size) to O(workspace_size).
_SHARED_MOE_WORKSPACES: _MoEWorkspaces | None = None


def get_shared_moe_workspaces() -> _MoEWorkspaces:
    """Get the shared MoE workspace instance, creating it if needed."""
    global _SHARED_MOE_WORKSPACES
    if _SHARED_MOE_WORKSPACES is None:
        _SHARED_MOE_WORKSPACES = _MoEWorkspaces()
    return _SHARED_MOE_WORKSPACES


def preallocate_shared_moe_workspaces(
    max_num_tokens: int,
    top_k: int,
    hidden_size: int,
    input_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-allocate shared MoE workspaces to ensure stable pointers for CUDA graphs.

    This must be called before capturing CUDA graphs. All FusedMoEModule instances
    share these workspaces, so this only needs to be called once.

    Args:
        max_num_tokens: Maximum tokens in any forward pass (typically max_seq_length - 1).
        top_k: Number of experts per token.
        hidden_size: MoE intermediate dimension (expert_inner_dim).
        input_size: Model hidden dimension.
        device: Target device.
        dtype: Data type for workspace tensors.
    """
    ws = get_shared_moe_workspaces()
    ws.up.get(
        (max_num_tokens, top_k, hidden_size * 2),
        device=device,
        dtype=dtype,
    )
    ws.activation.get(
        (max_num_tokens * top_k, hidden_size),
        device=device,
        dtype=dtype,
    )
    ws.down.get(
        (max_num_tokens, top_k, input_size),
        device=device,
        dtype=dtype,
    )
    ws.output.get(
        (max_num_tokens, input_size),
        device=device,
        dtype=dtype,
    )
    ws.lora_up.get(
        (max_num_tokens, top_k, hidden_size * 2),
        device=device,
        dtype=dtype,
    )
    ws.lora_down.get(
        (max_num_tokens, top_k, input_size),
        device=device,
        dtype=dtype,
    )
    ws.fp8_bits.get(
        (max_num_tokens * top_k, hidden_size),
        device=device,
        dtype=torch.uint8,
    )
    ws.fp8_scale.get(
        (max_num_tokens * top_k,),
        device=device,
        dtype=torch.float32,
    )


@dataclass
class FusedMoEConfig:
    block_size_m: int = 16
    block_size_n: int = 64
    block_size_k: int = 32
    group_size_m: int = 8
    num_warps: int = 4
    num_stages: int = 2
    allow_tf32: bool = True
    backend: str = "auto"  # "auto" | "cute" | "triton"
    auto_backend_token_threshold: int = 256
    lora_decode_shrink: dict[str, int] | None = field(
        default_factory=lambda: {
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 128,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        }
    )
    lora_decode_expand: dict[str, int] | None = field(
        default_factory=lambda: {
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 16,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        }
    )
    lora_prefill_shrink: dict[str, int] | None = field(
        default_factory=lambda: {
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 64,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        }
    )
    lora_prefill_expand: dict[str, int] | None = field(
        default_factory=lambda: {
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 16,
            "NUM_WARPS": 4,
            "NUM_STAGES": 3,
        }
    )

    def as_triton(self, *, block_size_m: int | None = None) -> dict[str, int]:
        config = {
            "BLOCK_SIZE_M": block_size_m or self.block_size_m,
            "BLOCK_SIZE_N": self.block_size_n,
            "BLOCK_SIZE_K": self.block_size_k,
            "GROUP_SIZE_M": self.group_size_m,
            "NUM_WARPS": self.num_warps,
            "NUM_STAGES": self.num_stages,
        }
        return config


@dataclass(frozen=True)
class _MoEExecutionContext:
    top_k: int
    hidden_size: int
    input_size: int
    num_experts: int
    config: FusedMoEConfig
    workspaces: _MoEWorkspaces


@dataclass
class _PreparedMoEForward:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    up_scale: torch.Tensor | None
    down_scale: torch.Tensor | None
    num_tokens: int
    block_size_m: int
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    up_out: torch.Tensor
    fp8_up_bits: torch.Tensor | None = None
    fp8_up_scale: torch.Tensor | None = None
    fp8_down_bits: torch.Tensor | None = None
    fp8_down_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class _PreparedSingleLoraRouting:
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    block_size_m: int


def _make_execution_context(owner: object) -> _MoEExecutionContext:
    return _MoEExecutionContext(
        top_k=owner.top_k,
        hidden_size=owner.hidden_size,
        input_size=owner.input_size,
        num_experts=owner.num_experts,
        config=owner.config,
        workspaces=owner._workspaces,
    )


def _get_block_m_for_routing(
    ctx: _MoEExecutionContext,
    *,
    device: torch.device,
    num_tokens: int,
    is_fp8_weights: bool,
) -> int:
    return _MOE.get_moe_block_m(
        num_tokens,
        num_experts=ctx.num_experts,
        input_size=ctx.input_size,
        hidden_size=ctx.hidden_size,
        is_fp8=is_fp8_weights,
        backend_pref=ctx.config.backend,
        auto_threshold=ctx.config.auto_backend_token_threshold,
        device=device,
    )


def _allocate_fp8_scratch(
    ctx: _MoEExecutionContext,
    *,
    device: torch.device,
    num_tokens: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    up_bits_shape = (num_tokens, ctx.input_size)
    down_bits_shape = (num_tokens * ctx.top_k, ctx.hidden_size)
    if prod(up_bits_shape) >= prod(down_bits_shape):
        fp8_up_bits = ctx.workspaces.fp8_bits.get(
            up_bits_shape,
            device=device,
            dtype=torch.uint8,
        )
        fp8_down_bits = ctx.workspaces.fp8_bits.get(
            down_bits_shape,
            device=device,
            dtype=torch.uint8,
        )
    else:
        fp8_down_bits = ctx.workspaces.fp8_bits.get(
            down_bits_shape,
            device=device,
            dtype=torch.uint8,
        )
        fp8_up_bits = ctx.workspaces.fp8_bits.get(
            up_bits_shape,
            device=device,
            dtype=torch.uint8,
        )
    fp8_down_scale = ctx.workspaces.fp8_scale.get(
        (num_tokens * ctx.top_k,),
        device=device,
        dtype=torch.float32,
    )
    fp8_up_scale = ctx.workspaces.fp8_scale.get(
        (num_tokens,),
        device=device,
        dtype=torch.float32,
    )
    return fp8_up_bits, fp8_up_scale, fp8_down_bits, fp8_down_scale


def _prepare_moe_forward(
    ctx: _MoEExecutionContext,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
    persistent_up_out: bool = False,
) -> _PreparedMoEForward:
    if hidden_states.device.type != "cuda":
        raise ValueError("Fused MoE backend only supports CUDA tensors")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Fused MoE backend requires fp16/bf16/fp32 inputs")
    if up_weight.device != hidden_states.device:
        raise ValueError("Expert weights must be on the same device as inputs")
    if down_weight.device != hidden_states.device:
        raise ValueError("Output expert weights must be on the input device")
    if topk_weights.dtype != hidden_states.dtype:
        raise ValueError("Top-k weights must match hidden state dtype")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be an integer tensor")

    up_weight_kernel, up_is_fp8w = _normalize_expert_weight(up_weight)
    down_weight_kernel, down_is_fp8w = _normalize_expert_weight(down_weight)
    if up_is_fp8w != down_is_fp8w:
        raise ValueError("Up and down expert weights must use the same dtype scheme")

    if up_is_fp8w:
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("FP8-weight MoE currently requires bfloat16 hidden states")
        if up_scale is None or down_scale is None:
            raise ValueError("FP8-weight experts must define scale tensors")
        if up_scale.device != hidden_states.device:
            raise ValueError("Up expert scales must be on the same device as inputs")
        if down_scale.device != hidden_states.device:
            raise ValueError("Down expert scales must be on the same device as inputs")
    else:
        if up_weight_kernel.dtype != hidden_states.dtype:
            raise ValueError("Expert weights must match hidden state dtype")
        if down_weight_kernel.dtype != hidden_states.dtype:
            raise ValueError("Output expert weights must match hidden state dtype")

    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous().to(torch.int32)

    num_tokens = hidden_states.size(0)
    block_size_m = _get_block_m_for_routing(
        ctx,
        device=hidden_states.device,
        num_tokens=num_tokens,
        is_fp8_weights=up_is_fp8w,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        ctx.num_experts,
    )

    if persistent_up_out:
        up_out = torch.empty(
            (num_tokens, ctx.top_k, ctx.hidden_size * 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        up_out = ctx.workspaces.up.get(
            (num_tokens, ctx.top_k, ctx.hidden_size * 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    fp8_up_bits = None
    fp8_up_scale = None
    fp8_down_bits = None
    fp8_down_scale = None
    if up_is_fp8w:
        (
            fp8_up_bits,
            fp8_up_scale,
            fp8_down_bits,
            fp8_down_scale,
        ) = _allocate_fp8_scratch(
            ctx,
            device=hidden_states.device,
            num_tokens=num_tokens,
        )

    return _PreparedMoEForward(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        up_weight=up_weight_kernel,
        down_weight=down_weight_kernel,
        up_scale=up_scale,
        down_scale=down_scale,
        num_tokens=num_tokens,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        up_out=up_out,
        fp8_up_bits=fp8_up_bits,
        fp8_up_scale=fp8_up_scale,
        fp8_down_bits=fp8_down_bits,
        fp8_down_scale=fp8_down_scale,
    )


def _prepare_single_lora_routing(
    ctx: _MoEExecutionContext,
    prepared: _PreparedMoEForward,
) -> _PreparedSingleLoraRouting:
    lora_block_m = _to_power_of_2(prepared.block_size_m)
    if lora_block_m == prepared.block_size_m:
        return _PreparedSingleLoraRouting(
            sorted_token_ids=prepared.sorted_token_ids,
            expert_ids=prepared.expert_ids,
            num_tokens_post_padded=prepared.num_tokens_post_padded,
            block_size_m=lora_block_m,
        )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        prepared.topk_ids,
        lora_block_m,
        ctx.num_experts,
    )
    return _PreparedSingleLoraRouting(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size_m=lora_block_m,
    )


def _apply_single_lora(
    ctx: _MoEExecutionContext,
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    routing: _PreparedSingleLoraRouting,
    lora_id: int,
    top_k: int,
    mul_routed_weight: bool,
    shrink_config: dict[str, int] | None,
    expand_config: dict[str, int] | None,
) -> None:
    apply_moe_lora_single(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        lora_a=lora_a,
        lora_b=lora_b,
        sorted_token_ids=routing.sorted_token_ids,
        expert_ids=routing.expert_ids,
        num_tokens_post_padded=routing.num_tokens_post_padded,
        lora_id=lora_id,
        top_k=top_k,
        num_experts=ctx.num_experts,
        block_size_m=routing.block_size_m,
        mul_routed_weight=mul_routed_weight,
        shrink_config=shrink_config,
        expand_config=expand_config,
    )


def _invoke_fused_moe_kernel(
    ctx: _MoEExecutionContext,
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor | None,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    a_fp8_bits: torch.Tensor | None = None,
    a_fp8_scale: torch.Tensor | None = None,
) -> None:
    _MOE.invoke_moe_gemm(
        A, B, C,
        B_scale=B_scale,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
        top_k=top_k,
        num_experts=ctx.num_experts,
        input_size=ctx.input_size,
        hidden_size=ctx.hidden_size,
        allow_tf32=ctx.config.allow_tf32,
        backend_pref=ctx.config.backend,
        auto_threshold=ctx.config.auto_backend_token_threshold,
        a_fp8_bits=a_fp8_bits,
        a_fp8_scale=a_fp8_scale,
    )


def _run_base_up(
    ctx: _MoEExecutionContext,
    prepared: _PreparedMoEForward,
) -> None:
    _invoke_fused_moe_kernel(
        ctx,
        prepared.hidden_states,
        prepared.up_weight,
        prepared.up_scale,
        prepared.up_out,
        topk_weights=None,
        sorted_token_ids=prepared.sorted_token_ids,
        expert_ids=prepared.expert_ids,
        num_tokens_post_padded=prepared.num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=ctx.top_k,
        a_fp8_bits=prepared.fp8_up_bits,
        a_fp8_scale=prepared.fp8_up_scale,
    )


def _run_moe_activation(
    ctx: _MoEExecutionContext,
    prepared: _PreparedMoEForward,
) -> torch.Tensor:
    activation_in = prepared.up_out.view(prepared.num_tokens * ctx.top_k, -1)
    activation_out = ctx.workspaces.activation.get(
        (prepared.num_tokens * ctx.top_k, ctx.hidden_size),
        device=prepared.hidden_states.device,
        dtype=prepared.hidden_states.dtype,
    )
    if activation_in.dtype != torch.bfloat16:
        raise ValueError(f"MoE activation expects bfloat16, got {activation_in.dtype}")
    _MOE.gelu_residual_cute(activation_out, activation_in)
    return activation_out


def _run_base_down(
    ctx: _MoEExecutionContext,
    prepared: _PreparedMoEForward,
    down_in: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    down_out = ctx.workspaces.down.get(
        (prepared.num_tokens, ctx.top_k, ctx.input_size),
        device=prepared.hidden_states.device,
        dtype=prepared.hidden_states.dtype,
    )
    _invoke_fused_moe_kernel(
        ctx,
        down_in,
        prepared.down_weight,
        prepared.down_scale,
        down_out,
        topk_weights=topk_weights,
        sorted_token_ids=prepared.sorted_token_ids,
        expert_ids=prepared.expert_ids,
        num_tokens_post_padded=prepared.num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        a_fp8_bits=prepared.fp8_down_bits,
        a_fp8_scale=prepared.fp8_down_scale,
    )
    return down_out


def _sum_moe_outputs(
    ctx: _MoEExecutionContext,
    prepared: _PreparedMoEForward,
    down_out: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    fused = output
    if fused is None:
        fused = ctx.workspaces.output.get(
            (prepared.num_tokens, ctx.input_size),
            device=prepared.hidden_states.device,
            dtype=prepared.hidden_states.dtype,
        )
    _MOE.moe_sum(down_out, fused)
    return fused


class FusedMoEModule(nn.Module):
    """Hybrid MoE backend that wraps vLLM's fused kernels for single-GPU use."""

    def __init__(
        self,
        up_experts: torch.nn.Module,
        down_experts: torch.nn.Module,
        *,
        top_k: int,
        hidden_size: int,
        input_size: int,
        num_experts: int,
        config: FusedMoEConfig | None = None,
    ) -> None:
        super().__init__()
        self.up_experts = up_experts
        self.down_experts = down_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_experts = num_experts
        self.config = config or FusedMoEConfig()
        self._lora_inputs_event = torch.cuda.Event(enable_timing=False)
        self._lora_activation_event = torch.cuda.Event(enable_timing=False)
        self._lora_up_event = torch.cuda.Event(enable_timing=False)
        self._lora_down_event = torch.cuda.Event(enable_timing=False)

    @property
    def _workspaces(self) -> _MoEWorkspaces:
        """Return the shared MoE workspaces."""
        return get_shared_moe_workspaces()

    def _compute_lora_routing(
        self,
        *,
        topk_ids: torch.Tensor,
        lora_slot_ids: torch.Tensor,
        block_size_m: int,
        lora_workspace: MoELoRALayerWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_lora_mapping = (lora_slot_ids - 1).to(torch.int32)
        max_loras = lora_workspace.up_a.shape[0] // self.num_experts
        return moe_lora_align_block_size(
            topk_ids.to(torch.int32),
            token_lora_mapping,
            block_size_m,
            self.num_experts,
            max_loras,
        )

    def _run_lora_up(
        self,
        *,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace,
        sorted_lora: torch.Tensor,
        expert_ids_lora: torch.Tensor,
        num_tokens_lora: torch.Tensor,
        block_size_m: int,
        shrink_config: dict[str, int] | None,
        expand_config: dict[str, int] | None,
    ) -> None:
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_workspace.up_a,
            lora_b=lora_workspace.up_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=self.top_k,
            num_experts=self.num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=False,
            shrink_config=shrink_config,
            expand_config=expand_config,
        )

    def _run_lora_down(
        self,
        *,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace,
        sorted_lora: torch.Tensor,
        expert_ids_lora: torch.Tensor,
        num_tokens_lora: torch.Tensor,
        block_size_m: int,
        shrink_config: dict[str, int] | None,
        expand_config: dict[str, int] | None,
    ) -> None:
        apply_moe_lora_batched(
            x=x,
            topk_weights=topk_weights,
            output=output,
            lora_a=lora_workspace.down_a,
            lora_b=lora_workspace.down_b,
            sorted_token_ids=sorted_lora,
            expert_ids=expert_ids_lora,
            num_tokens_post_padded=num_tokens_lora,
            top_k=1,  # Input is already per-expert [num_tokens * top_k, dim]
            num_experts=self.num_experts,
            block_size_m=block_size_m,
            mul_routed_weight=True,
            shrink_config=shrink_config,
            expand_config=expand_config,
        )

    @torch_compiler_disable()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace | None = None,
        lora_slot_ids: torch.Tensor | None = None,
        single_lora_id: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.size(0) == 0:
            return hidden_states

        ctx = _make_execution_context(self)
        prepared = _prepare_moe_forward(
            ctx,
            hidden_states,
            topk_weights,
            topk_ids,
            up_weight=self.up_experts.weight,
            down_weight=self.down_experts.weight,
            up_scale=getattr(self.up_experts, "scale", None),
            down_scale=getattr(self.down_experts, "scale", None),
        )

        # LoRA handling: dispatch based on call-local mode
        #
        # Single-LoRA mode (prefill): Use standard moe_align_block_size routing
        # with apply_moe_lora_single. No Z dimension overhead.
        #
        # Batched mode (decode): Use moe_lora_align_block_size for per-LoRA
        # routing with apply_moe_lora_batched. Handles mixed LoRA batches.
        #
        # The workspace stores weights as [max_loras * num_experts, rank, dim]
        # where max_loras = max_slots - 1 (slot 0 excluded).
        use_single_lora = (
            lora_workspace is not None
            and lora_slot_ids is not None
            and single_lora_id is not None
        )
        use_batched_lora = (
            lora_workspace is not None
            and lora_slot_ids is not None
            and single_lora_id is None
        )
        lora_stream = lora_workspace.stream if use_batched_lora else None
        compute_stream = torch.cuda.current_stream()
        use_lora_stream = lora_stream is not None and lora_stream != compute_stream

        is_prefill = use_single_lora
        lora_shrink_cfg = (
            self.config.lora_prefill_shrink
            if is_prefill
            else self.config.lora_decode_shrink
        )
        lora_expand_cfg = (
            self.config.lora_prefill_expand
            if is_prefill
            else self.config.lora_decode_expand
        )

        # For batched mode, prepare per-LoRA routing once (reused for up and down)
        sorted_lora = None
        expert_ids_lora = None
        num_tokens_lora = None

        lora_block_m = _to_power_of_2(prepared.block_size_m)
        lora_routing = None
        if use_single_lora:
            lora_routing = _prepare_single_lora_routing(ctx, prepared)

        # Launch batched LoRA up before base MoE so it can overlap if we have a
        # dedicated LoRA stream. Always compute into the LoRA buffer and add
        # into the base output after the fused kernel runs.
        lora_up_out = None
        if use_batched_lora:
            target_stream = lora_stream if use_lora_stream else compute_stream
            if use_lora_stream:
                self._lora_inputs_event.record(compute_stream)
            with torch.cuda.stream(target_stream):
                if use_lora_stream:
                    target_stream.wait_event(self._lora_inputs_event)
                sorted_lora, expert_ids_lora, num_tokens_lora = self._compute_lora_routing(
                    topk_ids=prepared.topk_ids,
                    lora_slot_ids=lora_slot_ids,
                    block_size_m=lora_block_m,
                    lora_workspace=lora_workspace,
                )
                lora_up_out = ctx.workspaces.lora_up.get(
                    (prepared.num_tokens, self.top_k, self.hidden_size * 2),
                    device=prepared.hidden_states.device,
                    dtype=prepared.hidden_states.dtype,
                )
                lora_up_out.zero_()
                self._run_lora_up(
                    x=prepared.hidden_states,
                    topk_weights=prepared.topk_weights,
                    output=lora_up_out,
                    lora_workspace=lora_workspace,
                    sorted_lora=sorted_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_lora=num_tokens_lora,
                    block_size_m=lora_block_m,
                    shrink_config=lora_shrink_cfg,
                    expand_config=lora_expand_cfg,
                )
                if use_lora_stream:
                    self._lora_up_event.record(target_stream)

        _run_base_up(ctx, prepared)

        if use_batched_lora:
            if use_lora_stream:
                compute_stream.wait_event(self._lora_up_event)
            if lora_up_out is not None:
                prepared.up_out.add_(lora_up_out)

        if use_single_lora:
            lora_up_out = ctx.workspaces.lora_up.get(
                (prepared.num_tokens, self.top_k, self.hidden_size * 2),
                device=prepared.hidden_states.device,
                dtype=prepared.hidden_states.dtype,
            )
            lora_up_out.zero_()
            _apply_single_lora(
                ctx,
                x=prepared.hidden_states,
                topk_ids=prepared.topk_ids,
                topk_weights=prepared.topk_weights,
                output=lora_up_out,
                lora_a=lora_workspace.up_a,
                lora_b=lora_workspace.up_b,
                routing=lora_routing,
                lora_id=single_lora_id,
                top_k=self.top_k,
                mul_routed_weight=False,
                shrink_config=lora_shrink_cfg,
                expand_config=lora_expand_cfg,
            )
            prepared.up_out.add_(lora_up_out)

        down_in = _run_moe_activation(ctx, prepared)
        lora_down_out = None
        if use_batched_lora:
            target_stream = lora_stream if use_lora_stream else compute_stream
            if use_lora_stream:
                self._lora_activation_event.record(compute_stream)
            with torch.cuda.stream(target_stream):
                if use_lora_stream:
                    target_stream.wait_event(self._lora_activation_event)
                lora_down_out = ctx.workspaces.lora_down.get(
                    (prepared.num_tokens, self.top_k, self.input_size),
                    device=prepared.hidden_states.device,
                    dtype=prepared.hidden_states.dtype,
                )
                lora_down_out.zero_()
                self._run_lora_down(
                    x=down_in,
                    topk_weights=prepared.topk_weights,
                    output=lora_down_out,
                    lora_workspace=lora_workspace,
                    sorted_lora=sorted_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_lora=num_tokens_lora,
                    block_size_m=lora_block_m,
                    shrink_config=lora_shrink_cfg,
                    expand_config=lora_expand_cfg,
                )
                if use_lora_stream:
                    self._lora_down_event.record(target_stream)

        down_out = _run_base_down(
            ctx,
            prepared,
            down_in,
            topk_weights=prepared.topk_weights.view(-1),
        )

        if use_batched_lora:
            if use_lora_stream:
                compute_stream.wait_event(self._lora_down_event)
            if lora_down_out is not None:
                down_out.add_(lora_down_out)

        if use_single_lora:
            lora_down_out = ctx.workspaces.lora_down.get(
                (prepared.num_tokens, self.top_k, self.input_size),
                device=prepared.hidden_states.device,
                dtype=prepared.hidden_states.dtype,
            )
            lora_down_out.zero_()
            _apply_single_lora(
                ctx,
                x=down_in,
                topk_ids=prepared.topk_ids,
                topk_weights=prepared.topk_weights,
                output=lora_down_out,
                lora_a=lora_workspace.down_a,
                lora_b=lora_workspace.down_b,
                routing=lora_routing,
                lora_id=single_lora_id,
                top_k=1,
                mul_routed_weight=True,
                shrink_config=lora_shrink_cfg,
                expand_config=lora_expand_cfg,
            )
            down_out.add_(lora_down_out)

        return _sum_moe_outputs(ctx, prepared, down_out)

def _normalize_expert_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if weight.dtype == torch.uint8:
        return weight, True
    if weight.dtype == torch.float8_e4m3fn:
        return weight.view(torch.uint8), True
    return weight, False
