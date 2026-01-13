
from dataclasses import dataclass, field
from math import prod
from typing import Literal

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from .kernels import (
    dtype_to_triton,
    invoke_fused_moe_kernel as invoke_fused_moe_kernel_triton,
    invoke_fused_moe_kernel_fp8_w8a8 as invoke_fused_moe_kernel_triton_fp8,
)
from .lora_kernels import apply_moe_lora_batched, apply_moe_lora_single
from .routing import moe_align_block_size, moe_lora_align_block_size


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

from kestrel.moondream.lora_workspace import MoELoRALayerWorkspace
from kestrel.utils.buffers import FixedBuffer
from kestrel_kernels.gelu_residual import gelu_residual_cute
from kestrel_kernels.moe_sum import moe_sum as moe_sum_cuda
from kestrel_kernels.cute_moe import (
    get_cute_moe_block_m,
    get_cute_moe_config,
    invoke_cute_moe_down,
    invoke_cute_moe_down_fp8,
    invoke_cute_moe_up,
    invoke_cute_moe_up_fp8,
)


_HARDCODED_CONFIGS: dict[tuple[int, int], dict[int, dict[str, int]]] = {
    (
        64,
        1024,
    ): {
        1: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        },
        2: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 4,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        },
        8: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
        16: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 5,
        },
        24: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        },
        32: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        48: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        64: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        96: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        128: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        256: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        512: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 3,
        },
        1024: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        },
        1536: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        },
        2048: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        },
        3072: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4,
        },
        4096: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        },
    }
}


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
        self._tuned_configs: dict[int, dict[str, int] | None] = {}
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

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace | None = None,
        lora_slot_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(hidden_states, topk_weights, topk_ids, lora_workspace, lora_slot_ids)

    @torch_compiler_disable()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace | None = None,
        lora_slot_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise ValueError("Fused MoE backend only supports CUDA tensors")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Fused MoE backend requires fp16/bf16/fp32 inputs")
        if self.up_experts.weight.device != hidden_states.device:
            raise ValueError("Expert weights must be on the same device as inputs")
        if self.down_experts.weight.device != hidden_states.device:
            raise ValueError("Output expert weights must be on the input device")
        if topk_weights.dtype != hidden_states.dtype:
            raise ValueError("Top-k weights must match hidden state dtype")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_ids must be an integer tensor")

        up_is_fp8w = self.up_experts.weight.dtype == torch.uint8
        down_is_fp8w = self.down_experts.weight.dtype == torch.uint8
        if up_is_fp8w != down_is_fp8w:
            raise ValueError("Up and down expert weights must use the same dtype scheme")

        if up_is_fp8w:
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("FP8-weight MoE currently requires bfloat16 hidden states")
            if not hasattr(self.up_experts, "scale") or not hasattr(self.down_experts, "scale"):
                raise ValueError("FP8-weight experts must define a `scale` tensor")
            if self.up_experts.scale.device != hidden_states.device:
                raise ValueError("Up expert scales must be on the same device as inputs")
            if self.down_experts.scale.device != hidden_states.device:
                raise ValueError("Down expert scales must be on the same device as inputs")
        else:
            if self.up_experts.weight.dtype != hidden_states.dtype:
                raise ValueError("Expert weights must match hidden state dtype")
            if self.down_experts.weight.dtype != hidden_states.dtype:
                raise ValueError("Output expert weights must match hidden state dtype")

        hidden_states = hidden_states.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()

        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states

        assignments = num_tokens * self.top_k
        triton_config = self._get_triton_config(
            num_tokens=num_tokens,
            assignments=assignments,
            dtype=hidden_states.dtype,
        )
        block_size_m = self._get_block_m_for_routing(
            num_tokens=num_tokens,
            is_fp8_weights=up_is_fp8w,
            triton_block_m=triton_config["BLOCK_SIZE_M"],
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, self.num_experts
        )

        up_out = self._workspaces.up.get(
            (num_tokens, self.top_k, self.hidden_size * 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Pre-allocate FP8 activation buffers once for both up and down projections.
        # Up projection: shape (num_tokens, input_size)
        # Down projection: shape (num_tokens * top_k, hidden_size)
        # We allocate them here to avoid repeated workspace.get() calls in the hot path.
        fp8_up_bits: torch.Tensor | None = None
        fp8_up_scale: torch.Tensor | None = None
        fp8_down_bits: torch.Tensor | None = None
        fp8_down_scale: torch.Tensor | None = None
        if up_is_fp8w:
            # W8A8: need FP8 activation buffers for both warp and WGMMA kernels
            fp8_up_bits = self._workspaces.fp8_bits.get(
                (num_tokens, self.input_size),
                device=hidden_states.device,
                dtype=torch.uint8,
            )
            fp8_up_scale = self._workspaces.fp8_scale.get(
                (num_tokens,),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            fp8_down_bits = self._workspaces.fp8_bits.get(
                (num_tokens * self.top_k, self.hidden_size),
                device=hidden_states.device,
                dtype=torch.uint8,
            )
            fp8_down_scale = self._workspaces.fp8_scale.get(
                (num_tokens * self.top_k,),
                device=hidden_states.device,
                dtype=torch.float32,
            )

        compute_type = dtype_to_triton(hidden_states.dtype)

        # LoRA handling: dispatch based on workspace mode
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
            and lora_workspace.single_lora_id is not None
        )
        use_batched_lora = (
            lora_workspace is not None
            and lora_slot_ids is not None
            and lora_workspace.single_lora_id is None
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

        # LoRA kernels use Triton which requires power-of-2 block sizes for tl.arange.
        # CuTe MoE configs can return non-power-of-2 values (e.g., 192 for FP8).
        # Round DOWN to ensure precompiled routing kernels exist (16, 32, 64, 128).
        lora_block_m = _to_power_of_2(block_size_m)

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
                    topk_ids=topk_ids,
                    lora_slot_ids=lora_slot_ids,
                    block_size_m=lora_block_m,  # Power-of-2 for Triton kernels
                    lora_workspace=lora_workspace,
                )
                lora_up_out = self._workspaces.lora_up.get(
                    (num_tokens, self.top_k, self.hidden_size * 2),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                lora_up_out.zero_()
                self._run_lora_up(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    output=lora_up_out,
                    lora_workspace=lora_workspace,
                    sorted_lora=sorted_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_lora=num_tokens_lora,
                    block_size_m=lora_block_m,  # Power-of-2 for Triton kernels
                    shrink_config=lora_shrink_cfg,
                    expand_config=lora_expand_cfg,
                )
                if use_lora_stream:
                    self._lora_up_event.record(target_stream)

        self._invoke_fused_moe_kernel(
            hidden_states,
            self.up_experts.weight,
            getattr(self.up_experts, "scale", None),
            up_out,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=self.top_k,
            triton_config=triton_config,
            compute_type=compute_type,
            a_fp8_bits=fp8_up_bits,
            a_fp8_scale=fp8_up_scale,
        )

        if use_batched_lora:
            if use_lora_stream:
                compute_stream.wait_event(self._lora_up_event)
            if lora_up_out is not None:
                up_out.add_(lora_up_out)

        if use_single_lora:
            # Single-LoRA path: use separate routing if block_m needs adjustment
            # for Triton's power-of-2 constraint.
            if lora_block_m == block_size_m:
                # Can reuse main MoE routing
                lora_sorted = sorted_token_ids
                lora_expert_ids = expert_ids
                lora_num_tokens = num_tokens_post_padded
            else:
                # Need separate routing with power-of-2 block_m
                lora_sorted, lora_expert_ids, lora_num_tokens = moe_align_block_size(
                    topk_ids, lora_block_m, self.num_experts
                )
            lora_up_out = self._workspaces.lora_up.get(
                (num_tokens, self.top_k, self.hidden_size * 2),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            lora_up_out.zero_()
            apply_moe_lora_single(
                x=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=lora_up_out,
                lora_a=lora_workspace.up_a,
                lora_b=lora_workspace.up_b,
                sorted_token_ids=lora_sorted,
                expert_ids=lora_expert_ids,
                num_tokens_post_padded=lora_num_tokens,
                lora_id=lora_workspace.single_lora_id,
                top_k=self.top_k,
                num_experts=self.num_experts,
                block_size_m=lora_block_m,  # Power-of-2 for Triton kernels
                mul_routed_weight=False,
                shrink_config=lora_shrink_cfg,
                expand_config=lora_expand_cfg,
            )
            up_out.add_(lora_up_out)

        activation_in = up_out.view(num_tokens * self.top_k, -1)
        activation_out = self._workspaces.activation.get(
            (num_tokens * self.top_k, self.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if activation_in.dtype != torch.bfloat16:
            raise ValueError(
                f"gelu_residual_cute only supports bfloat16, got {activation_in.dtype}"
            )
        gelu_residual_cute(activation_out, activation_in)

        down_in = activation_out
        lora_down_out = None
        if use_batched_lora:
            target_stream = lora_stream if use_lora_stream else compute_stream
            if use_lora_stream:
                self._lora_activation_event.record(compute_stream)
            with torch.cuda.stream(target_stream):
                if use_lora_stream:
                    target_stream.wait_event(self._lora_activation_event)
                lora_down_out = self._workspaces.lora_down.get(
                    (num_tokens, self.top_k, self.input_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                lora_down_out.zero_()
                self._run_lora_down(
                    x=down_in,
                    topk_weights=topk_weights,
                    output=lora_down_out,
                    lora_workspace=lora_workspace,
                    sorted_lora=sorted_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_lora=num_tokens_lora,
                    block_size_m=lora_block_m,  # Power-of-2 for Triton kernels
                    shrink_config=lora_shrink_cfg,
                    expand_config=lora_expand_cfg,
                )
                if use_lora_stream:
                    self._lora_down_event.record(target_stream)

        down_out = self._workspaces.down.get(
            (num_tokens, self.top_k, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self._invoke_fused_moe_kernel(
            down_in,
            self.down_experts.weight,
            getattr(self.down_experts, "scale", None),
            down_out,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            triton_config=triton_config,
            compute_type=compute_type,
            a_fp8_bits=fp8_down_bits,
            a_fp8_scale=fp8_down_scale,
        )

        if use_batched_lora:
            if use_lora_stream:
                compute_stream.wait_event(self._lora_down_event)
            if lora_down_out is not None:
                down_out.add_(lora_down_out)

        if use_single_lora:
            # Single-LoRA path for down projection
            # Reuse lora_sorted/lora_expert_ids/lora_num_tokens from up path
            lora_down_out = self._workspaces.lora_down.get(
                (num_tokens, self.top_k, self.input_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            lora_down_out.zero_()
            apply_moe_lora_single(
                x=down_in,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=lora_down_out,
                lora_a=lora_workspace.down_a,
                lora_b=lora_workspace.down_b,
                sorted_token_ids=lora_sorted,
                expert_ids=lora_expert_ids,
                num_tokens_post_padded=lora_num_tokens,
                lora_id=lora_workspace.single_lora_id,
                top_k=1,  # Input is already per-expert [num_tokens * top_k, dim]
                num_experts=self.num_experts,
                block_size_m=lora_block_m,  # Power-of-2 for Triton kernels
                mul_routed_weight=True,
                shrink_config=lora_shrink_cfg,
                expand_config=lora_expand_cfg,
            )
            down_out.add_(lora_down_out)

        fused = self._workspaces.output.get(
            (num_tokens, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        moe_sum_cuda(down_out, fused)
        return fused

    def _quantize_fp8_e4m3fn_rowwise(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-wise FP8(E4M3FN) quantization returning (uint8 bitview, fp32 scale)."""
        if x.dtype != torch.bfloat16:
            raise ValueError(f"FP8 activation quantization expects bf16 (got {x.dtype})")
        if x.ndim != 2:
            raise ValueError("Expected 2D activation matrix")

        # Use a custom CUDA kernel to keep this path CUDA-graph-capturable and avoid
        # per-call allocations.
        from kestrel_kernels.fp8_quant import fp8_e4m3fn_rowwise_quant_cuda

        out_bits = self._workspaces.fp8_bits.get(
            (int(x.shape[0]), int(x.shape[1])),
            device=x.device,
            dtype=torch.uint8,
        )
        out_scale = self._workspaces.fp8_scale.get(
            (int(x.shape[0]),),
            device=x.device,
            dtype=torch.float32,
        )
        fp8_e4m3fn_rowwise_quant_cuda(out_bits, out_scale, x)
        return out_bits, out_scale

    def _quantize_fp8_into(
        self,
        x: torch.Tensor,
        out_bits: torch.Tensor,
        out_scale: torch.Tensor,
    ) -> None:
        """Quantize x into pre-allocated FP8 buffers (fast path, no workspace.get)."""
        from kestrel_kernels.fp8_quant import fp8_e4m3fn_rowwise_quant_cuda

        fp8_e4m3fn_rowwise_quant_cuda(out_bits, out_scale, x)

    def _invoke_fused_moe_kernel(
        self,
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
        triton_config: dict[str, int],
        compute_type,
        a_fp8_bits: torch.Tensor | None = None,
        a_fp8_scale: torch.Tensor | None = None,
    ) -> None:
        b_is_fp8w = B.dtype == torch.uint8
        backend = self._resolve_backend(num_tokens=int(C.shape[0]), is_fp8=b_is_fp8w)

        if b_is_fp8w and backend == "triton":
            # Triton FP8 W8A8 path (SGLang-style): quantize activations per row
            # and do FP8 dot with per-output-channel weight scales.
            if B_scale is None:
                raise ValueError("B_scale is required for FP8-weight MoE")
            if a_fp8_bits is None or a_fp8_scale is None:
                raise ValueError("a_fp8_bits and a_fp8_scale are required for FP8 MoE")

            triton_fp8_config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "NUM_WARPS": 4,
                "NUM_STAGES": 4,
            }
            self._quantize_fp8_into(A, a_fp8_bits, a_fp8_scale)
            invoke_fused_moe_kernel_triton_fp8(
                a_fp8_bits.view(torch.float8_e4m3fn),
                a_fp8_scale,
                B.view(torch.float8_e4m3fn),
                B_scale,
                C,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=mul_routed_weight,
                top_k=top_k,
                config=triton_fp8_config,
                compute_type=compute_type,
            )
            return

        if backend == "cute":
            if b_is_fp8w:
                if B_scale is None:
                    raise ValueError("B_scale is required for FP8-weight MoE")
                if a_fp8_bits is None or a_fp8_scale is None:
                    raise ValueError("a_fp8_bits and a_fp8_scale are required for FP8 MoE")

                # W8A8 (FP8 activations + FP8 weights) via SM90 WGMMA.
                self._quantize_fp8_into(A, a_fp8_bits, a_fp8_scale)
                if mul_routed_weight:
                    if int(top_k) != 1:
                        raise ValueError("CuTe fp8 moe_down expects top_k=1")
                    if topk_weights is None:
                        raise ValueError("topk_weights is required when mul_routed_weight=True")
                    invoke_cute_moe_down_fp8(
                        a_fp8_bits,
                        a_fp8_scale,
                        B,
                        B_scale,
                        C,
                        topk_weights=topk_weights,
                        sorted_token_ids=sorted_token_ids,
                        expert_ids=expert_ids,
                        num_tokens_post_padded=num_tokens_post_padded,
                    )
                else:
                    if int(top_k) != 8:
                        raise ValueError("CuTe fp8 moe_up expects top_k=8")
                    invoke_cute_moe_up_fp8(
                        a_fp8_bits,
                        a_fp8_scale,
                        B,
                        B_scale,
                        C,
                        sorted_token_ids=sorted_token_ids,
                        expert_ids=expert_ids,
                        num_tokens_post_padded=num_tokens_post_padded,
                    )
                return

            # Unquantized BF16 CuTe path: use auto-config selection.
            if mul_routed_weight:
                if int(top_k) != 1:
                    raise ValueError("CuTe moe_down expects top_k=1")
                if topk_weights is None:
                    raise ValueError("topk_weights is required when mul_routed_weight=True")
                invoke_cute_moe_down(
                    A,
                    B,
                    C,
                    topk_weights=topk_weights,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    # config=None triggers auto-select from JSON configs
                )
            else:
                if int(top_k) != 8:
                    raise ValueError("CuTe moe_up expects top_k=8")
                invoke_cute_moe_up(
                    A,
                    B,
                    C,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    # config=None triggers auto-select from JSON configs
                )
            return

        if b_is_fp8w:
            # Safe-but-slower fallback: dequantize weights to match the Triton kernel.
            if B_scale is None:
                raise ValueError("B_scale is required for FP8-weight MoE")
            B_dequant = B.view(torch.float8_e4m3fn).to(A.dtype) * B_scale.to(A.dtype).unsqueeze(-1)
            B = B_dequant

        invoke_fused_moe_kernel_triton(
            A,
            B,
            C,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight,
            top_k=top_k,
            config=triton_config,
            compute_type=compute_type,
            bias=None,
            allow_tf32=self.config.allow_tf32,
        )

    def _select_block_size(self, assignments: int) -> int:
        block_m = self.config.block_size_m
        if assignments <= 16:
            return min(16, block_m)
        if assignments <= 32:
            return min(32, block_m)
        if assignments <= 64:
            return min(64, block_m)
        return block_m

    def _get_triton_config(
        self,
        *,
        num_tokens: int,
        assignments: int,
        dtype: torch.dtype,
    ) -> dict[str, int]:
        base = self.config.as_triton(
            block_size_m=self._select_block_size(assignments)
        ).copy()
        base.setdefault("NUM_WARPS", self.config.num_warps)
        base.setdefault("NUM_STAGES", self.config.num_stages)

        tuned = self._get_tuned_config(num_tokens=num_tokens, dtype=dtype)
        if tuned is not None:
            base.update(tuned)
        return base

    def _get_tuned_config(
        self,
        *,
        num_tokens: int,
        dtype: torch.dtype,
    ) -> dict[str, int] | None:
        if num_tokens in self._tuned_configs:
            return self._tuned_configs[num_tokens]

        key = (
            self.down_experts.weight.shape[0],
            self.down_experts.weight.shape[2],
        )
        hardcoded = _HARDCODED_CONFIGS.get(key)
        if not hardcoded:
            raise ValueError(
                f"No hardcoded MoE config for expert shape {key}. "
                "Add an entry to _HARDCODED_CONFIGS."
            )
        nearest = min(hardcoded.keys(), key=lambda m: abs(m - num_tokens))
        raw = hardcoded[nearest]
        tuned = {k.upper(): int(v) for k, v in raw.items()}

        self._tuned_configs[num_tokens] = tuned
        return tuned

    def _get_block_m_for_routing(
        self,
        *,
        num_tokens: int,
        is_fp8_weights: bool,
        triton_block_m: int,
    ) -> int:
        """Get block_m for moe_align_block_size based on backend.

        For both FP8 and BF16 weights with CuTe backend, use auto-selected config.
        For Triton backend, use Triton block_m.
        """
        backend = self._resolve_backend(num_tokens=num_tokens, is_fp8=is_fp8_weights)
        if backend == "cute":
            return get_cute_moe_block_m(
                num_tokens,
                num_experts=self.num_experts,
                hidden_size=self.input_size,
                intermediate_size=self.hidden_size,
                dtype="fp8" if is_fp8_weights else "bf16",
            )
        else:
            return triton_block_m

    def _resolve_backend(self, *, num_tokens: int, is_fp8: bool = False) -> str:
        # FP8 always uses CuTe backend (tuned configs available for all batch sizes)
        if is_fp8:
            return "cute"
        backend = self.config.backend.lower()
        if backend == "auto":
            return "triton" if num_tokens >= self.config.auto_backend_token_threshold else "cute"
        return backend

