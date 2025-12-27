
from dataclasses import dataclass
from math import prod

import os
import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from .kernels import dtype_to_triton, invoke_fused_moe_kernel as invoke_fused_moe_kernel_triton
from .lora_kernels import apply_moe_lora
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from .routing import moe_align_block_size

from kestrel.moondream.lora_workspace import MoELoRALayerWorkspace
from kestrel_kernels.activation import gelu_residual_cuda
from kestrel_kernels.moe_sum import moe_sum as moe_sum_cuda


class _ResizableBuffer:
    """Device-aware buffer that grows as needed and reuses storage."""

    def __init__(self) -> None:
        self._tensor: torch.Tensor | None = None

    def get(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = prod(shape)
        if numel == 0:
            return torch.empty(shape, device=device, dtype=dtype)

        if (
            self._tensor is None
            or self._tensor.numel() < numel
            or self._tensor.device != device
            or self._tensor.dtype != dtype
        ):
            self._tensor = torch.empty(numel, device=device, dtype=dtype)
        return self._tensor[:numel].view(*shape)


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
        self.up = _ResizableBuffer()
        self.down = _ResizableBuffer()
        self.output = _ResizableBuffer()
        self.activation = _ResizableBuffer()


@dataclass
class FusedMoEConfig:
    block_size_m: int = 16
    block_size_n: int = 64
    block_size_k: int = 32
    group_size_m: int = 8
    num_warps: int = 4
    num_stages: int = 2
    allow_tf32: bool = True
    backend: str = "triton"  # "triton" | "cute" | "auto"

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
        self._workspaces = _MoEWorkspaces()
        self._tuned_configs: dict[int, dict[str, int] | None] = {}

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
        if self.up_experts.weight.dtype != hidden_states.dtype:
            raise ValueError("Expert weights must match hidden state dtype")
        if self.down_experts.weight.device != hidden_states.device:
            raise ValueError("Output expert weights must be on the input device")
        if self.down_experts.weight.dtype != hidden_states.dtype:
            raise ValueError("Output expert weights must match hidden state dtype")
        if topk_weights.dtype != hidden_states.dtype:
            raise ValueError("Top-k weights must match hidden state dtype")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_ids must be an integer tensor")

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
        block_size_m = triton_config["BLOCK_SIZE_M"]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, self.num_experts
        )

        up_out = self._workspaces.up.get(
            (num_tokens, self.top_k, self.hidden_size * 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        compute_type = dtype_to_triton(hidden_states.dtype)

        self._invoke_fused_moe_kernel(
            hidden_states,
            self.up_experts.weight,
            up_out,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=self.top_k,
            triton_config=triton_config,
            compute_type=compute_type,
        )

        # For LoRA, run separate routing with super-expert IDs.
        # This keeps base MoE compute unchanged while enabling mixed-adapter batches.
        # Compute routing once here and reuse for both up and down LoRA.
        #
        # Super-expert mapping uses sentinel-based slot 0 filtering:
        # - Slot 0 (no LoRA): Set to sentinel value >= max_super_experts, which
        #   causes moe_align_block_size to skip these tokens (expert_id >= num_experts
        #   are ignored in the kernel). These tokens get no LoRA contribution.
        # - Slot N (N >= 1): Maps to super-expert (N-1)*num_experts + expert_id
        #
        # This allows the MoE workspace to exclude slot 0 entirely, reducing
        # max_super_experts from max_slots*num_experts to (max_slots-1)*num_experts.
        # With 64 experts, this keeps us under vLLM's 1024 super-expert limit
        # (floor(1023/64) = 15 usable adapter slots).
        sorted_lora = None
        expert_ids_lora = None
        num_tokens_lora = None
        combined_topk_ids = None
        if lora_workspace is not None and lora_slot_ids is not None:
            # Expand slot IDs: [M] -> [M, top_k] to match topk_ids shape
            expanded_slots = lora_slot_ids.unsqueeze(1).expand(-1, self.top_k)

            # Sentinel-based mapping: slot 0 -> sentinel (filtered), slot N -> (N-1)*E + expert
            max_super_experts = lora_workspace.up_a.shape[0]
            sentinel = max_super_experts  # Any value >= max_super_experts will be filtered
            combined_topk_ids = torch.where(
                expanded_slots > 0,
                topk_ids + (expanded_slots - 1) * self.num_experts,
                sentinel,
            ).to(torch.int32)

            sorted_lora, expert_ids_lora, num_tokens_lora = moe_align_block_size(
                combined_topk_ids, block_size_m, max_super_experts
            )

            apply_moe_lora(
                x=hidden_states,
                topk_ids=combined_topk_ids,
                topk_weights=topk_weights,
                output=up_out,
                lora_a=lora_workspace.up_a,
                lora_b=lora_workspace.up_b,
                sorted_token_ids=sorted_lora,
                expert_ids=expert_ids_lora,
                num_tokens_post_padded=num_tokens_lora,
                top_k=self.top_k,
                config=triton_config,
                mul_routed_weight=False,
            )

        activation_in = up_out.view(num_tokens * self.top_k, -1)
        activation_out = self._workspaces.activation.get(
            (num_tokens * self.top_k, self.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if activation_in.dtype != torch.bfloat16:
            raise ValueError(
                f"gelu_residual_cuda only supports bfloat16, got {activation_in.dtype}"
            )
        gelu_residual_cuda(activation_out, activation_in)

        down_in = activation_out
        down_out = self._workspaces.down.get(
            (num_tokens, self.top_k, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self._invoke_fused_moe_kernel(
            down_in,
            self.down_experts.weight,
            down_out,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            triton_config=triton_config,
            compute_type=compute_type,
        )

        if lora_workspace is not None and lora_slot_ids is not None:
            # For down projection, input is already per-expert [M * top_k, dim]
            # Reuse the same super-expert routing computed for up projection
            apply_moe_lora(
                x=down_in,
                topk_ids=combined_topk_ids,
                topk_weights=topk_weights,
                output=down_out,
                lora_a=lora_workspace.down_a,
                lora_b=lora_workspace.down_b,
                sorted_token_ids=sorted_lora,
                expert_ids=expert_ids_lora,
                num_tokens_post_padded=num_tokens_lora,
                top_k=1,  # Input is already per-expert [num_tokens * top_k, dim]
                config=triton_config,
                mul_routed_weight=True,
            )

        fused = self._workspaces.output.get(
            (num_tokens, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        moe_sum_cuda(down_out, fused)
        return fused

    def _invoke_fused_moe_kernel(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
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
    ) -> None:
        # Prefer explicit config flag, but allow quick env-var override.
        backend = os.getenv("KESTREL_FUSED_MOE_BACKEND", self.config.backend).lower()
        use_cute = backend in ("cute", "auto")

        if use_cute:
            try:
                from kestrel.ops.fused_moe_cute import (
                    FusedMoeCuTeConfig,
                    invoke_fused_moe_kernel_cute,
                )

                # Current CuTe kernel is tuned for the small-M decode regime where the routing
                # block size is 16. For other block sizes we fall back to Triton.
                block_m = triton_config["BLOCK_SIZE_M"]
                if block_m == 16:
                    # Allow CuTe-specific tuning for the decode regime.
                    # We keep correctness identical to Triton but may pick different tiles.
                    num_tokens = int(C.shape[0])
                    if num_tokens <= 8:
                        # Tiny decode batches (B~4, T~1) are extremely padding-heavy; we
                        # use CuTe-specific tiles tuned for H100 decode.
                        #
                        # NOTE: Up-proj and down-proj have different sweet spots:
                        # - Up: smaller BK helps shared memory footprint; 1 stage is enough.
                        # - Down: larger BK reduces K tiles and is faster even at 1 CTA/SM.
                        if mul_routed_weight:
                            cute_cfg = FusedMoeCuTeConfig(
                                block_m=16,
                                block_n=64,
                                block_k=256,
                                num_warps=4,
                                num_stages=3,
                            )
                        else:
                            cute_cfg = FusedMoeCuTeConfig(
                                block_m=16,
                                block_n=128,
                                block_k=128,
                                num_warps=2,
                                num_stages=1,
                            )
                    else:
                        # For down-proj (mul_routed_weight) we keep the Triton tiling but use
                        # fewer stages to reduce shared-memory pressure.
                        num_stages = int(triton_config["NUM_STAGES"])
                        if mul_routed_weight:
                            num_stages = min(2, num_stages)
                        cute_cfg = FusedMoeCuTeConfig(
                            block_m=block_m,
                            block_n=int(triton_config["BLOCK_SIZE_N"]),
                            block_k=int(triton_config["BLOCK_SIZE_K"]),
                            num_warps=int(triton_config["NUM_WARPS"]),
                            num_stages=num_stages,
                        )
                    invoke_fused_moe_kernel_cute(
                        A,
                        B,
                        C,
                        topk_weights=topk_weights,
                        sorted_token_ids=sorted_token_ids,
                        expert_ids=expert_ids,
                        num_tokens_post_padded=num_tokens_post_padded,
                        mul_routed_weight=mul_routed_weight,
                        top_k=top_k,
                        config=cute_cfg,
                    )
                    return
            except Exception:
                # Fall back to Triton if CuTe is unavailable (or config unsupported).
                pass

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

        hardcoded = _HARDCODED_CONFIGS.get(
            (
                self.down_experts.weight.shape[0],
                self.down_experts.weight.shape[2],
            )
        )
        if hardcoded:
            nearest = min(hardcoded.keys(), key=lambda m: abs(m - num_tokens))
            raw = hardcoded[nearest]
        else:
            config_name = FUSED_MOE_UNQUANTIZED_CONFIG.config_name(dtype)
            raw = try_get_optimal_moe_config(
                self.up_experts.weight.shape,
                self.down_experts.weight.shape,
                self.top_k,
                config_name,
                num_tokens,
            )
        tuned = {k.upper(): int(v) for k, v in raw.items()} if raw is not None else None

        self._tuned_configs[num_tokens] = tuned
        return tuned
