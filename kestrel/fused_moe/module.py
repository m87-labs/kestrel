
from dataclasses import dataclass
from math import prod

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable
from vllm import _custom_ops as ops

from .kernels import (
    dtype_to_triton,
    fused_gelu_and_mul,
    invoke_fused_moe_kernel,
)
from .lora_kernels import apply_moe_lora
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

from kestrel.moondream.lora import MoEMLPLoRA


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
        lora: MoEMLPLoRA | None = None,
    ) -> torch.Tensor:
        return self.forward(hidden_states, topk_weights, topk_ids, lora)

    @torch_compiler_disable()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        lora: MoEMLPLoRA | None = None,
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

        invoke_fused_moe_kernel(
            hidden_states,
            self.up_experts.weight,
            up_out,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=self.top_k,
            config=triton_config,
            compute_type=compute_type,
            bias=None,
            allow_tf32=self.config.allow_tf32,
        )

        if lora is not None:
            apply_moe_lora(
                x=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=up_out,
                lora_a=lora.up_a,
                lora_b=lora.up_b,
                scale=lora.scale,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
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
        fused_gelu_and_mul(activation_in, activation_out)

        down_in = activation_out
        down_out = self._workspaces.down.get(
            (num_tokens, self.top_k, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        invoke_fused_moe_kernel(
            down_in,
            self.down_experts.weight,
            down_out,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=triton_config,
            compute_type=compute_type,
            bias=None,
            allow_tf32=self.config.allow_tf32,
        )

        if lora is not None:
            apply_moe_lora(
                x=down_in,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=down_out,
                lora_a=lora.down_a,
                lora_b=lora.down_b,
                scale=lora.scale,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                top_k=1,  # Input is already per-expert [num_tokens * top_k, dim]
                config=triton_config,
                mul_routed_weight=True,
            )

        fused = self._workspaces.output.get(
            (num_tokens, self.input_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ops.moe_sum(down_out, fused)
        return fused

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
