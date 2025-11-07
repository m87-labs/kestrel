from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod

import torch
from vllm import _custom_ops as ops

from .kernels import (
    dtype_to_triton,
    gelu_and_mul_plus_one,
    invoke_fused_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


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


class _MoEWorkspaces:
    def __init__(self) -> None:
        self.up = _ResizableBuffer()
        self.down = _ResizableBuffer()
        self.output = _ResizableBuffer()


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


class FusedMoEModule:
    """Hybrid MoE backend that reuses ScatterMoE weights with vLLM-style kernels."""

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
        self.up_experts = up_experts
        self.down_experts = down_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_experts = num_experts
        self.config = config or FusedMoEConfig()
        self._disabled = os.environ.get("KESTREL_DISABLE_FUSED_MOE") == "1"
        self._metadata_debug_done = False
        self._debug_up: torch.Tensor | None = None
        self._debug_h: torch.Tensor | None = None
        self._workspaces = _MoEWorkspaces()

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(hidden_states, topk_weights, topk_ids)

    @property
    def available(self) -> bool:
        return (not self._disabled) and torch.cuda.is_available()

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.available:
            raise RuntimeError("Fused MoE backend is disabled or CUDA is unavailable")
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
        debug_enabled = os.getenv("KESTREL_COMPARE_MOE_BACKENDS") == "1"
        if not debug_enabled:
            self._debug_up = None
            self._debug_h = None

        num_tokens = hidden_states.size(0)
        if num_tokens == 0:
            return hidden_states

        assignments = num_tokens * self.top_k
        block_size_m = self._select_block_size(assignments)
        triton_config = self.config.as_triton(block_size_m=block_size_m)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, self.num_experts
        )
        if os.getenv("KESTREL_COMPARE_MOE_BACKENDS") == "1" and not self._metadata_debug_done:
            flat_expert = topk_ids.reshape(-1)
            valid_pairs = flat_expert.size(0)
            fused_order = sorted_token_ids[: num_tokens_post_padded.item()]
            mask = fused_order < valid_pairs
            fused_sorted = flat_expert.index_select(0, fused_order[mask])
            scatter_sorted, _ = torch.sort(flat_expert)
            meta_diff = (fused_sorted - scatter_sorted[: fused_sorted.size(0)]).abs().max()
            print(
                "[FusedMoE][compare] routing max_diff="
                f"{meta_diff.item():.6f}",
                flush=True,
            )
            self._metadata_debug_done = True

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
        if debug_enabled:
            self._debug_up = up_out[:num_tokens].detach().cpu()
        activated = gelu_and_mul_plus_one(up_out.view(num_tokens * self.top_k, -1))

        down_in = activated.view(num_tokens * self.top_k, self.hidden_size)
        if debug_enabled:
            self._debug_h = down_in.view(num_tokens, self.top_k, self.hidden_size).detach().cpu()
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
