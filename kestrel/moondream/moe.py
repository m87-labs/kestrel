from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from .lora_workspace import MoELoRALayerWorkspace
from kestrel_kernels import get_runtime, moe as _MOE_API

_SHARED_MOE_HANDLES: dict[tuple, _MOE_API.MoeHandle] = {}


class ExpertWeights(nn.Module):
    """Container for per-expert weight tensors."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size, dtype=dtype)
        )


@dataclass
class MoEConfig:
    allow_tf32: bool = True
    backend: str = "auto"  # "auto" | "cute" | "triton"
    auto_backend_token_threshold: int = 256


class MoEModule(nn.Module):
    """Moondream MoE wrapper backed by the kestrel-kernels runtime API."""

    def __init__(
        self,
        up_experts: torch.nn.Module,
        down_experts: torch.nn.Module,
        *,
        top_k: int,
        hidden_size: int,
        input_size: int,
        num_experts: int,
        config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        self.up_experts = up_experts
        self.down_experts = down_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_experts = num_experts
        self.config = config or MoEConfig()
        self._moe_runtime = get_runtime().moe

    def _spec_for_weights(
        self,
        *,
        dtype: torch.dtype,
        weight_format: _MOE_API.MoeWeightFormat,
    ) -> _MOE_API.MoeSpec:
        return _MOE_API.MoeSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.input_size,
            intermediate_size=self.hidden_size,
            activation="gelu",
            weight_format=weight_format,
            dtype=dtype,
            allow_tf32=self.config.allow_tf32,
            backend=self.config.backend,
            auto_backend_token_threshold=self.config.auto_backend_token_threshold,
        )

    @staticmethod
    def _capacity_tokens_for_mode(
        num_tokens: int,
        mode: Literal["prefill", "decode"],
    ) -> int:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if mode == "decode":
            return num_tokens
        return 1 << (num_tokens - 1).bit_length()

    def _get_moe_handle(
        self,
        *,
        hidden_states: torch.Tensor,
        weight_format: _MOE_API.MoeWeightFormat,
        max_loras: int,
        mode: Literal["prefill", "decode"],
        max_lora_rank: int = 0,
    ) -> _MOE_API.MoeHandle:
        spec = self._spec_for_weights(dtype=hidden_states.dtype, weight_format=weight_format)
        max_tokens = self._capacity_tokens_for_mode(int(hidden_states.shape[0]), mode)
        capacity = _MOE_API.MoeCapacity(
            max_tokens=max_tokens,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            mode=mode,
        )
        key = (
            str(hidden_states.device),
            hidden_states.dtype,
            weight_format,
            capacity.max_tokens,
            capacity.max_loras,
            capacity.max_lora_rank,
            capacity.mode,
            self.num_experts,
            self.top_k,
            self.input_size,
            self.hidden_size,
            self.config.allow_tf32,
            self.config.backend,
            self.config.auto_backend_token_threshold,
        )
        handle = _SHARED_MOE_HANDLES.get(key)
        if handle is None:
            handle = self._moe_runtime.prepare(
                spec,
                capacity,
                device=hidden_states.device,
            )
            _SHARED_MOE_HANDLES[key] = handle
        return handle

    @torch_compiler_disable()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        lora_workspace: MoELoRALayerWorkspace | None = None,
        lora_slot_ids: torch.Tensor | None = None,
        lora_route_ids: torch.Tensor | None = None,
        active_lora_ids: torch.Tensor | None = None,
        active_lora_meta: torch.Tensor | None = None,
        active_lora_token_counts: tuple[int, ...] | None = None,
        active_lora_max_rank: int | None = None,
        mode: Literal["prefill", "decode"] = "decode",
    ) -> torch.Tensor:
        if hidden_states.size(0) == 0:
            return hidden_states

        up_weight, up_is_fp8w = _normalize_expert_weight(self.up_experts.weight)
        down_weight, down_is_fp8w = _normalize_expert_weight(self.down_experts.weight)
        if up_is_fp8w != down_is_fp8w:
            raise ValueError("Up and down expert weights must use the same dtype scheme")

        weight_format: _MOE_API.MoeWeightFormat
        if up_is_fp8w:
            weight_format = "fp8_e4m3"
        elif up_weight.dtype == torch.float16:
            weight_format = "fp16"
        elif up_weight.dtype == torch.float32:
            weight_format = "fp32"
        else:
            weight_format = "bf16"

        max_loras = 0
        max_lora_rank = 0
        if lora_workspace is not None:
            max_loras = int(lora_workspace.up_a.shape[0]) // self.num_experts
            max_lora_rank = int(lora_workspace.up_a.shape[1])

        handle = self._get_moe_handle(
            hidden_states=hidden_states,
            weight_format=weight_format,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            mode=mode,
        )
        weights = _MOE_API.pack_weights(
            handle.spec,
            up=up_weight,
            down=down_weight,
            up_scale=getattr(self.up_experts, "scale", None),
            down_scale=getattr(self.down_experts, "scale", None),
        )

        lora_state = None
        if lora_workspace is not None and lora_slot_ids is not None:
            lora_state = _MOE_API.MoeLoraState(
                lora_slot_ids=lora_slot_ids,
                up_a=lora_workspace.up_a,
                up_b=lora_workspace.up_b,
                down_a=lora_workspace.down_a,
                down_b=lora_workspace.down_b,
                lora_ranks=lora_workspace.lora_ranks,
                lora_route_ids=lora_route_ids,
                active_lora_ids=active_lora_ids,
                active_lora_meta=active_lora_meta,
                active_lora_token_counts=active_lora_token_counts,
                active_lora_max_rank=active_lora_max_rank,
            )

        return self._moe_runtime.forward(
            handle,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            weights=weights,
            lora=lora_state,
        )


def _normalize_expert_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if weight.dtype == torch.uint8:
        return weight, True
    if weight.dtype == torch.float8_e4m3fn:
        return weight.view(torch.uint8), True
    return weight, False
