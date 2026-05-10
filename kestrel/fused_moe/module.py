from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from kestrel.moondream.lora_workspace import MoELoRALayerWorkspace
from kestrel_kernels import moe as _MOE_API

_SHARED_MOE_HANDLES: dict[tuple, _MOE_API.MoeHandle] = {}


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
        # Pipelined-LoRA events; only used when ``use_batched_lora`` or
        # ``use_single_lora`` is True at forward time. LoRA is disabled in
        # the v1 MPS path, so these are unreachable there — keep them
        # ``None`` rather than instantiating CUDA Events that would crash
        # on non-CUDA hosts.
        if torch.cuda.is_available():
            self._lora_inputs_event = torch.cuda.Event(enable_timing=False)
            self._lora_activation_event = torch.cuda.Event(enable_timing=False)
            self._lora_up_event = torch.cuda.Event(enable_timing=False)
            self._lora_down_event = torch.cuda.Event(enable_timing=False)
        else:
            self._lora_inputs_event = None
            self._lora_activation_event = None
            self._lora_up_event = None
            self._lora_down_event = None

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

    def _get_moe_handle(
        self,
        *,
        hidden_states: torch.Tensor,
        weight_format: _MOE_API.MoeWeightFormat,
        max_loras: int,
        mode: Literal["prefill", "decode"],
    ) -> _MOE_API.MoeHandle:
        spec = self._spec_for_weights(dtype=hidden_states.dtype, weight_format=weight_format)
        capacity = _MOE_API.MoeCapacity(
            max_tokens=int(hidden_states.shape[0]),
            max_loras=max_loras,
            mode=mode,
        )
        key = (
            str(hidden_states.device),
            hidden_states.dtype,
            weight_format,
            capacity.max_tokens,
            capacity.max_loras,
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
            handle = _MOE_API.prepare(spec, capacity, device=hidden_states.device)
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
        single_lora_id: int | None = None,
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
        if lora_workspace is not None:
            max_loras = int(lora_workspace.up_a.shape[0]) // self.num_experts

        handle = self._get_moe_handle(
            hidden_states=hidden_states,
            weight_format=weight_format,
            max_loras=max_loras,
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
            is_prefill = single_lora_id is not None
            lora_state = _MOE_API.MoeLoraState(
                lora_slot_ids=lora_slot_ids,
                up_a=lora_workspace.up_a,
                up_b=lora_workspace.up_b,
                down_a=lora_workspace.down_a,
                down_b=lora_workspace.down_b,
                stream=None if is_prefill else lora_workspace.stream,
                single_lora_id=single_lora_id,
                shrink_config=(
                    self.config.lora_prefill_shrink
                    if is_prefill
                    else self.config.lora_decode_shrink
                ),
                expand_config=(
                    self.config.lora_prefill_expand
                    if is_prefill
                    else self.config.lora_decode_expand
                ),
            )

        return _MOE_API.forward(
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
