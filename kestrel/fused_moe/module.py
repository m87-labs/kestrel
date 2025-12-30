
from dataclasses import dataclass
from math import prod

import torch
from torch import nn
from torch.compiler import disable as torch_compiler_disable

from .kernels import (
    dtype_to_triton,
    invoke_fused_moe_kernel as invoke_fused_moe_kernel_triton,
    invoke_fused_moe_kernel_fp8_w8a8 as invoke_fused_moe_kernel_triton_fp8,
)
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
        self.fp8_bits = _ResizableBuffer()
        self.fp8_scale = _ResizableBuffer()


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
        if up_is_fp8w:
            # SM90 FP8 WGMMA path prefers block_m=64 routing.
            triton_config = triton_config.copy()
            triton_config["BLOCK_SIZE_M"] = 64
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
    ) -> None:
        backend = self.config.backend.lower()
        use_cute = backend in ("cute", "auto")
        b_is_fp8w = B.dtype == torch.uint8

        if b_is_fp8w and backend in ("triton", "auto"):
            # Triton FP8 W8A8 path (SGLang-style): quantize activations per row
            # and do FP8 dot with per-output-channel weight scales.
            if B_scale is None:
                raise ValueError("B_scale is required for FP8-weight MoE")

            triton_fp8_config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "NUM_WARPS": 4,
                "NUM_STAGES": 4,
            }
            a_bits, a_scale = self._quantize_fp8_e4m3fn_rowwise(A)
            invoke_fused_moe_kernel_triton_fp8(
                a_bits.view(torch.float8_e4m3fn),
                a_scale,
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

        if use_cute:
            try:
                from kestrel.ops.fused_moe_cute import (
                    FusedMoeCuTeConfig,
                    invoke_fused_moe_kernel_cute_down_decode,
                    invoke_fused_moe_kernel_cute_down_decode_fp8,
                    invoke_fused_moe_kernel_cute_up_decode,
                    invoke_fused_moe_kernel_cute_up_decode_fp8,
                )

                block_m = triton_config["BLOCK_SIZE_M"]
                if b_is_fp8w:
                    # Prefer W8A8 (FP8 activations + FP8 weights) via SM90 WGMMA.
                    if block_m == 64:
                        if B_scale is None:
                            raise ValueError("B_scale is required for FP8-weight MoE")

                        num_tokens = int(C.shape[0])
                        # Match Triton's fp8 W8A8 tiling, but keep a smaller N tile
                        # for the single-token case where occupancy is otherwise poor.
                        block_n = 64 if (not mul_routed_weight) and (num_tokens == 1) else 128

                        # Prefer deeper pipelining for small decode batches to better hide
                        # DRAM latency; fall back to fewer stages for larger batches where
                        # occupancy becomes the limiting factor.
                        num_stages = 4 if num_tokens <= 16 else 2

                        cute_cfg = FusedMoeCuTeConfig(
                            block_m=64,
                            block_n=block_n,
                            block_k=128,
                            num_warps=4,
                            num_stages=num_stages,
                        )
                        a_bits, a_scale = self._quantize_fp8_e4m3fn_rowwise(A)
                        if mul_routed_weight:
                            if int(top_k) != 1:
                                raise ValueError("CuTe fp8 moe_down expects top_k=1")
                            if topk_weights is None:
                                raise ValueError("topk_weights is required when mul_routed_weight=True")
                            invoke_fused_moe_kernel_cute_down_decode_fp8(
                                a_bits,
                                a_scale,
                                B,
                                B_scale,
                                C,
                                topk_weights=topk_weights,
                                sorted_token_ids=sorted_token_ids,
                                expert_ids=expert_ids,
                                num_tokens_post_padded=num_tokens_post_padded,
                                config=cute_cfg,
                            )
                        else:
                            if int(top_k) != 8:
                                raise ValueError("CuTe fp8 moe_up expects top_k=8")
                            invoke_fused_moe_kernel_cute_up_decode_fp8(
                                a_bits,
                                a_scale,
                                B,
                                B_scale,
                                C,
                                sorted_token_ids=sorted_token_ids,
                                expert_ids=expert_ids,
                                num_tokens_post_padded=num_tokens_post_padded,
                                config=cute_cfg,
                            )
                        return

                # Unquantized BF16 CuTe path: tuned for decode with routing block_m=16.
                if block_m == 16:
                    num_tokens = int(C.shape[0])
                    if num_tokens <= 8:
                        if mul_routed_weight:
                            cute_cfg = FusedMoeCuTeConfig(
                                block_m=16,
                                block_n=64,
                                block_k=256,
                                num_warps=4,
                                num_stages=1,
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
                    if mul_routed_weight:
                        if int(top_k) != 1:
                            raise ValueError("CuTe moe_down expects top_k=1")
                        if topk_weights is None:
                            raise ValueError("topk_weights is required when mul_routed_weight=True")
                        invoke_fused_moe_kernel_cute_down_decode(
                            A,
                            B,
                            C,
                            topk_weights=topk_weights,
                            sorted_token_ids=sorted_token_ids,
                            expert_ids=expert_ids,
                            num_tokens_post_padded=num_tokens_post_padded,
                            config=cute_cfg,
                        )
                    else:
                        if int(top_k) != 8:
                            raise ValueError("CuTe moe_up expects top_k=8")
                        invoke_fused_moe_kernel_cute_up_decode(
                            A,
                            B,
                            C,
                            sorted_token_ids=sorted_token_ids,
                            expert_ids=expert_ids,
                            num_tokens_post_padded=num_tokens_post_padded,
                            config=cute_cfg,
                        )
                    return
            except Exception:
                # Fall back to Triton if CuTe is unavailable (or config unsupported).
                if backend == "cute":
                    raise

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
