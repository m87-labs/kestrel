"""Tests for CuTe fused MoE kernels vs Triton reference."""

import pytest
import torch
import triton


# All token counts from cute_moe config
CONFIG_TOKEN_COUNTS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestCuteMoeKernels:
    """Direct tests of CuTe MoE invoke functions against Triton."""

    # Moondream MoE shapes
    D_MODEL = 2048
    D_EXPERT = 1024
    NUM_EXPERTS = 64
    TOP_K = 8

    @pytest.mark.parametrize("num_tokens", CONFIG_TOKEN_COUNTS)
    def test_up_kernel(self, device, num_tokens: int):
        """Test CuTe up kernel matches Triton."""
        torch.manual_seed(42)

        from kestrel_kernels.cute_moe import invoke_cute_moe_up, get_cute_moe_config
        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.kernels import invoke_fused_moe_kernel

        hidden = torch.randn(num_tokens, self.D_MODEL, device=device, dtype=torch.bfloat16)
        up_weight = torch.randn(
            self.NUM_EXPERTS, self.D_EXPERT * 2, self.D_MODEL, device=device, dtype=torch.bfloat16
        ) * 0.02
        topk_ids = torch.randint(
            0, self.NUM_EXPERTS, (num_tokens, self.TOP_K), device=device, dtype=torch.int32
        )

        # Get config first to determine block_m for routing
        config = get_cute_moe_config(
            "up", num_tokens,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.D_MODEL,
            intermediate_size=self.D_EXPERT,
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config.block_m, self.NUM_EXPERTS
        )

        # CuTe output: 3D [M, top_k, N]
        out_cute = torch.zeros(
            num_tokens, self.TOP_K, self.D_EXPERT * 2, device=device, dtype=torch.bfloat16
        )

        # Triton output: 3D [M, top_k, N]
        out_triton = torch.zeros(
            num_tokens, self.TOP_K, self.D_EXPERT * 2, device=device, dtype=torch.bfloat16
        )

        invoke_cute_moe_up(
            hidden,
            up_weight,
            out_cute,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=config,
        )

        triton_cfg = {
            "BLOCK_SIZE_M": config.block_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        }
        invoke_fused_moe_kernel(
            hidden,
            up_weight,
            out_triton,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=self.TOP_K,
            config=triton_cfg,
            compute_type=triton.language.bfloat16,
        )

        torch.testing.assert_close(out_cute, out_triton, rtol=0, atol=0)

    @pytest.mark.parametrize("num_tokens", CONFIG_TOKEN_COUNTS)
    def test_up_kernel_fp8(self, device, num_tokens: int):
        """Test CuTe FP8 up kernel matches bf16 reference."""
        torch.manual_seed(42)

        from kestrel_kernels.cute_moe import (
            invoke_cute_moe_up,
            invoke_cute_moe_up_fp8,
            get_cute_moe_config,
            CuteMoeConfig,
        )
        from kestrel.fused_moe.routing import moe_align_block_size

        # Create bf16 inputs
        hidden = torch.randn(num_tokens, self.D_MODEL, device=device, dtype=torch.bfloat16)
        up_weight = torch.randn(
            self.NUM_EXPERTS, self.D_EXPERT * 2, self.D_MODEL, device=device, dtype=torch.bfloat16
        ) * 0.02
        topk_ids = torch.randint(
            0, self.NUM_EXPERTS, (num_tokens, self.TOP_K), device=device, dtype=torch.int32
        )

        # Quantize activations row-wise
        def quantize_fp8_rowwise(x: torch.Tensor):
            abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = abs_max / 448.0  # FP8 E4M3 max
            x_fp8 = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            return x_fp8, scale.squeeze(-1).to(torch.float32)

        # Quantize weights column-wise (per output channel)
        def quantize_fp8_colwise(w: torch.Tensor):
            # w: [E, N, K] -> scale per [E, N]
            abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = abs_max / 448.0
            w_fp8 = (w / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            return w_fp8, scale.squeeze(-1).to(torch.float32)

        hidden_fp8, hidden_scale = quantize_fp8_rowwise(hidden)
        up_weight_fp8, up_weight_scale = quantize_fp8_colwise(up_weight)

        # FP8 config: block_m=64 is standard for fp8 WGMMA
        config_fp8 = CuteMoeConfig(block_m=64, block_n=128, block_k=128, num_warps=4, num_stages=2, dtype="fp8")
        sorted_token_ids_fp8, expert_ids_fp8, num_tokens_post_padded_fp8 = moe_align_block_size(
            topk_ids, config_fp8.block_m, self.NUM_EXPERTS
        )

        # bf16 config from precompiled configs
        config_bf16 = get_cute_moe_config(
            "up", num_tokens,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.D_MODEL,
            intermediate_size=self.D_EXPERT,
        )
        sorted_token_ids_bf16, expert_ids_bf16, num_tokens_post_padded_bf16 = moe_align_block_size(
            topk_ids, config_bf16.block_m, self.NUM_EXPERTS
        )

        # FP8 output
        out_fp8 = torch.zeros(
            num_tokens, self.TOP_K, self.D_EXPERT * 2, device=device, dtype=torch.bfloat16
        )

        # bf16 reference output
        out_ref = torch.zeros(
            num_tokens, self.TOP_K, self.D_EXPERT * 2, device=device, dtype=torch.bfloat16
        )

        invoke_cute_moe_up_fp8(
            hidden_fp8,
            hidden_scale,
            up_weight_fp8,
            up_weight_scale,
            out_fp8,
            sorted_token_ids=sorted_token_ids_fp8,
            expert_ids=expert_ids_fp8,
            num_tokens_post_padded=num_tokens_post_padded_fp8,
            config=config_fp8,
        )

        invoke_cute_moe_up(
            hidden,
            up_weight,
            out_ref,
            sorted_token_ids=sorted_token_ids_bf16,
            expert_ids=expert_ids_bf16,
            num_tokens_post_padded=num_tokens_post_padded_bf16,
            config=config_bf16,
        )

        # FP8 has quantization error, so we use looser tolerances
        # rtol is not useful for values near zero, so focus on atol
        torch.testing.assert_close(out_fp8, out_ref, rtol=0.1, atol=0.2)

    @pytest.mark.parametrize("num_tokens", CONFIG_TOKEN_COUNTS)
    def test_down_kernel(self, device, num_tokens: int):
        """Test CuTe down kernel matches Triton."""
        torch.manual_seed(42)

        from kestrel_kernels.cute_moe import invoke_cute_moe_down, get_cute_moe_config
        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.kernels import invoke_fused_moe_kernel

        # Down kernel input is after activation: [M * top_k, D_EXPERT]
        activation = torch.randn(
            num_tokens * self.TOP_K, self.D_EXPERT, device=device, dtype=torch.bfloat16
        )
        down_weight = torch.randn(
            self.NUM_EXPERTS, self.D_MODEL, self.D_EXPERT, device=device, dtype=torch.bfloat16
        ) * 0.02
        topk_ids = torch.randint(
            0, self.NUM_EXPERTS, (num_tokens, self.TOP_K), device=device, dtype=torch.int32
        )
        topk_weights = torch.randn(num_tokens, self.TOP_K, device=device, dtype=torch.bfloat16)

        # Get config first to determine block_m for routing
        config = get_cute_moe_config(
            "down", num_tokens,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.D_MODEL,
            intermediate_size=self.D_EXPERT,
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config.block_m, self.NUM_EXPERTS
        )

        # CuTe output: 3D [M, top_k, N]
        out_cute = torch.zeros(
            num_tokens, self.TOP_K, self.D_MODEL, device=device, dtype=torch.bfloat16
        )

        # Triton output: 3D [M, top_k, N]
        out_triton = torch.zeros(
            num_tokens, self.TOP_K, self.D_MODEL, device=device, dtype=torch.bfloat16
        )

        invoke_cute_moe_down(
            activation,
            down_weight,
            out_cute,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=config,
        )

        triton_cfg = {
            "BLOCK_SIZE_M": config.block_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_WARPS": 4,
            "NUM_STAGES": 2,
        }
        invoke_fused_moe_kernel(
            activation,
            down_weight,
            out_triton,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,  # down uses top_k=1 for scatter
            config=triton_cfg,
            compute_type=triton.language.bfloat16,
        )

        torch.testing.assert_close(out_cute, out_triton, rtol=0, atol=0)

    @pytest.mark.parametrize("num_tokens", CONFIG_TOKEN_COUNTS)
    def test_down_kernel_fp8(self, device, num_tokens: int):
        """Test CuTe FP8 down kernel matches bf16 reference."""
        torch.manual_seed(42)

        from kestrel_kernels.cute_moe import (
            invoke_cute_moe_down,
            invoke_cute_moe_down_fp8,
            get_cute_moe_config,
            CuteMoeConfig,
        )
        from kestrel.fused_moe.routing import moe_align_block_size

        # Down kernel input is after activation: [M * top_k, D_EXPERT]
        activation = torch.randn(
            num_tokens * self.TOP_K, self.D_EXPERT, device=device, dtype=torch.bfloat16
        )
        down_weight = torch.randn(
            self.NUM_EXPERTS, self.D_MODEL, self.D_EXPERT, device=device, dtype=torch.bfloat16
        ) * 0.02
        topk_ids = torch.randint(
            0, self.NUM_EXPERTS, (num_tokens, self.TOP_K), device=device, dtype=torch.int32
        )
        topk_weights = torch.randn(num_tokens, self.TOP_K, device=device, dtype=torch.bfloat16)

        # Quantize activations row-wise
        def quantize_fp8_rowwise(x: torch.Tensor):
            abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = abs_max / 448.0  # FP8 E4M3 max
            x_fp8 = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            return x_fp8, scale.squeeze(-1).to(torch.float32)

        # Quantize weights column-wise (per output channel)
        def quantize_fp8_colwise(w: torch.Tensor):
            # w: [E, N, K] -> scale per [E, N]
            abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = abs_max / 448.0
            w_fp8 = (w / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            return w_fp8, scale.squeeze(-1).to(torch.float32)

        activation_fp8, activation_scale = quantize_fp8_rowwise(activation)
        down_weight_fp8, down_weight_scale = quantize_fp8_colwise(down_weight)

        # FP8 config: block_m=64 is standard for fp8 WGMMA
        config_fp8 = CuteMoeConfig(block_m=64, block_n=128, block_k=128, num_warps=4, num_stages=2, dtype="fp8")
        sorted_token_ids_fp8, expert_ids_fp8, num_tokens_post_padded_fp8 = moe_align_block_size(
            topk_ids, config_fp8.block_m, self.NUM_EXPERTS
        )

        # bf16 config from precompiled configs
        config_bf16 = get_cute_moe_config(
            "down", num_tokens,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.D_MODEL,
            intermediate_size=self.D_EXPERT,
        )
        sorted_token_ids_bf16, expert_ids_bf16, num_tokens_post_padded_bf16 = moe_align_block_size(
            topk_ids, config_bf16.block_m, self.NUM_EXPERTS
        )

        # FP8 output
        out_fp8 = torch.zeros(
            num_tokens, self.TOP_K, self.D_MODEL, device=device, dtype=torch.bfloat16
        )

        # bf16 reference output
        out_ref = torch.zeros(
            num_tokens, self.TOP_K, self.D_MODEL, device=device, dtype=torch.bfloat16
        )

        invoke_cute_moe_down_fp8(
            activation_fp8,
            activation_scale,
            down_weight_fp8,
            down_weight_scale,
            out_fp8,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids_fp8,
            expert_ids=expert_ids_fp8,
            num_tokens_post_padded=num_tokens_post_padded_fp8,
            config=config_fp8,
        )

        invoke_cute_moe_down(
            activation,
            down_weight,
            out_ref,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids_bf16,
            expert_ids=expert_ids_bf16,
            num_tokens_post_padded=num_tokens_post_padded_bf16,
            config=config_bf16,
        )

        # FP8 has quantization error, so we use looser tolerances
        # Down kernel accumulates more error than up kernel due to reduction
        torch.testing.assert_close(out_fp8, out_ref, rtol=0.2, atol=0.4)
