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
