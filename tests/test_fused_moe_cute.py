"""Tests for CuTe fused MoE matmul vs Triton reference."""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestFusedMoECuTe:
    @pytest.mark.parametrize("num_tokens", [4, 32])
    def test_matches_triton_decode_shapes(self, device, num_tokens: int):
        torch.manual_seed(0)

        # Moondream typical MoE shapes.
        d_model = 2048
        d_expert = 1024
        num_experts = 64
        top_k = 8

        # Cover both tiny decode batches (special CuTe tuning) and a larger
        # decode-like token count.

        from kestrel.fused_moe.module import FusedMoEConfig, FusedMoEModule
        from kestrel.fused_moe.weights import ExpertWeights

        up_experts = ExpertWeights(num_experts, d_model, d_expert * 2, dtype=torch.bfloat16).to(device)
        down_experts = ExpertWeights(num_experts, d_expert, d_model, dtype=torch.bfloat16).to(device)

        # Initialize weights with a small scale to keep outputs numerically stable.
        with torch.no_grad():
            up_experts.weight.normal_(mean=0.0, std=0.02)
            down_experts.weight.normal_(mean=0.0, std=0.02)

        hidden_states = torch.randn(num_tokens, d_model, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
        topk_weights = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)

        moe_triton = FusedMoEModule(
            up_experts,
            down_experts,
            top_k=top_k,
            hidden_size=d_expert,
            input_size=d_model,
            num_experts=num_experts,
            config=FusedMoEConfig(backend="triton"),
        )
        moe_cute = FusedMoEModule(
            up_experts,
            down_experts,
            top_k=top_k,
            hidden_size=d_expert,
            input_size=d_model,
            num_experts=num_experts,
            config=FusedMoEConfig(backend="cute"),
        )

        from kestrel.ops import fused_moe_cute as fused_moe_cute_mod

        # Ensure we exercise the CuTe path for *both* moe_up and moe_down (avoid silent fallback).
        fused_moe_cute_mod._COMPILE_CACHE.clear()

        out_triton = moe_triton(hidden_states, topk_weights, topk_ids)
        out_cute = moe_cute(hidden_states, topk_weights, topk_ids)

        keys = list(fused_moe_cute_mod._COMPILE_CACHE.keys())
        assert any(k[0] == "up" for k in keys), "CuTe moe_up did not compile/launch"
        assert any(k[0] == "down" for k in keys), "CuTe moe_down did not compile/launch"

        torch.testing.assert_close(out_cute, out_triton, rtol=2e-2, atol=2e-2)
