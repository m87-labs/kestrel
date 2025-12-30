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

    @pytest.mark.parametrize("num_tokens", [4, 32])
    def test_fp8_w8a8_matches_dequant_reference(self, device, num_tokens: int):
        torch.manual_seed(0)
        if torch.cuda.get_device_capability(device)[0] < 9:
            pytest.skip("FP8 WGMMA path requires SM90+")

        d_model = 2048
        d_expert = 1024
        num_experts = 64
        top_k = 8

        from kestrel.fused_moe.module import FusedMoEConfig, FusedMoEModule
        from kestrel.fused_moe.kernels import dtype_to_triton, invoke_fused_moe_kernel as invoke_fused_moe_kernel_triton
        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.weights import ExpertWeights, ExpertWeightsFp8E4M3FN
        from kestrel_kernels.activation import gelu_residual_cuda
        from kestrel_kernels.moe_sum import moe_sum as moe_sum_cuda

        # Build a BF16 reference, then quantize weights to fp8-e4m3fn bits + per-channel scales.
        up_bf16 = ExpertWeights(num_experts, d_model, d_expert * 2, dtype=torch.bfloat16).to(device)
        down_bf16 = ExpertWeights(num_experts, d_expert, d_model, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            up_bf16.weight.normal_(mean=0.0, std=0.02)
            down_bf16.weight.normal_(mean=0.0, std=0.02)

        def quantize_fp8_bits(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            # Per-(expert, out_channel) scale.
            scale = w_bf16.abs().amax(dim=2).to(torch.float32) / fp8_max
            scale = scale.clamp(min=1e-6)
            w_fp8 = (w_bf16 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return w_fp8.view(torch.uint8), scale

        def quantize_fp8_act_rowwise(a_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            scale = a_bf16.abs().amax(dim=1).to(torch.float32) / fp8_max
            scale = scale.clamp(min=1e-6)
            a_fp8 = (a_bf16 / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            return a_fp8.view(torch.uint8), scale

        up_bits, up_scale = quantize_fp8_bits(up_bf16.weight)
        down_bits, down_scale = quantize_fp8_bits(down_bf16.weight)

        up_fp8 = ExpertWeightsFp8E4M3FN(num_experts, d_model, d_expert * 2).to(device)
        down_fp8 = ExpertWeightsFp8E4M3FN(num_experts, d_expert, d_model).to(device)
        with torch.no_grad():
            up_fp8.weight.copy_(up_bits)
            up_fp8.scale.copy_(up_scale)
            down_fp8.weight.copy_(down_bits)
            down_fp8.scale.copy_(down_scale)

        # Reference weights are the dequantized bf16 view of the fp8 weights.
        up_dequant = ExpertWeights(num_experts, d_model, d_expert * 2, dtype=torch.bfloat16).to(device)
        down_dequant = ExpertWeights(num_experts, d_expert, d_model, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            up_dequant.weight.copy_(up_fp8.dequantize(dtype=torch.bfloat16))
            down_dequant.weight.copy_(down_fp8.dequantize(dtype=torch.bfloat16))

        hidden_states = torch.randn(num_tokens, d_model, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
        topk_weights = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)

        moe_fp8 = FusedMoEModule(
            up_fp8,
            down_fp8,
            top_k=top_k,
            hidden_size=d_expert,
            input_size=d_model,
            num_experts=num_experts,
            config=FusedMoEConfig(backend="cute"),
        )

        from kestrel.ops import fused_moe_cute as fused_moe_cute_mod

        fused_moe_cute_mod._COMPILE_CACHE_FP8.clear()

        # Reference: dequantize both activations and weights, then run Triton MoE.
        block_m = 64
        triton_cfg = FusedMoEConfig(backend="triton", block_size_m=block_m).as_triton(block_size_m=block_m)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        compute_type = dtype_to_triton(hidden_states.dtype)

        a_bits, a_scale = quantize_fp8_act_rowwise(hidden_states)
        a_dequant = a_bits.view(torch.float8_e4m3fn).to(torch.bfloat16) * a_scale.to(torch.bfloat16).unsqueeze(-1)
        up_out_ref = torch.empty((num_tokens, top_k, d_expert * 2), device=device, dtype=torch.bfloat16)
        invoke_fused_moe_kernel_triton(
            a_dequant,
            up_dequant.weight,
            up_out_ref,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=triton_cfg,
            compute_type=compute_type,
            bias=None,
            allow_tf32=True,
        )

        activation_in_ref = up_out_ref.view(num_tokens * top_k, -1)
        activation_out_ref = torch.empty((num_tokens * top_k, d_expert), device=device, dtype=torch.bfloat16)
        gelu_residual_cuda(activation_out_ref, activation_in_ref)

        down_bits_act, down_scale_act = quantize_fp8_act_rowwise(activation_out_ref)
        down_dequant_act = down_bits_act.view(torch.float8_e4m3fn).to(torch.bfloat16) * down_scale_act.to(torch.bfloat16).unsqueeze(-1)
        down_out_ref = torch.empty((num_tokens, top_k, d_model), device=device, dtype=torch.bfloat16)
        invoke_fused_moe_kernel_triton(
            down_dequant_act,
            down_dequant.weight,
            down_out_ref,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=triton_cfg,
            compute_type=compute_type,
            bias=None,
            allow_tf32=True,
        )
        out_ref = torch.empty((num_tokens, d_model), device=device, dtype=torch.bfloat16)
        moe_sum_cuda(down_out_ref, out_ref)

        out_fp8 = moe_fp8(hidden_states, topk_weights, topk_ids)

        keys = list(fused_moe_cute_mod._COMPILE_CACHE_FP8.keys())
        assert any(k[0] == "up" for k in keys), "CuTe FP8 moe_up did not compile/launch"
        assert any(k[0] == "down" for k in keys), "CuTe FP8 moe_down did not compile/launch"

        torch.testing.assert_close(out_fp8, out_ref, rtol=2e-1, atol=2e-1)

    @pytest.mark.parametrize("num_tokens", [4, 32])
    def test_fp8_w8a8_triton_backend_matches_dequant_reference(self, device, num_tokens: int):
        torch.manual_seed(0)
        if torch.cuda.get_device_capability(device)[0] < 9:
            pytest.skip("FP8 W8A8 path requires SM90+")

        d_model = 2048
        d_expert = 1024
        num_experts = 64
        top_k = 8

        from kestrel.fused_moe.module import FusedMoEConfig, FusedMoEModule
        from kestrel.fused_moe.kernels import (
            dtype_to_triton,
            invoke_fused_moe_kernel as invoke_fused_moe_kernel_triton,
        )
        from kestrel.fused_moe.routing import moe_align_block_size
        from kestrel.fused_moe.weights import ExpertWeights, ExpertWeightsFp8E4M3FN
        from kestrel_kernels.activation import gelu_residual_cuda
        from kestrel_kernels.moe_sum import moe_sum as moe_sum_cuda

        up_bf16 = ExpertWeights(num_experts, d_model, d_expert * 2, dtype=torch.bfloat16).to(device)
        down_bf16 = ExpertWeights(num_experts, d_expert, d_model, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            up_bf16.weight.normal_(mean=0.0, std=0.02)
            down_bf16.weight.normal_(mean=0.0, std=0.02)

        def quantize_fp8_bits(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            scale = w_bf16.abs().amax(dim=2).to(torch.float32) / fp8_max
            scale = scale.clamp(min=1e-6)
            w_fp8 = (w_bf16 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return w_fp8.view(torch.uint8), scale

        def quantize_fp8_act_rowwise(a_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            scale = a_bf16.abs().amax(dim=1).to(torch.float32) / fp8_max
            scale = scale.clamp(min=1e-6)
            a_fp8 = (a_bf16 / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            return a_fp8.view(torch.uint8), scale

        up_bits, up_scale = quantize_fp8_bits(up_bf16.weight)
        down_bits, down_scale = quantize_fp8_bits(down_bf16.weight)

        up_fp8 = ExpertWeightsFp8E4M3FN(num_experts, d_model, d_expert * 2).to(device)
        down_fp8 = ExpertWeightsFp8E4M3FN(num_experts, d_expert, d_model).to(device)
        with torch.no_grad():
            up_fp8.weight.copy_(up_bits)
            up_fp8.scale.copy_(up_scale)
            down_fp8.weight.copy_(down_bits)
            down_fp8.scale.copy_(down_scale)

        up_dequant = ExpertWeights(num_experts, d_model, d_expert * 2, dtype=torch.bfloat16).to(device)
        down_dequant = ExpertWeights(num_experts, d_expert, d_model, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            up_dequant.weight.copy_(up_fp8.dequantize(dtype=torch.bfloat16))
            down_dequant.weight.copy_(down_fp8.dequantize(dtype=torch.bfloat16))

        hidden_states = torch.randn(num_tokens, d_model, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
        topk_weights = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)

        moe_fp8_triton = FusedMoEModule(
            up_fp8,
            down_fp8,
            top_k=top_k,
            hidden_size=d_expert,
            input_size=d_model,
            num_experts=num_experts,
            config=FusedMoEConfig(backend="triton"),
        )

        # Reference: dequantize both activations and weights, then run Triton MoE.
        block_m = 64
        triton_cfg = FusedMoEConfig(backend="triton", block_size_m=block_m).as_triton(block_size_m=block_m)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        compute_type = dtype_to_triton(hidden_states.dtype)

        a_bits, a_scale = quantize_fp8_act_rowwise(hidden_states)
        a_dequant = a_bits.view(torch.float8_e4m3fn).to(torch.bfloat16) * a_scale.to(torch.bfloat16).unsqueeze(-1)
        up_out_ref = torch.empty((num_tokens, top_k, d_expert * 2), device=device, dtype=torch.bfloat16)
        invoke_fused_moe_kernel_triton(
            a_dequant,
            up_dequant.weight,
            up_out_ref,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=triton_cfg,
            compute_type=compute_type,
            bias=None,
            allow_tf32=True,
        )

        activation_in_ref = up_out_ref.view(num_tokens * top_k, -1)
        activation_out_ref = torch.empty((num_tokens * top_k, d_expert), device=device, dtype=torch.bfloat16)
        gelu_residual_cuda(activation_out_ref, activation_in_ref)

        down_bits_act, down_scale_act = quantize_fp8_act_rowwise(activation_out_ref)
        down_dequant_act = down_bits_act.view(torch.float8_e4m3fn).to(torch.bfloat16) * down_scale_act.to(torch.bfloat16).unsqueeze(-1)
        down_out_ref = torch.empty((num_tokens, top_k, d_model), device=device, dtype=torch.bfloat16)
        invoke_fused_moe_kernel_triton(
            down_dequant_act,
            down_dequant.weight,
            down_out_ref,
            topk_weights=topk_weights.view(-1),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=triton_cfg,
            compute_type=compute_type,
            bias=None,
            allow_tf32=True,
        )
        out_ref = torch.empty((num_tokens, d_model), device=device, dtype=torch.bfloat16)
        moe_sum_cuda(down_out_ref, out_ref)

        out_fp8 = moe_fp8_triton(hidden_states, topk_weights, topk_ids)
        torch.testing.assert_close(out_fp8, out_ref, rtol=2e-1, atol=2e-1)
