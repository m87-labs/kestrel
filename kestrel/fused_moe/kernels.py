"""Trimmed Triton kernels adapted from vLLM's fused MoE implementation."""


from typing import Dict

import torch
import triton
import triton.language as tl


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bbe,
    stride_bbn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, iters):
        k_base = k * BLOCK_SIZE_K
        k_mask = (k_base + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(
            b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn,
            mask=offs_bn < N,
            other=0.0,
        )
        accumulator = accumulator + bias[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_kernel_fp8_w8a8(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_bse,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_row = offs_token // top_k
    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, iters):
        k_base = k * BLOCK_SIZE_K
        k_mask = (k_base + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    a_scale = tl.load(
        a_scale_ptr + a_row * stride_asm, mask=token_mask, other=0.0
    )[:, None]
    b_scale = tl.load(
        b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn,
        mask=offs_bn[None, :] < N,
        other=0.0,
    )
    accumulator *= a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def dtype_to_triton(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype for fused MoE: {dtype}")


def invoke_fused_moe_kernel(
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
    config: Dict[str, int],
    compute_type: tl.dtype,
    bias: torch.Tensor | None = None,
    allow_tf32: bool = True,
) -> None:
    assert sorted_token_ids.stride(0) == 1, "sorted_token_ids must be contiguous"
    assert topk_weights is not None or not mul_routed_weight
    assert B.stride(-1) == 1, "Expert weights must be row-major"
    assert C.stride(-1) == 1, "Output tensor must be contiguous in the last dim"

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    block_m = config["BLOCK_SIZE_M"]
    if A.size(0) < block_m:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * block_m)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    _fused_moe_kernel[grid](
        A,
        B,
        C,
        bias,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HAS_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
        num_warps=config["NUM_WARPS"],
        num_stages=config["NUM_STAGES"],
    )


def invoke_fused_moe_kernel_fp8_w8a8(
    A_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, int],
    compute_type: tl.dtype,
) -> None:
    """FP8 W8A8 fused MoE kernel (Triton).

    Expects:
      - A_fp8: [M, K] float8_e4m3fn (quantized)
      - A_scale: [M] float32 (per-row scale)
      - B_fp8: [E, N, K] float8_e4m3fn (quantized)
      - B_scale: [E, N] float32 (per-output-channel scale)
      - C: [num_tokens, top_k, N] bf16/fp16 output (flattened via strides)
    """
    assert A_fp8.ndim == 2 and A_scale.ndim == 1
    assert B_fp8.ndim == 3 and B_scale.ndim == 2
    assert A_fp8.stride(1) == 1
    assert B_fp8.stride(-1) == 1
    assert C.stride(-1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert expert_ids.stride(0) == 1
    assert num_tokens_post_padded.numel() == 1

    if mul_routed_weight:
        assert topk_weights is not None
        assert topk_weights.stride(0) == 1

    EM = sorted_token_ids.size(0)
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B_fp8.size(1), META["BLOCK_SIZE_N"]),
    )

    topk_ptr = topk_weights if topk_weights is not None else A_scale
    _fused_moe_kernel_fp8_w8a8[grid](
        A_fp8,
        B_fp8,
        C,
        A_scale,
        B_scale,
        topk_ptr,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B_fp8.size(1),
        B_fp8.size(2),
        EM,
        A_fp8.size(0) * int(top_k),
        A_fp8.stride(0),
        A_fp8.stride(1),
        B_fp8.stride(0),
        B_fp8.stride(2),
        B_fp8.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0),
        B_scale.stride(0),
        B_scale.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        num_warps=config["NUM_WARPS"],
        num_stages=config["NUM_STAGES"],
    )
