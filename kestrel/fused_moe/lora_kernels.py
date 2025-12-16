import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_lora_kernel(
    # Input/output pointers
    x_ptr,  # [M, hidden_dim] or [M * top_k, hidden_dim]
    lora_a_ptr,  # [num_experts, rank, hidden_dim]
    lora_b_ptr,  # [num_experts, out_dim, rank]
    output_ptr,  # [M, top_k, out_dim]
    topk_weights_ptr,  # [M, top_k]
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    hidden_dim,
    rank,
    out_dim,
    EM,
    num_valid_tokens,
    # Strides for x
    stride_xm,
    stride_xk,
    # Strides for lora_a [num_experts, rank, hidden_dim]
    stride_ae,
    stride_ar,
    stride_ak,
    # Strides for lora_b [num_experts, out_dim, rank]
    stride_be,
    stride_bn,
    stride_br,
    # Strides for output [M, top_k, out_dim]
    stride_om,
    stride_on,
    # Constexprs
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_RANK: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    """Fused MoE LoRA kernel: computes x @ A.T @ B.T in one kernel.

    This fuses the shrink and expand phases to avoid intermediate buffer
    allocation and reduce kernel launch overhead.
    """
    pid_m = tl.program_id(0)  # Token block
    pid_n = tl.program_id(1)  # Output block

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    # Load token indices for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    # Phase 1: Shrink - compute x @ A.T -> [BLOCK_SIZE_M, rank]
    # Accumulate in float32 for precision
    intermediate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=tl.float32)

    for k_start in range(0, hidden_dim, BLOCK_SIZE_HIDDEN):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
        k_mask = k_offs < hidden_dim

        # Load x block: [BLOCK_SIZE_M, BLOCK_SIZE_HIDDEN]
        x_ptrs = x_ptr + (offs_token[:, None] // top_k) * stride_xm + k_offs[None, :] * stride_xk
        x_block = tl.load(x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Load A block: [BLOCK_SIZE_HIDDEN, BLOCK_SIZE_RANK] (transposed from [rank, hidden])
        # A is [num_experts, rank, hidden_dim], we want A[expert].T = [hidden_dim, rank]
        a_ptrs = lora_a_ptr + expert_id * stride_ae + tl.arange(0, BLOCK_SIZE_RANK)[None, :] * stride_ar + k_offs[:, None] * stride_ak
        a_block = tl.load(a_ptrs, mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_RANK)[None, :] < rank), other=0.0).to(tl.float32)

        intermediate += tl.dot(x_block, a_block)

    # Phase 2: Expand - compute intermediate @ B.T -> [BLOCK_SIZE_M, BLOCK_SIZE_OUT]
    # B is [num_experts, out_dim, rank], we want B[expert].T = [rank, out_dim]
    offs_n = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    n_mask = offs_n < out_dim

    output_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_OUT), dtype=tl.float32)

    for r_start in range(0, rank, BLOCK_SIZE_RANK):
        r_offs = r_start + tl.arange(0, BLOCK_SIZE_RANK)
        r_mask = r_offs < rank

        # Get intermediate slice: [BLOCK_SIZE_M, BLOCK_SIZE_RANK]
        if r_start == 0:
            inter_block = intermediate
        else:
            # For ranks > BLOCK_SIZE_RANK, we'd need to recompute or store
            # For typical LoRA ranks (8-32), this won't happen with BLOCK_SIZE_RANK=64
            inter_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=tl.float32)

        # Load B block: [BLOCK_SIZE_RANK, BLOCK_SIZE_OUT] (transposed from [out_dim, rank])
        b_ptrs = lora_b_ptr + expert_id * stride_be + offs_n[None, :] * stride_bn + r_offs[:, None] * stride_br
        b_block = tl.load(b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        output_acc += tl.dot(inter_block, b_block)

    # Apply routing weight if needed
    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        output_acc = output_acc * weights[:, None]

    # Store output (accumulate via atomic add)
    output_acc = output_acc.to(output_ptr.dtype.element_ty)
    out_ptrs = output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = token_mask[:, None] & n_mask[None, :]

    tl.atomic_add(out_ptrs, output_acc, mask=out_mask)


@torch.inference_mode()
def apply_moe_lora(
    x: torch.Tensor,  # [M, hidden_dim] or [M * top_k, hidden_dim]
    topk_ids: torch.Tensor,  # [M, top_k]
    topk_weights: torch.Tensor,  # [M, top_k]
    output: torch.Tensor,  # [M, top_k, out_dim]
    lora_a: torch.Tensor,  # [num_experts, rank, hidden_dim]
    lora_b: torch.Tensor,  # [num_experts, out_dim, rank]
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, int],
    mul_routed_weight: bool = False,
) -> None:
    """Apply MoE LoRA using a fused Triton kernel.

    Computes: output += (x @ A.T @ B.T) [* topk_weights]

    Args:
        x: Input activations
        topk_ids: Expert assignments [M, top_k]
        topk_weights: Router weights [M, top_k]
        output: Output tensor to accumulate into [M, top_k, out_dim]
        lora_a: LoRA A weights [num_experts, rank, hidden_dim]
        lora_b: LoRA B weights [num_experts, out_dim, rank]
        sorted_token_ids: Pre-sorted token indices from moe_align_block_size
        expert_ids: Expert ID per block from moe_align_block_size
        num_tokens_post_padded: Padded token count
        top_k: Number of experts per token (use 1 if x is already per-expert)
        config: Triton kernel config
        mul_routed_weight: Whether to multiply by router weights
    """
    M = topk_ids.shape[0]
    rank = lora_a.shape[1]
    hidden_dim = lora_a.shape[2]
    out_dim = lora_b.shape[1]
    EM = sorted_token_ids.shape[0]

    block_size_m = config["BLOCK_SIZE_M"]
    block_size_out = config.get("BLOCK_SIZE_N", 64)
    block_size_hidden = config.get("BLOCK_SIZE_K", 64)
    # For LoRA, rank is typically small (8-64), so we can fit it in one block
    block_size_rank = max(16, triton.next_power_of_2(rank))

    grid = (
        triton.cdiv(EM, block_size_m),
        triton.cdiv(out_dim, block_size_out),
    )

    _fused_moe_lora_kernel[grid](
        x,
        lora_a,
        lora_b,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        hidden_dim,
        rank,
        out_dim,
        EM,
        M * topk_ids.shape[1],  # num_valid_tokens
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        output.stride(1),  # stride for M (token*topk dim)
        output.stride(2),  # stride for N (out_dim)
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_RANK=block_size_rank,
        BLOCK_SIZE_HIDDEN=block_size_hidden,
        BLOCK_SIZE_OUT=block_size_out,
        num_warps=config.get("NUM_WARPS", 4),
        num_stages=config.get("NUM_STAGES", 2),
    )
