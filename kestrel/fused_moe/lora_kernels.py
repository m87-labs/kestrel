import torch
import triton
import triton.language as tl


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
        numel = 1
        for dim in shape:
            numel *= dim
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


_INTERMEDIATE_BUFFER = _ResizableBuffer()


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
    # Accumulate in float32 for precision. Keep operands in fp16/bf16 to enable
    # tensor core use in tl.dot (vLLM/Punica style).
    intermediate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=tl.float32)

    for k_start in range(0, hidden_dim, BLOCK_SIZE_HIDDEN):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
        k_mask = k_offs < hidden_dim

        # Load x block: [BLOCK_SIZE_M, BLOCK_SIZE_HIDDEN]
        x_ptrs = x_ptr + (offs_token[:, None] // top_k) * stride_xm + k_offs[None, :] * stride_xk
        x_block = tl.load(
            x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0
        )

        # Load A block: [BLOCK_SIZE_HIDDEN, BLOCK_SIZE_RANK] (transposed from [rank, hidden])
        # A is [num_experts, rank, hidden_dim], we want A[expert].T = [hidden_dim, rank]
        a_ptrs = lora_a_ptr + expert_id * stride_ae + tl.arange(0, BLOCK_SIZE_RANK)[None, :] * stride_ar + k_offs[:, None] * stride_ak
        a_block = tl.load(
            a_ptrs,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_RANK)[None, :] < rank),
            other=0.0,
        )

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
            # Cast intermediate to enable tensor core usage for the expand GEMM.
            inter_block = intermediate.to(output_ptr.dtype.element_ty)
        else:
            # For ranks > BLOCK_SIZE_RANK, we'd need to recompute or store
            # For typical LoRA ranks (8-32), this won't happen with BLOCK_SIZE_RANK=64
            inter_block = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=output_ptr.dtype.element_ty
            )

        # Load B block: [BLOCK_SIZE_RANK, BLOCK_SIZE_OUT] (transposed from [out_dim, rank])
        b_ptrs = lora_b_ptr + expert_id * stride_be + offs_n[None, :] * stride_bn + r_offs[:, None] * stride_br
        b_block = tl.load(
            b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0
        )

        output_acc += tl.dot(inter_block, b_block)

    # Apply routing weight if needed
    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        output_acc = output_acc * weights[:, None]

    # Store output (accumulate into existing output).
    #
    # NOTE: We use a non-atomic read+add+write here because each element of the
    # flattened [M * top_k, out_dim] output is written by exactly one program
    # instance (pid_m selects a disjoint BLOCK_SIZE_M range of sorted assignment
    # ids; pid_n selects a disjoint BLOCK_SIZE_OUT range of output columns).
    #
    # The previous atomic_add implementation scaled poorly at large M.
    out_ptrs = output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = token_mask[:, None] & n_mask[None, :]
    out_prev = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_new = (out_prev + output_acc).to(output_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_new, mask=out_mask)


@triton.jit
def _moe_lora_shrink_kernel(
    # Input pointers
    x_ptr,  # [M, hidden_dim] or [M * top_k, hidden_dim]
    lora_a_ptr,  # [num_experts, rank, hidden_dim]
    intermediate_ptr,  # [num_valid_tokens, rank] (fp32)
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    hidden_dim,
    rank,
    EM,
    num_valid_tokens,
    # Strides for x
    stride_xm,
    stride_xk,
    # Strides for lora_a [num_experts, rank, hidden_dim]
    stride_ae,
    stride_ar,
    stride_ak,
    # Strides for intermediate [num_valid_tokens, rank]
    stride_im,
    stride_ir,
    # Constexprs
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_RANK: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    pid_m = tl.program_id(0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    # Shrink: x @ A.T -> [BLOCK_SIZE_M, rank]
    intermediate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=tl.float32)

    for k_start in range(0, hidden_dim, BLOCK_SIZE_HIDDEN):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
        k_mask = k_offs < hidden_dim

        x_ptrs = x_ptr + (offs_token[:, None] // top_k) * stride_xm + k_offs[None, :] * stride_xk
        x_block = tl.load(x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

        a_ptrs = (
            lora_a_ptr
            + expert_id * stride_ae
            + tl.arange(0, BLOCK_SIZE_RANK)[None, :] * stride_ar
            + k_offs[:, None] * stride_ak
        )
        a_block = tl.load(
            a_ptrs,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_RANK)[None, :] < rank),
            other=0.0,
        )
        intermediate += tl.dot(x_block, a_block)

    r_offs = tl.arange(0, BLOCK_SIZE_RANK)
    r_mask = r_offs < rank
    inter_ptrs = intermediate_ptr + offs_token[:, None] * stride_im + r_offs[None, :] * stride_ir
    tl.store(inter_ptrs, intermediate, mask=token_mask[:, None] & r_mask[None, :])


@triton.jit
def _moe_lora_expand_kernel(
    # Input pointers
    intermediate_ptr,  # [num_valid_tokens, rank] (fp32)
    lora_b_ptr,  # [num_experts, out_dim, rank]
    output_ptr,  # [M, top_k, out_dim]
    topk_weights_ptr,  # [M, top_k]
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    rank,
    out_dim,
    EM,
    num_valid_tokens,
    # Strides for intermediate [num_valid_tokens, rank]
    stride_im,
    stride_ir,
    # Strides for lora_b [num_experts, out_dim, rank]
    stride_be,
    stride_bn,
    stride_br,
    # Strides for output [M, top_k, out_dim]
    stride_om,
    stride_on,
    # Constexprs
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_RANK: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    offs_n = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    n_mask = offs_n < out_dim

    r_offs = tl.arange(0, BLOCK_SIZE_RANK)
    r_mask = r_offs < rank

    # Load intermediate and cast to enable tensor core usage.
    inter_ptrs = intermediate_ptr + offs_token[:, None] * stride_im + r_offs[None, :] * stride_ir
    inter_block = tl.load(
        inter_ptrs, mask=token_mask[:, None] & r_mask[None, :], other=0.0
    ).to(output_ptr.dtype.element_ty)

    b_ptrs = lora_b_ptr + expert_id * stride_be + offs_n[None, :] * stride_bn + r_offs[:, None] * stride_br
    b_block = tl.load(b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0)

    output_acc = tl.dot(inter_block, b_block)

    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        output_acc = output_acc * weights[:, None]

    out_ptrs = output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = token_mask[:, None] & n_mask[None, :]
    out_prev = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_new = (out_prev + output_acc).to(output_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_new, mask=out_mask)


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
    num_valid_tokens = M * topk_ids.shape[1]

    block_size_m = config["BLOCK_SIZE_M"]
    block_size_out = config.get("BLOCK_SIZE_N", 64)
    block_size_hidden = config.get("BLOCK_SIZE_K", 64)
    # For LoRA, rank is typically small (8-64), so we can fit it in one block
    block_size_rank = max(16, triton.next_power_of_2(rank))
    num_warps = config.get("NUM_WARPS", config.get("num_warps", 4))
    num_stages = config.get("NUM_STAGES", config.get("num_stages", 2))

    # Heuristic:
    # - For small token counts (decode), a single fused kernel is often faster
    #   than paying the intermediate buffer traffic + second launch.
    # - For prefill (large M), the fused kernel is prohibitively expensive
    #   because it recomputes the shrink GEMM for every output tile.
    use_fused = num_valid_tokens <= 256

    if use_fused:
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
            num_valid_tokens,
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    # Split shrink+expand to avoid recomputing the shrink GEMM per output tile.
    intermediate = _INTERMEDIATE_BUFFER.get(
        (num_valid_tokens, rank),
        device=x.device,
        dtype=torch.float32,
    )

    shrink_grid = (triton.cdiv(EM, block_size_m),)
    _moe_lora_shrink_kernel[shrink_grid](
        x,
        lora_a,
        intermediate,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        hidden_dim,
        rank,
        EM,
        num_valid_tokens,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_RANK=block_size_rank,
        BLOCK_SIZE_HIDDEN=block_size_hidden,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    expand_grid = (
        triton.cdiv(EM, block_size_m),
        triton.cdiv(out_dim, block_size_out),
    )
    _moe_lora_expand_kernel[expand_grid](
        intermediate,
        lora_b,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        rank,
        out_dim,
        EM,
        num_valid_tokens,
        intermediate.stride(0),
        intermediate.stride(1),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        output.stride(1),  # stride for M (token*topk dim)
        output.stride(2),  # stride for N (out_dim)
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_RANK=block_size_rank,
        BLOCK_SIZE_OUT=block_size_out,
        num_warps=num_warps,
        num_stages=num_stages,
    )
