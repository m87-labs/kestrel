"""Triton MoE-LoRA kernels — imported only when the `triton` package is
available. Kept out of ``lora_kernels.py`` so the top-level module can
be imported in Triton-less environments (Windows, some Jetson images)
and transparently delegate to a plain-torch fallback."""


import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "EM",
        "num_valid_tokens",
        "max_tokens_padded",
        "stride_tl",
        "stride_el",
    ]
)
def _batched_moe_lora_kernel(
    # Input/output pointers
    a_ptr,  # Input: [num_valid_tokens, K] - x for shrink, intermediate for expand
    b_ptr,  # Weights: [max_loras * num_experts, N, K] - lora_a or lora_b
    c_ptr,  # Output: [num_valid_tokens, N] - intermediate for shrink, output for expand
    topk_weights_ptr,  # [M, top_k]
    sorted_token_ids_ptr,  # [max_loras, EM]
    expert_ids_ptr,  # [max_loras, num_blocks]
    num_tokens_post_padded_ptr,  # [max_loras]
    # Dimensions
    N,  # Output dim (rank for shrink, out_dim for expand)
    K,  # Input dim (hidden for shrink, rank for expand)
    EM,
    num_valid_tokens,
    num_experts,
    max_tokens_padded,
    # Strides for a [num_valid_tokens, K]
    stride_am,
    stride_ak,
    # Strides for b [max_loras * num_experts, N, K]
    stride_be,
    stride_bn,
    stride_bk,
    # Strides for c [num_valid_tokens, N]
    stride_cm,
    stride_cn,
    # Strides for sorted_token_ids [max_loras, EM]
    stride_tl,
    # Strides for expert_ids [max_loras, num_blocks]
    stride_el,
    # Constexprs
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_PDL: tl.constexpr,
    IS_PRIMARY: tl.constexpr,  # True for shrink (signal), False for expand (wait)
    MAX_LORAS: tl.constexpr,
):
    """Unified batched MoE LoRA kernel for both shrink and expand operations.

    Uses (USE_PDL, IS_PRIMARY) to control PDL behavior:
    - USE_PDL=True, IS_PRIMARY=True (shrink): signals gdc_launch_dependents
    - USE_PDL=True, IS_PRIMARY=False (expand): waits with gdc_wait
    """
    # Use natural 2D grid indexing (no swizzling overhead)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m_offset = pid_m * BLOCK_SIZE_M
    offs_m = pid_m_offset + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < EM
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Hint dependents once per CTA (primary/shrink only).
    if IS_PRIMARY and USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Wait once per CTA before the expand loop to ensure shrink is complete.
    if not IS_PRIMARY and USE_PDL:
        tl.extra.cuda.gdc_wait()

    for lora_idx in tl.static_range(0, MAX_LORAS):
        # Load per-LoRA token count and skip inactive CTAs
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_idx)
        active = pid_m_offset < num_tokens_post_padded

        if active:
            # Load expert_id for this block
            expert_id = tl.load(expert_ids_ptr + lora_idx * stride_el + pid_m)
            if expert_id != -1:
                # Compute super-expert index: lora_idx is used directly (identity mapping)
                super_expert_id = lora_idx * num_experts + expert_id

                # Load token indices
                offs_token = tl.load(
                    sorted_token_ids_ptr + lora_idx * stride_tl + offs_m,
                    mask=m_mask,
                    other=0,
                )
                token_mask = offs_token < num_valid_tokens

                # Compute pointers with pointer arithmetic for efficient K-loop
                a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
                b_ptrs = b_ptr + super_expert_id * stride_be + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

                # Initialize accumulator
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                # Main GEMM loop - load B first (prefetch weights), then A
                for k_start in range(0, K, BLOCK_SIZE_K):
                    k_remaining = K - k_start

                    # Load B block first (prefetch weights while waiting for A data)
                    # No N mask needed due to modulo - all threads access valid data
                    b_block = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < k_remaining,
                        other=0.0,
                    )

                    # Load A block
                    a_block = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0,
                    )

                    accumulator += tl.dot(a_block, b_block)

                    # Advance pointers
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk

                # Apply routing weight if needed
                if MUL_ROUTED_WEIGHT:
                    weights = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
                    accumulator = accumulator * weights[:, None]

                # Store output - use non-modulo offset for correct indexing
                c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
                # Direct store is safe for expand: sorted_token_ids is a per-LoRA permutation of
                # flattened [token, top_k] indices and this kernel does not split-K, so no cross-block
                # accumulation occurs.
                tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _batched_fused_moe_lora_kernel(
    # Input/output pointers
    x_ptr,  # [M, hidden_dim] or [M * top_k, hidden_dim]
    lora_a_ptr,  # [max_loras * num_experts, rank, hidden_dim]
    lora_b_ptr,  # [max_loras * num_experts, out_dim, rank]
    output_ptr,  # [M, top_k, out_dim]
    topk_weights_ptr,  # [M, top_k]
    sorted_token_ids_ptr,  # [max_loras, EM]
    expert_ids_ptr,  # [max_loras, num_blocks]
    num_tokens_post_padded_ptr,  # [max_loras]
    # Dimensions
    hidden_dim,
    rank,
    out_dim,
    EM,
    num_valid_tokens,
    num_experts,
    max_tokens_padded,  # For consistent grid decomposition
    # Strides for x
    stride_xm,
    stride_xk,
    # Strides for lora_a [max_loras * num_experts, rank, hidden_dim]
    stride_ae,
    stride_ar,
    stride_ak,
    # Strides for lora_b [max_loras * num_experts, out_dim, rank]
    stride_be,
    stride_bn,
    stride_br,
    # Strides for output [M, top_k, out_dim]
    stride_om,
    stride_on,
    # Strides for sorted_token_ids [max_loras, EM]
    stride_tl,
    # Strides for expert_ids [max_loras, num_blocks]
    stride_el,
    # Constexprs
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_RANK: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    """Batched fused MoE LoRA kernel with natural 2D grid indexing."""
    # Use natural 2D grid indexing (no swizzling overhead)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lora_idx = tl.program_id(2)

    # Load per-LoRA token count and early exit if inactive
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_idx)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Load expert_id for this block
    expert_id = tl.load(expert_ids_ptr + lora_idx * stride_el + pid_m)
    if expert_id == -1:
        return

    # Compute super-expert index: lora_idx is used directly (identity mapping)
    super_expert_id = lora_idx * num_experts + expert_id

    # Load token indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(
        sorted_token_ids_ptr + lora_idx * stride_tl + offs_m,
        mask=offs_m < EM,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    # Phase 1: Shrink - compute x @ A.T -> [BLOCK_SIZE_M, rank]
    intermediate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=tl.float32)

    for k_start in range(0, hidden_dim, BLOCK_SIZE_HIDDEN):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
        k_mask = k_offs < hidden_dim

        x_ptrs = x_ptr + (offs_token[:, None] // top_k) * stride_xm + k_offs[None, :] * stride_xk
        x_block = tl.load(
            x_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0
        )

        a_ptrs = lora_a_ptr + super_expert_id * stride_ae + tl.arange(0, BLOCK_SIZE_RANK)[None, :] * stride_ar + k_offs[:, None] * stride_ak
        a_block = tl.load(
            a_ptrs,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_RANK)[None, :] < rank),
            other=0.0,
        )

        intermediate += tl.dot(x_block, a_block)

    # Phase 2: Expand - compute intermediate @ B.T -> [BLOCK_SIZE_M, BLOCK_SIZE_OUT]
    offs_n = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    n_mask = offs_n < out_dim

    output_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_OUT), dtype=tl.float32)

    for r_start in range(0, rank, BLOCK_SIZE_RANK):
        r_offs = r_start + tl.arange(0, BLOCK_SIZE_RANK)
        r_mask = r_offs < rank

        if r_start == 0:
            inter_block = intermediate.to(output_ptr.dtype.element_ty)
        else:
            inter_block = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_RANK), dtype=output_ptr.dtype.element_ty
            )

        b_ptrs = lora_b_ptr + super_expert_id * stride_be + offs_n[None, :] * stride_bn + r_offs[:, None] * stride_br
        b_block = tl.load(
            b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0
        )

        output_acc += tl.dot(inter_block, b_block)

    # Apply routing weight if needed
    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        output_acc = output_acc * weights[:, None]

    # Read-modify-write to accumulate into output
    out_ptrs = output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = token_mask[:, None] & n_mask[None, :]
    out_prev = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_new = (out_prev + output_acc).to(output_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_new, mask=out_mask)
