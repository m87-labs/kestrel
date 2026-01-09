import torch
import triton
import triton.language as tl


class _FixedBuffer:
    """Device-aware buffer that is pre-allocated once and never resized.

    After the first allocation, requesting a larger size will raise an error.
    This ensures CUDA graph replay uses stable pointers.
    """

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

        if self._tensor is None:
            self._tensor = torch.empty(numel, device=device, dtype=dtype)
        elif self._tensor.numel() < numel:
            raise RuntimeError(
                f"LoRA buffer overflow: requested {numel} elements but "
                f"only {self._tensor.numel()} allocated. Increase max_seq_length "
                f"or ensure preallocate_lora_buffers() is called."
            )
        elif self._tensor.device != device or self._tensor.dtype != dtype:
            raise RuntimeError(
                f"LoRA buffer device/dtype mismatch: buffer on {self._tensor.device} "
                f"({self._tensor.dtype}), requested {device} ({dtype})."
            )
        return self._tensor[:numel].view(*shape)


# =============================================================================
# Batched MoE LoRA Kernels
# =============================================================================
#
# These kernels use a 2D grid (M, N) and loop over LoRA adapters inside each CTA.
# This approach uses moe_lora_align_block_size which produces per-LoRA routing:
#   sorted_token_ids: [max_loras, EM]
#   expert_ids: [max_loras, num_blocks]
#   num_tokens_post_padded: [max_loras]
#
# Benefits over super-expert approach:
#   - No expert-space explosion (avoids max_loras * num_experts routing)
#   - Cleaner slot-0 handling (filtered at routing time)
#   - Unified path for dense + MoE LoRA


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
    IS_PRIMARY: tl.constexpr,  # True for shrink (signal), False for expand (wait)
    MAX_LORAS: tl.constexpr,
):
    """Unified batched MoE LoRA kernel for both shrink and expand operations.

    Uses IS_PRIMARY to control PDL behavior:
    - IS_PRIMARY=True (shrink): signals gdc_launch_dependents after pointer setup
    - IS_PRIMARY=False (expand): waits with gdc_wait before loading input
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
    if IS_PRIMARY:
        tl.extra.cuda.gdc_launch_dependents()

    # Wait once per CTA before the expand loop to ensure shrink is complete.
    if not IS_PRIMARY:
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


_BATCHED_INTERMEDIATE_BUFFER = _FixedBuffer()
_BATCHED_LORA_OUTPUT_BUFFER = _FixedBuffer()  # Currently unused
_SINGLE_INTERMEDIATE_BUFFER = _FixedBuffer()


def preallocate_lora_buffers(
    max_num_tokens: int,
    top_k: int,
    max_lora_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-allocate LoRA intermediate buffers to ensure stable pointers.

    Args:
        max_num_tokens: Maximum tokens in any forward pass.
        top_k: Number of experts per token.
        max_lora_rank: Maximum LoRA rank used.
        device: Target device.
        dtype: Data type for buffers.
    """
    # Both buffers have shape (num_valid_tokens, rank) where num_valid_tokens = M * top_k
    max_valid_tokens = max_num_tokens * top_k
    _BATCHED_INTERMEDIATE_BUFFER.get(
        (max_valid_tokens, max_lora_rank),
        device=device,
        dtype=dtype,
    )
    _SINGLE_INTERMEDIATE_BUFFER.get(
        (max_valid_tokens, max_lora_rank),
        device=device,
        dtype=dtype,
    )


def _get_lora_kernel_params(
    config: dict[str, int] | None,
    *,
    block_size_out: int,
    block_size_hidden: int,
    num_warps: int,
    num_stages: int,
) -> tuple[int, int, int, int]:
    if not config:
        return block_size_out, block_size_hidden, num_warps, num_stages

    def pick(keys: tuple[str, ...], default: int) -> int:
        for key in keys:
            if key in config:
                return int(config[key])
        return default

    block_size_out = pick(("BLOCK_SIZE_N", "block_size_n"), block_size_out)
    block_size_hidden = pick(("BLOCK_SIZE_K", "block_size_k"), block_size_hidden)
    num_warps = pick(("NUM_WARPS", "num_warps"), num_warps)
    num_stages = pick(("NUM_STAGES", "num_stages"), num_stages)
    return block_size_out, block_size_hidden, num_warps, num_stages


@torch.inference_mode()
def apply_moe_lora_batched(
    x: torch.Tensor,  # [M, hidden_dim] or [M * top_k, hidden_dim]
    topk_weights: torch.Tensor,  # [M, top_k]
    output: torch.Tensor,  # [M, top_k, out_dim]
    lora_a: torch.Tensor,  # [max_loras * num_experts, rank, hidden_dim]
    lora_b: torch.Tensor,  # [max_loras * num_experts, out_dim, rank]
    sorted_token_ids: torch.Tensor,  # [max_loras, EM]
    expert_ids: torch.Tensor,  # [max_loras, num_blocks]
    num_tokens_post_padded: torch.Tensor,  # [max_loras]
    top_k: int,
    num_experts: int,
    block_size_m: int,
    *,
    mul_routed_weight: bool = False,
    shrink_config: dict[str, int] | None = None,
    expand_config: dict[str, int] | None = None,
    block_size_out: int = 64,
    block_size_hidden: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Apply batched MoE LoRA using per-LoRA routing from moe_lora_align_block_size.

    Uses a 2D grid and loops over LoRA adapters inside each CTA (lora_idx == lora_id).
    Optional shrink_config/expand_config override BLOCK_SIZE_N/K and NUM_WARPS/STAGES
    separately for the two phases.
    """
    M = topk_weights.shape[0]
    max_loras = sorted_token_ids.shape[0]
    EM = sorted_token_ids.shape[1]
    rank = lora_a.shape[1]
    hidden_dim = lora_a.shape[2]
    out_dim = lora_b.shape[1]
    num_valid_tokens = M * topk_weights.shape[1]

    # Use EM (sorted_token_ids dim 1) as max_tokens_padded - already computed by routing
    max_tokens_padded = EM
    if max_tokens_padded == 0:
        return  # No active LoRAs

    num_m_blocks = triton.cdiv(max_tokens_padded, block_size_m)
    shrink_block_size_out, shrink_block_size_hidden, shrink_num_warps, shrink_num_stages = (
        _get_lora_kernel_params(
            shrink_config,
            block_size_out=block_size_out,
            block_size_hidden=block_size_hidden,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    expand_block_size_out, expand_block_size_hidden, expand_num_warps, expand_num_stages = (
        _get_lora_kernel_params(
            expand_config,
            block_size_out=block_size_out,
            block_size_hidden=block_size_hidden,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )

    num_n_blocks = triton.cdiv(out_dim, expand_block_size_out)

    # Reshape output for kernel access: [M, top_k, out_dim] -> [M * top_k, out_dim]
    output_flat = output.view(num_valid_tokens, out_dim)

    # Split shrink+expand with PDL for larger token counts using unified kernel
    intermediate = _BATCHED_INTERMEDIATE_BUFFER.get(
        (num_valid_tokens, rank),
        device=x.device,
        dtype=output.dtype,  # Use output dtype, not float32
    )

    # Use 2D grid (M, N) and loop over LoRAs inside the kernel.
    num_rank_blocks = triton.cdiv(rank, shrink_block_size_out)
    shrink_grid = (num_m_blocks, num_rank_blocks)

    # Shrink: x @ lora_a.T -> intermediate
    # lora_a shape: [E, rank, hidden] -> N=rank, K=hidden
    _batched_moe_lora_kernel[shrink_grid](
        x,  # a_ptr: input
        lora_a,  # b_ptr: weights [E, N=rank, K=hidden]
        intermediate,  # c_ptr: output
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        rank,  # N
        hidden_dim,  # K
        EM,
        num_valid_tokens,
        num_experts,
        max_tokens_padded,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        top_k=top_k,
        MUL_ROUTED_WEIGHT=False,  # Never multiply in shrink
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=shrink_block_size_out,  # For rank dimension
        BLOCK_SIZE_K=shrink_block_size_hidden,
        IS_PRIMARY=True,
        MAX_LORAS=max_loras,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
        launch_pdl=True,
    )

    expand_grid = (num_m_blocks, num_n_blocks)

    # Expand: intermediate @ lora_b.T -> output (direct store)
    # lora_b shape: [E, out_dim, rank] -> N=out_dim, K=rank
    _batched_moe_lora_kernel[expand_grid](
        intermediate,  # a_ptr: input
        lora_b,  # b_ptr: weights [E, N=out_dim, K=rank]
        output_flat,  # c_ptr: output (kernel does read-modify-write)
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        out_dim,  # N
        rank,  # K
        EM,
        num_valid_tokens,
        num_experts,
        max_tokens_padded,
        intermediate.stride(0),
        intermediate.stride(1),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        output_flat.stride(0),
        output_flat.stride(1),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        top_k=1,  # Intermediate is already per-token*topk
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=expand_block_size_out,
        BLOCK_SIZE_K=expand_block_size_hidden,  # For rank - may need tuning
        IS_PRIMARY=False,
        MAX_LORAS=max_loras,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
        launch_pdl=True,
    )


@torch.inference_mode()
def apply_moe_lora_single(
    x: torch.Tensor,  # [M, hidden_dim]
    topk_ids: torch.Tensor,  # [M, top_k]
    topk_weights: torch.Tensor,  # [M, top_k]
    output: torch.Tensor,  # [M, top_k, out_dim]
    lora_a: torch.Tensor,  # [max_loras * num_experts, rank, hidden_dim]
    lora_b: torch.Tensor,  # [max_loras * num_experts, out_dim, rank]
    sorted_token_ids: torch.Tensor,  # [EM] - 1D from moe_align_block_size
    expert_ids: torch.Tensor,  # [num_blocks] - 1D from moe_align_block_size
    num_tokens_post_padded: torch.Tensor,  # Scalar
    lora_id: int,  # Which LoRA adapter to use
    top_k: int,
    num_experts: int,
    block_size_m: int,
    *,
    mul_routed_weight: bool = False,
    shrink_config: dict[str, int] | None = None,
    expand_config: dict[str, int] | None = None,
    block_size_out: int = 64,
    block_size_hidden: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Apply single-LoRA MoE using baseline routing.

    This is optimized for prefill where only one LoRA is active. Uses standard
    moe_align_block_size routing and reuses the batched kernel with MAX_LORAS=1.

    The expert_ids are offset by lora_id * num_experts so the kernel indexes
    into the correct slice of the weight tensors.

    Args:
        x: Input activations [M, hidden_dim]
        topk_ids: Expert assignments [M, top_k]
        topk_weights: Router weights [M, top_k]
        output: Output tensor to accumulate into [M, top_k, out_dim]
        lora_a: LoRA A weights [max_loras * num_experts, rank, hidden_dim]
        lora_b: LoRA B weights [max_loras * num_experts, out_dim, rank]
        sorted_token_ids: Pre-sorted token indices from moe_align_block_size [EM]
        expert_ids: Expert ID per block from moe_align_block_size [num_blocks]
        num_tokens_post_padded: Padded token count (scalar)
        lora_id: Which LoRA adapter (0-indexed)
        top_k: Number of experts per token
        num_experts: Number of experts
        block_size_m: Routing block size (must match moe_align_block_size).
        shrink_config/expand_config: Optional per-phase overrides for BLOCK_SIZE_N/K
            and NUM_WARPS/STAGES.
        mul_routed_weight: Whether to multiply by router weights
    """
    M = topk_ids.shape[0]
    rank = lora_a.shape[1]
    hidden_dim = lora_a.shape[2]
    out_dim = lora_b.shape[1]
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = M * topk_ids.shape[1]

    # Offset expert_ids by lora_id * num_experts so the kernel indexes
    # into the correct slice of weight tensors. With MAX_LORAS=1, lora_idx=0,
    # so super_expert_id = 0 * num_experts + expert_id = expert_id (already offset).
    expert_ids_offset = expert_ids + lora_id * num_experts

    # Reshape 1D routing to 2D with shape [1, ...] for batched kernel
    sorted_token_ids_2d = sorted_token_ids.unsqueeze(0)  # [1, EM]
    expert_ids_2d = expert_ids_offset.unsqueeze(0)  # [1, num_blocks]
    num_tokens_post_padded_1d = num_tokens_post_padded.view(1)  # [1]

    # Reshape output for kernel access: [M, top_k, out_dim] -> [M * top_k, out_dim]
    output_flat = output.view(num_valid_tokens, out_dim)

    shrink_block_size_out, shrink_block_size_hidden, shrink_num_warps, shrink_num_stages = (
        _get_lora_kernel_params(
            shrink_config,
            block_size_out=block_size_out,
            block_size_hidden=block_size_hidden,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    expand_block_size_out, expand_block_size_hidden, expand_num_warps, expand_num_stages = (
        _get_lora_kernel_params(
            expand_config,
            block_size_out=block_size_out,
            block_size_hidden=block_size_hidden,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )

    max_tokens_padded = EM
    num_m_blocks = triton.cdiv(max_tokens_padded, block_size_m)
    num_n_blocks = triton.cdiv(out_dim, expand_block_size_out)

    # Split shrink+expand with PDL for larger token counts
    intermediate = _SINGLE_INTERMEDIATE_BUFFER.get(
        (num_valid_tokens, rank),
        device=x.device,
        dtype=output.dtype,
    )

    num_rank_blocks = triton.cdiv(rank, shrink_block_size_out)
    shrink_grid = (num_m_blocks, num_rank_blocks)
    expand_grid = (num_m_blocks, num_n_blocks)

    # Shrink: x @ lora_a.T -> intermediate
    _batched_moe_lora_kernel[shrink_grid](
        x,
        lora_a,
        intermediate,
        topk_weights,
        sorted_token_ids_2d,
        expert_ids_2d,
        num_tokens_post_padded_1d,
        rank,  # N
        hidden_dim,  # K
        EM,
        num_valid_tokens,
        num_experts,
        max_tokens_padded,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        sorted_token_ids_2d.stride(0),
        expert_ids_2d.stride(0),
        top_k=top_k,
        MUL_ROUTED_WEIGHT=False,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=shrink_block_size_out,
        BLOCK_SIZE_K=shrink_block_size_hidden,
        IS_PRIMARY=True,
        MAX_LORAS=1,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
        launch_pdl=True,
    )

    # Expand: intermediate @ lora_b.T -> output
    _batched_moe_lora_kernel[expand_grid](
        intermediate,
        lora_b,
        output_flat,
        topk_weights,
        sorted_token_ids_2d,
        expert_ids_2d,
        num_tokens_post_padded_1d,
        out_dim,  # N
        rank,  # K
        EM,
        num_valid_tokens,
        num_experts,
        max_tokens_padded,
        intermediate.stride(0),
        intermediate.stride(1),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        output_flat.stride(0),
        output_flat.stride(1),
        sorted_token_ids_2d.stride(0),
        expert_ids_2d.stride(0),
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=expand_block_size_out,
        BLOCK_SIZE_K=expand_block_size_hidden,
        IS_PRIMARY=False,
        MAX_LORAS=1,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
        launch_pdl=True,
    )
