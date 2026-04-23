"""MoE LoRA apply-kernels: batched (multi-LoRA) and single (one LoRA).

Each public entry (``apply_moe_lora_batched`` / ``apply_moe_lora_single``)
executes shrink (x @ A.T → intermediate) then expand (intermediate @ B.T →
output) via the Triton kernels defined in ``_lora_triton_kernels``.

Triton is not available on every target (no Windows wheels; some Jetson
images also lack it). In those environments the module still imports —
``import kestrel.fused_moe`` is on the hot path even when no LoRA adapters
are attached — but ``apply_moe_lora_batched`` / ``apply_moe_lora_single``
raise a clear ``RuntimeError`` at call time. Base moondream2 / moondream3
inference works unchanged; only the LoRA-adapter codepath is gated.
"""


import torch

try:
    import triton

    _TRITON_AVAILABLE = True
    from ._lora_triton_kernels import (
        _batched_fused_moe_lora_kernel,
        _batched_moe_lora_kernel,
    )
except ImportError:
    triton = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


_NO_TRITON_MESSAGE = (
    "MoE LoRA adapters require the `triton` package, which is not "
    "available on this platform. Disable LoRA (run without adapters) "
    "or use a Linux / aarch64 build where triton is installable."
)


from kestrel.utils.buffers import FixedBuffer


def _supports_pdl(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9


_BATCHED_INTERMEDIATE_BUFFER = FixedBuffer("LoRA batched intermediate")
_SINGLE_INTERMEDIATE_BUFFER = FixedBuffer("LoRA single intermediate")


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


# ---------------------------------------------------------------------------
# Public entries
# ---------------------------------------------------------------------------


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
    if not _TRITON_AVAILABLE:
        raise RuntimeError(_NO_TRITON_MESSAGE)

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
    launch_pdl = _supports_pdl(x.device)

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
        USE_PDL=launch_pdl,
        IS_PRIMARY=True,
        MAX_LORAS=max_loras,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
        launch_pdl=launch_pdl,
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
        USE_PDL=launch_pdl,
        IS_PRIMARY=False,
        MAX_LORAS=max_loras,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
        launch_pdl=launch_pdl,
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

    if not _TRITON_AVAILABLE:
        raise RuntimeError(_NO_TRITON_MESSAGE)

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
    launch_pdl = _supports_pdl(x.device)

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
        USE_PDL=launch_pdl,
        IS_PRIMARY=True,
        MAX_LORAS=1,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
        launch_pdl=launch_pdl,
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
        USE_PDL=launch_pdl,
        IS_PRIMARY=False,
        MAX_LORAS=1,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
        launch_pdl=launch_pdl,
    )
