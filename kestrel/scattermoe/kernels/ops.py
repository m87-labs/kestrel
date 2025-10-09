import torch
import triton
import triton.language as tl

BLOCK_M = 128
ALLOW_TF32 = True


def compileable_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    counts = torch.zeros(minlength, dtype=torch.long, device=x.device)
    if x.numel() == 0:
        return counts
    ones = torch.ones_like(x, dtype=torch.long)
    counts.scatter_add_(0, x, ones)
    return counts


def flatten_and_sort(expert_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = expert_idxs.reshape(-1)
    return torch.sort(flattened)


def padded_block_indices(
    sorted_expert_idxs: torch.Tensor,
    k: int,
    N_BLOCK_SIZE: int = BLOCK_M,
    *,
    out: torch.Tensor | None = None,
    block_idx_template: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_counts = compileable_bincount(sorted_expert_idxs, minlength=k)
    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts

    total_blocks = padded_expert_block_end[-1]

    if out is None:
        block_idxs = torch.arange(
            total_blocks,
            dtype=sorted_expert_idxs.dtype,
            device=sorted_expert_idxs.device,
        )
    else:
        if out.dim() != 1 or out.dtype != torch.long:
            raise ValueError("Output buffer for padded_block_indices must be 1D long tensor")
        if block_idx_template is not None and block_idx_template.size(0) != out.size(0):
            raise ValueError("block_idx_template must match out buffer shape")
        torch._assert(
            torch.tensor(out.size(0), device=total_blocks.device, dtype=total_blocks.dtype)
            >= total_blocks,
            "Output buffer has fewer slots than required blocks",
        )
        block_idxs = (
            block_idx_template
            if block_idx_template is not None
            else torch.arange(
                out.size(0),
                dtype=sorted_expert_idxs.dtype,
                device=sorted_expert_idxs.device,
            )
        )

    block_mask = (block_idxs[:, None] < padded_expert_block_start) | (
        block_idxs[:, None] >= padded_expert_block_end
    )

    inactive_mask = block_idxs >= total_blocks
    block_mask = block_mask | inactive_mask[:, None]

    expanded_block_idxs = (
        N_BLOCK_SIZE * (block_idxs[:, None] - padded_expert_block_start)
        + expert_boundaries_start
    )
    sentinel = sorted_expert_idxs.size(0)
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, sentinel)
    expanded_block_idxs = expanded_block_idxs.min(dim=-1).values

    if out is not None:
        out.copy_(expanded_block_idxs)
        return out, expert_boundaries_end, total_blocks

    return expanded_block_idxs, expert_boundaries_end, total_blocks


def _scatter2scatter_configs():
    return [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    ]


@triton.autotune(configs=_scatter2scatter_configs(), key=["M", "N", "K"])
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def _scatter2scatter(
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    Y_ptr,
    stride_ym,
    stride_yn,
    grouped_idx_ptr,
    expert_idxs_ptr,
    block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    OUT_M,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr,
    y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + M_block_id)
    if block_start_idx >= (FAN_OUT * M):
        return
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < (FAN_OUT * M), other=E)
    E_idx = tl.min(E_idxs)
    valid_mask = E_idxs < E
    E_mask = (E_idxs == E_idx) & valid_mask
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = (
        W_ptr
        + K_block[:, None] * stride_wk
        + N_block[None, :] * stride_wn
        + E_idx * stride_we
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK or K_block_id < (iters - 1):
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])


def scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    k: int,
    padded_block_idxs: torch.Tensor,
    *,
    x_grouped: bool = False,
    y_grouped: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k

    y_dim = W.size(-1)
    length = sorted_expert_idxs.size(0)
    if out is None:
        out = torch.empty((length, y_dim), device=X.device, dtype=X.dtype)
    else:
        if out.size(0) != length or out.size(1) != y_dim:
            raise ValueError("Provided output tensor has incompatible shape")

    scatter2scatter_compileable(
        out,
        W,
        X,
        k,
        padded_block_idxs,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        x_grouped,
        y_grouped,
    )
    return out


@torch.library.custom_op("scattermoe::scatter2scatter", mutates_args={"O"})
def scatter2scatter_compileable(
    O: torch.Tensor,
    W: torch.Tensor,
    X: torch.Tensor,
    k: int,
    padded_block_idxs: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    x_grouped: bool,
    y_grouped: bool,
) -> None:
    def grid(meta):
        return (
            padded_block_idxs.size(0)
            * triton.cdiv(meta["N"], meta["BLOCK_N"]),
        )

    _scatter2scatter[grid](
        X,
        X.stride(0),
        X.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        O,
        O.stride(0),
        O.stride(1),
        grouped_idx_ptr=sorted_scattered_idxs,
        expert_idxs_ptr=sorted_expert_idxs,
        block_start_idx_ptr=padded_block_idxs,
        FAN_OUT=k,
        M=X.size(0),
        K=X.size(1),
        N=O.size(1),
        E=W.size(0),
        BLOCK_M=BLOCK_M,
        ACC_TYPE=tl.float32,
        OUT_M=O.size(0),
        allow_tf32=ALLOW_TF32,
        x_grouped=x_grouped,
        y_grouped=y_grouped,
    )


__all__ = [
    "flatten_and_sort",
    "padded_block_indices",
    "scatter2scatter",
]
