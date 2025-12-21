import torch

from ..jit import cpp_jit


@cpp_jit(function_name="moe_align_block_size")
def moe_align_block_size_cuda(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor,
) -> None: ...


__all__ = ["moe_align_block_size_cuda"]
