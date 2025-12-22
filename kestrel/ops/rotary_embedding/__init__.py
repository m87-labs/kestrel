import torch

from ..jit import cpp_jit


@cpp_jit(function_name="rotary_embedding")
def rotary_embedding_cuda(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> None: ...


__all__ = ["rotary_embedding_cuda"]
