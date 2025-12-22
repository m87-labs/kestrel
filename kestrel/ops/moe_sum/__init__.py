import torch

from ..jit import cpp_jit


@cpp_jit(function_name="moe_sum")
def moe_sum_cuda(input: torch.Tensor, output: torch.Tensor) -> None: ...


__all__ = ["moe_sum_cuda"]

