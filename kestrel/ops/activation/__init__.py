import torch

from ..jit import cpp_jit


@cpp_jit()
def gelu_residual_cuda(out: torch.Tensor, x: torch.Tensor) -> None: ...


__all__ = ["gelu_residual_cuda"]
