from __future__ import annotations

import torch
import torch.nn as nn

from . import kernels


def parallel_linear(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    k: int,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    padded_block_idxs: torch.Tensor,
    *,
    gates: torch.Tensor | None = None,
    grouped_in: bool = False,
    grouped_out: bool = False,
) -> torch.Tensor:
    """Scatter-gather linear layer used by the MoE MLP forward path."""

    output = kernels.ops.scatter2scatter(
        inputs,
        expert_weights,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        k,
        padded_block_idxs,
        x_grouped=grouped_in,
        y_grouped=grouped_out,
    )

    if gates is not None:
        expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
        output = torch.bmm(gates.unsqueeze(1), expanded).squeeze(1)
    return output


class ParallelExperts(nn.Module):
    """Batched expert weight wrapper for scatter-gather MoE."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size, dtype=dtype)
        )
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, input_size={self.input_size}, "
            f"output_size={self.output_size}"
        )

    def forward(
        self,
        inputs: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        padded_block_idxs: torch.Tensor,
        *,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        expert_weights = self.weight.permute(0, 2, 1)
        return parallel_linear(
            inputs,
            expert_weights,
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )


__all__ = ["ParallelExperts", "parallel_linear"]
