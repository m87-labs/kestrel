from __future__ import annotations

import torch
import torch.nn as nn

from . import kernels


class ParallelLinear(torch.autograd.Function):
    """Scatter-gather linear layer backed by Triton kernels."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        padded_block_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        with torch.device(x.device):
            output = kernels.ops.scatter2scatter(
                X=x,
                W=expert_weights,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                padded_block_idxs=padded_block_idxs,
                k=k,
                x_grouped=grouped_in,
                y_grouped=grouped_out,
            )
            if gates is not None:
                output_expanded = output.view(
                    gates.size(0), gates.size(1), output.size(-1)
                )
                output = torch.bmm(gates[:, None, :], output_expanded).squeeze(1)
            else:
                output_expanded = None

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        with torch.device(grad_out.device):
            if gates is not None:
                d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                grouped_grad_out = output_expanded.flatten(0, 1)
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None

            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = kernels.ops.group(
                    grad_out,
                    sorted_scattered_idxs,
                    fan_out=gate_fan,
                    coeff=gates_flat,
                    out=grouped_grad_out,
                )
            if grouped_in:
                grouped_x = x
                d_expanded_input = None
            else:
                grouped_x = kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
                d_expanded_input = grouped_x
            d_weights = kernels.ops.group_bwd_W(
                DY=grouped_grad_out,
                X=grouped_x,
                expert_offsets=expert_offsets,
                E=expert_weights.size(0),
            )
            d_expanded_input = kernels.ops.scatter2scatter(
                X=grouped_grad_out,
                x_grouped=True,
                W=expert_weights.permute(0, 2, 1),
                padded_block_idxs=padded_block_idxs,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
                y_grouped=grouped_in,
                out=d_expanded_input,
            )

            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(
                    x.size(0), k, d_expanded_input.size(-1)
                ).sum(-2)
        return (
            d_input,
            d_weights,
            None,
            None,
            None,
            None,
            None,
            d_gates,
            None,
            None,
        )


def parallel_linear(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    k: int,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    padded_block_idxs: torch.Tensor,
    expert_offsets: torch.Tensor,
    gates: torch.Tensor | None = None,
    grouped_in: bool = False,
    grouped_out: bool = False,
) -> torch.Tensor:
    return ParallelLinear.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )


class ParallelExperts(nn.Module):
    """Bundle of expert matrices with scatter-gather forward."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size, dtype=dtype))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, input_size={self.input_size}, "
            f"output_size={self.output_size}"
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        padded_block_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        return parallel_linear(
            inputs,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )
