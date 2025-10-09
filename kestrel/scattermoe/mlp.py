from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from . import kernels
from .parallel_experts import ParallelExperts
from .workspace import ScatterMoEWorkspace


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        activation: nn.Module | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(MLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(
            num_experts,
            input_size,
            hidden_size * 2,
            dtype=dtype,
        )
        self.output_experts = ParallelExperts(
            num_experts,
            hidden_size,
            input_size,
            dtype=dtype,
        )
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation or nn.GELU()
        self._workspace_tokens: int = 0
        self._workspace_device: torch.device | None = None
        self._workspace_dtype: torch.dtype | None = None
        self._workspace_up: Optional[ScatterMoEWorkspace] = None
        self._workspace_down: Optional[ScatterMoEWorkspace] = None

    def extra_repr(self):
        return "k={}".format(self.top_k)

    def ensure_workspaces(
        self, batch_tokens: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if batch_tokens <= 0:
            batch_tokens = 1

        reuse = (
            self._workspace_up is not None
            and self._workspace_down is not None
            and self._workspace_tokens >= batch_tokens
            and self._workspace_device == device
            and self._workspace_dtype == dtype
        )
        if reuse:
            return

        self._workspace_tokens = batch_tokens
        self._workspace_device = device
        self._workspace_dtype = dtype

        self._workspace_up = ScatterMoEWorkspace.allocate(
            batch_tokens=batch_tokens,
            hidden_size=self.hidden_size * 2,
            top_k=self.top_k,
            device=device,
            dtype=dtype,
        )
        self._workspace_down = ScatterMoEWorkspace.allocate(
            batch_tokens=batch_tokens,
            hidden_size=self.input_size,
            top_k=self.top_k,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor
    ):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(
                expert_idxs
            )

            use_workspace = (
                self._workspace_up is not None
                and self._workspace_down is not None
                and self._workspace_device == x.device
                and self._workspace_dtype == x.dtype
                and self._workspace_tokens >= x.size(0)
            )

            capturing = (
                x.is_cuda
                and torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            )

            if use_workspace:
                assert self._workspace_up is not None
                length = sorted_expert_idxs.size(0)
                sorted_expert_buffer = self._workspace_up.sorted_expert_idxs[:length]
                sorted_scattered_buffer = self._workspace_up.sorted_scattered_idxs[:length]
                sorted_expert_buffer.copy_(sorted_expert_idxs)
                sorted_scattered_buffer.copy_(sorted_scattered_idxs)
                # Always refresh the padded block plan so CUDA graph replay picks
                # up the current expert routing without changing tensor shapes.
                padded_block_idxs, _, _ = kernels.ops.padded_block_indices(
                    sorted_expert_buffer,
                    self.num_experts,
                    out=self._workspace_up.padded_block_idxs,
                    block_idx_template=self._workspace_up.block_idx_template,
                    capturing=capturing,
                )
                sorted_expert_for_kernel = sorted_expert_buffer
                sorted_scattered_for_kernel = sorted_scattered_buffer
            else:
                padded_block_idxs, _, _ = kernels.ops.padded_block_indices(
                    sorted_expert_idxs, self.num_experts, capturing=capturing
                )
                sorted_expert_for_kernel = sorted_expert_idxs
                sorted_scattered_for_kernel = sorted_scattered_idxs

        length = sorted_expert_idxs.size(0)

        expert_out = self.experts(
            x,
            self.top_k,
            sorted_expert_for_kernel,
            sorted_scattered_for_kernel,
            padded_block_idxs,
            grouped_out=True,
            out=(
                self._workspace_up.output[:length]
                if use_workspace and self._workspace_up is not None
                else None
            ),
        )
        h, g = expert_out[:length].chunk(2, dim=-1)
        h = self.activation(h) * (g + 1)
        y = self.output_experts(
            h,
            1,
            sorted_expert_for_kernel,
            sorted_scattered_for_kernel,
            padded_block_idxs,
            grouped_in=True,
            gates=expert_p,
            out=(
                self._workspace_down.output[:length]
                if use_workspace and self._workspace_down is not None
                else None
            ),
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y
