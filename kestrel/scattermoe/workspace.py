from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels import ops


@dataclass(slots=True)
class ScatterMoEWorkspace:
    """Reusable buffers to make ScatterMoE launches CUDA-graph friendly."""

    sorted_expert_idxs: torch.Tensor
    sorted_scattered_idxs: torch.Tensor
    padded_block_idxs: torch.Tensor
    block_idx_template: torch.Tensor
    output: torch.Tensor
    max_tokens: int
    top_k: int

    @classmethod
    def allocate(
        cls,
        *,
        batch_tokens: int,
        hidden_size: int,
        top_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ScatterMoEWorkspace:
        """Allocate all buffers required for a fixed-shape CUDA graph replay."""

        length = max(batch_tokens * top_k, 1)
        padded_block_capacity = length

        sorted_expert_idxs = torch.empty(length, dtype=torch.long, device=device)
        sorted_scattered_idxs = torch.empty(length, dtype=torch.long, device=device)
        padded_block_idxs = torch.empty(padded_block_capacity, dtype=torch.long, device=device)
        block_idx_template = torch.arange(
            padded_block_capacity, dtype=torch.long, device=device
        )
        output = torch.empty(length, hidden_size, dtype=dtype, device=device)

        return cls(
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            block_idx_template=block_idx_template,
            output=output,
            max_tokens=batch_tokens,
            top_k=top_k,
        )

    def compute_block_indices(self, num_experts: int) -> int:
        _, _, block_count = ops.padded_block_indices(
            self.sorted_expert_idxs,
            num_experts,
            out=self.padded_block_idxs,
            block_idx_template=self.block_idx_template,
        )
        return int(block_count.item())
