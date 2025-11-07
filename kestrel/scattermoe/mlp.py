from __future__ import annotations

import os
from typing import Literal, Optional

import torch
from torch import nn

from . import kernels
from .parallel_experts import ParallelExperts
from .workspace import ScatterMoEWorkspace
from ..fused_moe import FusedMoEModule


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
        decode_backend: Literal["auto", "scatter", "fused"] = "auto",
    ) -> None:
        super(MLP, self).__init__()

        if decode_backend not in {"auto", "scatter", "fused"}:
            raise ValueError("decode_backend must be one of {'auto','scatter','fused'}")

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
        self._decode_backend = decode_backend
        self._fused_backend = FusedMoEModule(
            self.experts,
            self.output_experts,
            top_k=self.top_k,
            hidden_size=hidden_size,
            input_size=input_size,
            num_experts=num_experts,
        )
        self._debug_dump_done = False

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
        self,
        x: torch.Tensor,
        expert_p: torch.Tensor,
        expert_idxs: torch.Tensor,
        *,
        mode: Literal["prefill", "decode"] = "decode",
        backend: Literal["auto", "scatter", "fused"] | None = None,
    ):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        expert_p = expert_p.reshape(-1, expert_p.size(-1))
        expert_idxs = expert_idxs.reshape(-1, expert_idxs.size(-1))

        fused_enabled = self._should_use_fused_backend(mode, backend)
        if fused_enabled and os.getenv("KESTREL_DEBUG_FUSED") == "1":
            print("[ScatterMoE] fused backend active", flush=True)

        if fused_enabled:
            dump_path = os.getenv("KESTREL_MOE_DUMP")
            compare = os.getenv("KESTREL_COMPARE_MOE_BACKENDS") == "1"
            if compare:
                scatter_result = self._forward_scatter(
                    x, expert_p, expert_idxs, return_intermediate=True
                )
                assert isinstance(scatter_result, tuple)
                scatter_y, scatter_expert_out, scatter_h = scatter_result
            else:
                scatter_y = None
                scatter_expert_out = None
                scatter_h = None

            y = self._fused_backend(x, expert_p, expert_idxs)
            if compare and scatter_y is not None:
                print("[ScatterMoE][compare] running checks", flush=True)
                diff = (y - scatter_y).abs()
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(
                    "[ScatterMoE][compare] max_diff="
                    f"{max_diff.item():.6f} mean_diff={mean_diff.item():.6f}",
                    flush=True,
                )
                debug_up = getattr(self._fused_backend, "_debug_up", None)
                if debug_up is not None and scatter_expert_out is not None:
                    up_diff = (debug_up - scatter_expert_out.detach().cpu()).abs()
                    print(
                        "[ScatterMoE][compare] up_out max_diff="
                        f"{up_diff.max().item():.6f}",
                        flush=True,
                    )
                debug_h = getattr(self._fused_backend, "_debug_h", None)
                if debug_h is not None and scatter_h is not None:
                    h_diff = (debug_h - scatter_h.detach().cpu()).abs()
                    print(
                        "[ScatterMoE][compare] act max_diff="
                        f"{h_diff.max().item():.6f}",
                        flush=True,
                    )
                if dump_path and not self._debug_dump_done:
                    cpu_blob = {
                        "x": x.detach().cpu(),
                        "expert_p": expert_p.detach().cpu(),
                        "expert_idxs": expert_idxs.detach().cpu(),
                        "fused": y.detach().cpu(),
                        "scatter": scatter_y.detach().cpu(),
                    }
                    torch.save(cpu_blob, dump_path)
                    self._debug_dump_done = True
            return y.view(*x_shape[:-1], y.size(-1))

        y = self._forward_scatter(x, expert_p, expert_idxs)
        return y.view(*x_shape[:-1], y.size(-1))

    def _forward_scatter(
        self,
        x: torch.Tensor,
        expert_p: torch.Tensor,
        expert_idxs: torch.Tensor,
        *,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            if use_workspace:
                assert self._workspace_up is not None
                length = sorted_expert_idxs.size(0)
                sorted_expert_buffer = self._workspace_up.sorted_expert_idxs[:length]
                sorted_scattered_buffer = self._workspace_up.sorted_scattered_idxs[:length]
                sorted_expert_buffer.copy_(sorted_expert_idxs)
                sorted_scattered_buffer.copy_(sorted_scattered_idxs)
                padded_block_idxs, _, _ = kernels.ops.padded_block_indices(
                    sorted_expert_buffer,
                    self.num_experts,
                    out=self._workspace_up.padded_block_idxs,
                    block_idx_template=self._workspace_up.block_idx_template,
                )
                sorted_expert_for_kernel = sorted_expert_buffer
                sorted_scattered_for_kernel = sorted_scattered_buffer
            else:
                padded_block_idxs, _, _ = kernels.ops.padded_block_indices(
                    sorted_expert_idxs, self.num_experts
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
        if return_intermediate:
            return y, expert_out[:length], h
        return y

    def _should_use_fused_backend(
        self,
        mode: Literal["prefill", "decode"],
        backend: Literal["auto", "scatter", "fused"] | None,
    ) -> bool:
        if self._fused_backend is None:
            return False

        choice = backend or ("fused" if mode == "decode" else "scatter")
        if choice == "auto":
            choice = self._decode_backend if mode == "decode" else "scatter"
        if choice != "fused":
            return False
        if not self._fused_backend.available:
            return False
        return True
