"""FlashInfer integration helpers for Moondream decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency validated at runtime
    import flashinfer  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled upstream
    raise RuntimeError(
        "FlashInfer is required for decoding. Please install the flashinfer package."
    ) from exc


@dataclass
class FlashInferBatchMetadata:
    """Per-step metadata required by FlashInfer batch decode kernels."""

    batch_size: int
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor

    @property
    def total_pages(self) -> int:
        return int(self.kv_indices.shape[0])


@dataclass
class _GraphState:
    wrapper: "flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper"
    workspace: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor


class FlashInferDecodeContext:
    """Stateful wrapper around FlashInfer batch decode kernels."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        page_size: int,
        max_batch_size: int,
        max_seq_len: int,
        use_cuda_graphs: bool,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.page_size = page_size
        self.use_cuda_graphs = use_cuda_graphs

        self._workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self._decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, "NHD"
        )

        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._graph_workspace_bytes = workspace_bytes
        self._graph_states: Dict[int, _GraphState] = {}
        self._active_graph_state: Optional[_GraphState] = None

    # ------------------------------------------------------------------
    # Planning

    def plan(
        self,
        metadata: FlashInferBatchMetadata,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_graph: bool,
    ) -> None:
        """Prepare FlashInfer for the given batch specification."""

        if metadata.batch_size == 0:
            return

        if use_graph:
            state = self._get_graph_state(metadata.batch_size)
            self._active_graph_state = state

            indptr_len = metadata.kv_indptr.shape[0]
            state.kv_indptr.zero_()
            state.kv_indices.zero_()
            state.kv_last_page_len.zero_()

            state.kv_indptr[:indptr_len].copy_(metadata.kv_indptr)
            state.kv_indices[: metadata.total_pages].copy_(metadata.kv_indices)
            state.kv_last_page_len[: metadata.batch_size].copy_(
                metadata.kv_last_page_len
            )

            state.wrapper.plan(
                state.kv_indptr[:indptr_len],
                state.kv_indices[: metadata.total_pages],
                state.kv_last_page_len[: metadata.batch_size],
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
        else:
            self._active_graph_state = None
            self._decode_wrapper.plan(
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.kv_last_page_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

    # ------------------------------------------------------------------
    # Execution

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        *,
        use_graph: bool,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError("FlashInfer expects q with shape [B, H, D]")

        if use_graph:
            if self._active_graph_state is None:
                raise RuntimeError("Graph state not initialized; call plan first")
            return self._active_graph_state.wrapper.run(q, kv_cache, out=out)
        return self._decode_wrapper.run(q, kv_cache, out=out)

    # ------------------------------------------------------------------
    # Internal helpers

    def _get_graph_state(self, batch_size: int) -> _GraphState:
        if batch_size < 1:
            raise ValueError("Graph batch size must be positive")
        max_effective = max(1, self._max_batch_size - 1)
        if batch_size > max_effective:
            raise ValueError(
                f"Requested graph batch size {batch_size} exceeds configured maximum {max_effective}"
            )

        state = self._graph_states.get(batch_size)
        if state is not None:
            return state

        max_pages_per_seq = max(1, self._max_seq_len // self.page_size)
        workspace = torch.empty(
            self._graph_workspace_bytes, dtype=torch.uint8, device=self.device
        )
        kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=self.device)
        kv_indices = torch.empty(
            batch_size * max_pages_per_seq, dtype=torch.int32, device=self.device
        )
        kv_last = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        wrapper = flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_indptr,
            kv_indices,
            kv_last,
            kv_layout="NHD",
        )
        state = _GraphState(
            wrapper=wrapper,
            workspace=workspace,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last,
        )
        self._graph_states[batch_size] = state
        return state


__all__ = [
    "FlashInferBatchMetadata",
    "FlashInferDecodeContext",
]
