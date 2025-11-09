"""FlashInfer integration helpers for Moondream prefill and decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch

import flashinfer  # type: ignore
from torch.compiler import disable as torch_compiler_disable

@dataclass
class FlashInferBatchMetadata:
    """Per-step metadata required by FlashInfer batch decode kernels."""

    batch_size: int
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    graph_state: Optional["_GraphState"] = None

    @property
    def total_pages(self) -> int:
        return int(self.kv_indices.shape[0])


@dataclass
class FlashInferPrefillBatchMetadata:
    """Per-step metadata required by FlashInfer batch prefill kernels."""

    batch_size: int
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor

    @property
    def total_queries(self) -> int:
        return int(self.qo_indptr[-1].item())


class FlashInferPrefillContext:
    """Stateful wrapper around FlashInfer batch prefill kernels."""

    def __init__(
        self,
        *,
        device: torch.device,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        page_size: int,
        workspace_bytes: int = 128 * 1024 * 1024,
        backend: str = "fa2",
    ) -> None:
        self.device = device
        self.q_dtype = q_dtype
        self.kv_dtype = kv_dtype
        self.page_size = page_size
        self._backend = backend

        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace,
            "HND",
            backend=self._backend,
        )

    def plan(
        self,
        metadata: FlashInferPrefillBatchMetadata,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if metadata.batch_size == 0:
            return
        sm_scale = 1.0 / math.sqrt(head_dim)
        self._prefill_wrapper.plan(
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_last_page_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            self.page_size,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            custom_mask=custom_mask,
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        *,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._prefill_wrapper.run(
            q,
            paged_kv_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
        )


@torch_compiler_disable()
def run_flashinfer_prefill(
    ctx: FlashInferPrefillContext,
    q: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    *,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return ctx.run(
        q,
        paged_kv_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        out=out,
    )


@dataclass
class _GraphState:
    wrapper: "flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper"
    workspace: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    batch_indices: torch.Tensor
    batch_capacity: int
    page_capacity: int
    kv_indptr_cpu: torch.Tensor
    kv_last_page_len_cpu: torch.Tensor
    num_pages_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    kv_indptr_np: np.ndarray
    kv_last_page_len_np: np.ndarray
    num_pages_np: np.ndarray
    seq_lens_np: np.ndarray
    pages_filled: int = 0


@dataclass
class FlashInferPlanBuffers:
    batch_capacity: int
    page_capacity: int
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    batch_indices: torch.Tensor
    kv_indptr_cpu: torch.Tensor
    kv_last_page_len_cpu: torch.Tensor
    num_pages_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    kv_indptr_np: np.ndarray
    kv_last_page_len_np: np.ndarray
    num_pages_np: np.ndarray
    seq_lens_np: np.ndarray
    graph_state: Optional[_GraphState] = None
    pages_filled: int = 0


class FlashInferDecodeContext:
    """Stateful wrapper around FlashInfer batch decode kernels."""

    def __init__(
        self,
        *,
        device: torch.device,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        page_size: int,
        max_batch_size: int,
        max_seq_len: int,
        use_cuda_graphs: bool,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.device = device
        self.q_dtype = q_dtype
        self.kv_dtype = kv_dtype
        self.page_size = page_size
        self.use_cuda_graphs = use_cuda_graphs

        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self._decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, "HND"
        )

        self._max_batch_size = max_batch_size
        self._max_effective_batch = max(1, max_batch_size - 1)
        self._max_pages_per_seq = max(1, max_seq_len // page_size)
        self._max_seq_len = max_seq_len
        self._graph_workspace_bytes = workspace_bytes
        self._graph_states: Dict[int, _GraphState] = {}
        self._active_graph_state: Optional[_GraphState] = None
        self._plan_buffer_pool: List[FlashInferPlanBuffers] = []

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
            state = metadata.graph_state or self._get_graph_state(metadata.batch_size)
            if metadata.graph_state is not state:
                raise ValueError(
                    "FlashInfer graph metadata must be built with the active graph state"
                )
            self._active_graph_state = state

            state.wrapper.plan(
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.kv_last_page_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
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
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )

    # ------------------------------------------------------------------
    # Buffer management

    def acquire_plan_buffers(
        self, batch_size: int, *, use_graph: bool
    ) -> FlashInferPlanBuffers:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        effective_size = max(batch_size, 1)
        if use_graph:
            state = self._get_graph_state(effective_size)
            return FlashInferPlanBuffers(
                batch_capacity=state.batch_capacity,
                page_capacity=state.page_capacity,
                kv_indptr=state.kv_indptr,
                kv_indices=state.kv_indices,
                kv_last_page_len=state.kv_last_page_len,
                batch_indices=state.batch_indices,
                kv_indptr_cpu=state.kv_indptr_cpu,
                kv_last_page_len_cpu=state.kv_last_page_len_cpu,
                num_pages_cpu=state.num_pages_cpu,
                seq_lens_cpu=state.seq_lens_cpu,
                kv_indptr_np=state.kv_indptr_np,
                kv_last_page_len_np=state.kv_last_page_len_np,
                num_pages_np=state.num_pages_np,
                seq_lens_np=state.seq_lens_np,
                graph_state=state,
                pages_filled=state.pages_filled,
            )

        return self._get_non_graph_plan_buffers(effective_size)

    def _get_non_graph_plan_buffers(self, batch_size: int) -> FlashInferPlanBuffers:
        for buffers in self._plan_buffer_pool:
            if buffers.batch_capacity >= batch_size:
                return buffers

        capacity = self._next_buffer_capacity(batch_size)
        buffers = self._allocate_plan_buffers(capacity)
        self._plan_buffer_pool.append(buffers)
        self._plan_buffer_pool.sort(key=lambda item: item.batch_capacity)
        return buffers

    def _allocate_plan_buffers(self, batch_capacity: int) -> FlashInferPlanBuffers:
        batch_capacity = max(1, min(batch_capacity, self._max_effective_batch))
        page_capacity = batch_capacity * self._max_pages_per_seq
        kv_indptr = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        kv_indices = torch.empty(page_capacity, dtype=torch.int32, device=self.device)
        kv_last_page_len = torch.empty(batch_capacity, dtype=torch.int32, device=self.device)
        batch_indices = torch.empty(batch_capacity, dtype=torch.long, device=self.device)

        pin_kwargs = dict(device="cpu", dtype=torch.int32, pin_memory=True)
        kv_indptr_cpu = torch.zeros(batch_capacity + 1, **pin_kwargs)
        kv_last_page_len_cpu = torch.zeros(batch_capacity, **pin_kwargs)
        num_pages_cpu = torch.zeros(batch_capacity, **pin_kwargs)
        seq_lens_cpu = torch.zeros(batch_capacity, **pin_kwargs)
        kv_indptr_np = kv_indptr_cpu.numpy()
        kv_last_page_len_np = kv_last_page_len_cpu.numpy()
        num_pages_np = num_pages_cpu.numpy()
        seq_lens_np = seq_lens_cpu.numpy()
        return FlashInferPlanBuffers(
            batch_capacity=batch_capacity,
            page_capacity=page_capacity,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            batch_indices=batch_indices,
            kv_indptr_cpu=kv_indptr_cpu,
            kv_last_page_len_cpu=kv_last_page_len_cpu,
            num_pages_cpu=num_pages_cpu,
            seq_lens_cpu=seq_lens_cpu,
            kv_indptr_np=kv_indptr_np,
            kv_last_page_len_np=kv_last_page_len_np,
            num_pages_np=num_pages_np,
            seq_lens_np=seq_lens_np,
            pages_filled=0,
        )

    def _next_buffer_capacity(self, batch_size: int) -> int:
        if batch_size >= self._max_effective_batch:
            return self._max_effective_batch
        if batch_size <= 1:
            return 1
        # round up to next power of two to limit reallocations
        return 1 << (batch_size - 1).bit_length()

    # ------------------------------------------------------------------
    # Execution

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        *,
        use_graph: bool,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError("FlashInfer expects q with shape [B, H, D]")

        run_kwargs: dict[str, float] = {}
        if k_scale is not None:
            run_kwargs["k_scale"] = float(k_scale)
        if v_scale is not None:
            run_kwargs["v_scale"] = float(v_scale)

        if use_graph:
            if self._active_graph_state is None:
                raise RuntimeError("Graph state not initialized; call plan first")
            return self._active_graph_state.wrapper.run(
                q, kv_cache, out=out, **run_kwargs
            )
        return self._decode_wrapper.run(q, kv_cache, out=out, **run_kwargs)

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

        workspace = torch.empty(
            self._graph_workspace_bytes, dtype=torch.uint8, device=self.device
        )
        plan_buffers = self._allocate_plan_buffers(batch_size)
        wrapper = flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            workspace,
            plan_buffers.kv_indptr,
            plan_buffers.kv_indices,
            plan_buffers.kv_last_page_len,
            kv_layout="HND",
        )
        state = _GraphState(
            wrapper=wrapper,
            workspace=workspace,
            kv_indptr=plan_buffers.kv_indptr,
            kv_indices=plan_buffers.kv_indices,
            kv_last_page_len=plan_buffers.kv_last_page_len,
            batch_indices=plan_buffers.batch_indices,
            batch_capacity=plan_buffers.batch_capacity,
            page_capacity=plan_buffers.page_capacity,
            kv_indptr_cpu=plan_buffers.kv_indptr_cpu,
            kv_last_page_len_cpu=plan_buffers.kv_last_page_len_cpu,
            num_pages_cpu=plan_buffers.num_pages_cpu,
            seq_lens_cpu=plan_buffers.seq_lens_cpu,
            kv_indptr_np=plan_buffers.kv_indptr_np,
            kv_last_page_len_np=plan_buffers.kv_last_page_len_np,
            num_pages_np=plan_buffers.num_pages_np,
            seq_lens_np=plan_buffers.seq_lens_np,
            pages_filled=0,
        )
        self._graph_states[batch_size] = state
        return state


__all__ = [
    "FlashInferBatchMetadata",
    "FlashInferPrefillBatchMetadata",
    "FlashInferDecodeContext",
    "FlashInferPrefillContext",
    "run_flashinfer_prefill",
]
