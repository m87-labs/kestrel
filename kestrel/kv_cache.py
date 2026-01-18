
from contextlib import nullcontext
from typing import Optional

import torch
import triton
import triton.language as tl


from kestrel_kernels.kv_cache_write import reshape_and_cache_flash as reshape_and_cache_flash_cuda

from kestrel.utils import CpuGpuBuffer


def _maybe_stream_context(stream: torch.cuda.Stream | None):
    """Return a stream context manager, or nullcontext if stream is None."""
    if stream is not None:
        return torch.cuda.stream(stream)
    return nullcontext()


def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
    return (x + multiple - 1) // multiple


class PagedKVCache(torch.nn.Module):
    def __init__(
        self,
        page_table,
        n_heads,
        head_dim,
        dtype,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        super().__init__()
        cache_shape = (
            page_table.n_pages,
            n_heads,
            page_table.page_size,
            head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

        self.page_table = page_table
        self.quantized = dtype == torch.float8_e4m3fn
        self._kv_cache_dtype = "fp8_e4m3" if self.quantized else "auto"
        if self.quantized:
            if k_scale is None or v_scale is None:
                raise ValueError("FP8 KV cache requires per-layer k/v scales")
            self.k_scale = float(k_scale)
            self.v_scale = float(v_scale)
        else:
            self.k_scale = 1.0
            self.v_scale = 1.0
        self.register_buffer(
            "k_scale_tensor", torch.tensor(self.k_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "v_scale_tensor", torch.tensor(self.v_scale, dtype=torch.float32)
        )

    def update(
        self,
        input_pos,
        k_val,
        v_val,
        *,
        slot_mapping: torch.Tensor,
    ):
        k_view = k_val.view(-1, k_val.shape[2], k_val.shape[3])
        v_view = v_val.view(-1, v_val.shape[2], v_val.shape[3])

        if slot_mapping.shape != input_pos.shape:
            raise ValueError("slot_mapping must match input_pos shape")
        slot_mapping = slot_mapping.to(device=k_view.device, dtype=torch.int64)

        flat = slot_mapping.numel()
        if flat == 0:
            return k_val, v_val
        if flat != k_view.shape[0]:
            raise RuntimeError("PagedKVCache.update slot size mismatch")

        slot_mapping = slot_mapping.contiguous().view(-1)

        key_cache = self.k_cache.permute(0, 2, 1, 3)
        value_cache = self.v_cache.permute(0, 2, 1, 3)
        if self._kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(torch.uint8)
            value_cache = value_cache.view(torch.uint8)

        reshape_and_cache_flash_cuda(
            k_view,
            v_view,
            key_cache,
            value_cache,
            slot_mapping,
            self._kv_cache_dtype,
            self.k_scale_tensor,
            self.v_scale_tensor,
        )

        return k_val, v_val


class PageTable:
    """
    PageTable is a modified version of PagedAttention from attention-gym.

    PageTable improves it by:
    - maintaining a cpu copy of the page table, to avoid device-to-host transfers
    - support batch prefill
    - fix the bug in the original code in mask_mod and score_mod by mapping physical batch index to logical batch index
    - subsuming the free_batch_idx into the page table, so we don't need to maintain it separately
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
        prefix_cache=None,
        *,
        h2d_stream: torch.cuda.Stream | None = None,
    ):
        self.n_pages = n_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.device = device
        # Stream for H2D copies (ensures proper synchronization with graph replay)
        self._h2d_stream = h2d_stream

        # page table: [logical_batch_idx, logical_block_idx] -> physical_page_idx
        self._page_table_buffer = CpuGpuBuffer(
            max_batch_size,
            self.n_pages,
            dtype=torch.int32,
            device=torch.device(device),
            pin_memory=True,
        )
        self.page_table = self._page_table_buffer.gpu
        self._page_table_cpu_tensor = self._page_table_buffer.cpu
        self._page_table_cpu_tensor.fill_(-1)
        self._sync_full_page_table()
        self._page_table_cpu_tensor[0, :].fill_(0)  # reserve row 0 for bookkeeping
        self._sync_page_table_row(0)
        self.page_table_cpu = [[] for _ in range(max_batch_size)]

        self.capacity = [
            0 for _ in range(max_batch_size)
        ]  # capacity: batch_idx -> number of pages allocated * page size
        self.free_pages = list(reversed(range(1, n_pages)))  # page 0 stays reserved
        self.free_batch_idx = list(
            reversed(range(1, max_batch_size))
        )  # batch_idx 0 is reserved for no-op

        # [logical_batch_idx, physical_page_idx] -> logical_page_idx
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

        # Prefix cache for on-demand eviction (optional)
        self.prefix_cache = None
        if prefix_cache is not None:
            assert page_size == 1, (
                f"Prefix caching requires page_size=1, got {page_size}. "
                "With page_size>1, partial page reuse would cause leaks or corruption."
            )
            # Wire eviction to return pages to our free pool
            prefix_cache._free_pages_sink = self.free_pages_to_pool
            self.prefix_cache = prefix_cache

    def can_reserve(self, size: int, batch_idx_int: int | None = None) -> bool:
        """check if we can reserve new pages for an existing request or a new request, without gpu operations"""
        if batch_idx_int is None:
            # check if we can schedule a new request
            return (
                self.pages_available * self.page_size >= size
                and len(self.free_batch_idx) > 0
            )
        else:
            # check if we can reserve new pages for an existing request
            return self.reserve(batch_idx_int, None, size, dry_run=True)

    def allocate(self) -> int:
        """allocate a new batch"""
        batch_idx = self.free_batch_idx.pop()

        self.capacity[batch_idx] = 0
        self.physical_to_logical[batch_idx, :] = -1
        self._page_table_cpu_tensor[batch_idx, :].fill_(-1)
        self._sync_page_table_row(batch_idx)
        return batch_idx

    @property
    def pages_available(self) -> int:
        return len(self.free_pages)

    def reserve(
        self,
        batch_idx_int: int,
        batch_idx: torch.Tensor,
        seq_len: int,
        dry_run: bool = False,
    ) -> bool:
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx_int (int): batch index to be reserved;
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.

        Returns:
            bool: True if the reservation was successful, False if the reservation was not successful (no space, and in this case, no update is done)
        """

        if seq_len <= self.capacity[batch_idx_int]:
            return True

        num_pages_to_allocate = _cdiv(
            seq_len - self.capacity[batch_idx_int], self.page_size
        )

        available = self.pages_available
        can_allocate = num_pages_to_allocate <= available

        if dry_run:
            return can_allocate

        # Evict from prefix cache if needed
        if not can_allocate and self.prefix_cache is not None:
            needed = num_pages_to_allocate - available
            self.prefix_cache.evict(needed)
            available = self.pages_available
            can_allocate = num_pages_to_allocate <= available

        if not can_allocate:
            msg = (
                f"Cannot reserve {num_pages_to_allocate} pages for a sequence of length {seq_len} "
                f"in batch {batch_idx_int}. Only {self.pages_available} pages available. "
                f"Current capacity is {self.capacity[batch_idx_int]} tokens."
            )
            if self.prefix_cache is not None:
                msg += " All cached prefixes are locked by active sequences."
            raise RuntimeError(msg)

        start_page_idx = self.capacity[batch_idx_int] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages_list = self.free_pages[-num_pages_to_allocate:]
        allocated_pages_cpu = torch.as_tensor(allocated_pages_list, dtype=torch.int32)
        # update page table on host first, then sync the touched slice once
        self._page_table_cpu_tensor[
            batch_idx_int, start_page_idx:end_page_idx
        ] = allocated_pages_cpu
        self._sync_page_table_row(batch_idx_int, start_page_idx, end_page_idx)

        with _maybe_stream_context(self._h2d_stream):
            allocated_pages_idx = allocated_pages_cpu.to(
                device=self.device, dtype=torch.long
            )
            self.physical_to_logical[batch_idx, allocated_pages_idx] = torch.arange(
                start_page_idx,
                end_page_idx,
                device=self.device,
            )
        # update cpu side metadata
        self.page_table_cpu[batch_idx_int] += allocated_pages_list
        self.free_pages = self.free_pages[:-num_pages_to_allocate]
        self.capacity[batch_idx_int] += num_pages_to_allocate * self.page_size
        return True

    def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx: Batch index to be removed.
            cached_page_count: Number of leading pages owned by prefix cache.
                These are NOT returned to free pool (they belong to cache tree).
        """
        # NOTE: the GPU side data will only be reset/overwritten when we allocate it for a new batch
        self.free_batch_idx.append(batch_idx)
        allocated_pages_cpu = self.page_table_cpu[batch_idx]
        # Skip cached pages (they belong to the prefix cache tree)
        pages_to_free = allocated_pages_cpu[cached_page_count:]
        self.free_pages.extend(reversed(pages_to_free))
        self.page_table_cpu[batch_idx] = []
        self.capacity[batch_idx] = 0

    def populate_fa3_decode_metadata(
        self,
        *,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        out_page_table: torch.Tensor,
        out_seqused_k: torch.Tensor,
    ) -> None:
        """Populate FA3 paged-KV metadata buffers using a fused Triton kernel."""
        if batch_idx.ndim != 1:
            raise ValueError("batch_idx must be 1D for FA3 metadata")
        if input_pos.ndim != 1:
            raise ValueError("input_pos must be 1D for FA3 metadata")
        if batch_idx.shape[0] != input_pos.shape[0]:
            raise ValueError("batch_idx and input_pos must have the same length")
        if out_page_table.ndim != 2:
            raise ValueError("out_page_table must be 2D")
        if out_seqused_k.ndim != 1:
            raise ValueError("out_seqused_k must be 1D")

        batch_size = batch_idx.shape[0]
        if out_page_table.shape[0] < batch_size:
            raise ValueError("out_page_table has insufficient batch capacity")
        if out_page_table.shape[1] < self.n_pages:
            raise ValueError("out_page_table has insufficient page capacity")
        if out_seqused_k.shape[0] < batch_size:
            raise ValueError("out_seqused_k has insufficient batch capacity")

        if out_page_table.dtype != torch.int32:
            raise ValueError("out_page_table must be int32")
        if out_seqused_k.dtype != torch.int32:
            raise ValueError("out_seqused_k must be int32")
        if input_pos.dtype != torch.int32:
            raise ValueError("input_pos must be int32")
        if self.page_table.dtype != torch.int32:
            raise ValueError("page_table must be int32")
        if out_page_table.stride(-1) != 1:
            raise ValueError("out_page_table must be contiguous in the last dimension")

        device = self.page_table.device
        if not (batch_idx.is_cuda and input_pos.is_cuda):
            raise ValueError("batch_idx and input_pos must be CUDA tensors")
        if not (out_page_table.is_cuda and out_seqused_k.is_cuda):
            raise ValueError("out_page_table and out_seqused_k must be CUDA tensors")

        batch_idx = batch_idx.to(device=device)
        input_pos = input_pos.to(device=device)
        if not batch_idx.is_contiguous():
            batch_idx = batch_idx.contiguous()
        if not input_pos.is_contiguous():
            input_pos = input_pos.contiguous()

        if batch_size == 0:
            return

        BLOCK_PAGES = 128
        grid = (batch_size, triton.cdiv(self.n_pages, BLOCK_PAGES))
        _build_fa3_decode_metadata_kernel[grid](
            self.page_table,
            self.page_table.stride(0),
            batch_idx,
            input_pos,
            out_page_table,
            out_page_table.stride(0),
            out_seqused_k,
            self.n_pages,
            batch_size,
            BLOCK_PAGES=BLOCK_PAGES,
            PAGE_SIZE=self.page_size,
            num_warps=4,
        )

    def build_slot_mapping(
        self, batch_idx: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if batch_idx.ndim == 1:
            batch_tiled = batch_idx.view(-1, 1).expand_as(positions)
        else:
            if batch_idx.shape != positions.shape:
                raise ValueError("batch_idx and positions shape mismatch")
            batch_tiled = batch_idx

        page_size = self.page_size
        logical_block_idx = positions // page_size
        logical_block_offset = positions % page_size

        block_flat = logical_block_idx.to(torch.long).reshape(-1)
        batch_flat = batch_tiled.to(torch.long).reshape(-1)
        offset_flat = logical_block_offset.to(torch.long).reshape(-1)

        physical_block_idx = self.page_table[batch_flat, block_flat]
        slots = physical_block_idx * page_size + offset_flat
        return slots.to(dtype=torch.int64).view_as(positions)

    def _sync_page_table_row(
        self, batch_idx: int, start: int = 0, end: Optional[int] = None
    ) -> None:
        if end is None:
            end = self.n_pages
        if start >= end:
            return
        gpu_slice = self.page_table[batch_idx, start:end]
        cpu_slice = self._page_table_cpu_tensor[batch_idx, start:end]
        # Use the designated H2D stream to avoid races with graph replay.
        with _maybe_stream_context(self._h2d_stream):
            gpu_slice.copy_(cpu_slice, non_blocking=True)

    def _sync_full_page_table(self) -> None:
        # Use the designated H2D stream to avoid races with graph replay.
        with _maybe_stream_context(self._h2d_stream):
            self._page_table_buffer.copy_to_gpu()

    # =========================================================================
    # Prefix cache integration methods
    # =========================================================================

    def allocate_pages(self, count: int) -> list[int]:
        """Allocate physical pages from free pool, evicting from cache if needed.

        Unlike reserve(), this returns unbound physical pages that must be
        mapped via map_pages(). Used for suffix allocation after cache hit.

        Args:
            count: Number of pages to allocate.

        Returns:
            List of physical page indices.

        Raises:
            RuntimeError: If not enough pages available even after eviction.
        """
        available = len(self.free_pages)

        if available < count and self.prefix_cache is not None:
            needed = count - available
            # evict() sends freed pages directly to free_pages_to_pool via sink
            self.prefix_cache.evict(needed)
            available = len(self.free_pages)

        if available < count:
            msg = f"Cannot allocate {count} pages: {available} available"
            if self.prefix_cache is not None:
                msg += ", all cached prefixes are locked by active sequences"
            raise RuntimeError(msg)

        return [self.free_pages.pop() for _ in range(count)]

    def map_pages(
        self,
        batch_idx: int,
        logical_start: int,
        physical_pages: list[int],
    ) -> None:
        """Map physical pages into a batch's page table at specified positions.

        Used for:
        1. Mapping cached pages at logical_start=0
        2. Mapping newly allocated suffix pages at logical_start=skip_positions

        INVARIANT: map_pages() must be called sequentially - logical_start must
        equal the current number of mapped pages for this batch. This is because
        page_table_cpu[batch_idx].extend() assumes non-overlapping, sequential
        mappings.

        Args:
            batch_idx: Target batch slot.
            logical_start: First logical page index (must match current length).
            physical_pages: Physical page indices to map.
        """
        # Enforce sequential, non-overlapping invariant
        current_len = len(self.page_table_cpu[batch_idx])
        assert logical_start == current_len, (
            f"map_pages must be called sequentially: expected logical_start="
            f"{current_len}, got {logical_start}"
        )

        if not physical_pages:
            return

        # Update CPU page table tensor
        end = logical_start + len(physical_pages)
        pages_tensor = torch.as_tensor(physical_pages, dtype=torch.int32)
        self._page_table_cpu_tensor[batch_idx, logical_start:end] = pages_tensor

        # Sync to GPU
        self._sync_page_table_row(batch_idx, start=logical_start, end=end)

        with _maybe_stream_context(self._h2d_stream):
            pages_gpu = pages_tensor.to(device=self.device, dtype=torch.long)
            self.physical_to_logical[batch_idx, pages_gpu] = torch.arange(
                logical_start, end, device=self.device
            )

        # Update capacity to cover all mapped pages
        new_capacity = end * self.page_size
        self.capacity[batch_idx] = max(self.capacity[batch_idx], new_capacity)

        # Track pages for this batch (for erase())
        self.page_table_cpu[batch_idx].extend(physical_pages)

    def get_pages(self, batch_idx: int, start: int, end: int) -> list[int]:
        """Get physical page indices for a range of logical pages.

        Used after prefill to get the physical pages for cache insertion.

        Args:
            batch_idx: Batch slot to query.
            start: Start logical page index (inclusive).
            end: End logical page index (exclusive).

        Returns:
            List of physical page indices.
        """
        return [
            self._page_table_cpu_tensor[batch_idx, i].item()
            for i in range(start, end)
        ]

    def free_pages_to_pool(self, pages: tuple[int, ...] | list[int]) -> None:
        """Return pages to free pool. Called by cache eviction.

        Args:
            pages: Physical page indices to free.
        """
        self.free_pages.extend(reversed(pages))

    def can_reserve_with_eviction(
        self, size: int, batch_idx_int: int | None = None
    ) -> bool:
        """Check if we can reserve pages, considering evictable cache pages.

        Extends existing can_reserve() to account for prefix cache.

        Args:
            size: Number of KV positions needed.
            batch_idx_int: Existing batch slot, or None for new request.

        Returns:
            True if reservation is possible (with eviction if needed).
        """
        if batch_idx_int is None:
            # New request: need batch slot + pages
            if len(self.free_batch_idx) == 0:
                return False
            pages_needed = (size + self.page_size - 1) // self.page_size
        else:
            # Existing request: just need pages
            current = self.capacity[batch_idx_int]
            if size <= current:
                return True
            pages_needed = (size - current + self.page_size - 1) // self.page_size

        available = self.pages_available
        if available >= pages_needed:
            return True

        # Check if eviction could free enough
        if self.prefix_cache is not None:
            evictable = self.prefix_cache.evictable_page_count()
            return available + evictable >= pages_needed
        return False


@triton.jit
def _build_fa3_decode_metadata_kernel(
    page_table_ptr,
    page_table_stride,
    batch_idx_ptr,
    input_pos_ptr,
    out_page_table_ptr,
    out_page_table_stride,
    out_seqused_k_ptr,
    n_pages,
    batch_size,
    BLOCK_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    batch_mask = batch_id < batch_size
    batch_idx = tl.load(batch_idx_ptr + batch_id, mask=batch_mask, other=0).to(tl.int64)
    seqlen = tl.load(input_pos_ptr + batch_id, mask=batch_mask, other=0).to(tl.int64)
    seqlen = seqlen + 1
    num_pages = (seqlen + (PAGE_SIZE - 1)) // PAGE_SIZE
    n_pages_i64 = tl.full((), n_pages, dtype=tl.int64)
    num_pages = tl.where(num_pages < n_pages_i64, num_pages, n_pages_i64)

    page_stride = tl.full((), page_table_stride, dtype=tl.int64)
    out_stride = tl.full((), out_page_table_stride, dtype=tl.int64)

    offs = tile_id * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    offs = offs.to(tl.int64)
    mask = offs < num_pages

    in_ptr = page_table_ptr + batch_idx * page_stride + offs
    out_ptr = out_page_table_ptr + batch_id * out_stride + offs
    values = tl.load(in_ptr, mask=mask & batch_mask, other=0)
    tl.store(out_ptr, values, mask=mask & batch_mask)

    write_seq = (tile_id == 0) & batch_mask
    tl.store(out_seqused_k_ptr + batch_id, seqlen, mask=write_seq)
