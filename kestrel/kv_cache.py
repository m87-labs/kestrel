import torch


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
        if self.quantized:
            if k_scale is None or v_scale is None:
                raise ValueError("FP8 KV cache requires per-layer k/v scales")
            self.k_scale = float(k_scale)
            self.v_scale = float(v_scale)
            self._k_inv_scale = 1.0 / self.k_scale
            self._v_inv_scale = 1.0 / self.v_scale
            self.register_buffer(
                "k_inv_scale_tensor",
                torch.tensor(self._k_inv_scale, dtype=torch.float32),
            )
            self.register_buffer(
                "v_inv_scale_tensor",
                torch.tensor(self._v_inv_scale, dtype=torch.float32),
            )
        else:
            self.k_scale = None
            self.v_scale = None
            self._k_inv_scale = None
            self._v_inv_scale = None
            self.register_buffer(
                "k_inv_scale_tensor", torch.tensor(1.0, dtype=torch.float32)
            )
            self.register_buffer(
                "v_inv_scale_tensor", torch.tensor(1.0, dtype=torch.float32)
            )

    def _prepare_key_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return tensor
        scale = self.k_inv_scale_tensor.to(dtype=tensor.dtype)
        return (tensor * scale).to(self.k_cache.dtype)

    def _prepare_value_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return tensor
        scale = self.v_inv_scale_tensor.to(dtype=tensor.dtype)
        return (tensor * scale).to(self.v_cache.dtype)

    def update(self, input_pos, k_val, v_val, batch_idx=None):
        assert (
            batch_idx is not None
        ), "batch_idx is required for paged kv cache, are you using non-paged attention?"

        if batch_idx.ndim == 1:
            batch_indices = batch_idx.view(-1, 1).expand_as(input_pos)
        else:
            assert batch_idx.ndim == 2, "batch_idx must be 1D or 2D"
            batch_indices = batch_idx

        k_store = self._prepare_key_tensor(k_val)
        v_store = self._prepare_value_tensor(v_val)

        page_size = self.page_table.page_size
        logical_block_idx = input_pos // page_size
        logical_block_offset = input_pos % page_size

        batch_flat = batch_indices.to(torch.long).reshape(-1)
        block_flat = logical_block_idx.to(torch.long).reshape(-1)
        offset_flat = logical_block_offset.to(torch.long).reshape(-1)

        physical_block_idx = self.page_table.page_table[batch_flat, block_flat]
        page_idx = physical_block_idx.to(torch.long)
        slot_idx = offset_flat

        k_view = (
            k_store.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, k_store.shape[1], k_store.shape[3])
        )
        v_view = (
            v_store.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, v_store.shape[1], v_store.shape[3])
        )

        self.k_cache[page_idx, :, slot_idx, :] = k_view
        self.v_cache[page_idx, :, slot_idx, :] = v_view

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
    ):
        self.n_pages = n_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.device = device

        # page table: [logical_batch_idx, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )
        self.page_table[0, :] = 0  # keep page 0 reserved for internal bookkeeping
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
        self.page_table[batch_idx, :] = -1
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

        can_allocate = num_pages_to_allocate <= self.pages_available
        if dry_run:
            return can_allocate

        if not can_allocate:
            raise RuntimeError(
                f"Cannot reserve {num_pages_to_allocate} pages for a sequence of length {seq_len} "
                f"in batch {batch_idx_int}. Only {self.pages_available} pages available. "
                f"Current capacity is {self.capacity[batch_idx_int]} tokens."
            )

        start_page_idx = self.capacity[batch_idx_int] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages_list = self.free_pages[-num_pages_to_allocate:]
        allocated_pages = torch.tensor(allocated_pages_list, device=self.device)
        # update page table
        self.page_table[batch_idx, start_page_idx:end_page_idx] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx,
            end_page_idx,
            device=self.device,
        )
        # update cpu side metadata
        self.page_table_cpu[batch_idx_int] += allocated_pages_list
        self.free_pages = self.free_pages[:-num_pages_to_allocate]
        self.capacity[batch_idx_int] += num_pages_to_allocate * self.page_size
        return True

    def erase(self, batch_idx: int) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (int): batch index to be removed;
        """
        # NOTE: the GPU side data will only be reset/overwritten when we allocate it for a new batch
        self.free_batch_idx.append(batch_idx)
        allocated_pages_cpu = self.page_table_cpu[batch_idx]
        self.free_pages.extend(reversed(allocated_pages_cpu))
        self.page_table_cpu[batch_idx] = []

    def build_flashinfer_kv_metadata(
        self,
        batch_idx: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct per-request metadata tensors for FlashInfer paged decoding."""

        if batch_idx.ndim != 1:
            raise ValueError("batch_idx must be 1D for FlashInfer metadata")
        if seq_lens.ndim != 1:
            raise ValueError("seq_lens must be 1D for FlashInfer metadata")
        if batch_idx.shape[0] != seq_lens.shape[0]:
            raise ValueError(
                "batch_idx and seq_lens must have matching leading dimensions"
            )

        device = self.page_table.device
        batch_size = batch_idx.shape[0]
        if batch_size == 0:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return (
                torch.zeros(1, dtype=torch.int32, device=device),
                empty,
                empty,
            )

        seq_lens = seq_lens.to(device=device, dtype=torch.int32)
        num_pages = torch.div(
            seq_lens + (self.page_size - 1),
            self.page_size,
            rounding_mode="floor",
        )
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        if num_pages.numel() > 0:
            kv_indptr[1:] = torch.cumsum(num_pages, dim=0)

        total_pages = int(kv_indptr[-1].item())
        if total_pages == 0:
            kv_indices = torch.zeros(0, dtype=torch.int32, device=device)
        else:
            max_pages = int(num_pages.max().item())
            page_rows = self.page_table[batch_idx.to(torch.long), :max_pages]
            page_rows = page_rows.to(torch.int32)
            arange_pages = torch.arange(max_pages, device=device, dtype=torch.int32)
            mask = arange_pages.unsqueeze(0) < num_pages.unsqueeze(1)
            kv_indices = torch.masked_select(page_rows, mask)

        kv_last_page_len = torch.where(
            num_pages > 0,
            ((seq_lens - 1) % self.page_size) + 1,
            torch.zeros_like(seq_lens),
        )

        return kv_indptr, kv_indices.to(torch.int32), kv_last_page_len
