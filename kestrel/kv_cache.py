# Adapted from attention-gym
# Original source: https://github.com/pytorch-labs/attention-gym
# License: BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
# Copyright (c) 2023, Driss Guessous

# the original implementation has some bugs and has some feature that lives outside of the PageTable class

from typing import Optional
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    noop_mask,
    create_block_mask,
)

create_block_mask = torch.compile(create_block_mask)


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
        self.page_table[0, :] = (
            0  # page 0 is reserved for simpler code in assign_prefill_no_paging
        )
        self.page_table_cpu = [[] for _ in range(max_batch_size)]

        self.capacity = [
            0 for _ in range(max_batch_size)
        ]  # capacity: batch_idx -> number of pages allocated * page size
        self.free_pages = list(
            reversed(range(1, n_pages))
        )  # page 0 is reserved for simpler code in assign_prefill_no_paging
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

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
        cache (Tensor): the cache to store the values; shape:`(MAX_PAGES, H, PAGE_SIZE, D)`
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        _, H_cache, _, D_cache = k_cache.shape
        assert H_cache == H, "number of heads must match"
        assert D_cache == K_D, "hidden dim must match"
        assert input_pos.shape == (B, S), "input_pos must have the same shape as val"
        assert batch_idx.shape == (B,), "batch_idx must have one dimension only"

        V_D = v_val.shape[3]
        if B != batch_idx.shape[0]:
            raise RuntimeError(
                f"Expect val and batch_idx have the same batch size but got B={B} and B={batch_idx.shape[0]}."
            )
        if S != input_pos.shape[1]:
            raise RuntimeError(
                f"Expect val and input_pos has the same length but got S={S} and S={input_pos.shape[0]}."
            )
        if K_D != D_cache:
            raise RuntimeError(
                f"Expect k_val and k_cache has the same hidden dim but got D={K_D} and D={D_cache}."
            )
        if V_D != v_cache.shape[3]:
            raise RuntimeError(
                f"Expect v_val and v_cache has the same hidden dim but got D={V_D} and D={v_cache.shape[3]}."
            )

        # find address
        logical_block_idx = input_pos // self.page_size  # [B, S]
        logical_block_offset = input_pos % self.page_size  # [B, S]

        # NOTE: this code path is only used for decoding. For batch prefill, use assign_prefill_no_paging() instead
        physical_block_idx = torch.gather(
            self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)
        ).to(
            torch.int32
        )  # [B, S]

        page_idx = physical_block_idx.reshape(-1).to(torch.long)
        slot_idx = logical_block_offset.reshape(-1).to(torch.long)

        k_store = self._prepare_key_tensor(k_val)
        v_store = self._prepare_value_tensor(v_val)

        self.page_table.assign(
            batch_idx,
            input_pos,
            k_store,
            v_store,
            k_cache,
            v_cache,
        )

        return k_val, v_val

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_sizebut got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)

        assert batch_idx.ndim == 1, "batch_idx must be a 1D tensor"
        assert (
            batch_idx.shape[0] == B
        ), "batch_idx must have the same shape as block_mask"
        assert (
            B <= self.max_batch_size
        ), "batch_idx must be less than or equal to max_batch_size"

        page_table = self.page_table[batch_idx]

        def transform(num_blocks, indices):
            """
            transform the block mask from [B, H, num_q_blocks, num_logical_kv_blocks]
            to [B, H, num_q_blocks, num_physical_kv_blocks]

            kv_num_blocks: [B, H, num_q_blocks] -> unchanged
            kv_indices: [B, H, num_q_blocks, num_logical_kv_blocks] -> [B, H, num_q_blocks, num_physical_kv_blocks]
            """
            if num_blocks is None:
                return None, None
            new_kv_num_blocks = num_blocks.clone()
            new_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(page_table, 1, indices.view(B, -1).to(torch.int64))
                .view(block_mask.kv_indices.shape)
                .to(torch.int32)
            )
            return new_kv_num_blocks, new_kv_indices

        new_kv_num_blocks, new_kv_indices = transform(
            block_mask.kv_num_blocks, block_mask.kv_indices
        )
        new_full_kv_num_blocks, new_full_kv_indices = transform(
            block_mask.full_kv_num_blocks, block_mask.full_kv_indices
        )

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, batch_idx)

        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_logical_kv_idx(
        self,
        physical_batch_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
        batch_idx: torch.Tensor,
    ):
        logical_batch_idx = batch_idx[physical_batch_idx]
        physical_kv_block = physical_kv_idx // self.page_size
        physical_kv_offset = physical_kv_idx % self.page_size
        logical_block_idx = self.physical_to_logical[
            logical_batch_idx, physical_kv_block
        ]
        logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
        is_valid = logical_block_idx >= 0
        safe_logical_kv_idx = logical_kv_idx.clamp(min=0)
        return is_valid, safe_logical_kv_idx

    def get_mask_mod(
        self, mask_mod: Optional[_mask_mod_signature], batch_idx: torch.Tensor
    ) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            is_valid, safe_logical_kv_idx = self.get_logical_kv_idx(
                b, physical_kv_idx, batch_idx
            )
            return torch.where(
                is_valid, mask_mod(b, h, q_idx, safe_logical_kv_idx), False
            )

        return new_mask_mod

    def create_causal_blockmask(self, B, L):
        """A minimal, unoptimized causal block mask creation function"""

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(
            causal,
            B=B,
            H=None,
            Q_LEN=L,
            KV_LEN=L,
            BLOCK_SIZE=self.page_size,
            device=self.device,
        )

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

    def create_prefill_blockmask_no_paging(
        self, batch_idx: Tensor, BLOCK_SIZE: int = 128
    ):
        """
        there's no prefix sharing implemented, batch_idx is the document id, batch_idx is not guaranteed to be sorted
        """
        assert batch_idx.ndim == 2, "batch_idx must be a 2D tensor"
        assert batch_idx.shape[0] == 1, "batch_idx must have batch size 1"
        L = batch_idx.shape[1]
        docs = batch_idx.view(-1)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        return create_block_mask(
            document_causal, B=1, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=BLOCK_SIZE
        )

    # we assign prefill to the cache, similar to assign(), except we don't return the k_cache, v_cache, we only return the k_val, v_val
    def assign_prefill_no_paging(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        assigns kv and returns the original kv

        batch_idx: [1, L]
        input_pos: [1, L]
        k_val: [1, H, L, D]
        v_val: [1, H, L, D]
        k_cache: [MAX_PAGES, H, PAGE_SIZE, D]
        v_cache: [MAX_PAGES, H, PAGE_SIZE, D]
        """

        assert batch_idx.ndim == 2, "batch_idx must be a 2D tensor"
        assert input_pos.ndim == 2, "input_pos must be a 2D tensor"
        assert k_val.ndim == 4, "k_val must be a 4D tensor"
        assert v_val.ndim == 4, "v_val must be a 4D tensor"
        assert k_cache.ndim == 4, "k_cache must be a 4D tensor"
        assert v_cache.ndim == 4, "v_cache must be a 4D tensor"
        assert batch_idx.shape[0] == 1, "batch_idx must have batch size 1"

        input_pos_block_idx = input_pos // self.page_size
        input_pos_offset_in_block = input_pos % self.page_size
        physical_page = self.page_table[batch_idx, input_pos_block_idx].to(torch.long)
        page_idx = physical_page.view(-1)
        slot_idx = input_pos_offset_in_block.view(-1).to(torch.long)

        k_store = (
            k_val.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, k_val.shape[1], k_val.shape[3])
        )
        v_store = (
            v_val.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, v_val.shape[1], v_val.shape[3])
        )

        k_cache[page_idx, :, slot_idx, :] = k_store
        v_cache[page_idx, :, slot_idx, :] = v_store

        return k_val, v_val
