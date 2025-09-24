"""Text-only Moondream runtime built on Kestrel's paged KV cache."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from tokenizers import Tokenizer

from kestrel.config import RuntimeConfig
from kestrel.kv_cache import PageTable, PagedKVCache

from kestrel.moondream import (
    DEFAULT_MOONDREAM3_CONFIG,
    MoondreamTextConfig,
    MoondreamTextModel,
    load_text_weights,
)
from kestrel.moondream.text import lm_head, text_decoder, text_encoder
from torch.nn.attention.flex_attention import BlockMask


DEFAULT_MAX_TOKENS = 768


@dataclass
class SequenceState:
    """Metadata for an active text request."""

    batch_idx: int
    length: int
    max_length: int
    prompt_length: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_length is None:
            self.prompt_length = self.length

    def advance(self, tokens: int = 1) -> None:
        self.length += tokens

    @property
    def output_length(self) -> int:
        return self.length - (self.prompt_length or 0)

    def mark_prefilled(self, prompt_len: int) -> None:
        self.prompt_length = prompt_len
        self.length = prompt_len

    def at_capacity(self) -> bool:
        return self.length >= self.max_length

    def remaining_new_tokens(self) -> int:
        return max(self.max_length - self.length, 0)


class _LayerPagedCache(torch.nn.Module):
    """Adapter that wires :class:`PagedKVCache` into the text blocks."""

    def __init__(
        self,
        page_table: PageTable,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cache = PagedKVCache(
            page_table,
            n_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(device)
        self._batch_idx_tensor: Optional[Tensor] = None

    def bind(self, batch_idx: int, device: torch.device) -> None:
        self.set_active_batch(torch.tensor([batch_idx], device=device, dtype=torch.int64))

    def set_active_batch(self, batch_indices: Tensor) -> None:
        if batch_indices.ndim == 0:
            batch_indices = batch_indices.view(1)
        if batch_indices.ndim != 1:
            raise ValueError(
                f"batch_indices must be 1D, received shape {batch_indices.shape}"
            )
        self._batch_idx_tensor = batch_indices.to(
            device=self.cache.k_cache.device, dtype=torch.int64
        )

    def update(self, pos_ids: Tensor, k_val: Tensor, v_val: Tensor):
        if self._batch_idx_tensor is None:
            raise RuntimeError("Paged cache must be bound before update calls")

        if pos_ids.ndim == 1:
            input_pos = pos_ids.unsqueeze(0)
        elif pos_ids.ndim == 2:
            input_pos = pos_ids
        else:
            raise ValueError(f"Unsupported position shape: {pos_ids.shape}")

        input_pos = input_pos.to(dtype=torch.int32, device=k_val.device)
        if k_val.shape[2] != input_pos.shape[1]:
            raise ValueError(
                f"KV sequence length {k_val.shape[2]} does not match position tensor {input_pos.shape}"
            )
        batch_idx = self._batch_idx_tensor
        batch_count, seq_len = input_pos.shape

        if seq_len == 1:
            if batch_idx.numel() == 1:
                batch_idx_arg = batch_idx.expand(batch_count)
            elif batch_idx.numel() == batch_count:
                batch_idx_arg = batch_idx
            else:
                raise ValueError(
                    f"Batch index count {batch_idx.numel()} does not match decode batch {batch_count}"
                )
        else:
            if batch_count != 1:
                raise ValueError(
                    f"Prefill expects single sequence, got input_pos shape {input_pos.shape}"
                )
            if batch_idx.numel() == 0:
                raise ValueError("No batch index bound for prefill update")
            batch_idx_scalar = batch_idx.view(1)[0]
            batch_idx_arg = batch_idx_scalar.view(1, 1).expand(1, seq_len)

        return self.cache.update(
            input_pos=input_pos,
            k_val=k_val,
            v_val=v_val,
            batch_idx=batch_idx_arg,
        )


class MoondreamTextRuntime:
    """High-level runtime for paged text-only Moondream inference."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self._cfg = cfg
        self.device = cfg.resolved_device()
        self.dtype = cfg.resolved_dtype()

        if cfg.model_paths.config_json:
            with Path(cfg.model_paths.config_json).open("r", encoding="utf-8") as fp:
                raw_config = json.load(fp)
        else:
            raw_config = deepcopy(DEFAULT_MOONDREAM3_CONFIG)
        self.config = MoondreamTextConfig.from_dict(raw_config)

        self.max_seq_length = cfg.max_seq_length or self.config.text.max_context
        if self.max_seq_length % cfg.page_size != 0:
            raise ValueError("max_seq_length must be divisible by page_size")

        self.page_size = cfg.page_size
        if cfg.max_batch_size < 2:
            raise ValueError(
                "max_batch_size must be at least 2; index 0 is reserved for the page table."
            )
        self.max_batch_size = cfg.max_batch_size
        n_pages = self.max_seq_length // self.page_size

        self.page_table = PageTable(
            n_pages=n_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_size,
            device=str(self.device),
        )

        self.model = MoondreamTextModel(
            self.config,
            dtype=self.dtype,
            device=self.device,
            setup_caches=False,
        ).eval()
        load_text_weights(str(cfg.model_paths.weights), self.model)

        tokenizer_path = cfg.model_paths.tokenizer or "moondream/starmie-v1"
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

        head_dim = self.config.text.dim // self.config.text.n_heads
        self.layer_caches: list[_LayerPagedCache] = []
        for block in self.model.text.blocks:
            cache = _LayerPagedCache(
                page_table=self.page_table,
                n_kv_heads=self.config.text.n_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            block.kv_cache = cache
            self.layer_caches.append(cache)

        self.block_mask = self.page_table.create_causal_blockmask(
            B=self.max_batch_size, L=self.max_seq_length
        )
        self.input_positions = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        self.active_sequences: Dict[int, SequenceState] = {}

    # ------------------------------------------------------------------
    # Capacity helpers

    def can_reserve(self, total_length: int) -> bool:
        """Return True if a request of ``total_length`` tokens can be admitted."""

        return self.page_table.can_reserve(total_length)

    # ------------------------------------------------------------------
    # Prompt helpers

    def build_prompt_tokens(self, question: str) -> Tensor:
        prefix = self.config.tokenizer.templates["query"]["prefix"]
        suffix = self.config.tokenizer.templates["query"]["suffix"]
        ids = (
            [self.config.tokenizer.bos_id]
            + prefix
            + self.tokenizer.encode(question).ids
            + suffix
        )
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Sequence lifecycle

    def start_sequence(
        self, question: Optional[str] = None, *, prompt_tokens: Optional[Tensor] = None, max_new_tokens: Optional[int] = None
    ) -> tuple[SequenceState, Tensor]:
        if prompt_tokens is None:
            if question is None:
                raise ValueError('Either question or prompt_tokens must be provided.')
            prompt_tokens = self.build_prompt_tokens(question)
        prompt_len = prompt_tokens.shape[1]
        max_new = max_new_tokens or DEFAULT_MAX_TOKENS
        target_length = prompt_len + max_new
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds max_seq_length={self.max_seq_length}."
            )

        batch_idx = self.page_table.allocate()
        batch_tensor = torch.tensor([batch_idx], device=self.device, dtype=torch.int64)
        self.page_table.reserve(
            batch_idx_int=batch_idx,
            batch_idx=batch_tensor,
            seq_len=target_length,
        )
        for cache in self.layer_caches:
            cache.bind(batch_idx=batch_idx, device=self.device)

        logits = self._prefill(prompt_tokens)
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
        )
        self.active_sequences[batch_idx] = state
        return state, logits

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self.page_table.erase(state.batch_idx)

    # ------------------------------------------------------------------
    # Core forward paths

    def greedy_generate(
        self, question: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS
    ) -> tuple[str, list[int]]:
        state, logits = self.start_sequence(question, max_new_tokens=max_new_tokens)
        generated: list[int] = []
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        try:
            while True:
                token_id = next_token.view(-1)[0].item()
                if token_id == self.config.tokenizer.eos_id:
                    break
                generated.append(token_id)
                if len(generated) >= max_new_tokens:
                    break
                logits = self.decode(state, next_token.view(1, 1))
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
        finally:
            self.release_sequence(state)

        text = self.tokenizer.decode(generated) if generated else ""
        return text, generated

    @torch.inference_mode()
    def _prefill(self, input_ids: Tensor) -> Tensor:
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        embeds = text_encoder(input_ids, self.model.text)
        attn_mask = torch.tril(
            torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=self.device)
        )
        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask,
            pos_ids,
            self.config.text,
        )
        logits = lm_head(hidden, self.model.text)
        return logits

    @torch.inference_mode()
    def decode(self, state: SequenceState, token_id: Tensor) -> Tensor:
        token_vector = token_id.view(-1)
        logits = self.decode_batch([state], token_vector)[0]
        state.advance()
        return logits.unsqueeze(0)

    def _build_decode_block_mask(
        self, state: SequenceState, pos_tensor: Tensor
    ) -> Tensor:
        batch_idx_tensor = torch.tensor(
            [state.batch_idx], device=self.device, dtype=torch.int64
        )
        pos_tensor = pos_tensor.to(device=self.device, dtype=torch.int32)
        self.input_positions.zero_()
        self.input_positions[state.batch_idx] = pos_tensor.item()
        offsets = self.input_positions[batch_idx_tensor]
        return self._make_decode_block_mask(batch_idx_tensor, offsets)

    def _make_decode_block_mask(
        self, batch_idx: Tensor, offsets: Tensor
    ) -> BlockMask:
        """Construct a decode-time block mask for the provided batch indices."""

        if batch_idx.ndim != 1:
            raise ValueError("batch_idx must be a 1D tensor")
        if offsets.ndim != 1:
            raise ValueError("offsets must be a 1D tensor")
        if batch_idx.shape[0] != offsets.shape[0]:
            raise ValueError("batch_idx and offsets must have the same shape")

        block_mask = self.block_mask
        input_block_idx = offsets // block_mask.BLOCK_SIZE[0]

        def causal_offset(off: Tensor):
            def offset_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
                return q_idx + off[b] >= kv_idx

            return offset_fn

        kv_num_blocks = (
            block_mask.kv_num_blocks[batch_idx, :, input_block_idx]
            .view(offsets.shape[0], 1, 1)
        )
        kv_indices = (
            block_mask.kv_indices[batch_idx, :, input_block_idx]
            .view(offsets.shape[0], 1, 1, -1)
        )
        full_kv_num_blocks = None
        full_kv_indices = None
        if block_mask.full_kv_num_blocks is not None:
            full_kv_num_blocks = (
                block_mask.full_kv_num_blocks[batch_idx, :, input_block_idx]
                .view(offsets.shape[0], 1, 1)
            )
            full_kv_indices = (
                block_mask.full_kv_indices[batch_idx, :, input_block_idx]
                .view(offsets.shape[0], 1, 1, -1)
            )

        seq_lengths = (1, block_mask.seq_lengths[1])
        logical_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=causal_offset(offsets),
            seq_lengths=seq_lengths,
        )
        return self.page_table.convert_logical_block_mask(logical_mask, batch_idx)

    @torch.inference_mode()
    def decode_batch(
        self,
        states: Sequence[SequenceState],
        token_ids: Tensor,
    ) -> Tensor:
        if not states:
            raise ValueError("states must not be empty")

        if token_ids.ndim > 1:
            token_ids = token_ids.view(-1)

        if token_ids.shape[0] != len(states):
            raise ValueError("token_ids and states must have matching batch dimensions")

        batch_idx = torch.tensor(
            [state.batch_idx for state in states],
            device=self.device,
            dtype=torch.int64,
        )
        input_pos = torch.tensor(
            [state.length for state in states],
            device=self.device,
            dtype=torch.int32,
        )

        self.input_positions.zero_()
        self.input_positions[batch_idx] = input_pos
        offsets = self.input_positions[batch_idx]
        block_mask = self._make_decode_block_mask(batch_idx, offsets)

        for cache in self.layer_caches:
            cache.set_active_batch(batch_idx)

        tokens = token_ids.to(device=self.device, dtype=torch.long)
        embeds = text_encoder(tokens.view(-1, 1), self.model.text)
        position_ids = input_pos.to(dtype=torch.long).view(-1, 1)
        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask=None,
            position_ids=position_ids,
            config=self.config.text,
            flex_block_mask_slice=block_mask,
        )
        logits = lm_head(hidden, self.model.text)
        return logits


__all__ = ["MoondreamTextRuntime", "SequenceState", "DEFAULT_MAX_TOKENS"]
