"""Moondream runtime with paged KV cache and optional image prefixes."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import warnings

import torch
from torch import Tensor
from PIL import Image

from tokenizers import Tokenizer

from kestrel.config import RuntimeConfig
from kestrel.kv_cache import PageTable, PagedKVCache

from kestrel.moondream import (
    DEFAULT_MOONDREAM_CONFIG,
    MoondreamConfig,
    MoondreamModel,
    load_moondream_weights,
)
from kestrel.moondream.text import (
    lm_head,
    text_decoder,
    text_encoder,
)
from kestrel.moondream.vision import encode_image
from kestrel.utils import log_gpu_memory, reset_peak_gpu_memory
from torch.nn.attention.flex_attention import BlockMask


DEFAULT_MAX_TOKENS = 768


@dataclass
class SequenceState:
    """Metadata for an active text request."""

    batch_idx: int
    length: int
    max_length: int
    prompt_length: int | None = None
    image_length: int = 0

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


@dataclass
class _GraphWorkspace:
    token_buffer: Tensor
    batch_idx_buffer: Tensor
    position_buffer: Tensor
    output_buffer: Tensor


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
        idx = torch.atleast_1d(batch_indices)
        if idx.ndim != 1:
            raise ValueError(
                f"batch_indices must be 1D, received shape {batch_indices.shape}"
            )
        self._batch_idx_tensor = idx.to(
            device=self.cache.k_cache.device, dtype=torch.int64
        )

    def update(self, pos_ids: Tensor, k_val: Tensor, v_val: Tensor):
        if self._batch_idx_tensor is None:
            raise RuntimeError("Paged cache must be bound before update calls")

        if k_val.shape[2] != v_val.shape[2]:
            raise ValueError("k_val and v_val must share the sequence dimension")

        input_pos = torch.atleast_2d(pos_ids).to(
            dtype=torch.int32, device=k_val.device
        )
        if input_pos.shape[0] != 1 and input_pos.shape[0] != k_val.shape[0]:
            raise ValueError(
                f"Unsupported position shape {pos_ids.shape} for batch size {k_val.shape[0]}"
            )

        seq_len = input_pos.shape[1]
        if k_val.shape[2] != seq_len:
            raise ValueError(
                f"KV sequence length {k_val.shape[2]} does not match position tensor {input_pos.shape}"
            )

        batch_idx = self._batch_idx_tensor
        if batch_idx is None:
            raise RuntimeError("Paged cache must be bound before update calls")

        if seq_len == 1:
            batch_idx_arg = batch_idx.expand(k_val.shape[0])
        else:
            if batch_idx.numel() == 0:
                raise ValueError("No batch index bound for prefill update")
            batch_idx_arg = batch_idx.view(1).expand_as(input_pos)

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
            raw_config = deepcopy(DEFAULT_MOONDREAM_CONFIG)
        self.config = MoondreamConfig.from_dict(raw_config)

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

        self.model = MoondreamModel(
            self.config,
            dtype=self.dtype,
            device=self.device,
            setup_caches=False,
        ).eval()
        self.image_prefix_length = (
            self.model.vision.pos_emb.shape[1]
            if hasattr(self.model, "vision")
            else 0
        )
        load_moondream_weights(str(cfg.model_paths.weights), self.model)

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

        self.active_sequences: dict[int, SequenceState] = {}
        self._use_cuda_graphs = (
            cfg.enable_cuda_graphs
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

        self._graph_workspace: _GraphWorkspace | None = None
        self._cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_batch_sizes: list[int] = []
        self._graph_pool: object | None = None

        self._prefill_fn = self._prefill_impl
        if cfg.enable_compile:
            compile_kwargs: dict[str, object] = {"dynamic": True}
            if cfg.compile_mode:
                compile_kwargs["mode"] = cfg.compile_mode
            try:
                self._prefill_fn = torch.compile(self._prefill_impl, **compile_kwargs)
            except Exception as exc:  # pragma: no cover - torch.compile optional path
                warnings.warn(
                    f"torch.compile failed for prefill path, continuing without compilation: {exc}"
                )
                self._prefill_fn = self._prefill_impl

        if self._use_cuda_graphs:
            self._ensure_cuda_graphs_ready()

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

    def encode_image(self, image: Image.Image) -> Tensor:
        return encode_image(
            image,
            self.model.vision,
            self.config.vision,
            device=self.device,
            dtype=self.dtype,
        )

    def start_sequence(
        self,
        question: Optional[str] = None,
        *,
        prompt_tokens: Optional[Tensor] = None,
        image: Optional[Image.Image] = None,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[SequenceState, Tensor]:
        reset_peak_gpu_memory(self.device)
        log_gpu_memory("start_sequence:begin", self.device)
        if prompt_tokens is None:
            if question is None:
                raise ValueError("Either question or prompt_tokens must be provided.")
            prompt_tokens = self.build_prompt_tokens(question)

        prompt_tokens = prompt_tokens.to(device=self.device, dtype=torch.long)
        if prompt_tokens.ndim != 2:
            raise ValueError(
                f"prompt_tokens must have shape (1, N); received {prompt_tokens.shape}"
            )

        embeddings: list[Tensor] = []
        image_length = 0
        if image is not None:
            image_proj = self.encode_image(image).unsqueeze(0)
            embeddings.append(image_proj)
            image_length = image_proj.shape[1]
            log_gpu_memory("start_sequence:after_image_encode", self.device)

        token_embeds = text_encoder(prompt_tokens, self.model.text)
        embeddings.append(token_embeds)
        inputs_embeds = torch.cat(embeddings, dim=1)

        prompt_len = inputs_embeds.shape[1]
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

        attention_mask = torch.tril(
            torch.ones(1, 1, prompt_len, prompt_len, dtype=torch.bool, device=self.device)
        )
        if image_length:
            attention_mask[:, :, :image_length, :image_length] = True
        position_ids = torch.arange(
            prompt_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        logits = self._prefill(inputs_embeds, attention_mask, position_ids)
        log_gpu_memory("start_sequence:after_prefill", self.device)
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=image_length,
        )
        self.active_sequences[batch_idx] = state
        return state, logits

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self.page_table.erase(state.batch_idx)

    # ------------------------------------------------------------------
    # Core forward paths

    def greedy_generate(
        self,
        question: str,
        *,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[str, list[int]]:
        state, logits = self.start_sequence(
            question,
            image=image,
            max_new_tokens=max_new_tokens,
        )
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
    def _prefill(self, inputs_embeds: Tensor, attn_mask: Tensor, position_ids: Tensor) -> Tensor:
        return self._prefill_fn(inputs_embeds, attn_mask, position_ids)

    def _prefill_impl(
        self,
        inputs_embeds: Tensor,
        attn_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        hidden = text_decoder(
            inputs_embeds,
            self.model.text,
            attn_mask,
            position_ids,
            self.config.text,
            mode="prefill",
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

        tokens = token_ids.to(device=self.device, dtype=torch.long)
        batch_size = batch_idx.shape[0]
        if self._use_cuda_graphs and batch_size > 0:
            graph_batch_size = self._select_graph_batch_size(batch_size)
            if graph_batch_size is not None and self._cuda_graphs:
                return self._decode_with_graph(
                    batch_idx, tokens, input_pos, graph_batch_size
                )
        return self._decode_step(batch_idx, tokens, input_pos)

    def _decode_step(
        self, batch_idx: Tensor, tokens: Tensor, input_pos: Tensor
    ) -> Tensor:
        self.input_positions.zero_()
        self.input_positions[batch_idx] = input_pos
        offsets = self.input_positions[batch_idx]
        block_mask = self._make_decode_block_mask(batch_idx, offsets)

        for cache in self.layer_caches:
            cache.set_active_batch(batch_idx)

        embeds = text_encoder(tokens.view(-1, 1), self.model.text)
        position_ids = input_pos.to(dtype=torch.long).view(-1, 1)
        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask=None,
            position_ids=position_ids,
            config=self.config.text,
            flex_block_mask_slice=block_mask,
            mode="decode",
        )
        logits = lm_head(hidden, self.model.text)
        return logits

    def _decode_with_graph(
        self,
        batch_idx: Tensor,
        tokens: Tensor,
        input_pos: Tensor,
        graph_batch_size: int,
    ) -> Tensor:
        workspace = self._graph_workspace
        if workspace is None:
            raise RuntimeError("CUDA graph workspace is not initialized")
        if graph_batch_size not in self._cuda_graphs:
            raise RuntimeError(
                f"No CUDA graph captured for batch size {graph_batch_size}"
            )

        batch_size = batch_idx.shape[0]
        self._clear_graph_inputs(graph_batch_size)
        workspace.token_buffer[:batch_size, 0].copy_(tokens)
        workspace.batch_idx_buffer[:batch_size].copy_(batch_idx)
        workspace.position_buffer[:batch_size].copy_(input_pos)

        graph = self._cuda_graphs[graph_batch_size]
        graph.replay()
        return workspace.output_buffer[:batch_size].clone()

    def _ensure_cuda_graphs_ready(self) -> None:
        if not self._use_cuda_graphs or self._cuda_graphs:
            return
        self._initialize_graph_workspace()
        self._capture_decode_graphs()

    def _initialize_graph_workspace(self) -> None:
        if self._graph_workspace is not None:
            return
        max_effective_batch = max(1, self.max_batch_size - 1)
        vocab = self.model.text.lm_head.weight.shape[0]
        token_buffer = torch.zeros(
            (max_effective_batch, 1), device=self.device, dtype=torch.long
        )
        batch_idx_buffer = torch.zeros(
            (max_effective_batch,), device=self.device, dtype=torch.long
        )
        position_buffer = torch.zeros(
            (max_effective_batch,), device=self.device, dtype=torch.int32
        )
        output_buffer = torch.zeros(
            (max_effective_batch, vocab),
            device=self.device,
            dtype=self.model.text.lm_head.weight.dtype,
        )
        self._graph_workspace = _GraphWorkspace(
            token_buffer=token_buffer,
            batch_idx_buffer=batch_idx_buffer,
            position_buffer=position_buffer,
            output_buffer=output_buffer,
        )
        self._graph_batch_sizes = self._make_graph_batch_sizes(max_effective_batch)

    def _make_graph_batch_sizes(self, max_batch: int) -> list[int]:
        seeds = [size for size in (1, 2, 4, 8) if size <= max_batch]
        ramps = list(range(16, max_batch + 1, 16))
        sizes = sorted({*seeds, *ramps, max_batch})
        return sizes

    def _capture_decode_graphs(self) -> None:
        workspace = self._graph_workspace
        if workspace is None:
            raise RuntimeError("CUDA graph workspace must be initialized before capture")
        max_batch = workspace.token_buffer.shape[0]
        if max_batch == 0:
            return

        device = self.device
        batch_indices = torch.arange(1, max_batch + 1, device=device, dtype=torch.long)
        workspace.batch_idx_buffer.copy_(batch_indices)
        workspace.token_buffer.zero_()
        workspace.position_buffer.zero_()

        allocated_batches: list[int] = []
        try:
            for idx in batch_indices.tolist():
                allocated = self.page_table.allocate()
                if allocated != idx:
                    raise RuntimeError(
                        f"Expected batch index {idx} during CUDA graph capture, got {allocated}"
                    )
                batch_tensor = torch.tensor([idx], device=device, dtype=torch.int64)
                self.page_table.reserve(
                    batch_idx_int=idx,
                    batch_idx=batch_tensor,
                    seq_len=1,
                )
                allocated_batches.append(idx)

            torch.cuda.synchronize(device=device)
            for bs in reversed(self._graph_batch_sizes):
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode():
                    warmup = self._decode_step(
                        workspace.batch_idx_buffer[:bs],
                        workspace.token_buffer[:bs, 0],
                        workspace.position_buffer[:bs],
                    )
                    workspace.output_buffer[:bs].copy_(warmup)

                    torch.cuda.synchronize(device=device)

                    with torch.cuda.graph(graph, self._graph_pool):
                        out = self._decode_step(
                            workspace.batch_idx_buffer[:bs],
                            workspace.token_buffer[:bs, 0],
                            workspace.position_buffer[:bs],
                        )
                        workspace.output_buffer[:bs].copy_(out)

                if self._graph_pool is None:
                    self._graph_pool = graph.pool()
                self._cuda_graphs[bs] = graph
                torch.cuda.synchronize(device=device)
        finally:
            for idx in reversed(allocated_batches):
                self.page_table.erase(idx)
            self._clear_graph_inputs(0)

    def _select_graph_batch_size(self, batch_size: int) -> int | None:
        for size in self._graph_batch_sizes:
            if size >= batch_size:
                return size
        return None

    def _clear_graph_inputs(self, limit: int) -> None:
        workspace = self._graph_workspace
        if workspace is None:
            return
        if limit <= 0:
            limit = workspace.token_buffer.shape[0]
        workspace.token_buffer[:limit].zero_()
        workspace.batch_idx_buffer[:limit].zero_()
        workspace.position_buffer[:limit].zero_()


__all__ = ["MoondreamTextRuntime", "SequenceState", "DEFAULT_MAX_TOKENS"]
