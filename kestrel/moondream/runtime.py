"""Moondream runtime with paged KV cache and optional image prefixes."""


import functools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, cast

import warnings
import threading

import numpy as np
import pyvips
import torch
from torch import Tensor

from kestrel.utils import CpuGpuBuffer

from tokenizers import Tokenizer

from kestrel.config import RuntimeConfig
from kestrel.kv_cache import PageTable, PagedKVCache

from .config import DEFAULT_MOONDREAM_CONFIG, MoondreamConfig
from .model import MoondreamModel
from .weights import load_moondream_weights
from .text import (
    lm_head,
    text_decoder,
    text_encoder,
)
from .vision import encode_image
from .lora import LoRA
from .image_crops import OverlapCropOutput
from .flashinfer import (
    FlashInferBatchMetadata,
    FlashInferDecodeContext,
    FlashInferPrefillBatchMetadata,
    FlashInferPrefillContext,
)
from .region import (
    build_region_module,
    encode_coordinate,
    encode_size,
    decode_coordinate,
    decode_size,
)
from ..hqsam_head_refiner import HQSAMHeadRefiner



DEFAULT_MAX_TOKENS = 768


class TextToken(NamedTuple):
    """Discrete text token represented by its vocabulary id."""

    token_id: int


class CoordToken(NamedTuple):
    """Normalized positional token emitted or consumed by the region model."""

    pos: float


class SizeToken(NamedTuple):
    """Normalized width/height token emitted or consumed by the region model."""

    width: float
    height: float


Token = TextToken | CoordToken | SizeToken


class RuntimeDecodeResult(NamedTuple):
    logits: Tensor
    hidden: Tensor


@dataclass
class SequenceState:
    """Metadata for an active text request."""

    batch_idx: int
    length: int
    max_length: int
    prompt_length: int | None = None
    image_length: int = 0
    last_hidden: Tensor | None = None

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
    hidden_buffer: Tensor


@dataclass
class _BatchBinding:
    tensor: Tensor | None = None


class _LayerPagedCache(torch.nn.Module):
    """Adapter that wires :class:`PagedKVCache` into the text blocks."""

    def __init__(
        self,
        page_table: PageTable,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.cache = PagedKVCache(
            page_table,
            n_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        ).to(device)
        self._batch_binding = _BatchBinding()

    def attach_batch_binding(self, binding: _BatchBinding) -> None:
        self._batch_binding = binding

    def update(
        self,
        pos_ids: Tensor,
        k_val: Tensor,
        v_val: Tensor,
        *,
        slot_mapping: Tensor,
    ):
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

        batch_idx = self._batch_binding.tensor

        if seq_len == 1:
            batch_idx_arg = batch_idx.expand(k_val.shape[0])
        else:
            batch_idx_arg = batch_idx.view(1).expand_as(input_pos)

        return self.cache.update(
            input_pos=input_pos,
            k_val=k_val,
            v_val=v_val,
            batch_idx=batch_idx_arg,
            slot_mapping=slot_mapping,
        )


class MoondreamRuntime:
    """High-level runtime for paged text-only Moondream inference."""

    def __init__(self, cfg: RuntimeConfig, *, lora: Optional[LoRA] = None) -> None:
        self._cfg = cfg
        self._lora = lora
        self.device = cfg.resolved_device()
        self.dtype = cfg.resolved_dtype()
        torch.cuda.set_device(self.device)
        # Guards CUDA graph capture so other threads avoid device-wide sync during capture.
        self.graph_capture_lock = threading.RLock()

        if cfg.model_paths.config_json:
            with Path(cfg.model_paths.config_json).open("r", encoding="utf-8") as fp:
                raw_config = json.load(fp)
        else:
            raw_config = deepcopy(DEFAULT_MOONDREAM_CONFIG)

        text_section = raw_config.setdefault("text", {})
        default_context = int(
            text_section.get("max_context", DEFAULT_MOONDREAM_CONFIG["text"]["max_context"])
        )
        requested_context = cfg.max_seq_length
        if requested_context is not None and requested_context != default_context:
            text_section["max_context"] = int(requested_context)

        raw_config["refiner_iters"] = cfg.refiner_iters

        self.config = MoondreamConfig.from_dict(raw_config)

        self._kv_layer_k_scales: list[float] | None = None
        self._kv_layer_v_scales: list[float] | None = None

        self.max_seq_length = int(
            cfg.max_seq_length if cfg.max_seq_length is not None else self.config.text.max_context
        )
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
        self.region = build_region_module(self.config.region, self.dtype).to(self.device)
        self.image_prefix_length = (
            self.model.vision.pos_emb.shape[1]
            if hasattr(self.model, "vision")
            else 0
        )
        n_layers = self.config.text.n_layers
        captured_k_scales: list[Optional[float]] = [None] * n_layers
        captured_v_scales: list[Optional[float]] = [None] * n_layers

        def _capture_kv_scale(name: str, tensor: torch.Tensor) -> None:
            if not name.startswith("text_model.transformer.h."):
                return
            parts = name.split(".")
            if len(parts) < 6:
                return
            try:
                layer_idx = int(parts[3])
            except ValueError:
                return
            if not (0 <= layer_idx < n_layers):
                return
            if parts[4] != "kv_quantizer":
                return
            target = parts[5]
            value_tensor = tensor.detach()
            if value_tensor.numel() != 1:
                return
            value = float(value_tensor.cpu().item())
            if target == "k_scale":
                captured_k_scales[layer_idx] = value
            elif target == "v_scale":
                captured_v_scales[layer_idx] = value

        load_moondream_weights(
            str(cfg.model_paths.weights),
            self.model,
            tensor_hook=_capture_kv_scale,
            region=self.region,
        )

        if all(val is not None for val in captured_k_scales) and all(
            val is not None for val in captured_v_scales
        ):
            self._kv_layer_k_scales = [cast(float, val) for val in captured_k_scales]
            self._kv_layer_v_scales = [cast(float, val) for val in captured_v_scales]
        elif any(val is not None for val in captured_k_scales) or any(
            val is not None for val in captured_v_scales
        ):
            warnings.warn(
                "Partial KV scales found in checkpoint; falling back to standard KV cache.",
                stacklevel=2,
            )

        if (
            self._kv_layer_k_scales is not None
            and self._kv_layer_v_scales is not None
        ):
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            self.kv_cache_dtype = self.dtype

        tokenizer_path = cfg.model_paths.tokenizer or "moondream/starmie-v1"
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

        head_dim = self.config.text.dim // self.config.text.n_heads
        self.head_dim = head_dim
        self.layer_caches: list[_LayerPagedCache] = []
        for block in self.model.text.blocks:
            layer_idx = len(self.layer_caches)
            k_scale = None
            v_scale = None
            if (
                self._kv_layer_k_scales is not None
                and self._kv_layer_v_scales is not None
            ):
                k_scale = self._kv_layer_k_scales[layer_idx]
                v_scale = self._kv_layer_v_scales[layer_idx]
            cache = _LayerPagedCache(
                page_table=self.page_table,
                n_kv_heads=self.config.text.n_kv_heads,
                head_dim=head_dim,
                dtype=self.kv_cache_dtype,
                device=self.device,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            block.kv_cache = cache
            self.layer_caches.append(cache)

        self.active_sequences: dict[int, SequenceState] = {}
        self._use_cuda_graphs = (
            cfg.enable_cuda_graphs
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

        self._flashinfer_ctx = FlashInferDecodeContext(
            device=self.device,
            q_dtype=self.dtype,
            kv_dtype=self.kv_cache_dtype,
            page_size=self.page_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_length,
            use_cuda_graphs=self._use_cuda_graphs,
        )
        # Force fa2 backend: fa3 lacks mixed-precision attention (see flashinfer#2038).
        self._flashinfer_prefill_ctx = FlashInferPrefillContext(
            device=self.device,
            q_dtype=self.dtype,
            kv_dtype=self.kv_cache_dtype,
            page_size=self.page_size,
        )
        self._active_prefill_metadata: Optional[FlashInferPrefillBatchMetadata] = None
        self._active_prefill_batch_idx: Optional[Tensor] = None

        self._graph_workspace: _GraphWorkspace | None = None
        self._cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_batch_sizes: list[int] = []
        self._graph_pool: object | None = None
        self._batch_idx = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int64,
            device=self.device,
            pin_memory=True,
        )
        self._batch_binding: _BatchBinding = _BatchBinding()
        self._input_pos = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            device=self.device,
            pin_memory=True,
        )
        self._tokens = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.long,
            device=self.device,
            pin_memory=True,
        )
        self._prefill_batch_idx = CpuGpuBuffer(
            1,
            dtype=torch.int64,
            device=self.device,
            pin_memory=True,
            with_numpy=False,
        )
        self._prefill_query_lens = CpuGpuBuffer(
            1,
            dtype=torch.int32,
            device=self.device,
            pin_memory=True,
            with_numpy=False,
        )

        for cache in self.layer_caches:
            cache.attach_batch_binding(self._batch_binding)

        self._prefill_fn = self._prefill_impl

        self.hqsam_head_refiner = HQSAMHeadRefiner(device=self.device)

        if self._use_cuda_graphs:
            self._ensure_cuda_graphs_ready()

    # ------------------------------------------------------------------
    # Capacity helpers

    def can_reserve(self, total_length: int) -> bool:
        """Return True if a request of ``total_length`` tokens can be admitted."""

        return self.page_table.can_reserve(total_length)

    # ------------------------------------------------------------------
    # Prompt helpers

    @functools.cached_property
    def bos_embed(self) -> Tensor:
        bos = torch.tensor(
            [[self.config.tokenizer.bos_id]],
            device=self.device,
            dtype=torch.long,
        )
        return text_encoder(bos, self.model.text)

    def _embed_tokens(self, tokens: Sequence[Token]) -> Tensor:
        """Embed an in-order prompt (single sequence) into shape (1, L, dim)."""

        if not tokens:
            dim = self.bos_embed.shape[-1]
            return torch.empty((1, 0, dim), device=self.device, dtype=self.dtype)

        length = len(tokens)
        width = self.bos_embed.shape[-1]
        out = torch.empty((1, length, width), device=self.device, dtype=self.dtype)

        text_pos: list[int] = []
        coord_pos: list[int] = []
        size_pos: list[int] = []
        text_ids: list[int] = []
        coord_vals: list[float] = []
        size_vals: list[tuple[float, float]] = []

        for idx, token in enumerate(tokens):
            if isinstance(token, TextToken):
                text_pos.append(idx)
                text_ids.append(token.token_id)
            elif isinstance(token, CoordToken):
                coord_pos.append(idx)
                coord_vals.append(token.pos)
            elif isinstance(token, SizeToken):
                size_pos.append(idx)
                size_vals.append((token.width, token.height))
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported token type: {type(token)!r}")

        if text_ids:
            ids = torch.tensor([text_ids], device=self.device, dtype=torch.long)
            text_emb = text_encoder(ids, self.model.text)
            out[:, text_pos, :] = text_emb

        if coord_vals:
            coords = torch.tensor(
                coord_vals,
                device=self.device,
                dtype=self.region.coord_features.dtype,
            ).view(-1, 1)
            coord_emb = encode_coordinate(coords, self.region)
            out[:, coord_pos, :] = coord_emb.unsqueeze(0)

        if size_vals:
            sizes = torch.tensor(
                size_vals,
                device=self.device,
                dtype=self.region.size_features.dtype,
            )
            size_emb = encode_size(sizes, self.region)
            out[:, size_pos, :] = size_emb.unsqueeze(0)

        return out

    def _embed_token_batch(self, tokens: Sequence[Token]) -> Tensor:
        """Embed N pending tokens (one per active sequence) into (N, 1, dim)."""
        if not tokens:
            dim = self.bos_embed.shape[-1]
            return torch.empty((0, 1, dim), device=self.device, dtype=self.dtype)

        batch = len(tokens)
        dim = self.bos_embed.shape[-1]
        out = torch.empty((batch, 1, dim), device=self.device, dtype=self.dtype)

        text_idx: list[int] = []
        text_ids: list[int] = []
        coord_idx: list[int] = []
        coord_vals: list[float] = []
        size_idx: list[int] = []
        size_vals: list[tuple[float, float]] = []

        for i, token in enumerate(tokens):
            if isinstance(token, TextToken):
                text_idx.append(i)
                text_ids.append(token.token_id)
            elif isinstance(token, CoordToken):
                coord_idx.append(i)
                coord_vals.append(token.pos)
            elif isinstance(token, SizeToken):
                size_idx.append(i)
                size_vals.append((token.width, token.height))
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported token type: {type(token)!r}")

        if text_idx:
            ids = torch.tensor(text_ids, device=self.device, dtype=torch.long).view(
                len(text_idx), 1
            )
            embeds = text_encoder(ids, self.model.text)
            out[text_idx, :, :] = embeds

        if coord_idx:
            coords = torch.tensor(
                coord_vals,
                device=self.device,
                dtype=self.region.coord_features.dtype,
            ).view(len(coord_idx), 1)
            embeds = encode_coordinate(coords, self.region)
            out[coord_idx, 0, :] = embeds

        if size_idx:
            sizes = torch.tensor(
                size_vals,
                device=self.device,
                dtype=self.region.size_features.dtype,
            )
            embeds = encode_size(sizes, self.region)
            out[size_idx, 0, :] = embeds

        return out

    def render_token(self, token_id: int, hidden: Tensor) -> Token:
        """Materialise a sampled id into its typed token, decoding coords/sizes."""
        coord_id = self.config.tokenizer.coord_id
        size_id = self.config.tokenizer.size_id

        hidden_row = hidden.squeeze(0) if hidden.ndim == 2 else hidden
        hidden_batch = hidden_row.unsqueeze(0)

        if token_id == coord_id:
            logits = decode_coordinate(hidden_batch, self.region).squeeze(0)
            bins = logits.shape[-1]
            index = torch.argmax(logits).item()
            denom = max(bins - 1, 1)
            pos = float(min(max(index / denom, 0.0), 1.0))
            return CoordToken(pos=pos)

        if token_id == size_id:
            logits = decode_size(hidden_batch, self.region)
            width_logits = logits[0]
            height_logits = logits[1]
            bins = width_logits.shape[-1]
            width_bin = torch.argmax(width_logits).item()
            height_bin = torch.argmax(height_logits).item()
            scale = float(bins - 1) if bins > 1 else 1.0
            width = 2.0 ** ((width_bin / scale) * 10.0 - 10.0)
            height = 2.0 ** ((height_bin / scale) * 10.0 - 10.0)
            width = float(min(max(width, 0.0), 1.0))
            height = float(min(max(height, 0.0), 1.0))
            return SizeToken(width=width, height=height)

        return TextToken(token_id=token_id)

    def encode_image(
        self,
        image: Optional[pyvips.Image | np.ndarray],
        *,
        overlap: Optional[OverlapCropOutput] = None,
        adapter: Optional[LoRA] = None,
    ) -> Tensor:
        return encode_image(
            image,
            self.model.vision,
            self.config.vision,
            device=self.device,
            dtype=self.dtype,
            overlap=overlap,
            adapter=adapter,
        )

    def start_sequence(
        self,
        prompt_tokens: Tensor | Sequence[Token],
        *,
        image: Optional[pyvips.Image | np.ndarray] = None,
        image_crops: Optional[OverlapCropOutput] = None,
        max_new_tokens: Optional[int] = None,
        adapter: Optional[LoRA] = None,
    ) -> tuple[SequenceState, Tensor]:
        if isinstance(prompt_tokens, Tensor):
            tokens_view = prompt_tokens.to(device=self.device, dtype=torch.long)
            if tokens_view.ndim != 2:
                raise ValueError(
                    f"prompt_tokens must have shape (1, N); received {tokens_view.shape}"
                )
            prompt_embed = (
                text_encoder(tokens_view, self.model.text)
                if tokens_view.shape[1]
                else None
            )
        else:
            prompt_embed = self._embed_tokens(list(prompt_tokens))
            if prompt_embed.shape[1] == 0:
                prompt_embed = None

        segments: list[Tensor] = [self.bos_embed]
        image_length = 0
        if image is not None or image_crops is not None:
            image_embed = self.encode_image(
                image, overlap=image_crops, adapter=adapter
            ).unsqueeze(0)
            segments.append(image_embed)
            image_length = image_embed.shape[1]
        if prompt_embed is not None:
            segments.append(prompt_embed)

        inputs_embeds = torch.cat(segments, dim=1)

        prompt_len = inputs_embeds.shape[1]
        max_new = max_new_tokens or DEFAULT_MAX_TOKENS
        target_length = prompt_len + max_new
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds max_seq_length={self.max_seq_length}."
            )

        batch_idx = self.page_table.allocate()
        self._prefill_batch_idx.cpu[0] = batch_idx
        batch_tensor = self._prefill_batch_idx.copy_to_gpu()
        self.page_table.reserve(
            batch_idx_int=batch_idx,
            batch_idx=batch_tensor,
            seq_len=target_length,
        )
        self._batch_binding.tensor = batch_tensor

        attention_mask = None
        position_ids = torch.arange(
            prompt_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        self._prefill_query_lens.cpu[0] = prompt_len
        query_lens = self._prefill_query_lens.copy_to_gpu()
        prefill_metadata = self._build_flashinfer_prefill_metadata(
            batch_tensor, query_lens
        )
        custom_mask = None
        if image_length:
            mask_matrix = torch.tril(
                torch.ones(
                    (prompt_len, prompt_len),
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            prefix_len = image_length + 1
            mask_matrix[:prefix_len, :prefix_len] = True
            custom_mask = mask_matrix.flatten()
        self._flashinfer_prefill_ctx.plan(
            prefill_metadata,
            num_q_heads=self.config.text.n_heads,
            num_kv_heads=self.config.text.n_kv_heads,
            head_dim=self.head_dim,
            custom_mask=custom_mask,
        )
        self._active_prefill_metadata = prefill_metadata
        self._active_prefill_batch_idx = batch_tensor
        try:
            hidden, logits = self._prefill(inputs_embeds, attention_mask, position_ids)
        finally:
            self._active_prefill_metadata = None
            self._active_prefill_batch_idx = None
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=image_length,
            last_hidden=hidden[:, -1, :].squeeze(0).detach(),
        )
        self.active_sequences[batch_idx] = state
        return state, logits

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self.page_table.erase(state.batch_idx)

    # ------------------------------------------------------------------
    # Core forward paths

    def _prefill(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden, logits = self._prefill_fn(inputs_embeds, attn_mask, position_ids)
        return hidden, logits

    def _prefill_impl(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prefill_metadata = self._active_prefill_metadata
        use_flashinfer_prefill = prefill_metadata is not None
        batch_idx = self._active_prefill_batch_idx
        if batch_idx is None:
            raise RuntimeError("Prefill batch index missing during warmup")
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx, positions=position_ids
        )

        hidden = text_decoder(
            inputs_embeds,
            self.model.text,
            attn_mask,
            position_ids,
            self.config.text,
            slot_mapping=slot_mapping,
            mode="prefill",
            flashinfer_prefill_ctx=(
                self._flashinfer_prefill_ctx if use_flashinfer_prefill else None
            ),
            flashinfer_prefill_metadata=prefill_metadata,
            use_flashinfer_prefill=use_flashinfer_prefill,
            text_lora=self._lora.text if self._lora else None,
        )
        logits = lm_head(hidden, self.model.text)
        return hidden, logits

    def decode(self, state: SequenceState, token_id: Tensor | Token) -> None:
        self.decode_batch([state], token_id)
        state.advance()

    def decode_batch(
        self,
        states: Sequence[SequenceState],
        token_inputs: Tensor | Sequence[Token],
    ) -> Tensor:
        if not states:
            raise ValueError("states must not be empty")

        batch_size = len(states)
        self._batch_idx.np[:batch_size] = [state.batch_idx for state in states]
        self._input_pos.np[:batch_size] = [state.length for state in states]
        batch_idx = self._batch_idx.copy_to_gpu(batch_size)
        input_pos = self._input_pos.copy_to_gpu(batch_size)

        tokens_tensor: Optional[Tensor] = None
        token_seq: Optional[Sequence[Token]] = None

        if isinstance(token_inputs, Tensor):
            tokens_tensor = token_inputs
            if tokens_tensor.ndim > 1:
                tokens_tensor = tokens_tensor.view(-1)
            if tokens_tensor.shape[0] != len(states):
                raise ValueError(
                    "token_ids and states must have matching batch dimensions"
                )
            if tokens_tensor.device.type != "cpu":
                tokens_tensor = tokens_tensor.to(device="cpu")
            tokens_tensor = tokens_tensor.to(dtype=torch.long)
            host_tokens = self._tokens.cpu[:batch_size]
            host_tokens.copy_(tokens_tensor)
            tokens_tensor = self._tokens.copy_to_gpu(batch_size)
        else:
            token_seq = list(token_inputs)
            if len(token_seq) != len(states):
                raise ValueError(
                    "token list and states must have matching batch dimensions"
                )

        batch_size = batch_idx.shape[0]
        result: RuntimeDecodeResult
        if (
            tokens_tensor is not None
            and self._use_cuda_graphs
            and batch_size > 0
        ):
            graph_batch_size = self._select_graph_batch_size(batch_size)
            if graph_batch_size is None:
                raise RuntimeError(
                    f"Requested batch size {batch_size} exceeds captured graph capacity "
                    f"{self._graph_batch_sizes[-1] if self._graph_batch_sizes else 0}"
                )
            if not self._cuda_graphs:
                raise RuntimeError(
                    "CUDA graphs requested but none were captured; initialization likely failed."
                )
            if graph_batch_size not in self._cuda_graphs:
                raise RuntimeError(
                    f"No CUDA graph captured for batch size {graph_batch_size}"
                )
            result = self._decode_with_graph(
                batch_idx, tokens_tensor, input_pos, graph_batch_size
            )
        else:
            token_arg: Tensor | Sequence[Token]
            token_arg = tokens_tensor if tokens_tensor is not None else token_seq  # type: ignore[assignment]
            result = self._decode_step(batch_idx, token_arg, input_pos)

        hidden_last = result.hidden[:, -1, :]
        for state, vec in zip(states, hidden_last):
            state.last_hidden = vec.detach()

        return result.logits

    def _build_flashinfer_metadata(
        self,
        batch_idx: Tensor,
        input_pos: Tensor,
        *,
        full_batch_size: Optional[int] = None,
        use_graph: bool = False,
    ) -> FlashInferBatchMetadata:
        if batch_idx.ndim != 1:
            raise ValueError("batch_idx must be 1D")

        batch_size = batch_idx.shape[0]
        target_batch = full_batch_size or batch_size
        buffers = self._flashinfer_ctx.acquire_plan_buffers(
            target_batch, use_graph=use_graph
        )

        seq_lens_np = buffers.seq_lens_np[:target_batch]
        seq_lens_np.fill(0)
        if batch_size:
            seq_lens_cpu_tensor = buffers.seq_lens_cpu[:batch_size]
            seq_lens_cpu_tensor.copy_(input_pos[:batch_size])
            seq_lens_np[:batch_size] += 1

        num_pages_np = buffers.num_pages_np[:target_batch]
        if target_batch:
            np.add(seq_lens_np, self.page_size - 1, out=num_pages_np)
            np.floor_divide(num_pages_np, self.page_size, out=num_pages_np)
        else:
            num_pages_np.fill(0)

        kv_indptr_np = buffers.kv_indptr_np[: target_batch + 1]
        kv_indptr_np.fill(0)
        if target_batch:
            kv_indptr_np[1:] = np.cumsum(num_pages_np, dtype=np.int32)

        kv_last_page_len_np = buffers.kv_last_page_len_np[:target_batch]
        if target_batch:
            np.subtract(seq_lens_np[:target_batch], 1, out=kv_last_page_len_np)
            np.mod(kv_last_page_len_np, self.page_size, out=kv_last_page_len_np)
            kv_last_page_len_np += 1
            kv_last_page_len_np[num_pages_np == 0] = 0
        else:
            kv_last_page_len_np.fill(0)

        batch_count = target_batch

        kv_indptr = buffers._kv_indptr.copy_to_gpu(target_batch + 1)
        kv_last_page_len = buffers._kv_last_page_len.copy_to_gpu(target_batch)

        if buffers.graph_state is not None or target_batch != batch_size:
            batch_buf = buffers.batch_indices[:target_batch]
            batch_buf.zero_()
            if batch_size:
                batch_buf[:batch_size].copy_(batch_idx)
            expanded_batch = batch_buf
        else:
            expanded_batch = batch_idx

        total_pages = int(buffers.kv_indptr_cpu[target_batch].item())
        if total_pages > buffers.page_capacity:
            raise RuntimeError(
                f"FlashInfer plan buffer overflow: need {total_pages} pages but capacity is {buffers.page_capacity}"
            )

        kv_indices = buffers.kv_indices[:total_pages]
        if total_pages > 0:
            self.page_table.populate_flashinfer_kv_indices(
                batch_idx=expanded_batch,
                kv_indptr=kv_indptr,
                out_kv_indices=kv_indices,
            )

        pages_filled = total_pages
        buffers.pages_filled = pages_filled
        if buffers.graph_state is not None:
            buffers.graph_state.pages_filled = pages_filled
        return FlashInferBatchMetadata(
            batch_size=batch_count,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            graph_state=buffers.graph_state if use_graph else None,
        )

    def _build_flashinfer_prefill_metadata(
        self,
        batch_idx: Tensor,
        query_lens: Tensor,
    ) -> FlashInferPrefillBatchMetadata:
        if batch_idx.ndim != 1:
            raise ValueError("batch_idx must be 1D for FlashInfer prefill metadata")
        if query_lens.ndim != 1:
            raise ValueError("query_lens must be 1D for FlashInfer prefill metadata")

        seq_lens = query_lens.to(dtype=torch.int32, device=self.device)
        kv_indptr, kv_indices, kv_last_page_len = (
            self.page_table.build_flashinfer_kv_metadata(batch_idx, seq_lens)
        )
        qo_indptr = torch.zeros(
            batch_idx.shape[0] + 1, dtype=torch.int32, device=self.device
        )
        if query_lens.numel():
            qo_indptr[1:] = torch.cumsum(
                query_lens.to(dtype=torch.int32, device=self.device), dim=0
            )

        return FlashInferPrefillBatchMetadata(
            batch_size=batch_idx.shape[0],
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
        )

    def _decode_step(
        self,
        batch_idx: Tensor,
        token_inputs: Tensor | Sequence[Token],
        input_pos: Tensor,
        *,
        use_graph: bool = False,
        full_batch_size: Optional[int] = None,
        flashinfer_metadata: Optional[FlashInferBatchMetadata] = None,
        skip_plan: bool = False,
    ) -> RuntimeDecodeResult:
        self._batch_binding.tensor = batch_idx

        if isinstance(token_inputs, Tensor):
            embeds = text_encoder(token_inputs.view(-1, 1), self.model.text)
        else:
            embeds = self._embed_token_batch(token_inputs)
        position_ids = input_pos.to(dtype=torch.long).view(-1, 1)
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx.view(-1, 1), positions=position_ids
        )

        metadata = flashinfer_metadata
        if metadata is None:
            metadata = self._build_flashinfer_metadata(
                batch_idx,
                input_pos,
                full_batch_size=full_batch_size,
                use_graph=use_graph,
            )

        if not skip_plan:
            self._flashinfer_ctx.plan(
                metadata,
                num_q_heads=self.config.text.n_heads,
                num_kv_heads=self.config.text.n_kv_heads,
                head_dim=self.head_dim,
                use_graph=use_graph,
            )
        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask=None,
            position_ids=position_ids,
            config=self.config.text,
            flashinfer_ctx=self._flashinfer_ctx,
            flashinfer_metadata=metadata,
            use_flashinfer=True,
            use_graph=use_graph,
            mode="decode",
            slot_mapping=slot_mapping,
            text_lora=self._lora.text if self._lora else None,
        )
        logits = lm_head(hidden, self.model.text)
        return RuntimeDecodeResult(logits=logits, hidden=hidden)

    def _decode_with_graph(
        self,
        batch_idx: Tensor,
        tokens: Tensor,
        input_pos: Tensor,
        graph_batch_size: int,
    ) -> RuntimeDecodeResult:
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
        metadata = self._build_flashinfer_metadata(
            workspace.batch_idx_buffer[:graph_batch_size],
            workspace.position_buffer[:graph_batch_size],
            full_batch_size=graph_batch_size,
            use_graph=True,
        )
        self._batch_binding.tensor = workspace.batch_idx_buffer[:graph_batch_size]
        self._flashinfer_ctx.plan(
            metadata,
            num_q_heads=self.config.text.n_heads,
            num_kv_heads=self.config.text.n_kv_heads,
            head_dim=self.head_dim,
            use_graph=True,
        )

        graph = self._cuda_graphs[graph_batch_size]
        graph.replay()
        self._batch_binding.tensor = batch_idx
        logits = workspace.output_buffer[:batch_size]
        hidden = workspace.hidden_buffer[:batch_size].unsqueeze(1)
        return RuntimeDecodeResult(logits=logits, hidden=hidden)

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
        hidden_buffer = torch.zeros(
            (max_effective_batch, self.model.text.lm_head.weight.shape[1]),
            device=self.device,
            dtype=self.model.text.lm_head.weight.dtype,
        )
        self._graph_workspace = _GraphWorkspace(
            token_buffer=token_buffer,
            batch_idx_buffer=batch_idx_buffer,
            position_buffer=position_buffer,
            output_buffer=output_buffer,
            hidden_buffer=hidden_buffer,
        )
        self._graph_batch_sizes = self._make_graph_batch_sizes(max_effective_batch)

    def _make_graph_batch_sizes(self, max_batch: int) -> list[int]:
        seeds = [size for size in (1, 2, 4, 8) if size <= max_batch]
        ramps = list(range(16, max_batch + 1, 16))
        sizes = sorted({*seeds, *ramps, max_batch})
        return sizes

    def _capture_decode_graphs(self) -> None:
        with self.graph_capture_lock:
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
                        metadata = self._build_flashinfer_metadata(
                            workspace.batch_idx_buffer[:bs],
                            workspace.position_buffer[:bs],
                            full_batch_size=bs,
                            use_graph=True,
                        )
                        warmup = self._decode_step(
                            workspace.batch_idx_buffer[:bs],
                            workspace.token_buffer[:bs, 0],
                            workspace.position_buffer[:bs],
                            use_graph=True,
                            full_batch_size=bs,
                            flashinfer_metadata=metadata,
                            skip_plan=False,
                        )
                        workspace.output_buffer[:bs].copy_(warmup.logits)
                        workspace.hidden_buffer[:bs].copy_(warmup.hidden[:, 0, :])

                        torch.cuda.synchronize(device=device)

                        with torch.cuda.graph(graph, self._graph_pool):
                            out = self._decode_step(
                                workspace.batch_idx_buffer[:bs],
                                workspace.token_buffer[:bs, 0],
                                workspace.position_buffer[:bs],
                                use_graph=True,
                                full_batch_size=bs,
                                flashinfer_metadata=metadata,
                                skip_plan=True,
                            )
                            workspace.output_buffer[:bs].copy_(out.logits)
                            workspace.hidden_buffer[:bs].copy_(out.hidden[:, 0, :])

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


__all__ = ["MoondreamRuntime", "SequenceState", "DEFAULT_MAX_TOKENS"]
