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
from .lora_workspace import AdapterSlotManager, TextLoRAWorkspace
from .image_crops import OverlapCropOutput
from .region import (
    build_region_module,
    build_spatial_decode_tables,
    encode_coordinate,
    encode_size,
)
from ..seg_refiner import SegmentRefiner
from .decode_slot import DecodeSlot, create_decode_slot


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
    lora_slot: int = 0  # 0 = no LoRA, >0 = slot in TextLoRAWorkspace

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
        if k_val.shape != v_val.shape:
            raise ValueError("k_val and v_val must match shape")

        input_pos = torch.atleast_2d(pos_ids).to(
            dtype=torch.int32, device=k_val.device
        )
        if input_pos.shape[0] != 1 and input_pos.shape[0] != k_val.shape[0]:
            raise ValueError(
                f"Unsupported position shape {pos_ids.shape} for batch size {k_val.shape[0]}"
            )

        seq_len = input_pos.shape[1]
        if k_val.shape[1] != seq_len:
            raise ValueError(
                f"KV sequence length {k_val.shape[1]} does not match position tensor {input_pos.shape}"
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

    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        max_lora_rank: int | None = None,
    ) -> None:
        self._cfg = cfg
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
        self.spatial_tables = build_spatial_decode_tables(self.region)
        self.image_prefix_length = self.model.vision.pos_emb.shape[1]
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
            and self.page_size == 1
            and hasattr(torch, "float8_e4m3fn")
        ):
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            if (
                self._kv_layer_k_scales is not None
                and self._kv_layer_v_scales is not None
                and self.page_size != 1
            ):
                warnings.warn(
                    "KV scales found in checkpoint but FP8 KV cache currently requires page_size==1; "
                    "falling back to standard KV cache.",
                    stacklevel=2,
                )
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

        # Shared streams for pipelined decoding. All decode forwards serialize on
        # the compute stream (preserves KV ordering and _pending_* dependencies).
        # D2H copies share the copy stream for simpler ordering guarantees.
        self._decode_compute_stream = torch.cuda.Stream(device=self.device)
        self._decode_lora_stream = torch.cuda.Stream(device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)

        self._active_prefill_batch_idx: Optional[Tensor] = None

        # CUDA graph batch sizes for decode (same for all slots).
        self._graph_batch_sizes: list[int] = []
        self._graph_pool: object | None = None
        self._batch_binding: _BatchBinding = _BatchBinding()
        coord_dtype = self.region.coord_features.dtype
        size_dtype = self.region.size_features.dtype
        self._prefill_batch_idx = CpuGpuBuffer(
            1,
            dtype=torch.int64,
            device=self.device,
            pin_memory=True,
            with_numpy=False,
        )

        for cache in self.layer_caches:
            cache.attach_batch_binding(self._batch_binding)

        self._prefill_fn = self._prefill_impl

        self.seg_refiner = SegmentRefiner(self.model.vision, self.config.vision, self.device)

        # Multi-slot LoRA workspace and slot manager.
        # Slot 0 represents "no LoRA". Active adapters are loaded into slots 1+.
        # With max_slots = max_batch_size, we have max_batch_size - 1 usable adapter
        # slots, which matches effective batch size (page 0 is reserved in page table).
        self._lora_workspace: TextLoRAWorkspace | None = None
        self._slot_manager: AdapterSlotManager | None = None
        self._max_lora_rank: int | None = max_lora_rank
        if max_lora_rank is not None:
            max_slots = cfg.max_batch_size

            # Validate MoE super-expert limit.
            # vLLM's moe_align_block_size kernel requires num_experts < 1024 due to
            # CUB BlockScan using 1024 threads. With sentinel-based slot 0 filtering,
            # max_super_experts = (max_slots - 1) * num_experts.
            moe_cfg = self.config.text.moe
            if moe_cfg is not None:
                max_super_experts = (max_slots - 1) * moe_cfg.num_experts
                if max_super_experts >= 1024:
                    max_allowed = 1 + (1023 // moe_cfg.num_experts)
                    raise ValueError(
                        f"max_batch_size ({cfg.max_batch_size}) is too large for MoE LoRA. "
                        f"With {moe_cfg.num_experts} experts, max_super_experts = "
                        f"(max_batch_size - 1) * {moe_cfg.num_experts} = {max_super_experts}, "
                        f"which exceeds vLLM's moe_align_block_size limit of 1024. "
                        f"Maximum allowed max_batch_size: {max_allowed}"
                    )

            self._lora_workspace = TextLoRAWorkspace(
                text_config=self.config.text,
                max_slots=max_slots,
                max_rank=max_lora_rank,
                device=self.device,
                dtype=self.dtype,
                lora_stream=self._decode_lora_stream,
            )
            self._slot_manager = AdapterSlotManager(max_slots)

        # Create two ping-pong decode slots for pipelined decoding.
        # Each slot has its own staging buffers, FA3 paged-KV metadata buffers,
        # and RenderBuffer, but they share the decode compute stream and copy stream.
        vocab_size = self.model.text.lm_head.weight.shape[0]
        hidden_dim = self.model.text.lm_head.weight.shape[1]
        self._decode_slots: list[DecodeSlot] = [
            create_decode_slot(
                slot_id=slot_id,
                device=self.device,
                dtype=self.dtype,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_length,
                page_size=self.page_size,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                coord_dtype=coord_dtype,
                size_dtype=size_dtype,
                compute_stream=self._decode_compute_stream,
                copy_stream=self._copy_stream,
            )
            for slot_id in range(2)
        ]

        if self._use_cuda_graphs:
            self._ensure_cuda_graphs_ready()

    # ------------------------------------------------------------------
    # Capacity helpers

    def can_reserve(self, total_length: int) -> bool:
        """Return True if a request of ``total_length`` tokens can be admitted."""

        return self.page_table.can_reserve(total_length)

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """Shared copy stream for D2H transfers."""
        return self._copy_stream

    @property
    def decode_compute_stream(self) -> torch.cuda.Stream:
        """Shared decode compute stream for all decode forwards."""
        return self._decode_compute_stream

    @property
    def decode_slots(self) -> list[DecodeSlot]:
        """Two ping-pong decode slots for pipelined decoding."""
        return self._decode_slots

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

    def _embed_packed_token_batch(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
    ) -> Tensor:
        """Embed pending decode tokens from packed id/value tensors.

        `coord_values`/`size_values` are meaningful only for rows whose token id is
        the corresponding special token (coord_id/size_id); other rows should be
        zero-filled.
        """

        if token_ids.ndim != 1:
            token_ids = token_ids.view(-1)

        batch = int(token_ids.shape[0])
        if batch == 0:
            dim = self.bos_embed.shape[-1]
            return torch.empty((0, 1, dim), device=self.device, dtype=self.dtype)

        ids = token_ids.to(dtype=torch.long)
        text_emb = text_encoder(ids.view(-1, 1), self.model.text)

        coord_emb = encode_coordinate(coord_values, self.region).unsqueeze(1)
        size_emb = encode_size(size_values, self.region).unsqueeze(1)

        coord_id = self.config.tokenizer.coord_id
        size_id = self.config.tokenizer.size_id
        coord_mask = (ids == coord_id).view(-1, 1, 1)
        size_mask = (ids == size_id).view(-1, 1, 1)

        out = torch.where(coord_mask, coord_emb, text_emb)
        out = torch.where(size_mask, size_emb, out)
        return out

    def encode_image(
        self,
        image: Optional[pyvips.Image | np.ndarray],
        *,
        overlap: Optional[OverlapCropOutput] = None,
    ) -> Tensor:
        return encode_image(
            image,
            self.model.vision,
            self.config.vision,
            device=self.device,
            dtype=self.dtype,
            overlap=overlap,
        )

    def start_sequence(
        self,
        prompt_tokens: Tensor | Sequence[Token],
        *,
        image: Optional[pyvips.Image | np.ndarray] = None,
        image_crops: Optional[OverlapCropOutput] = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
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
            image_embed = self.encode_image(image, overlap=image_crops).unsqueeze(0)
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

        self._active_prefill_batch_idx = batch_tensor
        try:
            hidden, logits = self._prefill(
                inputs_embeds,
                attention_mask,
                position_ids,
                lora_slot,
                use_prefix_attn=bool(image_length),
            )
        finally:
            self._active_prefill_batch_idx = None
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=image_length,
            last_hidden=hidden[:, -1, :].squeeze(0).detach(),
            lora_slot=lora_slot,
        )
        self.active_sequences[batch_idx] = state
        return state, logits

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self.page_table.erase(state.batch_idx)
        # Release the adapter slot (no-op if lora_slot == 0)
        self.release_adapter_slot(state.lora_slot)

    # ------------------------------------------------------------------
    # Core forward paths

    def _prefill(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
        lora_slot: int = 0,
        *,
        use_prefix_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        hidden, logits = self._prefill_fn(
            inputs_embeds,
            attn_mask,
            position_ids,
            lora_slot,
            use_prefix_attn=use_prefix_attn,
        )
        return hidden, logits

    def _prefill_impl(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
        lora_slot: int = 0,
        *,
        use_prefix_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        batch_idx = self._active_prefill_batch_idx
        if batch_idx is None:
            raise RuntimeError("Prefill batch index missing during warmup")
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx, positions=position_ids
        )

        # Build FA3 paged attention metadata for prefill
        seqlen = position_ids.max().item() + 1
        fa3_page_table = self.page_table.page_table[batch_idx : batch_idx + 1]
        fa3_seqused_k = torch.tensor([seqlen], dtype=torch.int32, device=self.device)

        # For no-adapter prefill, skip LoRA entirely to avoid redundant work
        if lora_slot == 0:
            lora_workspace = None
            lora_slot_ids = None
        else:
            lora_workspace = self._lora_workspace
            lora_slot_ids = torch.tensor([lora_slot], dtype=torch.int32, device=self.device)
            # Enable single-LoRA mode for prefill (optimized kernel path)
            if lora_workspace is not None:
                lora_workspace.set_prefill_mode(lora_slot)

        hidden = text_decoder(
            inputs_embeds,
            self.model.text,
            attn_mask,
            position_ids,
            self.config.text,
            slot_mapping=slot_mapping,
            mode="prefill",
            use_prefix_attn=use_prefix_attn,
            page_table=fa3_page_table,
            fa3_seqused_k=fa3_seqused_k,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
        )

        # Reset to decode mode after prefill (batched kernel path)
        if lora_workspace is not None:
            lora_workspace.set_decode_mode()

        logits = lm_head(hidden, self.model.text)
        return hidden, logits

    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        """Run batched decode forward pass using per-slot resources.

        IMPORTANT: Caller must ensure:
        - Already on slot.compute_stream (via `with torch.cuda.stream()`)
        - Inputs staged in slot buffers (decode_token_ids, decode_coord_values, etc.)
        - Metadata copied to GPU (batch_idx, input_pos, lora_slot_ids)

        This method runs the forward pass and writes results to slot.logits
        and slot.hidden_last.
        """
        self._decode_with_slot(slot, batch_size)

    def _decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        """Unified decode using slot buffers. Writes results to slot.logits/hidden_last.

        This method provides identical preparation for both graph and non-graph paths:
        1. Clear padding region (for graph batch size alignment)
        2. Build FA3 paged-KV metadata buffers
        3. Execute: either graph.replay() or eager forward

        The only difference between paths is step 3. This ensures debugging with
        graphs disabled produces identical behavior to the graph path.

        Args:
            slot: DecodeSlot with inputs already staged in its buffers.
            batch_size: Actual number of sequences (before padding).
        """
        use_graph = self._use_cuda_graphs and slot.cuda_graphs is not None

        # Determine padded batch size for graph alignment
        if use_graph:
            graph_batch_size = self._select_graph_batch_size(batch_size)
            if graph_batch_size is None:
                raise RuntimeError(
                    f"Batch size {batch_size} exceeds max graph capacity "
                    f"{self._graph_batch_sizes[-1] if self._graph_batch_sizes else 0}"
                )
            if graph_batch_size not in slot.cuda_graphs:
                raise RuntimeError(
                    f"No CUDA graph captured for batch size {graph_batch_size}"
                )
        else:
            graph_batch_size = batch_size

        # Clear padding region for deterministic graph behavior
        if graph_batch_size > batch_size:
            slot.decode_token_ids[batch_size:graph_batch_size].zero_()
            slot.decode_coord_values[batch_size:graph_batch_size].zero_()
            slot.decode_size_values[batch_size:graph_batch_size].zero_()
            slot.meta.batch_idx.gpu[batch_size:graph_batch_size].zero_()
            slot.meta.input_pos.gpu[batch_size:graph_batch_size].zero_()
            slot.meta.lora_slot_ids.gpu[batch_size:graph_batch_size].zero_()

        # Build FA3 per-step metadata buffers (identical for both paths)
        # - page_table rows: [B, num_pages]
        # - seqused_k: [B] (KV length including the current token after update)
        batch_idx = slot.meta.batch_idx.gpu[:graph_batch_size]
        self.page_table.populate_fa3_decode_metadata(
            batch_idx=batch_idx,
            input_pos=slot.meta.input_pos.gpu[:graph_batch_size],
            out_page_table=slot.fa3_page_table[:graph_batch_size],
            out_seqused_k=slot.fa3_seqused_k[:graph_batch_size],
        )

        # Set batch binding (identical for both paths)
        self._batch_binding.tensor = slot.meta.batch_idx.gpu[:graph_batch_size]

        # Execute (only difference between paths)
        if use_graph:
            slot.cuda_graphs[graph_batch_size].replay()
        else:
            self._run_decode_forward(slot, graph_batch_size)

        # Restore batch binding to actual batch size
        self._batch_binding.tensor = slot.meta.batch_idx.gpu[:batch_size]

    def _run_decode_forward(
        self,
        slot: DecodeSlot,
        batch_size: int,
    ) -> None:
        """Run decode forward pass and write results to slot output buffers.

        This is the core forward computation, used by both eager decode and
        CUDA graph replay.

        Args:
            slot: DecodeSlot with inputs in its buffers.
            batch_size: Batch size (may be padded for graph capture).
        """
        embeds = self._embed_packed_token_batch(
            slot.decode_token_ids[:batch_size],
            slot.decode_coord_values[:batch_size],
            slot.decode_size_values[:batch_size],
        )
        position_ids = slot.meta.input_pos.gpu[:batch_size].to(torch.long).view(-1, 1)
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=slot.meta.batch_idx.gpu[:batch_size].view(-1, 1),
            positions=position_ids,
        )
        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask=None,
            position_ids=position_ids,
            config=self.config.text,
            mode="decode",
            slot_mapping=slot_mapping,
            page_table=slot.fa3_page_table[:batch_size],
            fa3_seqused_k=slot.fa3_seqused_k[:batch_size],
            lora_workspace=self._lora_workspace,
            lora_slot_ids=slot.meta.lora_slot_ids.gpu[:batch_size],
        )
        logits = lm_head(hidden, self.model.text)

        # Write to slot output buffers (stable addresses for graph capture)
        slot.logits[:batch_size].copy_(logits)
        slot.hidden_last[:batch_size].copy_(hidden[:, 0, :])

    def acquire_adapter_slot(self, adapter_id: str, adapter: LoRA) -> int:
        """Acquire a slot for an adapter, loading weights if necessary.

        Uses the slot manager to either reuse an existing slot (if the adapter is
        already resident) or allocate a new one. If newly allocated, copies the
        adapter weights into the workspace.

        Args:
            adapter_id: Identifier for the adapter (from settings.adapter).
            adapter: The LoRA adapter to load (must be on same device/dtype).

        Returns:
            The slot number assigned to this adapter.

        Raises:
            NotImplementedError: If no adapter provider is configured, vision LoRA
                is provided, or adapter is on wrong device/dtype.
            ValueError: If adapter rank exceeds max_lora_rank.
            RuntimeError: If no free slots are available.
        """
        if self._lora_workspace is None or self._slot_manager is None:
            raise NotImplementedError(
                "Adapter provider is not configured for this runtime."
            )
        if adapter.vision is not None:
            raise NotImplementedError("Vision LoRA is not supported.")

        if adapter.text.rank > self._max_lora_rank:
            raise ValueError(
                f"Adapter rank ({adapter.text.rank}) exceeds max_lora_rank ({self._max_lora_rank})."
            )

        # Require CUDA tensors and exact dtype/device matches.
        try:
            sample_param = next(adapter.text.parameters())
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("Adapter contains no parameters") from exc
        if sample_param.device != self.device:
            raise NotImplementedError(
                f"Adapter must be on device {self.device}; received {sample_param.device}."
            )
        if sample_param.dtype != self.dtype:
            raise NotImplementedError(
                f"Adapter must have dtype {self.dtype}; received {sample_param.dtype}."
            )

        # Acquire slot (reuse if already resident, allocate if new)
        slot, is_new = self._slot_manager.acquire(adapter_id)

        if is_new:
            try:
                self._lora_workspace.load_slot_(slot, adapter)
            except Exception:
                # Rollback on load failure
                self._slot_manager.release_on_error(slot)
                raise

        return slot

    def release_adapter_slot(self, slot: int) -> None:
        """Release a reference to an adapter slot.

        Decrements the slot's refcount. When the last reference is released,
        the slot is returned to the free pool.

        Args:
            slot: The slot to release (0 is a no-op).
        """
        if slot == 0:
            return  # No LoRA, nothing to release

        if self._slot_manager is None:
            return

        self._slot_manager.release(slot)

    def rebuild_cuda_graphs(self) -> None:
        """Reset and recapture CUDA graphs used for decode.

        This is intended for workflows that mutate runtime-owned tensors
        (e.g. weight tying or hot-swapping checkpoints) where a previously
        captured CUDA graph might replay with stale tensor pointers.

        Callers must ensure no CUDA work is in flight (e.g. pause the engine)
        before invoking this method.
        """

        if not self._use_cuda_graphs:
            return

        with self.graph_capture_lock:
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            # Clear per-slot graph state
            for slot in self._decode_slots:
                slot.cuda_graphs = None

            self._graph_batch_sizes = []
            self._graph_pool = None

            self._ensure_cuda_graphs_ready()

    def _ensure_cuda_graphs_ready(self) -> None:
        """Capture CUDA graphs for all slots using slot buffers directly."""
        if not self._use_cuda_graphs:
            return
        # Check if graphs already captured (first slot has graphs)
        if self._decode_slots and self._decode_slots[0].cuda_graphs is not None:
            return

        # Initialize graph batch sizes once
        max_effective_batch = max(1, self.max_batch_size - 1)
        self._graph_batch_sizes = self._make_graph_batch_sizes(max_effective_batch)

        # Capture graphs for each slot using its own buffers
        for slot in self._decode_slots:
            slot.cuda_graphs = self._capture_decode_graphs_for_slot(slot)

    def _make_graph_batch_sizes(self, max_batch: int) -> list[int]:
        seeds = [size for size in (1, 2, 4, 8) if size <= max_batch]
        ramps = list(range(16, max_batch + 1, 16))
        sizes = sorted({*seeds, *ramps, max_batch})
        return sizes

    def _capture_decode_graphs_for_slot(
        self,
        slot: DecodeSlot,
    ) -> dict[int, torch.cuda.CUDAGraph]:
        """Capture CUDA graphs for a decode slot using its own buffers.

        Graphs are captured using the slot's staging buffers (decode_token_ids,
        meta.batch_idx.gpu, etc.) and output buffers (logits, hidden_last).
        This ensures graph replay reads from and writes to the same addresses
        that the non-graph path uses, making behavior identical.
        """
        cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        with self.graph_capture_lock:
            max_batch = slot.decode_token_ids.shape[0]
            if max_batch == 0:
                return cuda_graphs

            device = self.device

            # Graph capture must happen on the same stream we use for decode replay.
            # Otherwise, replayed kernels may read stale metadata (page tables / seqused_k)
            # written on a different stream, producing incorrect results.
            with torch.cuda.stream(slot.compute_stream):
                # Zero all slot buffers for capture.
                # Use batch index 0 for all entries - row 0 in the page table is
                # pre-initialized and provides valid memory access patterns.
                slot.decode_token_ids.zero_()
                slot.decode_coord_values.zero_()
                slot.decode_size_values.zero_()
                slot.meta.batch_idx.gpu.zero_()
                slot.meta.input_pos.gpu.zero_()
                slot.meta.input_pos.cpu.zero_()
                slot.meta.lora_slot_ids.gpu.zero_()
                slot.fa3_page_table.zero_()
                slot.fa3_seqused_k.zero_()

                try:
                    torch.cuda.synchronize(device=device)
                    for bs in reversed(self._graph_batch_sizes):
                        graph = torch.cuda.CUDAGraph()
                        with torch.inference_mode():
                            # Build FA3 per-step metadata buffers
                            batch_idx = slot.meta.batch_idx.gpu[:bs]
                            self.page_table.populate_fa3_decode_metadata(
                                batch_idx=batch_idx,
                                input_pos=slot.meta.input_pos.gpu[:bs],
                                out_page_table=slot.fa3_page_table[:bs],
                                out_seqused_k=slot.fa3_seqused_k[:bs],
                            )

                            # Set batch binding
                            self._batch_binding.tensor = slot.meta.batch_idx.gpu[:bs]

                            # Warmup run (not captured)
                            self._run_decode_forward(slot, bs)
                            torch.cuda.synchronize(device=device)

                            # Capture the graph
                            with torch.cuda.graph(graph, self._graph_pool):
                                self._run_decode_forward(slot, bs)

                        if self._graph_pool is None:
                            self._graph_pool = graph.pool()
                        cuda_graphs[bs] = graph
                        torch.cuda.synchronize(device=device)
                finally:
                    # Clear slot buffers after capture
                    slot.decode_token_ids.zero_()
                    slot.meta.batch_idx.gpu.zero_()
                    slot.meta.input_pos.gpu.zero_()
                    slot.meta.lora_slot_ids.gpu.zero_()
                    slot.fa3_page_table.zero_()
                    slot.fa3_seqused_k.zero_()

        return cuda_graphs

    def _select_graph_batch_size(self, batch_size: int) -> int | None:
        for size in self._graph_batch_sizes:
            if size >= batch_size:
                return size
        return None


__all__ = ["MoondreamRuntime", "SequenceState", "DEFAULT_MAX_TOKENS"]
