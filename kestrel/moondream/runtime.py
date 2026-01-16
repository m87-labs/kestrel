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


from tokenizers import Tokenizer

from kestrel.config import RuntimeConfig
from kestrel.kv_cache import PageTable, PagedKVCache
from kestrel.prefix_cache import (
    CacheNamespace,
    CacheToken,
    ImageToken,
    MatchResult,
    RadixPrefixCache,
    TreeNode,
)

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

    def cache_key(self) -> tuple:
        """Cache key: (0, token_id) - 0 discriminates from other token types."""
        return (0, self.token_id)

    def kv_length(self) -> int:
        """Text tokens occupy exactly 1 KV position."""
        return 1


class CoordToken(NamedTuple):
    """Normalized positional token emitted or consumed by the region model."""

    pos: float

    def cache_key(self) -> tuple:
        """Cache key: (1, pos) - 1 discriminates from text tokens."""
        return (1, self.pos)

    def kv_length(self) -> int:
        """Coord tokens occupy exactly 1 KV position."""
        return 1


class SizeToken(NamedTuple):
    """Normalized width/height token emitted or consumed by the region model."""

    width: float
    height: float

    def cache_key(self) -> tuple:
        """Cache key: (2, width, height) - 2 discriminates from coord tokens."""
        return (2, self.width, self.height)

    def kv_length(self) -> int:
        """Size tokens occupy exactly 1 KV position."""
        return 1


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
    # DEPRECATED: Use image_regions instead. Kept for backward compatibility with
    # scheduler code. Will be removed once scheduler is migrated to image_regions.
    image_length: int = 0
    last_hidden: Tensor | None = None
    lora_slot: int = 0  # 0 = no LoRA, >0 = slot in TextLoRAWorkspace

    # Prefix cache fields
    cache_tokens: list[CacheToken] | None = None
    cache_lock_node: TreeNode | None = None
    cache_owned_page_count: int = 0  # Pages belonging to cache (not freed on release)
    reused_page_count: int = 0  # Pages reused from cache hit (for metrics)
    # List of (start, end) KV positions for bidirectional attention (image regions).
    # For single-image: [(1, 1+image_kv_length)]. Empty for text-only.
    # This will replace image_length once multi-image is supported.
    image_regions: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.prompt_length is None:
            self.prompt_length = self.length
        # Validate consistency between image_length and image_regions
        if self.image_regions:
            computed_length = sum(end - start for start, end in self.image_regions)
            assert self.image_length == computed_length, (
                f"image_length ({self.image_length}) inconsistent with "
                f"image_regions ({self.image_regions}, computed={computed_length})"
            )

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
class _CacheLookupResult:
    """Result of prefix cache lookup in start_sequence."""

    match: MatchResult | None
    skip_positions: int
    temp_lock_node: TreeNode | None
    can_reuse: bool
    namespace: CacheNamespace | None


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
        if cfg.max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be at least 1; batch_idx 0 is reserved internally."
            )
        # max_batch_size is the effective user-facing batch capacity.
        # We reserve batch_idx 0 for internal bookkeeping, so allocate +1 slot.
        self.max_batch_size = cfg.max_batch_size
        self.max_batch_slots = cfg.max_batch_size + 1
        n_pages = self.max_seq_length // self.page_size

        # Create prefix cache if enabled
        self.prefix_cache: RadixPrefixCache | None = None
        if cfg.enable_prefix_cache:
            self.prefix_cache = RadixPrefixCache()

        # Primary stream for all GPU operations. Created early because PageTable
        # needs it for H2D copies that must synchronize with graph replay.
        self._primary_stream = torch.cuda.Stream(device=self.device)

        self.page_table = PageTable(
            n_pages=n_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=self.prefix_cache,
            h2d_stream=self._primary_stream,
        )

        self.model = MoondreamModel(
            self.config,
            dtype=self.dtype,
            device=self.device,
            setup_caches=False,
        ).eval()
        self.region = build_region_module(self.config.region, self.dtype).to(self.device)
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
            str(cfg.model_path),
            self.model,
            tensor_hook=_capture_kv_scale,
            region=self.region,
        )

        # Build spatial decode tables after region weight loading; otherwise the
        # concatenated weights/biases depend on random init/seed rather than the
        # checkpoint.
        self.spatial_tables = build_spatial_decode_tables(self.region)

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

        tokenizer_path = "moondream/starmie-v1"
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

        # Additional streams: LoRA operations and D2H copies.
        # (Primary stream was created earlier for PageTable H2D sync.)
        self._lora_stream = torch.cuda.Stream(device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)

        self._active_prefill_batch_idx: Optional[Tensor] = None

        # CUDA graph batch sizes for decode (same for all slots).
        self._graph_batch_sizes: list[int] = []
        self._batch_binding: _BatchBinding = _BatchBinding()
        coord_dtype = self.region.coord_features.dtype
        size_dtype = self.region.size_features.dtype
        self._prefill_batch_idx = torch.empty(
            (1,), dtype=torch.int64, device=self.device
        )

        for cache in self.layer_caches:
            cache.attach_batch_binding(self._batch_binding)

        self._prefill_fn = self._prefill_impl

        self.seg_refiner = SegmentRefiner(self.model.vision, self.config.vision, self.device)

        # Multi-slot LoRA workspace and slot manager.
        # Slot 0 represents "no LoRA". Active adapters are loaded into slots 1+.
        # With max_slots = max_batch_slots, we have max_batch_size usable adapter
        # slots (slot 0 reserved), matching effective batch size.
        self._lora_workspace: TextLoRAWorkspace | None = None
        self._slot_manager: AdapterSlotManager | None = None
        self._max_lora_rank: int | None = max_lora_rank
        if max_lora_rank is not None:
            max_slots = self.max_batch_slots


            self._lora_workspace = TextLoRAWorkspace(
                text_config=self.config.text,
                max_slots=max_slots,
                max_rank=max_lora_rank,
                device=self.device,
                dtype=self.dtype,
                lora_stream=self._lora_stream,
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
                max_batch_slots=self.max_batch_slots,
                max_seq_len=self.max_seq_length,
                page_size=self.page_size,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                coord_dtype=coord_dtype,
                size_dtype=size_dtype,
                compute_stream=self._primary_stream,
                copy_stream=self._copy_stream,
            )
            for slot_id in range(2)
        ]

        # Pre-allocate workspaces unconditionally (needed for both graph and non-graph paths)
        self._preallocate_workspaces()

        if self._use_cuda_graphs:
            self._ensure_cuda_graphs_ready()

    # ------------------------------------------------------------------
    # Capacity helpers

    def can_reserve(self, total_length: int) -> bool:
        """Return True if a request of ``total_length`` tokens can be admitted."""

        return self.page_table.can_reserve_with_eviction(total_length)

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """Shared copy stream for D2H transfers."""
        return self._copy_stream

    @property
    def primary_stream(self) -> torch.cuda.Stream:
        """Primary compute stream for all GPU operations."""
        return self._primary_stream

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

    # ------------------------------------------------------------------
    # start_sequence helpers

    def _normalize_prompt_tokens(
        self, prompt_tokens: Tensor | Sequence[Token]
    ) -> list[Token]:
        """Convert prompt_tokens to a list of Token objects."""
        if isinstance(prompt_tokens, Tensor):
            tokens_view = prompt_tokens.to(device=self.device, dtype=torch.long)
            if tokens_view.ndim != 2:
                raise ValueError(
                    f"prompt_tokens must have shape (1, N); received {tokens_view.shape}"
                )
            return [TextToken(int(tid)) for tid in tokens_view[0].tolist()]
        return list(prompt_tokens)

    def check_prefix_cache(
        self,
        tokens_list: list[Token],
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> bool:
        """Check if an image+tokens combo would hit the prefix cache.

        This is a lightweight check that does not acquire locks or map pages.
        Used for early cache lookup to skip crop computation on cache hits.

        Returns True if the cache would hit and cover the full image prefix.
        """
        if self.prefix_cache is None:
            return False
        if image_hash is None:
            return False

        image_kv_length = self.image_prefix_length
        cache_tokens = self._build_cache_tokens(tokens_list, image_hash, image_kv_length)

        # Build namespace
        image_hash_int = int.from_bytes(image_hash[:16], "big")
        namespace = CacheNamespace(
            lora_id=adapter_id,
            image_hash=image_hash_int,
        )

        match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)

        # Cache hit must cover at least BOS + full image prefix
        min_hit_length = 1 + image_kv_length
        return match.matched_kv_length >= min_hit_length

    def _build_cache_tokens(
        self,
        tokens_list: list[Token],
        image_hash: bytes | None,
        image_kv_length: int,
    ) -> list[CacheToken]:
        """Build cache token sequence: [BOS, ImageToken?, text tokens...]."""
        cache_tokens: list[CacheToken] = []
        if tokens_list:
            cache_tokens.append(tokens_list[0])  # BOS
        if image_hash is not None:
            cache_tokens.append(
                ImageToken(
                    content_hash=int.from_bytes(image_hash[:16], "big"),
                    kv_length_=image_kv_length,
                )
            )
        if len(tokens_list) > 1:
            cache_tokens.extend(tokens_list[1:])
        return cache_tokens

    def _lookup_prefix_cache(
        self,
        cache_tokens: list[CacheToken],
        adapter_id: str | None,
        image_hash: bytes | None,
        image_kv_length: int,
        prompt_len: int,
        batch_idx: int,
    ) -> _CacheLookupResult:
        """Lookup prefix cache, map cached pages, and acquire temp lock."""
        if self.prefix_cache is None or not cache_tokens:
            return _CacheLookupResult(
                match=None,
                skip_positions=0,
                temp_lock_node=None,
                can_reuse=False,
                namespace=None,
            )

        # Build namespace from adapter identity and image hash
        image_hash_int = (
            int.from_bytes(image_hash[:16], "big") if image_hash else None
        )
        namespace = CacheNamespace(
            lora_id=adapter_id,
            image_hash=image_hash_int,
        )
        match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)
        can_reuse = match.matched_kv_length > 0

        # Invariant: In image namespace, any hit must include BOS+image.
        if image_kv_length > 0 and can_reuse:
            assert match.matched_kv_length >= (1 + image_kv_length), (
                f"Invariant violated: image namespace hit ({match.matched_kv_length} KV) "
                f"must include BOS+image ({1 + image_kv_length} KV)"
            )

        skip_positions = 0
        temp_lock_node: TreeNode | None = None

        if can_reuse:
            # Cap skip_positions to ensure at least one suffix KV position
            skip_positions = min(match.matched_kv_length, prompt_len - 1)

            # Map cached pages
            cached_pages = match.matched_pages[:skip_positions]
            self.page_table.map_pages(batch_idx, 0, cached_pages)

            # Lock matched prefix during prefill
            temp_lock_node = match.last_node
            self.prefix_cache.lock(temp_lock_node)

        return _CacheLookupResult(
            match=match,
            skip_positions=skip_positions,
            temp_lock_node=temp_lock_node,
            can_reuse=can_reuse,
            namespace=namespace,
        )

    def _prepare_append_prefill_inputs(
        self,
        tokens_list: list[Token],
        skip_positions: int,
        image_kv_length: int,
        prompt_len: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare inputs for append prefill (cache hit path)."""
        # Derive suffix tokens from skip_positions
        if image_kv_length > 0:
            # KV layout: [BOS(1)] [Image(image_kv_length)] [Text tokens...]
            prefix_kv = 1 + image_kv_length
            text_kv_cached = skip_positions - prefix_kv
            # tokens_list[0] = BOS, tokens_list[1:] = text after image
            suffix_tokens = tokens_list[1 + text_kv_cached :]
        else:
            suffix_tokens = tokens_list[skip_positions:]

        # Embed suffix tokens
        if suffix_tokens:
            inputs_embeds = self._embed_tokens(suffix_tokens)
        else:
            # This shouldn't happen due to skip_positions capping
            inputs_embeds = torch.empty(
                (1, 0, self.bos_embed.shape[-1]),
                device=self.device,
                dtype=self.dtype,
            )

        position_ids = torch.arange(
            skip_positions, prompt_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # FA3 needs to know total KV length (cached + new)
        fa3_seqused_k = torch.tensor(
            [prompt_len], dtype=torch.int32, device=self.device
        )

        return inputs_embeds, position_ids, fa3_seqused_k

    def _prepare_full_prefill_inputs(
        self,
        tokens_list: list[Token],
        image: Optional[pyvips.Image | np.ndarray],
        image_crops: Optional[OverlapCropOutput],
        prompt_len: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare inputs for full prefill (cache miss path)."""
        # Embed prompt tokens
        if len(tokens_list) > 0:
            prompt_embed = self._embed_tokens(tokens_list)
        else:
            prompt_embed = None

        segments: list[Tensor] = []
        if prompt_embed is not None and len(tokens_list) > 0:
            # Add BOS embed (first token)
            segments.append(prompt_embed[:, :1, :])

        if image is not None or image_crops is not None:
            image_embed = self.encode_image(image, overlap=image_crops).unsqueeze(0)
            segments.append(image_embed)

        if prompt_embed is not None and len(tokens_list) > 1:
            # Add remaining tokens after BOS
            segments.append(prompt_embed[:, 1:, :])

        if not segments:
            segments = [self.bos_embed]

        inputs_embeds = torch.cat(segments, dim=1)
        position_ids = torch.arange(
            prompt_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        fa3_seqused_k = torch.tensor(
            [prompt_len], dtype=torch.int32, device=self.device
        )

        return inputs_embeds, position_ids, fa3_seqused_k

    def _finalize_cache_after_prefill(
        self,
        cache_tokens: list[CacheToken],
        cache_result: _CacheLookupResult,
        prompt_len: int,
        batch_idx: int,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> tuple[TreeNode | None, int]:
        """Insert into cache and handle lock transfer after prefill."""
        if self.prefix_cache is None or not cache_tokens:
            return None, 0

        full_prompt_cached = (
            cache_result.can_reuse
            and cache_result.match is not None
            and cache_result.match.matched_kv_length >= prompt_len
        )

        if full_prompt_cached:
            # Full prompt was cached - don't insert
            return cache_result.temp_lock_node, cache_result.skip_positions

        # Insert prompt into cache
        prompt_pages = self.page_table.get_pages(batch_idx, 0, prompt_len)
        namespace = CacheNamespace(
            lora_id=adapter_id,
            image_hash=(
                int.from_bytes(image_hash[:16], "big") if image_hash else None
            ),
        )
        insert_result = self.prefix_cache.insert(
            cache_tokens,
            prompt_pages,
            namespace=namespace,
            from_node=cache_result.match.last_node if cache_result.match else None,
            from_token_idx=(
                cache_result.match.matched_token_count if cache_result.match else 0
            ),
            from_page_idx=cache_result.skip_positions,
        )

        # Duplicate prefill: another sequence inserted the full prompt before we did.
        # In this case, inserted_pages == 0 and the cache already owns a different
        # set of pages for the full prompt. Do NOT transfer the lock to that node
        # (we didn't use those pages); keep the existing temp lock if we reused
        # cached prefix pages, otherwise return no lock so our pages can be freed.
        if insert_result.inserted_pages == 0:
            if cache_result.temp_lock_node is None:
                return None, 0
            return cache_result.temp_lock_node, cache_result.skip_positions

        # Lock transfer
        temp_lock_node = cache_result.temp_lock_node
        if temp_lock_node is None:
            # Miss path
            self.prefix_cache.lock(insert_result.node)
            cache_lock_node = insert_result.node
        elif insert_result.node is temp_lock_node:
            # Identity case
            cache_lock_node = temp_lock_node
        else:
            # Partial hit path
            self.prefix_cache.lock(insert_result.node)
            self.prefix_cache.unlock(temp_lock_node)
            cache_lock_node = insert_result.node

        cache_owned_page_count = cache_result.skip_positions + insert_result.inserted_pages
        return cache_lock_node, cache_owned_page_count

    # ------------------------------------------------------------------

    def start_sequence(
        self,
        prompt_tokens: Tensor | Sequence[Token],
        *,
        image: Optional[pyvips.Image | np.ndarray] = None,
        image_crops: Optional[OverlapCropOutput] = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> tuple[SequenceState, Tensor]:
        # 1. Normalize inputs
        tokens_list = self._normalize_prompt_tokens(prompt_tokens)

        # 2. Validate image/hash consistency
        if image is None:
            assert image_hash is None, "image_hash must be None when image is None"
        else:
            # Image prompts must have at least one text token after BOS to ensure
            # correct suffix slicing on cache hit (text_kv_cached >= 0).
            if len(tokens_list) < 2:
                raise ValueError(
                    "Image prompts must include at least one text token after BOS"
                )
            if self.prefix_cache is not None:
                assert image_hash is not None, (
                    "image_hash must be provided when image is not None and prefix cache is enabled"
                )

        # 3. Compute dimensions
        image_kv_length = self.image_prefix_length if image is not None else 0
        prompt_len = len(tokens_list) + image_kv_length

        # Build image_regions for attention masking. Until multi-image support,
        # this must be either [] (no image) or [(1, 1+image_kv_length)] (single image).
        if image is not None:
            image_regions: list[tuple[int, int]] = [(1, 1 + image_kv_length)]
            # Validate single-image invariant
            assert len(image_regions) == 1, "Multi-image not yet supported"
            expected_region = (1, 1 + self.image_prefix_length)
            assert image_regions[0] == expected_region, (
                f"Unexpected image region {image_regions[0]}, expected {expected_region}"
            )
        else:
            image_regions = []

        max_new = max_new_tokens or DEFAULT_MAX_TOKENS
        target_length = prompt_len + max_new
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds max_seq_length={self.max_seq_length}."
            )

        # 4. Build cache tokens
        cache_tokens = self._build_cache_tokens(tokens_list, image_hash, image_kv_length)

        # 5. Allocate batch slot
        batch_idx = self.page_table.allocate()
        # GPU-only buffer avoids async H2D races on shared pinned host memory.
        # Fill runs on primary stream to synchronize with page table H2D copies.
        batch_tensor = self._prefill_batch_idx
        with torch.cuda.stream(self._primary_stream):
            batch_tensor.fill_(batch_idx)

        # 6. Cache lookup (maps pages, acquires temp lock)
        cache_result = self._lookup_prefix_cache(
            cache_tokens, adapter_id, image_hash, image_kv_length, prompt_len, batch_idx
        )

        # Steps 7-10 can fail; ensure we clean up batch slot and cache lock on error.
        try:
            # 7. Reserve remaining pages for decode. On cache hit, map_pages() in step 6
            # already set capacity to skip_positions, so reserve() only allocates suffix
            # pages (target_length - skip_positions), not the full target_length.
            self.page_table.reserve(
                batch_idx_int=batch_idx,
                batch_idx=batch_tensor,
                seq_len=target_length,
            )
            self._batch_binding.tensor = batch_tensor

            # 8-9. GPU work: embedding, vision encoding, and prefill forward pass.
            # Use primary stream to ensure ordering with shared buffers.
            with torch.cuda.stream(self._primary_stream):
                # 8. Prepare inputs (branch on cache hit)
                if cache_result.can_reuse:
                    inputs_embeds, position_ids, fa3_seqused_k = self._prepare_append_prefill_inputs(
                        tokens_list, cache_result.skip_positions, image_kv_length, prompt_len
                    )
                else:
                    inputs_embeds, position_ids, fa3_seqused_k = self._prepare_full_prefill_inputs(
                        tokens_list, image, image_crops, prompt_len
                    )

                # 9. Run prefill
                self._active_prefill_batch_idx = batch_tensor
                try:
                    hidden, logits = self._prefill(
                        inputs_embeds,
                        None,  # attention_mask
                        position_ids,
                        lora_slot,
                        use_prefix_attn=bool(image_kv_length) and not cache_result.can_reuse,
                        fa3_seqused_k=fa3_seqused_k,
                    )
                finally:
                    self._active_prefill_batch_idx = None

            # 10. Finalize cache (insert + lock transfer)
            cache_lock_node, cache_owned_page_count = self._finalize_cache_after_prefill(
                cache_tokens, cache_result, prompt_len, batch_idx, adapter_id, image_hash
            )
        except Exception:
            # Release temp cache lock if held
            if (
                self.prefix_cache is not None
                and cache_result.temp_lock_node is not None
            ):
                self.prefix_cache.unlock(cache_result.temp_lock_node)
            # Release batch slot (erase frees non-cache pages; 0 = no cache-owned pages yet)
            self.page_table.erase(batch_idx, 0)
            raise

        # 11. Create state
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=image_kv_length,
            last_hidden=hidden[:, -1, :].squeeze(0).detach(),
            lora_slot=lora_slot,
            cache_tokens=cache_tokens if self.prefix_cache else None,
            cache_lock_node=cache_lock_node,
            cache_owned_page_count=cache_owned_page_count,
            reused_page_count=cache_result.skip_positions,
            image_regions=image_regions if image_regions else None,
        )
        self.active_sequences[batch_idx] = state
        return state, logits

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)

        # Unlock the cached prefix (exactly one unlock per sequence)
        if self.prefix_cache is not None and state.cache_lock_node is not None:
            self.prefix_cache.unlock(state.cache_lock_node)

        # Release batch slot
        # Don't free cache-owned pages - they belong to the prefix cache tree
        self.page_table.erase(state.batch_idx, state.cache_owned_page_count)

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
        use_prefix_attn: bool,
        fa3_seqused_k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden, logits = self._prefill_fn(
            inputs_embeds,
            attn_mask,
            position_ids,
            lora_slot,
            use_prefix_attn=use_prefix_attn,
            fa3_seqused_k=fa3_seqused_k,
        )
        return hidden, logits

    def _prefill_impl(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
        lora_slot: int = 0,
        *,
        use_prefix_attn: bool,
        fa3_seqused_k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_idx = self._active_prefill_batch_idx
        if batch_idx is None:
            raise RuntimeError("Prefill batch index missing during warmup")
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx, positions=position_ids
        )

        # Build FA3 paged attention metadata for prefill
        fa3_page_table = self.page_table.page_table[batch_idx : batch_idx + 1]

        # For no-adapter prefill, skip LoRA entirely to avoid redundant work
        if lora_slot == 0:
            lora_workspace = None
            lora_slot_ids = None
            single_lora_id = None
        else:
            lora_workspace = self._lora_workspace
            lora_slot_ids = torch.tensor([lora_slot], dtype=torch.int32, device=self.device)
            # Single-LoRA prefill path uses a fixed lora_id (slot N -> lora_id N-1)
            single_lora_id = lora_slot - 1

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
            single_lora_id=single_lora_id,
        )

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
                with torch.cuda.stream(self._primary_stream):
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

            self._ensure_cuda_graphs_ready()

    def _ensure_cuda_graphs_ready(self) -> None:
        """Capture CUDA graphs for all slots using slot buffers directly."""
        if not self._use_cuda_graphs:
            return
        # Check if graphs already captured (first slot has graphs)
        if self._decode_slots and self._decode_slots[0].cuda_graphs is not None:
            return

        # Initialize graph batch sizes once
        max_effective_batch = max(1, self.max_batch_size)
        self._graph_batch_sizes = self._make_graph_batch_sizes(max_effective_batch)

        # Capture graphs for each slot using its own buffers
        for slot in self._decode_slots:
            slot.cuda_graphs = self._capture_decode_graphs_for_slot(slot)

    def _preallocate_workspaces(self) -> None:
        """Pre-allocate all shared workspaces to ensure stable pointers for CUDA graphs.

        All MoE layers share a single set of workspace buffers since they execute
        sequentially. This reduces memory from O(num_layers * workspace) to O(workspace).
        The buffers are fixed-size; requesting more tokens than allocated raises an error.
        """
        max_tokens = self.max_seq_length - 1

        # Pre-allocate vision fused MLP workspace
        # max_tokens = (max_crops + 1) * patches_per_crop
        # The +1 accounts for the global/overview crop added by overlap_crop_image
        vision_cfg = self.config.vision
        patches_per_crop = (vision_cfg.crop_size // vision_cfg.enc_patch_size) ** 2
        max_vision_tokens = (vision_cfg.max_crops + 1) * patches_per_crop
        from kestrel.ops.fused_mlp import preallocate_fused_mlp_workspaces
        preallocate_fused_mlp_workspaces(
            max_num_tokens=max_vision_tokens,
            hidden_dim=vision_cfg.enc_ff_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Pre-allocate MoE workspaces if MoE is enabled
        if self.config.text.moe is not None:
            from kestrel.fused_moe import preallocate_shared_moe_workspaces
            from kestrel.fused_moe.lora_kernels import preallocate_lora_buffers

            moe_cfg = self.config.text.moe
            preallocate_shared_moe_workspaces(
                max_num_tokens=max_tokens,
                top_k=moe_cfg.experts_per_token,
                hidden_size=moe_cfg.expert_inner_dim,
                input_size=self.config.text.dim,
                device=self.device,
                dtype=self.dtype,
            )

            # Pre-allocate LoRA buffers if LoRA is enabled
            if self._max_lora_rank is not None:
                preallocate_lora_buffers(
                    max_num_tokens=max_tokens,
                    top_k=moe_cfg.experts_per_token,
                    max_lora_rank=self._max_lora_rank,
                    device=self.device,
                    dtype=self.dtype,
                )

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

                            # Capture the graph (each graph gets its own pool for isolation)
                            with torch.cuda.graph(graph):
                                self._run_decode_forward(slot, bs)

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
