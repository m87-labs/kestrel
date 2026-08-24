"""Qwen 3.5/3.6 runtime for the Kestrel inference engine."""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from kestrel_kernels import get_runtime
from torch import nn

from kestrel.kv_cache import KVMemoryPool, PageTable, allocate_paged_kv_layers
from kestrel.runtime.decode_slot import DecodeSlot, create_decode_slot
from kestrel.runtime.paged_resources import bound_kv_cache_pages
from kestrel.runtime.tokenizer import load_tokenizer
from kestrel.runtime.preprocessing import (
    derive_image_insertion_offset,
    derive_preprocessing_workers,
)
from kestrel.runtime.uncached_paged import UncachedPagedRuntime

from .cache import (
    Qwen35InferenceCache,
    Qwen35LinearStatePool,
    qwen_paged_kv_specs,
)
from .prefill_slot import (
    Qwen35PrefillScratch,
    create_qwen35_prefill_slot,
)
from .prompt_template import (
    IMAGE_PAD_ID,
    IM_START_ID,
    Qwen35PromptTemplate,
    VISION_END_ID,
    VISION_START_ID,
    _NEWLINE_ID,
    _USER_ID,
)
from .qwen_image import preprocess_image


_PREFILL_SCRATCH_TOKENS = 1024


@dataclass
class QwenImageInputs:
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    num_image_tokens: int


@dataclass
class _PackedPrefillBatch:
    input_ids: torch.Tensor
    cache_position_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seq_lens_q: Optional[torch.Tensor]
    sequence_lengths: tuple[int, ...]
    topology_token: object
    seq_idx: Optional[torch.Tensor]
    batch_indices: torch.Tensor
    max_length: int
    last_token_offsets: torch.Tensor
    paged_kv_page_table: torch.Tensor
    paged_kv_seqlens_k: torch.Tensor
    slot_mapping: torch.Tensor
    rope_deltas: torch.Tensor
    image_token_spans: tuple[tuple[int, int], ...]
    pixel_values: Optional[torch.Tensor] = None
    vision_bilinear_indices: Optional[torch.Tensor] = None
    vision_bilinear_weights: Optional[torch.Tensor] = None
    vision_position_ids: Optional[torch.Tensor] = None
    vision_cu_seqlens: Optional[torch.Tensor] = None


class _QwenImagePreprocessor:
    def __init__(self, *, num_workers: int) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="qwen35-image"
        )

    def process(self, image: Any) -> QwenImageInputs:
        # Accept one image or an ordered list (chat may carry several). Each
        # image contributes one [1, H, W] grid row; concatenated pixel_values
        # plus a [N, 3] grid is exactly what the packed-prefill path expects.
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if not images:
            raise RuntimeError("Qwen image preprocessing got no images")
        pixel_values_parts = []
        grid_parts = []
        for one in images:
            pv, grid = preprocess_image(one)
            pixel_values_parts.append(pv)
            grid_parts.append(grid)
        if len(images) == 1:
            pixel_values = pixel_values_parts[0]
            image_grid_thw = grid_parts[0]
        else:
            pixel_values = torch.cat(pixel_values_parts, dim=0)
            image_grid_thw = torch.cat(grid_parts, dim=0)
        num_image_tokens = int(image_grid_thw.prod(-1).sum().item()) // 4
        if num_image_tokens <= 0:
            raise RuntimeError("Qwen image preprocessing produced no image tokens")
        return QwenImageInputs(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_image_tokens=num_image_tokens,
        )

    def submit(self, image: Any) -> Future[QwenImageInputs]:
        return self._executor.submit(self.process, image)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def _write_vision_grid_metadata(
    *,
    grid_thw: np.ndarray,
    bilinear_indices: np.ndarray,
    bilinear_weights: np.ndarray,
    position_ids: np.ndarray,
    cu_seqlens: np.ndarray,
    token_offset: int,
    sequence_offset: int,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[int, int]:
    side = int(num_grid_per_side)
    merge_size = int(spatial_merge_size)
    cu_value = int(cu_seqlens[sequence_offset])

    for grid_t_raw, grid_h_raw, grid_w_raw in grid_thw:
        grid_t = int(grid_t_raw)
        grid_h = int(grid_h_raw)
        grid_w = int(grid_w_raw)
        token_count = grid_t * grid_h * grid_w
        token_end = token_offset + token_count

        h_grid = np.linspace(0, side - 1, grid_h, dtype=np.float32)
        w_grid = np.linspace(0, side - 1, grid_w, dtype=np.float32)
        h_floor = h_grid.astype(np.int64)
        w_floor = w_grid.astype(np.int64)
        h_ceil = np.minimum(h_floor + 1, side - 1)
        w_ceil = np.minimum(w_floor + 1, side - 1)
        h_frac = h_grid - h_floor.astype(np.float32)
        w_frac = w_grid - w_floor.astype(np.float32)

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = (
            (h_floor_offset[:, None] + w_floor[None, :]).reshape(-1),
            (h_floor_offset[:, None] + w_ceil[None, :]).reshape(-1),
            (h_ceil_offset[:, None] + w_floor[None, :]).reshape(-1),
            (h_ceil_offset[:, None] + w_ceil[None, :]).reshape(-1),
        )
        corner_weights = (
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).reshape(-1),
            ((1 - h_frac)[:, None] * w_frac[None, :]).reshape(-1),
            (h_frac[:, None] * (1 - w_frac)[None, :]).reshape(-1),
            (h_frac[:, None] * w_frac[None, :]).reshape(-1),
        )

        h_idx = np.arange(grid_h, dtype=np.int64).reshape(
            grid_h // merge_size, merge_size
        )
        w_idx = np.arange(grid_w, dtype=np.int64).reshape(
            grid_w // merge_size, merge_size
        )
        reorder = (
            (h_idx[:, :, None, None] * grid_w + w_idx[None, None, :, :])
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        if grid_t != 1:
            reorder = np.tile(reorder, grid_t)

        for idx in range(4):
            bilinear_indices[idx, token_offset:token_end] = corner_indices[idx][
                reorder
            ]
            bilinear_weights[idx, token_offset:token_end] = corner_weights[idx][
                reorder
            ]

        hpos_ids = np.broadcast_to(
            np.arange(grid_h, dtype=np.int64).reshape(grid_h, 1),
            (grid_h, grid_w),
        )
        hpos_ids = (
            hpos_ids.reshape(
                grid_h // merge_size,
                merge_size,
                grid_w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        wpos_ids = np.broadcast_to(
            np.arange(grid_w, dtype=np.int64).reshape(1, grid_w),
            (grid_h, grid_w),
        )
        wpos_ids = (
            wpos_ids.reshape(
                grid_h // merge_size,
                merge_size,
                grid_w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        grid_position_ids = np.stack((hpos_ids, wpos_ids), axis=-1)
        if grid_t != 1:
            grid_position_ids = np.tile(grid_position_ids, (grid_t, 1))
        position_ids[token_offset:token_end] = grid_position_ids

        for _ in range(grid_t):
            sequence_offset += 1
            cu_value += grid_h * grid_w
            cu_seqlens[sequence_offset] = cu_value

        token_offset = token_end

    return token_offset, sequence_offset


class Qwen35Runtime(UncachedPagedRuntime):
    """Runtime wrapping upstream Qwen 3.5 modeling for Kestrel."""

    def __init__(
        self,
        cfg: Any,
        *,
        max_lora_rank: Optional[int] = None,
        kv_pool: KVMemoryPool,
        compute_stream: torch.cuda.Stream | None = None,
    ) -> None:
        from kestrel.runtime import ExecutionShape

        self._cfg = cfg
        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        # The runtime protocol requires an explicit speculative capability.
        self.spec = None
        self.device = (
            cfg.resolved_device()
            if hasattr(cfg, "resolved_device")
            else torch.device(cfg.device)
        )
        self.dtype = (
            cfg.resolved_dtype()
            if hasattr(cfg, "resolved_dtype")
            else getattr(cfg, "dtype", torch.bfloat16)
        )
        from kestrel.models.registry import get_spec

        self._spec = get_spec(cfg.model)
        self._model_name = cfg.model
        self.decode_path = getattr(cfg, "decode_path", "auto")
        if self.decode_path == "native":
            raise ValueError(
                "Qwen 3.5/3.6 requires generated decode; native decode was removed"
            )
        self.max_batch_size = getattr(cfg, "max_batch_size", 1)
        self.max_batch_slots = self.max_batch_size + 2
        self._padding_batch_idx = self.max_batch_slots - 1

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        model_source = (
            Path(cfg.model_path).expanduser()
            if getattr(cfg, "model_path", None) is not None
            else self._spec.repo_id
        )
        if model_source is None:
            raise ValueError("Qwen model spec must declare repo_id")
        self._generated_weight_storage = None
        self.model = self._load_model(model_source).eval()
        self.architecture = self.model.config
        text_cfg = self.architecture.text_config
        self.max_seq_length = int(text_cfg.max_position_embeddings)
        self.vocab_size = int(text_cfg.vocab_size)

        self.tokenizer = load_tokenizer(
            self._spec.tokenizer_id,
            getattr(cfg, "tokenizer_path", None),
            revision=(
                self._spec.revision
                if self._spec.tokenizer_id == self._spec.repo_id
                else None
            ),
        )
        self.tokenizer.post_processor = None
        self.prompt_template = Qwen35PromptTemplate()
        self.eos_token_ids = (self.prompt_template.eos_id,)

        # Conservative fixed reservation for image requests. Actual prompt
        # insertion uses the per-image processor count.
        self.image_prefix_length = int(
            os.environ.get("KESTREL_QWEN35_IMAGE_PREFIX_LENGTH", "4096")
        )
        self._image_preprocessor = _QwenImagePreprocessor(
            num_workers=derive_preprocessing_workers(self.max_batch_size)
        )
        # batch_idx -> Qwen image inputs preprocessed in prepare_sequence for the
        # chat path (admission doesn't precompute them); consumed once by
        # launch_prepared_batch so each image is preprocessed exactly once.
        self._chat_image_crops: dict[int, QwenImageInputs] = {}

        self._compute_stream = compute_stream or (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.graph_capture_lock = threading.RLock()

        if getattr(cfg, "enable_prefix_cache", False):
            warnings.warn(
                "Qwen 3.5 prefix cache is disabled until GDN recurrent state "
                "snapshots are cached alongside KV pages.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.page_size = int(getattr(cfg, "page_size", 1))
        self._kv_cache_pages = bound_kv_cache_pages(
            int(getattr(cfg, "kv_cache_pages", 65536)),
            page_size=self.page_size,
            max_batch_size=self.max_batch_size,
            max_seq_length=self.max_seq_length,
        )
        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=None,
            h2d_stream=self._compute_stream,
        )
        self.page_table.free_batch_idx.remove(self._padding_batch_idx)
        self.page_table.reserve(self._padding_batch_idx, 1)
        self.page_table.commit_block_table([self._padding_batch_idx])
        self._kv_pool = kv_pool
        if self._kv_pool.device != self.device:
            raise ValueError(
                f"kv_pool.device ({self._kv_pool.device}) must match runtime "
                f"device ({self.device})"
            )
        self._paged_kv = allocate_paged_kv_layers(
            layer_specs=qwen_paged_kv_specs(text_cfg),
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )
        self._linear_state_pool = Qwen35LinearStatePool(
            config=text_cfg,
            max_batch_slots=self.max_batch_slots,
            device=self.device,
        )
        self._linear_state_pool.initialize_from_config(
            text_cfg, dtype=self.dtype
        )
        self._decode_rope_deltas = torch.zeros(
            (self.max_batch_slots, 1),
            dtype=torch.long,
            device=self.device,
        )

        self._prefill_slots = tuple(
            create_qwen35_prefill_slot(
                slot_id=i,
                device=self.device,
                max_batch_size=self.max_batch_size,
                kv_cache_pages=self._kv_cache_pages,
                dtype=self.dtype,
                token_capacity=_PREFILL_SCRATCH_TOKENS,
            )
            for i in range(2)
        )
        self._prefill_slot_free = list(self._prefill_slots)
        self.prefill_slots: Sequence[Any] = self._prefill_slots

        from .generated_decode import generated_decode_slot_capacity

        generated_decode_capacity = generated_decode_slot_capacity(
            self,
            required=True,
        )
        self._decode_row_capacity = max(
            self.max_batch_slots,
            generated_decode_capacity or 0,
        )

        self._decode_slots = tuple(
            create_decode_slot(
                slot_id=i,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=self._decode_row_capacity,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=int(text_cfg.vocab_size),
                hidden_dim=int(text_cfg.hidden_size),
                position_shape=(4, self._decode_row_capacity, 1),
                scratch_specs={
                    "rope_deltas": (
                        (self._decode_row_capacity, 1),
                        torch.long,
                    ),
                },
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for i in range(2)
        )
        self.decode_slots: Sequence[Any] = self._decode_slots
        self.active_sequences: dict[int, Any] = {}

        self.spatial_tables = None

        self._initialize_generated_decode()

    def _initialize_generated_decode(self) -> None:
        """Construct the required generated Qwen decode implementation."""

        from .generated_decode import create_generated_decode

        self.generated_decode = create_generated_decode(
            self,
            required=True,
        )

    def image_kv_length(
        self,
        prompt_tokens: Sequence[Any],
        image: Any,
        image_crops: Any,
    ) -> int:
        """Return the exact Qwen vision expansion after preprocessing."""
        if image is None:
            return 0
        if not isinstance(image_crops, QwenImageInputs):
            return super().image_kv_length(prompt_tokens, image, image_crops)

        from kestrel.runtime.tokens import ImageMarker

        num_images = int(image_crops.image_grid_thw.shape[0])
        grid_tokens = int(image_crops.image_grid_thw.prod(-1).sum().item()) // 4
        if grid_tokens != int(image_crops.num_image_tokens):
            raise ValueError(
                "Qwen preprocessed image token count does not match its grid"
            )
        marker_count = sum(isinstance(token, ImageMarker) for token in prompt_tokens)
        if marker_count not in (0, num_images):
            raise ValueError(
                "Qwen image marker count must be zero or match preprocessed images"
            )
        inserted_tokens = int(image_crops.num_image_tokens) + 2 * num_images
        return inserted_tokens - marker_count

    def _load_model(self, source: str | Path) -> nn.Module:
        from .qwen_loader import load_qwen35_model

        from kestrel.runtime.generated_decode import (
            finalize_generated_weight_storage_after_loading,
            prepare_generated_weight_storage_for_loading,
        )
        from .generated_decode import _WEIGHT_LAYER_PREFIX

        def prepare_model(model: nn.Module) -> None:
            self._generated_weight_storage = (
                prepare_generated_weight_storage_for_loading(
                    self,
                    model,
                    label="Qwen",
                    layer_prefix=_WEIGHT_LAYER_PREFIX,
                    required_batch_sizes=range(1, self.max_batch_size + 1),
                    required=True,
                )
            )

        def finalize_model(model: nn.Module) -> None:
            if self._generated_weight_storage is not None:
                self._generated_weight_storage = (
                    finalize_generated_weight_storage_after_loading(
                        self,
                        model,
                        self._generated_weight_storage,
                        label="Qwen",
                        layer_prefix=_WEIGHT_LAYER_PREFIX,
                        required_batch_sizes=range(
                            1, self.max_batch_size + 1
                        ),
                    )
                )

        return load_qwen35_model(
            source,
            device=self.device,
            dtype=self.dtype,
            revision=self._spec.revision,
            prepare_model=prepare_model,
            finalize_model=finalize_model,
        )

    def acquire_prefill_slot(self, slot_id: int) -> Any:
        if slot_id < 0 or slot_id >= len(self._prefill_slots):
            raise ValueError(f"Invalid prefill_slot_id {slot_id}")

        for idx in range(len(self._prefill_slot_free) - 1, -1, -1):
            slot = self._prefill_slot_free[idx]
            if slot.slot_id == slot_id:
                slot = self._prefill_slot_free.pop(idx)
                self._wait_prefill_slot_ready(slot)
                return slot
        raise RuntimeError(f"Prefill slot {slot_id} is already in use")

    def release_prefill_slot(self, slot: Any) -> None:
        if slot in self._prefill_slot_free:
            raise RuntimeError(f"Prefill slot {slot.slot_id} is already free")
        self._prefill_slot_free.append(slot)

    def _wait_prefill_slot_ready(self, slot: Any) -> None:
        if not getattr(slot, "step_done_event_pending", False):
            return
        slot.step_done_event.synchronize()
        slot.step_done_event_pending = False

    def _record_prefill_slot_done(self, slot: Any) -> None:
        event = getattr(slot, "step_done_event", None)
        if event is None:
            return
        event.record()
        slot.step_done_event_pending = True

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Any],
        *,
        image: Optional[Any] = None,
        image_crops: Optional[QwenImageInputs] = None,
        encoder_input: object | None = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
        image_hash: Optional[bytes] = None,
        adapter_id: Optional[str] = None,
    ) -> Any:
        if encoder_input is not None:
            raise ValueError("Qwen35Runtime does not support encoder inputs")

        from kestrel.runtime.tokens import TextToken
        from kestrel.runtime.tokens import ImageMarker

        tokens_list = list(prompt_tokens)
        num_image_tokens = 0
        num_images = 0
        chat_crops = None
        if image is not None:
            if image_crops is None:
                # Chat path: admission decodes the images but doesn't precompute
                # Qwen inputs (the query path does, async). Preprocess once here;
                # launch_prepared_batch reuses it via _chat_image_crops so the
                # image is never preprocessed twice.
                image_crops = self._image_preprocessor.process(image)
                chat_crops = image_crops
            grids = image_crops.image_grid_thw
            num_images = int(grids.shape[0])
            spatial = int(
                self.architecture.vision_config.spatial_merge_size
            )
            # Per-image token count, computed the same way the position-id /
            # vision-metadata paths expect (so the expanded pad count matches).
            per_image = []
            for i in range(num_images):
                gt, gh, gw = (int(v) for v in grids[i])
                per_image.append(gt * (gh // spatial) * (gw // spatial))
            num_image_tokens = sum(per_image)

            markers = [
                (i, t.index)
                for i, t in enumerate(tokens_list)
                if isinstance(t, ImageMarker)
            ]
            if markers:
                # Chat path: the chat skill emitted one ImageMarker per image at
                # its content position. Replace each with that image's vision
                # block: <|vision_start|> + <|image_pad|>×N + <|vision_end|>.
                if len(markers) != num_images:
                    raise RuntimeError(
                        f"Qwen chat prompt has {len(markers)} image marker(s) "
                        f"but {num_images} image(s) were provided"
                    )
                marker_indices = [index for _, index in markers]
                if marker_indices != list(range(num_images)):
                    raise RuntimeError(
                        "Qwen image markers must appear in image input order"
                    )
                for pos, idx in sorted(markers, reverse=True):
                    block = (
                        [TextToken(token_id=VISION_START_ID)]
                        + [TextToken(token_id=IMAGE_PAD_ID)] * per_image[idx]
                        + [TextToken(token_id=VISION_END_ID)]
                    )
                    tokens_list[pos : pos + 1] = block
            else:
                # Query path: no marker in the prompt; splice a single vision
                # block after the first user-turn opener.
                if num_images != 1:
                    raise RuntimeError(
                        "Qwen prompt without image markers supports exactly one image"
                    )
                image_block = (
                    [TextToken(token_id=VISION_START_ID)]
                    + [TextToken(token_id=IMAGE_PAD_ID)] * num_image_tokens
                    + [TextToken(token_id=VISION_END_ID)]
                )
                query_template = self.prompt_template.query()
                offset = derive_image_insertion_offset(
                    tokens_list,
                    user_turn_opener=(IM_START_ID, _USER_ID, _NEWLINE_ID),
                    fallback_offset=1 + (
                        len(query_template.prefix) if query_template else 0
                    ),
                )
                tokens_list = tokens_list[:offset] + image_block + tokens_list[offset:]

        new_tokens = 128 if max_new_tokens is None else max_new_tokens
        target_length = len(tokens_list) + new_tokens
        prepared = self._prepare_uncached_sequence(
            tokens=tokens_list,
            target_length=target_length,
            image_length=(
                num_image_tokens + 2 * num_images if image is not None else 0
            ),
            lora_slot=lora_slot,
            adapter_id=adapter_id,
            image_hash=image_hash,
        )
        if chat_crops is not None:
            self._chat_image_crops[int(prepared.state.batch_idx)] = chat_crops
        return prepared

    @torch.inference_mode()
    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[Any],
        prefill_slot: Any,
        *,
        images: Optional[Sequence[Any]] = None,
        image_crops_list: Optional[Sequence[Optional[QwenImageInputs]]] = None,
        encoder_inputs: Optional[Sequence[object | None]] = None,
    ) -> torch.Tensor:
        if encoder_inputs is not None and any(
            item is not None for item in encoder_inputs
        ):
            raise ValueError("Qwen35Runtime does not support encoder inputs")
        batch_size = len(prepared_sequences)
        if batch_size == 0:
            raise ValueError("prepared_sequences must be non-empty")
        if batch_size > self.max_batch_size:
            raise NotImplementedError(
                f"Qwen35Runtime prefill: batch_size={batch_size} > "
                f"max_batch_size={self.max_batch_size}"
            )
        if images is None:
            images = [None] * batch_size
        if image_crops_list is None:
            image_crops_list = [None] * batch_size
        if len(images) != batch_size:
            raise ValueError("images length must match prepared_sequences")
        if len(image_crops_list) != batch_size:
            raise ValueError("image_crops_list length must match prepared_sequences")
        # Chat path: prepare_sequence already preprocessed the image (for the
        # grids) and stashed the result; reuse it here so the image isn't
        # preprocessed a second time. (Query precomputes crops at admission and
        # passes them in via image_crops_list.)
        image_crops_list = [
            crops
            if crops is not None
            else (
                self._chat_image_crops.pop(int(prepared.state.batch_idx), None)
                if img is not None
                else None
            )
            for crops, img, prepared in zip(image_crops_list, images, prepared_sequences)
        ]

        batch_indices = [int(prepared.state.batch_idx) for prepared in prepared_sequences]
        self.page_table.commit_block_table(batch_indices)
        try:
            packed = self._build_packed_prefill_batch(
                prepared_sequences,
                prefill_slot=prefill_slot,
                image_crops_list=image_crops_list,
                batch_indices=batch_indices,
            )
            last_hidden, cache = self._forward_packed_prefill(packed)
            hidden_rows = last_hidden[0].index_select(0, packed.last_token_offsets)
            self._store_packed_sequence_caches(
                packed.batch_indices,
                cache,
                rope_deltas=packed.rope_deltas,
                host_batch_indices=batch_indices,
            )

            for row, prepared in enumerate(prepared_sequences):
                prepared.state.last_hidden = hidden_rows[row].detach()

            return self.model.lm_head(hidden_rows)
        finally:
            self._record_prefill_slot_done(prefill_slot)

    def _release_runtime_state(self, batch_idx: int) -> None:
        self._chat_image_crops.pop(batch_idx, None)
        self._clear_decode_state(batch_idx)

    @torch.inference_mode()
    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        megakernel = self.generated_decode
        if megakernel is None:
            raise RuntimeError("Qwen generated decode is not initialized")
        if not megakernel.supports(batch_size):
            raise RuntimeError(
                "selected generated Qwen decode does not cover "
                f"active batch size {batch_size}"
            )
        stream = getattr(slot, "compute_stream", None)
        stream_context = (
            torch.cuda.stream(stream) if stream is not None else nullcontext())
        with stream_context:
            megakernel.run(slot, batch_size)

    def _gather_decode_rope_deltas(
        self,
        slot: DecodeSlot,
        batch_size: int,
    ) -> None:
        torch.index_select(
            self._decode_rope_deltas,
            0,
            slot.meta.batch_idx.gpu[:batch_size],
            out=slot.scratch["rope_deltas"][:batch_size],
        )

    def _prepare_decode_position_ids(
        self,
        slot: DecodeSlot,
        batch_size: int,
    ) -> None:
        # Row 0 carries the text position; the three spatial M-RoPE rows carry
        # the same position offset by the per-sequence rope delta. Broadcast the
        # text position into all four rows, then add the delta into the spatial
        # rows in place. Two kernels, no temporary allocation.
        cache_position_ids = slot.cache_position_ids[:batch_size]
        position_ids = slot.position_ids[:, :batch_size, :]
        position_ids.copy_(cache_position_ids)
        position_ids[1:].add_(slot.scratch["rope_deltas"][:batch_size])

    def _prefill_scratch_for(
        self,
        slot: Any,
        *,
        total_tokens: int,
        batch_size: int,
        pixel_rows: int,
        pixel_dim: int,
        vision_sequence_count: int,
    ) -> Qwen35PrefillScratch:
        scratch = getattr(slot, "scratch", None)
        if scratch is None:
            scratch = Qwen35PrefillScratch.create(
                token_capacity=total_tokens,
                batch_capacity=batch_size,
                kv_cache_pages=self.page_table.n_pages,
                dtype=self.dtype,
                device=self.device,
            )
            setattr(slot, "scratch", scratch)

        grown = scratch.ensure_size(
            total_tokens=total_tokens,
            batch_size=batch_size,
            pixel_rows=pixel_rows,
            pixel_dim=pixel_dim,
            vision_sequence_count=vision_sequence_count,
        )
        if grown is not scratch:
            setattr(slot, "scratch", grown)
            scratch = grown
        return scratch

    def _fill_text_position_ids(
        self,
        out: np.ndarray,
        *,
        start: int,
        end: int,
    ) -> None:
        positions = np.arange(end - start, dtype=np.int64)
        out[:, 0, start:end] = positions

    def _fill_multimodal_position_ids(
        self,
        out: np.ndarray,
        *,
        start: int,
        end: int,
        mm_token_type_ids: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        spatial_merge_size = int(
            self.architecture.vision_config.spatial_merge_size
        )
        cursor = start
        current_pos = 0
        image_idx = 0
        image_token_spans: list[tuple[int, int]] = []
        while cursor < end:
            modality_type = int(mm_token_type_ids[cursor - start])
            group_end = cursor + 1
            while (
                group_end < end
                and int(mm_token_type_ids[group_end - start]) == modality_type
            ):
                group_end += 1

            if modality_type == 0:
                text_len = group_end - cursor
                positions = np.arange(
                    current_pos, current_pos + text_len, dtype=np.int64
                )
                out[:, 0, cursor:group_end] = positions
                current_pos += text_len
            elif modality_type == 1:
                if image_idx >= image_grid_thw.shape[0]:
                    raise RuntimeError("Qwen image grid metadata ended early")
                grid_t, grid_h_raw, grid_w_raw = (
                    int(v) for v in image_grid_thw[image_idx]
                )
                image_idx += 1
                grid_h = grid_h_raw // spatial_merge_size
                grid_w = grid_w_raw // spatial_merge_size
                image_len = grid_t * grid_h * grid_w
                if image_len != group_end - cursor:
                    raise RuntimeError(
                        "Qwen image token count does not match image grid: "
                        f"tokens={group_end - cursor}, grid={image_len}"
                    )
                image_token_spans.append((cursor, group_end))
                offset = current_pos
                temporal = np.repeat(np.arange(grid_t, dtype=np.int64), grid_h * grid_w)
                height = np.tile(
                    np.repeat(np.arange(grid_h, dtype=np.int64), grid_w),
                    grid_t,
                )
                width = np.tile(np.arange(grid_w, dtype=np.int64), grid_h * grid_t)
                out[0, 0, cursor:group_end] = temporal + offset
                out[1, 0, cursor:group_end] = height + offset
                out[2, 0, cursor:group_end] = width + offset
                current_pos += max(grid_h_raw, grid_w_raw) // spatial_merge_size
            else:
                raise NotImplementedError("Qwen packed prefill does not support video")

            cursor = group_end

        if image_idx != image_grid_thw.shape[0]:
            raise RuntimeError("Qwen image grid metadata has unused rows")
        rope_delta = int(out[:, 0, start:end].max()) + 1 - (end - start)
        return rope_delta, tuple(image_token_spans)

    def _fill_vision_metadata(
        self,
        scratch: Qwen35PrefillScratch,
        crop_grid_rows: Sequence[np.ndarray | None],
        *,
        pixel_rows: int,
        vision_sequence_count: int,
    ) -> None:
        if pixel_rows == 0:
            return
        if (
            scratch.vision_bilinear_indices is None
            or scratch.vision_bilinear_weights is None
            or scratch.vision_position_ids is None
            or scratch.vision_cu_seqlens is None
        ):
            raise RuntimeError("Qwen vision metadata scratch was not allocated")

        vision_config = self.architecture.vision_config
        num_positions = int(vision_config.num_position_embeddings)
        num_grid_per_side = int(num_positions**0.5)
        spatial_merge_size = int(vision_config.spatial_merge_size)

        bilinear_indices = scratch.vision_bilinear_indices.np[:, :pixel_rows]
        bilinear_weights = scratch.vision_bilinear_weights.np[:, :pixel_rows]
        position_ids = scratch.vision_position_ids.np[:pixel_rows]
        cu_seqlens = scratch.vision_cu_seqlens.np[: vision_sequence_count + 1]
        cu_seqlens[0] = 0

        token_offset = 0
        sequence_offset = 0
        for grid_np in crop_grid_rows:
            if grid_np is None:
                continue
            token_offset, sequence_offset = _write_vision_grid_metadata(
                grid_thw=grid_np,
                bilinear_indices=bilinear_indices,
                bilinear_weights=bilinear_weights,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                token_offset=token_offset,
                sequence_offset=sequence_offset,
                num_grid_per_side=num_grid_per_side,
                spatial_merge_size=spatial_merge_size,
            )

        if token_offset != pixel_rows:
            raise RuntimeError(
                "Qwen vision metadata token count mismatch: "
                f"metadata={token_offset}, pixels={pixel_rows}"
            )
        if sequence_offset != vision_sequence_count:
            raise RuntimeError(
                "Qwen vision metadata sequence count mismatch: "
                f"metadata={sequence_offset}, expected={vision_sequence_count}"
            )

    def _fill_prefill_slot_mapping(
        self,
        out: np.ndarray,
        *,
        start: int,
        length: int,
        batch_idx: int,
    ) -> None:
        pages = self.page_table.page_table_cpu[int(batch_idx)]
        page_size = int(self.page_table.page_size)
        if page_size == 1:
            if len(pages) < length:
                raise RuntimeError("Qwen page table row is shorter than prompt length")
            out[0, start : start + length] = np.asarray(
                pages[:length], dtype=np.int64
            )
            return

        num_blocks = (length + page_size - 1) // page_size
        if len(pages) < num_blocks:
            raise RuntimeError("Qwen page table row is shorter than prompt length")
        positions = np.arange(length, dtype=np.int64)
        block_idx = positions // page_size
        block_offset = positions % page_size
        physical_blocks = np.asarray(pages[:num_blocks], dtype=np.int64)[block_idx]
        out[0, start : start + length] = physical_blocks * page_size + block_offset

    def _copy_prefill_metadata_to_gpu(
        self,
        scratch: Qwen35PrefillScratch,
        *,
        total_tokens: int,
        batch_size: int,
        has_images: bool,
        pixel_rows: int,
        vision_sequence_count: int,
    ) -> None:
        # One H2D for every text-metadata field (input ids, positions, slot
        # mapping, seq idx, batch indices, cu-seqlens, rope deltas, …) instead
        # of a dozen separate cudaMemcpyAsync launches. Consumers slice each
        # field to its live length, so shipping the whole packed buffer
        # (including unused tails) is safe.
        scratch.text_meta.copy_to_gpu()
        if has_images:
            if scratch.pixel_values is None:
                raise RuntimeError("Qwen pixel scratch was not allocated")
            scratch.pixel_values.copy_to_gpu(pixel_rows)
            if (
                scratch.vision_bilinear_indices is None
                or scratch.vision_bilinear_weights is None
                or scratch.vision_position_ids is None
                or scratch.vision_cu_seqlens is None
            ):
                raise RuntimeError("Qwen vision metadata scratch was not allocated")
            scratch.vision_bilinear_indices.gpu[:, :pixel_rows].copy_(
                scratch.vision_bilinear_indices.cpu[:, :pixel_rows],
                non_blocking=True,
            )
            scratch.vision_bilinear_weights.gpu[:, :pixel_rows].copy_(
                scratch.vision_bilinear_weights.cpu[:, :pixel_rows],
                non_blocking=True,
            )
            scratch.vision_position_ids.copy_to_gpu(pixel_rows)
            scratch.vision_cu_seqlens.copy_to_gpu(vision_sequence_count + 1)

    def _populate_prefill_paged_metadata(
        self,
        scratch: Qwen35PrefillScratch,
        *,
        batch_size: int,
    ) -> None:
        if self.device.type == "cuda":
            self.page_table.populate_paged_kv_metadata(
                batch_idx=scratch.text_meta.batch_indices.gpu[:batch_size],
                input_pos=scratch.text_meta.last_positions.gpu[:batch_size],
                out_page_table=scratch.paged_kv_page_table[:batch_size],
                out_seqused_k=scratch.paged_kv_seqlens_k[:batch_size],
            )
            return

        for row, batch_idx in enumerate(scratch.text_meta.batch_indices.cpu[:batch_size].tolist()):
            scratch.paged_kv_page_table[row].copy_(self.page_table.page_table[int(batch_idx)])
        scratch.paged_kv_seqlens_k[:batch_size].copy_(
            scratch.text_meta.last_positions.gpu[:batch_size] + 1
        )

    def _build_packed_prefill_batch(
        self,
        prepared_sequences: Sequence[Any],
        *,
        prefill_slot: Any,
        image_crops_list: Sequence[Optional[QwenImageInputs]],
        batch_indices: Sequence[int],
    ) -> _PackedPrefillBatch:
        from kestrel.runtime.tokens import TextToken

        if len(prepared_sequences) != len(image_crops_list):
            raise ValueError("image_crops_list length must match prepared_sequences")
        if len(prepared_sequences) != len(batch_indices):
            raise ValueError("batch_indices length must match prepared_sequences")

        token_rows: list[list[int]] = []
        crop_grid_rows: list[np.ndarray | None] = []
        lengths: list[int] = []
        total_tokens = 0
        pixel_rows = 0
        pixel_dim = 0
        vision_sequence_count = 0
        has_images = False
        image_token_spans: list[tuple[int, int]] = []

        for prepared, crops in zip(prepared_sequences, image_crops_list):
            tokens = prepared.tokens_list
            if not all(isinstance(t, TextToken) for t in tokens):
                raise ValueError(
                    "Qwen35Runtime prefill only supports TextToken prompts"
                )
            token_ids = [int(t.token_id) for t in tokens]
            if not token_ids:
                raise ValueError("Prefill prompt must contain at least one token")

            length = len(token_ids)
            token_rows.append(token_ids)
            lengths.append(length)
            total_tokens += length
            if crops is not None:
                has_images = True
                crop_pixels = crops.pixel_values
                if crop_pixels.ndim != 2:
                    raise ValueError("Qwen image pixel_values must be rank-2")
                if pixel_dim == 0:
                    pixel_dim = int(crop_pixels.shape[1])
                elif pixel_dim != int(crop_pixels.shape[1]):
                    raise ValueError("Qwen image pixel feature dimension changed")
                pixel_rows += int(crop_pixels.shape[0])
                grid_np = crops.image_grid_thw.detach().cpu().numpy().astype(
                    np.int64, copy=False
                )
                if grid_np.ndim != 2 or grid_np.shape[1] != 3:
                    raise ValueError("Qwen image_grid_thw must have shape [N, 3]")
                crop_grid_rows.append(grid_np)
                vision_sequence_count += int(grid_np[:, 0].sum())
            else:
                if IMAGE_PAD_ID in token_ids:
                    raise ValueError(
                        "Qwen prefill row has image tokens without image inputs"
                    )
                crop_grid_rows.append(None)

        scratch = self._prefill_scratch_for(
            prefill_slot,
            total_tokens=total_tokens,
            batch_size=len(prepared_sequences),
            pixel_rows=pixel_rows,
            pixel_dim=pixel_dim,
            vision_sequence_count=vision_sequence_count,
        )

        offset = 0
        pixel_offset = 0
        scratch.text_meta.cu_seq_lens_q.np[0] = 0

        for row, (token_ids, crops, grid_np, batch_idx) in enumerate(
            zip(token_rows, image_crops_list, crop_grid_rows, batch_indices)
        ):
            length = lengths[row]
            end = offset + length
            ids_np = np.asarray(token_ids, dtype=np.int64)
            scratch.text_meta.input_ids.np[0, offset:end] = ids_np
            scratch.text_meta.cache_position_ids.np[0, offset:end] = np.arange(
                length, dtype=np.int64
            )
            scratch.text_meta.seq_idx.np[0, offset:end] = row
            scratch.text_meta.batch_indices.np[row] = int(batch_idx)
            scratch.text_meta.last_positions.np[row] = length - 1
            scratch.text_meta.last_token_offsets.np[row] = end - 1
            scratch.text_meta.cu_seq_lens_q.np[row + 1] = end
            self._fill_prefill_slot_mapping(
                scratch.text_meta.slot_mapping.np,
                start=offset,
                length=length,
                batch_idx=int(batch_idx),
            )
            mm_types = scratch.text_meta.mm_token_type_ids.np[0, offset:end]
            mm_types.fill(0)
            if crops is None:
                self._fill_text_position_ids(
                    scratch.text_meta.position_ids.np,
                    start=offset,
                    end=end,
                )
                scratch.text_meta.rope_deltas.np[row, 0] = 0
            else:
                mm_types[ids_np == IMAGE_PAD_ID] = 1
                assert grid_np is not None
                rope_delta, row_image_token_spans = self._fill_multimodal_position_ids(
                    scratch.text_meta.position_ids.np,
                    start=offset,
                    end=end,
                    mm_token_type_ids=mm_types,
                    image_grid_thw=grid_np,
                )
                scratch.text_meta.rope_deltas.np[row, 0] = rope_delta
                image_token_spans.extend(row_image_token_spans)
                pixel_end = pixel_offset + int(crops.pixel_values.shape[0])
                if scratch.pixel_values is None:
                    raise RuntimeError("Qwen pixel scratch was not allocated")
                scratch.pixel_values.cpu[pixel_offset:pixel_end].copy_(
                    crops.pixel_values
                )
                pixel_offset = pixel_end
            offset += length

        if has_images:
            self._fill_vision_metadata(
                scratch,
                crop_grid_rows,
                pixel_rows=pixel_rows,
                vision_sequence_count=vision_sequence_count,
            )

        self._copy_prefill_metadata_to_gpu(
            scratch,
            total_tokens=total_tokens,
            batch_size=len(prepared_sequences),
            has_images=has_images,
            pixel_rows=pixel_rows,
            vision_sequence_count=vision_sequence_count,
        )
        prefill_slot.batch_idx[: len(prepared_sequences)].copy_(
            scratch.text_meta.batch_indices.gpu[: len(prepared_sequences)]
        )
        self._populate_prefill_paged_metadata(
            scratch,
            batch_size=len(prepared_sequences),
        )

        batch_size = len(prepared_sequences)
        sequence_lengths = tuple(lengths)
        cu_seq_lens_q, topology_token = (
            get_runtime().gated_delta.bind_packed_prefill_topology(
                sequence_lengths=sequence_lengths,
                device=self.device,
            )
        )

        return _PackedPrefillBatch(
            input_ids=scratch.text_meta.input_ids.gpu[:, :total_tokens],
            cache_position_ids=scratch.text_meta.cache_position_ids.gpu[:, :total_tokens],
            position_ids=scratch.text_meta.position_ids.gpu[:, :, :total_tokens],
            # Always pass packed metadata, even for a single sequence: text_meta
            # is staged to the GPU unconditionally, so cu_seq_lens_q ([0, total])
            # and seq_idx are valid for batch_size == 1. This lets a single
            # sequence take the same packed_prefill path as batched prefill
            # instead of the separate uniform_native_prefill branch.
            cu_seq_lens_q=cu_seq_lens_q,
            sequence_lengths=sequence_lengths,
            topology_token=topology_token,
            seq_idx=scratch.text_meta.seq_idx.gpu[:, :total_tokens],
            batch_indices=scratch.text_meta.batch_indices.gpu[:batch_size],
            max_length=max(lengths),
            last_token_offsets=scratch.text_meta.last_token_offsets.gpu[:batch_size],
            paged_kv_page_table=scratch.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=scratch.paged_kv_seqlens_k[:batch_size],
            slot_mapping=scratch.text_meta.slot_mapping.gpu[:, :total_tokens],
            rope_deltas=scratch.text_meta.rope_deltas.gpu[:batch_size],
            image_token_spans=tuple(image_token_spans),
            pixel_values=(
                scratch.pixel_values.gpu[:pixel_rows]
                if has_images and scratch.pixel_values is not None
                else None
            ),
            vision_bilinear_indices=(
                scratch.vision_bilinear_indices.gpu[:, :pixel_rows]
                if has_images and scratch.vision_bilinear_indices is not None
                else None
            ),
            vision_bilinear_weights=(
                scratch.vision_bilinear_weights.gpu[:, :pixel_rows]
                if has_images and scratch.vision_bilinear_weights is not None
                else None
            ),
            vision_position_ids=(
                scratch.vision_position_ids.gpu[:pixel_rows]
                if has_images and scratch.vision_position_ids is not None
                else None
            ),
            vision_cu_seqlens=(
                scratch.vision_cu_seqlens.gpu[: vision_sequence_count + 1]
                if has_images and scratch.vision_cu_seqlens is not None
                else None
            ),
        )

    def _forward_packed_prefill(
        self,
        packed: _PackedPrefillBatch,
    ) -> tuple[torch.Tensor, Qwen35InferenceCache]:
        cache = self._new_cache()
        self._linear_state_pool.bind_prefill_state(cache)
        outputs = self.model.model(
            input_ids=packed.input_ids,
            past_key_values=cache,
            pixel_values=packed.pixel_values,
            position_ids=packed.position_ids,
            vision_bilinear_indices=packed.vision_bilinear_indices,
            vision_bilinear_weights=packed.vision_bilinear_weights,
            vision_position_ids=packed.vision_position_ids,
            vision_cu_seqlens=packed.vision_cu_seqlens,
            image_token_spans=packed.image_token_spans,
            cache_position_ids=packed.cache_position_ids,
            slot_mapping=packed.slot_mapping,
            page_table=packed.paged_kv_page_table,
            paged_kv_seqlens_k=packed.paged_kv_seqlens_k,
            cu_seq_lens_q=packed.cu_seq_lens_q,
            sequence_lengths=packed.sequence_lengths,
            topology_token=packed.topology_token,
            seq_idx=packed.seq_idx,
            gdn_state_indices=packed.batch_indices,
            gdn_state_indices_allocator_owned=True,
        )
        outputs.past_key_values.advance_to(packed.max_length)
        return outputs.last_hidden_state, outputs.past_key_values

    def _new_cache(self) -> Qwen35InferenceCache:
        return Qwen35InferenceCache(
            config=self.architecture.text_config,
            paged_kv=self._paged_kv,
        )

    def _store_packed_sequence_caches(
        self,
        batch_idx: torch.Tensor,
        cache: Qwen35InferenceCache,
        *,
        rope_deltas: torch.Tensor,
        host_batch_indices: Sequence[int],
    ) -> None:
        if not isinstance(cache, Qwen35InferenceCache):
            raise RuntimeError("Qwen engine decode requires paged hybrid caches")
        indices = batch_idx.to(device=self.device, dtype=torch.long).view(-1)
        batch_size = int(indices.shape[0])
        if len(host_batch_indices) != batch_size:
            raise ValueError("host batch indices must match packed batch size")
        self._linear_state_pool.capture_batch_from_cache(
            indices,
            cache,
            batch_size=batch_size,
        )
        rope_deltas = rope_deltas.to(device=self.device, dtype=torch.long)
        if rope_deltas.ndim == 1:
            rope_deltas = rope_deltas.view(-1, 1)
        if rope_deltas.shape != (batch_size, 1):
            raise RuntimeError(
                "Qwen packed M-RoPE deltas must have shape [batch, 1]"
            )
        self._decode_rope_deltas.index_copy_(0, indices, rope_deltas)

    def _clear_decode_state(self, batch_idx: int) -> None:
        if hasattr(self, "_decode_rope_deltas"):
            self._decode_rope_deltas[int(batch_idx)].zero_()
        if hasattr(self, "_linear_state_pool"):
            self._linear_state_pool.clear(int(batch_idx))

__all__ = ["Qwen35Runtime", "QwenImageInputs"]
