"""Qwen 3.5 runtime for the Kestrel inference engine."""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from kestrel.kv_cache import KVMemoryPool, PageTable
from kestrel.runtime.decode_graph import DecodeGraphManager
from kestrel.utils import CpuGpuBuffer
from kestrel.runtime.preprocessing import (
    derive_image_insertion_offset,
    derive_preprocessing_workers,
)

from .decode_slot import Qwen35DecodeSlot, create_qwen35_decode_slot
from .paged_cache import Qwen35LinearStatePool, Qwen35PagedHybridCache
from .prefill_slot import (
    Qwen35PrefillScratch,
    create_qwen35_prefill_slot,
)
from .prompt_template import (
    END_OF_TEXT_ID,
    IMAGE_PAD_ID,
    IM_END_ID,
    IM_START_ID,
    Qwen35PromptTemplate,
    VISION_END_ID,
    VISION_START_ID,
    _NEWLINE_ID,
    _USER_ID,
)
from .qwen_image import preprocess_image
from .qwen_loader import load_qwen35_model


_PREFILL_SCRATCH_TOKENS = 1024


def _native_decode_state_requirements(generated, linear_state_pool):
    from mkl.megakernel.state_runtime import StateRepresentationRequirement

    native = []
    for requirement in generated:
        form = (
            linear_state_pool.replay_recurrent_form
            if requirement.buffer == "gdn_recurrent_state"
            else requirement.physical_form
        )
        native.append(StateRepresentationRequirement(
            requirement.buffer,
            form.representation,
            form.storage_axis_order,
            form.storage_dtype,
        ))
    return tuple(native)


@dataclass
class _EncodeResult:
    ids: list[int]


@dataclass
class _TextConfigView:
    vocab_size: int


@dataclass
class _RuntimeConfigView:
    hf_config: Any
    text: _TextConfigView

    def __getattr__(self, name: str) -> Any:
        return getattr(self.hf_config, name)


@dataclass
class _QwenForwardCache:
    past_key_values: Any
    rope_deltas: Optional[torch.Tensor] = None
    linear_state_row_indices: Optional[torch.Tensor] = None


class _TokenizerShim:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def encode(self, text: str) -> _EncodeResult:
        enc = self._tok.encode(text, add_special_tokens=False)
        return _EncodeResult(ids=list(enc.ids))

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        return self._tok.decode(
            list(ids), skip_special_tokens=skip_special_tokens, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)


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
    seq_idx: Optional[torch.Tensor]
    batch_indices: torch.Tensor
    max_length: int
    last_token_offsets: torch.Tensor
    paged_kv_page_table: torch.Tensor
    paged_kv_seqlens_k: torch.Tensor
    slot_mapping: torch.Tensor
    rope_deltas: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    mm_token_type_ids: Optional[torch.Tensor] = None
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
        pixel_values = torch.cat(pixel_values_parts, dim=0)
        image_grid_thw = torch.cat(grid_parts, dim=0)
        num_image_tokens = int(image_grid_thw.prod(-1).sum().item()) // 4
        if num_image_tokens <= 0:
            raise RuntimeError("Qwen image preprocessing produced no image tokens")
        return QwenImageInputs(
            pixel_values=pixel_values.detach().cpu(),
            image_grid_thw=image_grid_thw.detach().cpu(),
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


class Qwen35Runtime:
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
        # Speculative-decoding capability. ``None`` => one token per decode step
        # (identical to non-spec behavior). Set by ``_maybe_init_spec_decode``
        # at the end of __init__ when a drafter is configured.
        self.spec = None
        self._spec_runner = None
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
        self._model_name = cfg.model
        self.max_batch_size = getattr(cfg, "max_batch_size", 1)
        self.max_batch_slots = self.max_batch_size + 2
        self._padding_batch_idx = self.max_batch_slots - 1

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        requested_cuda_graphs = bool(getattr(cfg, "enable_cuda_graphs", False))
        self._use_cuda_graphs = (
            requested_cuda_graphs
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        if requested_cuda_graphs and not self._use_cuda_graphs:
            warnings.warn(
                "Qwen 3.5 CUDA graphs are disabled because CUDA is unavailable "
                "or the runtime device is not CUDA.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.model = self._load_model(self._model_name).eval()
        self.hf_config = self.model.config
        text_cfg = _text_config(self.hf_config)
        self.max_seq_length = int(getattr(text_cfg, "max_position_embeddings"))
        self.config = _RuntimeConfigView(
            hf_config=self.hf_config,
            text=_TextConfigView(vocab_size=int(getattr(text_cfg, "vocab_size"))),
        )

        from tokenizers import Tokenizer

        self.tokenizer = _TokenizerShim(Tokenizer.from_pretrained(self._model_name))
        self.prompt_template = Qwen35PromptTemplate()
        self._eos_ids = {IM_END_ID, END_OF_TEXT_ID}
        eos_cfg = getattr(text_cfg, "eos_token_id", None)
        if isinstance(eos_cfg, int):
            self._eos_ids.add(eos_cfg)
        elif eos_cfg is not None:
            self._eos_ids.update(int(tid) for tid in eos_cfg)

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

        self.primary_stream = compute_stream or (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.graph_capture_lock = threading.RLock()

        self.prefix_cache = None
        if getattr(cfg, "enable_prefix_cache", False):
            warnings.warn(
                "Qwen 3.5 prefix cache is disabled until GDN recurrent state "
                "snapshots are cached alongside KV pages.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.page_size = int(getattr(cfg, "page_size", 1))
        self._kv_cache_pages = int(getattr(cfg, "kv_cache_pages", 65536))
        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=None,
            h2d_stream=self.primary_stream,
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
        self._shared_paged_layers = Qwen35PagedHybridCache.build_shared_paged_layers(
            config=text_cfg,
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )
        self._linear_state_pool = Qwen35LinearStatePool(
            config=text_cfg,
            max_batch_slots=self.max_batch_slots,
            device=self.device,
        )
        self._decode_rope_deltas = torch.zeros(
            (self.max_batch_slots, 1),
            dtype=torch.long,
            device=self.device,
        )
        self._cache_batch_idx_staging = CpuGpuBuffer(
            self.max_batch_slots,
            dtype=torch.int64,
            device=self.device,
            pin_memory=self.device.type == "cuda",
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

        self._decode_slots = tuple(
            create_qwen35_decode_slot(
                slot_id=i,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=self.max_batch_slots,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=int(getattr(text_cfg, "vocab_size")),
                hidden_dim=int(getattr(text_cfg, "hidden_size")),
                compute_stream=self.primary_stream,
                copy_stream=self.copy_stream,
            )
            for i in range(2)
        )
        self.decode_slots: Sequence[Any] = self._decode_slots
        self._decode_caches = tuple(self._new_cache() for _ in self._decode_slots)
        self._decode_megakernel = None
        self._decode_graphs = DecodeGraphManager[Qwen35DecodeSlot](
            enabled=self._use_cuda_graphs,
            device=self.device,
            max_batch=self.max_batch_size,
            graph_capture_lock=self.graph_capture_lock,
            compute_stream=self.primary_stream,
            run_forward=self._run_decode_forward,
            prepare_step=self._prepare_decode_slot,
            zero_padding=self._zero_decode_graph_padding,
            zero_for_capture=self._zero_decode_graph_capture_buffers,
        )
        self.active_sequences: dict[int, Any] = {}
        self._caches: dict[int, Any] = {}

        self.spatial_tables = None

        # Size the persistent GDN (ReplaySSM) pool for spec decode BEFORE the
        # decode graph is captured. The captured ``decode_with_slot`` graph binds
        # the linear pool's replay tensors, and ``_maybe_init_spec_decode``
        # reallocates them to ``flush_cap`` (changing their addresses/shapes); if
        # that resize ran AFTER capture, a later fallback to ``decode_with_slot``
        # would replay kernels bound to the freed/stale buffers. Resize first so
        # the graph captures against the final tensors.
        _spec_cfg_for_pool = getattr(self._cfg, "spec_decode", None)
        if _spec_cfg_for_pool:
            self._resize_linear_pool_for_spec(
                self._spec_flush_cap(_spec_cfg_for_pool)
            )

        if self.device.type == "cuda":
            from .megakernel_decode import Qwen35DecodeMegakernel

            self._decode_megakernel = Qwen35DecodeMegakernel.try_create(self)
        self._decode_state_coordinator = None
        self._native_decode_state_requirements = ()
        if self._decode_megakernel is not None:
            from mkl.megakernel.state_runtime import CarriedStateCoordinator

            native_requirements = {
                _native_decode_state_requirements(
                    generated, self._linear_state_pool)
                for generated
                in self._decode_megakernel.state_requirements_by_capacity.values()
            }
            if len(native_requirements) != 1:
                raise RuntimeError(
                    "generated decode capacities imply different native state forms")
            self._native_decode_state_requirements = next(
                iter(native_requirements))
            self._decode_state_coordinator = CarriedStateCoordinator(
                buffers=self._decode_megakernel.state_buffers,
                rows=range(self.max_batch_slots),
                transitions={
                    "gdn_recurrent_state": (
                        self._linear_state_pool.transition_recurrent_form
                    ),
                },
            )

        if self._use_cuda_graphs:
            self._decode_graphs.ensure_ready(self._decode_slots)

        self._maybe_init_spec_decode()

    def _maybe_init_spec_decode(self) -> None:
        """Build the spec-decode capability when a drafter is configured.

        The drafter is a kestrel.models.qwen35 concern (the gated DFlash checkpoint),
        so it is configured via a ``spec_decode`` dict on the runtime config (or
        a pre-built drafter passed in) rather than the engine's shared
        ``RuntimeConfig`` schema. When unset, ``self.spec`` stays ``None`` and
        the runtime decodes one token per step exactly as before.

        Config (``cfg.spec_decode``), all optional except a drafter source:
          * ``drafter`` / ``drafter_repo``: a built ``DFlashDraftModel`` (+ its
            ``DFlashConfig`` via ``drafter_config``), or a HF repo id to load.
          * ``flush_cap`` (default 32), ``max_seq_len`` (default
            ``max_seq_length``).
        """
        spec_cfg = getattr(self._cfg, "spec_decode", None)
        if not spec_cfg:
            return
        from .dflash import (
            DFlashConfig,
            DFlashDraftModel,
            SpecStepRunner,
            load_dflash_drafter,
        )
        from kestrel.runtime.spec import SpecDecodeCaps

        drafter = spec_cfg.get("drafter")
        dcfg = spec_cfg.get("drafter_config")
        if drafter is None:
            repo = spec_cfg.get("drafter_repo")
            if repo is None:
                raise ValueError("spec_decode config needs 'drafter' or 'drafter_repo'")
            drafter, dcfg = load_dflash_drafter(
                repo, device=self.device, dtype=self.dtype
            )
        if not isinstance(drafter, DFlashDraftModel):
            raise TypeError("spec_decode 'drafter' must be a DFlashDraftModel")
        if dcfg is None:
            raise ValueError("spec_decode config needs 'drafter_config' for a prebuilt drafter")

        flush_cap = self._spec_flush_cap(spec_cfg)
        # ``SpecRunner`` reserves ``max_seq_len`` KV pages for EACH of the
        # ``max_batch_size`` fixed spec rows at construction and never frees them
        # (the rows must be address-stable for graph capture). Defaulting this to
        # the model context (``max_seq_length``) would reserve
        # ``max_batch_size * max_position_embeddings`` pages up front, which on
        # large-context models exceeds the serving KV page budget and makes
        # ``page_table.reserve`` raise (or starve the pool) even for short
        # prompts. The non-spec path only reserves each request's actual target
        # length, so default the spec budget to the per-row share of the serving
        # KV pages instead of the model maximum (still overridable explicitly).
        explicit_max = spec_cfg.get("max_seq_len")
        if explicit_max is not None:
            max_seq_len = int(explicit_max)
        else:
            max_seq_len = self._default_spec_max_seq_len(flush_cap)

        # The persistent GDN pool must be sized at flush_cap so the verify
        # kernel + flush shapes agree. ``__init__`` already resized it to this
        # capacity BEFORE the decode-graph capture (see ``_resize_linear_pool_for_spec``);
        # re-run it here (idempotent) so the standalone / no-graph path is sized too.
        self._resize_linear_pool_for_spec(flush_cap)

        runner = SpecStepRunner(
            self,
            drafter,
            dcfg,
            batch_size=self.max_batch_size,
            max_seq_len=max_seq_len,
            flush_cap=flush_cap,
            use_graphs=bool(spec_cfg.get("use_graphs", True)),
            # Capture the draft graph as per-position LOGITS so the runtime spec
            # path serves BOTH greedy and sampled (``temperature > 0``) requests.
            # The default ``sampling=False`` captures greedy draft *tokens*, so a
            # sampled request admitted through this runner would hit ``step()``'s
            # ``NotImplementedError`` (no logits to run rejection sampling from).
            # ``sampling=True`` is a superset: greedy rows still take the exact
            # argmax/accept path, so this keeps greedy bit-exact while unlocking
            # non-greedy spec decode end-to-end. Overridable via the spec config.
            sampling=bool(spec_cfg.get("sampling", True)),
        )
        self._spec_runner = runner
        self.spec = SpecDecodeCaps(
            proposer=runner.proposer,
            capture_hidden_layers=tuple(dcfg.target_layer_ids),
            decoder=runner,
        )

    def _default_spec_max_seq_len(self, flush_cap: int) -> int:
        """Per-row spec KV reservation derived from the serving KV budget.

        ``SpecRunner`` reserves this many pages for every one of the
        ``max_batch_size`` fixed rows up front and holds them for the life of the
        runner. To keep that total within the serving KV page pool (rather than
        the model's full context), size each row to its share of the available
        pages, clamped to the model context above and to ``flush_cap`` below (a
        row must at least span one GDN flush window).

        Crucially, the share is taken over ``max_batch_size + 1`` slots, not
        ``max_batch_size``: the serving ``admit`` contract starts each request as
        a *transient* prefill ``batch_idx`` that ``prepare_sequence``
        ``page_table.reserve``s its own pages for (up to ``target_length`` <=
        ``max_seq_len`` pages) BEFORE ``admit`` re-points ``state.batch_idx`` at a
        persistent spec row and erases that transient row. If the ``B`` persistent
        rows already claimed the whole pool (minus a 2-page margin), that
        transient reservation -- and hence ``prepare_sequence`` / ``can_reserve``
        for an ordinary prompt -- would fail / starve the pool even though the
        request will fit once admitted. Reserving an extra row-sized share
        (``// (batch + 1)``) leaves headroom for exactly one in-flight transient
        prefill (the scheduler admits one sequence at a time, erasing its
        transient slot before preparing the next), so admission succeeds at pool
        capacity. ``margin`` keeps the couple of pages for the padding row / page
        0 that are already reserved.
        """
        batch = max(1, int(self.max_batch_size))
        # Pages already spoken for before the spec runner reserves: page 0 (never
        # handed out) + the padding batch row. Leave a small headroom so the spec
        # reservation cannot consume the very last pages of the pool.
        reserved_margin = 2
        usable_pages = max(0, int(self._kv_cache_pages) - reserved_margin)
        # Divide among B persistent rows PLUS one transient prefill slot. Taking
        # the floor share over ``B + 1`` is exactly "reserve one row-sized slot
        # (== max_seq_len == per_row_pages) for the in-flight transient, split the
        # rest among the B held rows": after the B rows each claim
        # ``per_row_pages``, the leftover ``usable - B*per_row_pages`` is provably
        # ``>= per_row_pages`` (since ``per_row_pages <= usable/(B+1)``), so a
        # transient whose reservation is itself an *admissible* request -- one that
        # will fit a spec row, i.e. ``target_length <= max_seq_len ==
        # per_row_tokens`` -- always fits alongside the B held rows at admit time.
        # (A request larger than ``max_seq_len`` cannot occupy a spec row at all
        # and is rejected at admit by ``_prefill_row``.)
        per_row_pages = usable_pages // (batch + 1)
        per_row_tokens = per_row_pages * int(self.page_size)
        # A spec row must hold at least one flush window; if the pool is too small
        # to give every row even that, fall back to the flush cap and let
        # ``SpecRunner`` / ``page_table.reserve`` surface the real shortage.
        budget = max(int(flush_cap), per_row_tokens)
        return min(int(self.max_seq_length), budget)

    @staticmethod
    def _spec_flush_cap(spec_cfg: Any) -> int:
        """The GDN replay-ring capacity for the spec runner (default 32).

        Read in two places that must agree: the early ``__init__`` pre-pass that
        resizes the linear-state pool *before* the decode graph is captured, and
        the later ``_maybe_init_spec_decode`` that builds the runner.
        """
        return int(spec_cfg.get("flush_cap", 32))

    def _resize_linear_pool_for_spec(self, flush_cap: int) -> None:
        """Resize the persistent GDN (ReplaySSM) pool to ``flush_cap`` capacity.

        The spec verify kernel + flush/reset shapes are keyed to the replay-ring
        capacity, so the pool must be (re)allocated at ``flush_cap`` (the default
        ``linear_replay_capacity`` is small, e.g. 16). This frees the old replay
        tensors and re-initializes them at the new capacity, which CHANGES their
        device addresses and shapes.

        It is therefore called from ``__init__`` BEFORE
        ``_decode_graphs.ensure_ready`` captures the normal ``decode_with_slot``
        graph: that graph binds the linear pool's replay tensors
        (``_prepare_decode_slot`` -> ``bind_to_cache``), so capturing it against
        the small default buffers and then resizing here would leave any later
        fallback to ``decode_with_slot`` replaying kernels bound to freed/stale
        replay-buffer addresses. Sizing the pool first makes the captured graph
        bind the final ``flush_cap`` tensors.

        Idempotent: ``__init__`` runs this BEFORE the decode-graph capture and
        ``_maybe_init_spec_decode`` runs it AGAIN afterwards (for the no-graph /
        standalone path). The second call must NOT drop and reallocate the
        replay tensors -- that would change their device addresses out from under
        the already-captured ``decode_with_slot`` graph, exactly the breakage the
        pre-capture resize exists to avoid. So when the pool is already at the
        requested ``flush_cap``, skip the drop/realloc entirely and return.
        """
        flush_cap = int(flush_cap)
        if int(self._linear_state_pool.replay_capacity) == flush_cap:
            # Already sized (e.g. by the pre-capture ``__init__`` call); the
            # tensors the captured graph bound must stay put.
            return
        text_cfg = _text_config(self.hf_config)
        text_cfg.linear_replay_capacity = flush_cap
        self._linear_state_pool.replay_capacity = flush_cap
        for st in self._linear_state_pool.layers:
            if st is not None:
                st.replay_checkpoint_states = None
                st.replay_k = st.replay_u = st.replay_g = st.replay_lengths = None
        self._linear_state_pool.initialize_from_config(text_cfg, dtype=self.dtype)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def cuda_graphs_enabled(self) -> bool:
        return self._use_cuda_graphs

    @property
    def kv_pool(self) -> KVMemoryPool:
        return self._kv_pool

    @property
    def compute_stream(self) -> torch.cuda.Stream | None:
        return self.primary_stream

    def skills(self) -> Any:
        from .skills import build_skill_registry

        return build_skill_registry()

    def _load_model(self, repo_id: str) -> nn.Module:
        attn_impl = os.environ.get("KESTREL_QWEN35_ATTN_IMPL", "sdpa")
        return load_qwen35_model(
            repo_id,
            device=self.device,
            dtype=self.dtype,
            attn_implementation=attn_impl,
        )

    def preprocess_image_async(self, image: Any) -> Future[QwenImageInputs]:
        return self._image_preprocessor.submit(image)

    def shutdown_image_preprocessor(self) -> None:
        self._image_preprocessor.shutdown(wait=True)

    def shutdown(self) -> None:
        self.shutdown_image_preprocessor()

    def _admission_max_seq_len(self) -> int:
        """Largest total sequence length a request may be admitted with.

        Normally the model context (``max_seq_length``). When a spec runner is
        active it reserved each fixed spec row for only ``SpecRunner.max_seq_len``
        tokens (which may be set BELOW the model context, e.g. an explicit
        ``spec_decode.max_seq_len`` or the per-row KV-budget share). ``admit``
        re-points the sequence onto one of those fixed rows, so a request whose
        prompt+generation exceeds that row reservation cannot be served: ``step()``
        would build slot mappings PAST the reserved pages. Cap admission to the
        spec row length so such requests are rejected up front rather than
        corrupting KV at decode time.

        The cap further reserves ``block_size + 4`` tokens of verify headroom below
        ``SpecRunner.max_seq_len``: ``step()`` verifies a full ``block_size`` from
        the current KV cursor, so the final macro-step of a request whose
        ``target_length == max_seq_len`` starts with the pending token at
        ``ctx_len == max_seq_len - 1`` and builds slot mappings through
        ``ctx_len + block_size - 1`` -- past the reserved pages. Admitting only up
        to ``max_seq_len - (block_size + 4)`` leaves room for that final block,
        matching the ``+ block_size + 4`` budget ``decode_batch`` / ``_prefill_row``
        already enforce on the prefill path."""
        limit = int(self.max_seq_length)
        runner = self._spec_runner
        if runner is not None:
            headroom = int(runner.block_size) + 4
            limit = min(limit, max(0, int(runner.max_seq_len) - headroom))
        return limit

    def can_reserve(self, total_length: int) -> bool:
        return (
            total_length <= self._admission_max_seq_len()
            and self.page_table.can_reserve_with_eviction(total_length)
            and self._available_batch_slots() > 0
        )

    def prefill_budget(self) -> tuple[int, int]:
        return (self.page_table.pages_available, self._available_batch_slots())

    def _available_batch_slots(self) -> int:
        active_headroom = max(0, self.max_batch_size - len(self.active_sequences))
        return min(active_headroom, len(self.page_table.free_batch_idx))

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

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int:
        raise NotImplementedError("LoRA adapters not supported on Qwen35Runtime yet")

    def release_adapter_slot(self, slot: int) -> None:
        raise NotImplementedError("LoRA adapters not supported on Qwen35Runtime yet")

    def classify_prefill(
        self,
        prompt_tokens: Sequence[Any],
        *,
        has_image: bool = False,
        image_hash: Optional[bytes] = None,
        adapter_id: Optional[str] = None,
    ) -> Any:
        from kestrel.runtime.state import PrefillClassification

        return PrefillClassification(
            prompt_length=len(prompt_tokens),
            skip_positions=0,
            can_reuse=False,
            use_prefix_attn=False,
        )

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Any],
        *,
        image: Optional[Any] = None,
        image_crops: Optional[QwenImageInputs] = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
        image_hash: Optional[bytes] = None,
        adapter_id: Optional[str] = None,
    ) -> Any:
        from kestrel.runtime.tokens import TextToken
        from kestrel.runtime.tokens import ImageMarker
        from kestrel.runtime.state import (
            PreparedSequence,
            SequenceState,
            _CacheLookupResult,
        )

        tokens_list = list(prompt_tokens)
        text_only_len = len(tokens_list)
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
                getattr(self.hf_config.vision_config, "spatial_merge_size", 2)
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

        prompt_len = len(tokens_list)
        budget_for_finalize = (
            text_only_len
            + (self.image_prefix_length if image is not None else 0)
            + (max_new_tokens or 128)
        )
        actual_kv_budget = prompt_len + (max_new_tokens or 128)
        target_length = max(budget_for_finalize, actual_kv_budget)
        admit_limit = self._admission_max_seq_len()
        if target_length > admit_limit:
            # When a spec runner is active ``admit_limit`` is the fixed spec row
            # reservation (``SpecRunner.max_seq_len``), which may be below the
            # model context; rejecting here keeps ``step()`` from writing KV past
            # the row's reserved pages after the sequence is moved onto it.
            raise ValueError(
                f"Requested length {target_length} exceeds "
                f"max_seq_length={admit_limit}"
            )
        if self._available_batch_slots() <= 0:
            raise RuntimeError("Cannot reserve Qwen batch slot")
        batch_idx = self.page_table.allocate()
        try:
            self.page_table.reserve(batch_idx, target_length)
        except Exception:
            self.page_table.erase(batch_idx, 0)
            raise
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=(num_image_tokens + 2 * num_images) if image is not None else 0,
            last_hidden=None,
            lora_slot=lora_slot,
            cache_tokens=None,
            cache_lock_node=None,
            cache_owned_page_count=0,
            reused_page_count=0,
        )
        cache_result = _CacheLookupResult(
            match=None,
            skip_positions=0,
            temp_lock_node=None,
            can_reuse=False,
            namespace=None,
        )
        if chat_crops is not None:
            self._chat_image_crops[int(batch_idx)] = chat_crops
        return PreparedSequence(
            state=state,
            tokens_list=tokens_list,
            cache_tokens=[],
            cache_result=cache_result,
            adapter_id=adapter_id,
            image_hash=image_hash,
        )

    @torch.inference_mode()
    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[Any],
        prefill_slot: Any,
        *,
        images: Optional[Sequence[Any]] = None,
        image_crops_list: Optional[Sequence[Optional[QwenImageInputs]]] = None,
    ) -> torch.Tensor:
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
                host_batch_indices=batch_indices,
            )

            for row, prepared in enumerate(prepared_sequences):
                prepared.state.last_hidden = hidden_rows[row].detach()

            return self.model.lm_head(hidden_rows)
        finally:
            self._record_prefill_slot_done(prefill_slot)

    def finalize_prepared_sequence_after_prefill(self, prepared: Any) -> None:
        self.active_sequences[prepared.state.batch_idx] = prepared.state
        return None

    def abort_prepared_sequence(self, prepared: Any) -> None:
        self._release_batch_idx(prepared.state.batch_idx)

    def retain_sequence_prefix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def release_sequence(self, state: Any) -> None:
        self._release_batch_idx(state.batch_idx)

    def _release_batch_idx(self, batch_idx: int) -> None:
        # A spec-admitted sequence has had ``state.batch_idx`` re-pointed by
        # ``SpecRunner.admit`` at one of the runner's FIXED persistent spec rows --
        # rows reserved ONCE at runner construction whose page-table addresses + GDN
        # pool slots are captured into the spec CUDA graphs (so they must stay
        # allocated and address-stable for the runner's lifetime). The spec runner's
        # own ``retire()`` is the authoritative cleanup for those rows (it returns the
        # graph row to the runner's free list and resets the row's mask/cursor/GDN
        # state); it deliberately does NOT free the batch_idx. So when the normal
        # ``release_sequence`` cleanup later runs for that same ``state``, skip it
        # entirely for a persistent spec row: ``page_table.erase`` would free the
        # fixed row back to the pool (letting it be reallocated under the live spec
        # graphs -> page-table corruption), and ``_clear_decode_state`` would clobber
        # that row's GDN linear state + RoPE deltas in the shared ``_linear_state_pool``
        # (addressed by the same batch_idx). Mirrors the ``-1`` sentinel guard in
        # ``admit``: a batch_idx the runtime does not own must not be erased.
        runner = self._spec_runner
        if runner is not None and int(batch_idx) in runner._persistent_batch_idx:
            return
        self.active_sequences.pop(batch_idx, None)
        self._caches.pop(batch_idx, None)
        self._clear_decode_state(batch_idx)
        if batch_idx not in self.page_table.free_batch_idx:
            self.page_table.erase(batch_idx, 0)

    @torch.inference_mode()
    def decode_with_slot(self, slot: Qwen35DecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        megakernel = getattr(self, "_decode_megakernel", None)
        coordinator = getattr(self, "_decode_state_coordinator", None)
        rows = (
            tuple(
                int(row)
                for row in slot.meta.batch_idx.cpu[:batch_size].tolist()
            )
            if coordinator is not None
            else ()
        )
        if megakernel is not None and megakernel.supports(batch_size):
            stream = getattr(slot, "compute_stream", None)
            stream_context = (
                torch.cuda.stream(stream) if stream is not None else nullcontext())
            with stream_context:
                if coordinator is not None:
                    coordinator.prepare(
                        megakernel.state_requirements_for(batch_size), rows)
                megakernel.run(slot, batch_size)
            return
        if coordinator is not None:
            stream = getattr(slot, "compute_stream", None)
            stream_context = (
                torch.cuda.stream(stream) if stream is not None else nullcontext())
            with stream_context:
                coordinator.prepare(
                    self._native_decode_state_requirements, rows)
        self._decode_graphs.run(slot, batch_size)

    def _zero_decode_graph_padding(
        self,
        slot: Qwen35DecodeSlot,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        padding_batch_idx = int(self._padding_batch_idx)
        slot.decode_token_ids[batch_size:graph_batch_size].zero_()
        slot.meta.batch_idx.gpu[batch_size:graph_batch_size].fill_(padding_batch_idx)
        slot.meta.batch_idx.cpu[batch_size:graph_batch_size].fill_(padding_batch_idx)
        slot.meta.input_pos.gpu[batch_size:graph_batch_size].zero_()
        slot.meta.input_pos.cpu[batch_size:graph_batch_size].zero_()
        slot.meta.lora_slot_ids.gpu[batch_size:graph_batch_size].zero_()
        slot.meta.lora_slot_ids.cpu[batch_size:graph_batch_size].zero_()

    def _zero_decode_graph_capture_buffers(self, slot: Qwen35DecodeSlot) -> None:
        text_cfg = _text_config(self.hf_config)
        self._linear_state_pool.initialize_from_config(text_cfg, dtype=self.dtype)
        self._linear_state_pool.zero_all()
        self._decode_rope_deltas.zero_()
        slot.decode_token_ids.zero_()
        slot.meta.batch_idx.gpu.zero_()
        slot.meta.batch_idx.cpu.zero_()
        slot.meta.input_pos.gpu.zero_()
        slot.meta.input_pos.cpu.zero_()
        slot.meta.lora_slot_ids.gpu.zero_()
        slot.meta.lora_slot_ids.cpu.zero_()
        slot.paged_kv_page_table.zero_()
        slot.paged_kv_seqlens_k.zero_()
        slot.slot_mapping.zero_()
        slot.cache_position_ids.zero_()
        slot.position_ids.zero_()
        slot.rope_deltas.zero_()
        slot.sampled_ids.zero_()
        slot.sampled_logprobs.zero_()
        slot.logits.zero_()
        slot.hidden_last.zero_()

    def _prepare_decode_slot(self, slot: Qwen35DecodeSlot, batch_size: int) -> None:
        # Pure-Python step: bind the cache's linear (GDN) layers to the
        # runtime-owned persistent state. All GPU metadata prep lives in
        # ``_build_decode_metadata`` and runs inside ``_run_decode_forward`` so
        # it is captured by the decode CUDA graph (one cudaGraphLaunch) instead
        # of relaunching the per-step copy/index/fill kernels eagerly. The
        # captured kernels read the same fixed slot buffers (batch_idx/input_pos
        # written by the engine before replay; the page table is pre-reserved at
        # prefill and static during decode), so results are bit-identical.
        cache = self._decode_cache_for_slot(slot)
        self._linear_state_pool.bind_to_cache(cache)

    def _build_decode_metadata(
        self, slot: Qwen35DecodeSlot, batch_size: int
    ) -> None:
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]
        input_pos = slot.meta.input_pos.gpu[:batch_size]
        slot.cache_position_ids[:batch_size, 0].copy_(input_pos)
        self.page_table.populate_paged_kv_metadata(
            batch_idx=batch_idx,
            input_pos=input_pos,
            out_page_table=slot.paged_kv_page_table[:batch_size],
            out_seqused_k=slot.paged_kv_seqlens_k[:batch_size],
        )
        slot.slot_mapping[:batch_size].copy_(
            self.page_table.build_slot_mapping(
                batch_idx=batch_idx,
                positions=slot.cache_position_ids[:batch_size],
            )
        )
        self._gather_decode_rope_deltas(slot, batch_size)
        self._prepare_decode_position_ids(slot, batch_size)

    def _gather_decode_rope_deltas(
        self,
        slot: Qwen35DecodeSlot,
        batch_size: int,
    ) -> None:
        torch.index_select(
            self._decode_rope_deltas,
            0,
            slot.meta.batch_idx.gpu[:batch_size],
            out=slot.rope_deltas[:batch_size],
        )

    def _prepare_decode_position_ids(
        self,
        slot: Qwen35DecodeSlot,
        batch_size: int,
    ) -> None:
        # Row 0 carries the text position; the three spatial M-RoPE rows carry
        # the same position offset by the per-sequence rope delta. Broadcast the
        # text position into all four rows, then add the delta into the spatial
        # rows in place. Two kernels, no temporary allocation.
        cache_position_ids = slot.cache_position_ids[:batch_size]
        position_ids = slot.position_ids[:, :batch_size, :]
        position_ids.copy_(cache_position_ids)
        position_ids[1:].add_(slot.rope_deltas[:batch_size])

    def _run_decode_forward(self, slot: Qwen35DecodeSlot, batch_size: int) -> None:
        self._build_decode_metadata(slot, batch_size)
        cache = self._decode_cache_for_slot(slot)
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]
        cache_position_ids = slot.cache_position_ids[:batch_size]
        input_ids = slot.decode_token_ids[:batch_size].view(-1, 1)
        self.model.model.rope_deltas = None
        outputs = self.model.model(
            input_ids=input_ids,
            past_key_values=cache,
            position_ids=slot.position_ids[:, :batch_size, :],
            cache_position_ids=cache_position_ids,
            slot_mapping=slot.slot_mapping[:batch_size],
            page_table=slot.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
            gdn_state_indices=batch_idx,
        )
        last_hidden = outputs.last_hidden_state
        hidden = last_hidden[:, 0, :]
        slot.hidden_last[:batch_size].copy_(hidden)
        slot.logits[:batch_size].copy_(self.model.lm_head(hidden))

    def _host_long_tensor(
        self,
        values: Sequence[int] | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        first = values[0] if values else ()
        rows = (
            [list(row) for row in values]
            if isinstance(first, (list, tuple))
            else [list(values)]  # type: ignore[list-item]
        )
        row_count = len(rows)
        col_count = len(rows[0]) if rows else 0
        cpu = torch.empty(
            (row_count, col_count),
            dtype=torch.long,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        for row_idx, row in enumerate(rows):
            if len(row) != col_count:
                raise ValueError("Qwen host tensor rows must have equal length")
            for col_idx, value in enumerate(row):
                cpu[row_idx, col_idx] = int(value)
        return cpu.to(device=self.device, non_blocking=True)

    def _device_long_scalar(self, shape: tuple[int, ...], value: int) -> torch.Tensor:
        out = torch.empty(shape, dtype=torch.long, device=self.device)
        out.fill_(int(value))
        return out

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        image: Optional[Any] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning: bool = False,
    ) -> str:
        query = self.prompt_template.query()
        if query is None:
            raise RuntimeError("Qwen 3.5 prompt template missing query()")
        user_ids = self.tokenizer.encode(prompt).ids
        opener = query.reasoning_prefix if reasoning else query.answer_prefix
        ids = [self.prompt_template.bos_id] + list(query.prefix) + user_ids + list(opener)
        image_inputs = self._image_preprocessor.process(image) if image is not None else None
        if image_inputs is not None:
            offset = 1 + len(query.prefix)
            image_block = (
                [VISION_START_ID]
                + [IMAGE_PAD_ID] * int(image_inputs.num_image_tokens)
                + [VISION_END_ID]
            )
            ids = ids[:offset] + image_block + ids[offset:]

        batch_idx = self.page_table.allocate()
        cache = self._new_cache()
        try:
            self.page_table.reserve(batch_idx, len(ids) + max_new_tokens)
            self.page_table.commit_block_table([batch_idx])
            input_ids = self._host_long_tensor([ids])
            image_kwargs = self._image_forward_kwargs(input_ids, image_inputs)
            cache_position_ids = torch.arange(
                len(ids), dtype=torch.long, device=self.device
            ).view(1, -1)
            last_hidden, cache = self._forward_base(
                input_ids=input_ids,
                past_key_values=cache,
                batch_idx=batch_idx,
                cache_position_ids=cache_position_ids,
                **image_kwargs,
            )
            logits = self._logits_for_last(last_hidden[0, -1])
            generated: list[int] = []
            for _ in range(max_new_tokens):
                next_id = self._sample_next(logits, temperature=temperature, top_p=top_p)
                if next_id in self._eos_ids:
                    break
                generated.append(next_id)
                step_ids = self._device_long_scalar((1, 1), next_id)
                pos = self._device_long_scalar(
                    (1, 1),
                    int(cache.past_key_values.get_seq_length()),
                )
                last_hidden, cache = self._forward_base(
                    input_ids=step_ids,
                    past_key_values=cache,
                    batch_idx=batch_idx,
                    cache_position_ids=pos,
                )
                logits = self._logits_for_last(last_hidden[0, 0])
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        finally:
            self.page_table.erase(batch_idx, 0)

    def _image_forward_kwargs(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[QwenImageInputs],
    ) -> dict[str, torch.Tensor]:
        if image_inputs is None:
            return {}
        pixel_values = image_inputs.pixel_values.to(
            device=self.device, dtype=self.dtype
        )
        image_grid_thw = image_inputs.image_grid_thw.to(device=self.device)
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[input_ids == IMAGE_PAD_ID] = 1
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

    def _prefill_scratch_for(
        self,
        slot: Any,
        *,
        total_tokens: int,
        batch_size: int,
        pixel_rows: int,
        pixel_dim: int,
        image_grid_rows: int,
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
            image_grid_rows=image_grid_rows,
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
    ) -> int:
        spatial_merge_size = int(
            getattr(self.hf_config.vision_config, "spatial_merge_size", 2)
        )
        cursor = start
        current_pos = 0
        image_idx = 0
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
        return int(out[:, 0, start:end].max()) + 1 - (end - start)

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

        model = getattr(self, "model", None)
        visual = getattr(getattr(model, "model", None), "visual", None)
        if visual is not None:
            num_grid_per_side = int(visual.num_grid_per_side)
            spatial_merge_size = int(visual.config.spatial_merge_size)
        else:
            vision_config = self.hf_config.vision_config
            num_positions = int(getattr(vision_config, "num_position_embeddings", 2304))
            num_grid_per_side = int(num_positions**0.5)
            spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))


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
        image_grid_rows: int,
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
            if scratch.image_grid_thw is None:
                raise RuntimeError("Qwen image grid scratch was not allocated")
            scratch.image_grid_thw.copy_to_gpu(image_grid_rows)
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
        image_grid_rows = 0
        vision_sequence_count = 0
        has_images = False

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
                image_grid_rows += int(grid_np.shape[0])
                vision_sequence_count += int(grid_np[:, 0].sum())
            else:
                crop_grid_rows.append(None)

        scratch = self._prefill_scratch_for(
            prefill_slot,
            total_tokens=total_tokens,
            batch_size=len(prepared_sequences),
            pixel_rows=pixel_rows,
            pixel_dim=pixel_dim,
            image_grid_rows=image_grid_rows,
            vision_sequence_count=vision_sequence_count,
        )

        offset = 0
        pixel_offset = 0
        grid_offset = 0
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
                grid_end = grid_offset + grid_np.shape[0]
                if scratch.image_grid_thw is None:
                    raise RuntimeError("Qwen image grid scratch was not allocated")
                scratch.image_grid_thw.np[grid_offset:grid_end] = grid_np
                scratch.text_meta.rope_deltas.np[row, 0] = self._fill_multimodal_position_ids(
                    scratch.text_meta.position_ids.np,
                    start=offset,
                    end=end,
                    mm_token_type_ids=mm_types,
                    image_grid_thw=grid_np,
                )
                pixel_end = pixel_offset + int(crops.pixel_values.shape[0])
                if scratch.pixel_values is None:
                    raise RuntimeError("Qwen pixel scratch was not allocated")
                scratch.pixel_values.cpu[pixel_offset:pixel_end].copy_(
                    crops.pixel_values
                )
                pixel_offset = pixel_end
                grid_offset = grid_end
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
            image_grid_rows=image_grid_rows,
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

        return _PackedPrefillBatch(
            input_ids=scratch.text_meta.input_ids.gpu[:, :total_tokens],
            cache_position_ids=scratch.text_meta.cache_position_ids.gpu[:, :total_tokens],
            position_ids=scratch.text_meta.position_ids.gpu[:, :, :total_tokens],
            # Always pass packed metadata, even for a single sequence: text_meta
            # is staged to the GPU unconditionally, so cu_seq_lens_q ([0, total])
            # and seq_idx are valid for batch_size == 1. This lets a single
            # sequence take the same packed_prefill path as batched prefill
            # instead of the separate uniform_native_prefill branch.
            cu_seq_lens_q=scratch.text_meta.cu_seq_lens_q.gpu[: batch_size + 1],
            seq_idx=scratch.text_meta.seq_idx.gpu[:, :total_tokens],
            batch_indices=scratch.text_meta.batch_indices.gpu[:batch_size],
            max_length=max(lengths),
            last_token_offsets=scratch.text_meta.last_token_offsets.gpu[:batch_size],
            paged_kv_page_table=scratch.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=scratch.paged_kv_seqlens_k[:batch_size],
            slot_mapping=scratch.text_meta.slot_mapping.gpu[:, :total_tokens],
            rope_deltas=scratch.text_meta.rope_deltas.gpu[:batch_size],
            pixel_values=(
                scratch.pixel_values.gpu[:pixel_rows]
                if has_images and scratch.pixel_values is not None
                else None
            ),
            image_grid_thw=(
                scratch.image_grid_thw.gpu[:image_grid_rows]
                if has_images and scratch.image_grid_thw is not None
                else None
            ),
            mm_token_type_ids=(
                scratch.text_meta.mm_token_type_ids.gpu[:, :total_tokens] if has_images else None
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
    ) -> tuple[torch.Tensor, _QwenForwardCache]:
        cache = self._new_cache()
        self.model.model.rope_deltas = None
        outputs = self.model.model(
            input_ids=packed.input_ids,
            past_key_values=cache,
            pixel_values=packed.pixel_values,
            image_grid_thw=packed.image_grid_thw,
            mm_token_type_ids=packed.mm_token_type_ids,
            position_ids=packed.position_ids,
            vision_bilinear_indices=packed.vision_bilinear_indices,
            vision_bilinear_weights=packed.vision_bilinear_weights,
            vision_position_ids=packed.vision_position_ids,
            vision_cu_seqlens=packed.vision_cu_seqlens,
            cache_position_ids=packed.cache_position_ids,
            slot_mapping=packed.slot_mapping,
            page_table=packed.paged_kv_page_table,
            paged_kv_seqlens_k=packed.paged_kv_seqlens_k,
            cu_seq_lens_q=packed.cu_seq_lens_q,
            seq_idx=packed.seq_idx,
        )
        last_hidden = outputs.last_hidden_state
        if hasattr(outputs.past_key_values, "advance_to"):
            outputs.past_key_values.advance_to(packed.max_length)
        return last_hidden, _QwenForwardCache(
            outputs.past_key_values,
            packed.rope_deltas,
        )

    def _forward_base(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any],
        batch_idx: Optional[int | Sequence[int] | torch.Tensor] = None,
        cache_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        spec_verify: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        cache_state = (
            past_key_values
            if isinstance(past_key_values, _QwenForwardCache)
            else _QwenForwardCache(past_key_values)
        )
        if batch_idx is not None:
            if cache_state.past_key_values is None:
                cache_state.past_key_values = self._new_cache()
            cache_batch_idx = self._cache_batch_idx_tensor(batch_idx)
            batch_count = int(cache_batch_idx.shape[0])
            if input_ids.shape[0] != batch_count:
                raise ValueError(
                    "input_ids batch dimension must match cache batch indices"
                )
            if cache_position_ids is None:
                if batch_count != 1:
                    raise RuntimeError(
                        "Batched Qwen decode requires explicit cache positions"
                    )
                start = int(cache_state.past_key_values.get_seq_length())
                cache_position_ids = torch.arange(
                    start,
                    start + input_ids.shape[1],
                    dtype=torch.long,
                    device=self.device,
                ).view(1, -1)
            cache_position_ids = cache_position_ids.to(
                device=self.device, dtype=torch.long
            )
            if cache_position_ids.ndim == 1:
                cache_position_ids = cache_position_ids.view(batch_count, -1)
            if cache_position_ids.shape[0] != batch_count:
                raise ValueError(
                    "cache_position_ids batch dimension must match batch_idx"
                )
            slot_mapping = self.page_table.build_slot_mapping(
                batch_idx=cache_batch_idx,
                positions=cache_position_ids,
            )
            paged_kv_page_table = torch.index_select(
                self.page_table.page_table,
                0,
                cache_batch_idx.to(dtype=torch.long),
            )
            paged_kv_seqlens_k = (
                cache_position_ids.max(dim=1).values.to(dtype=torch.int32) + 1
            )
            gdn_state_indices = self._gdn_state_indices_for_cache(
                cache_state,
                cache_batch_idx=cache_batch_idx,
                batch_count=batch_count,
            )
            cache_kwargs = {
                "cache_position_ids": cache_position_ids,
                "slot_mapping": slot_mapping,
                "page_table": paged_kv_page_table,
                "paged_kv_seqlens_k": paged_kv_seqlens_k,
                "gdn_state_indices": gdn_state_indices,
                "spec_verify": spec_verify,
            }
            # Single-sequence multi-token prefill (e.g. generate()): pass
            # cu_seq_lens_q = [0, seq_len] so GDN takes the native packed_prefill
            # path (the model derives seq_idx from it), matching
            # _forward_packed_prefill. Single-token decode steps skip this and
            # stay on the decode path.
            if input_ids.shape[0] == 1 and input_ids.shape[1] > 1:
                cache_kwargs["cu_seq_lens_q"] = torch.tensor(
                    [0, input_ids.shape[1]],
                    dtype=torch.int32,
                    device=self.device,
                )
        else:
            cache_kwargs = {}
        has_multimodal_inputs = (
            pixel_values is not None
            or image_grid_thw is not None
            or mm_token_type_ids is not None
        )
        use_multimodal_positions = (
            has_multimodal_inputs or cache_state.rope_deltas is not None
        )
        if not use_multimodal_positions:
            outputs = self.model.model.language_model(
                input_ids=input_ids,
                position_ids=cache_position_ids if batch_idx is not None else None,
                past_key_values=cache_state.past_key_values,
                **cache_kwargs,
            )
            rope_deltas = None
        else:
            self.model.model.rope_deltas = cache_state.rope_deltas
            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=cache_state.past_key_values,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                position_ids=position_ids,
                **cache_kwargs,
            )
            rope_deltas = getattr(outputs, "rope_deltas", None)
        last_hidden = outputs.last_hidden_state
        if (
            batch_idx is not None
            and cache_position_ids is not None
            and hasattr(outputs.past_key_values, "advance_to")
        ):
            outputs.past_key_values.advance_to(
                int(cache_position_ids.max().item()) + 1
            )
        return last_hidden, _QwenForwardCache(
            outputs.past_key_values,
            rope_deltas,
            cache_state.linear_state_row_indices,
        )

    def _new_cache(self) -> Qwen35PagedHybridCache:
        return Qwen35PagedHybridCache(
            config=_text_config(self.hf_config),
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
            shared_paged_layers=self._shared_paged_layers,
        )

    def _store_sequence_cache(self, batch_idx: int, cache: Any) -> None:
        cache_state = self._as_forward_cache(cache)
        if not isinstance(cache_state.past_key_values, Qwen35PagedHybridCache):
            raise RuntimeError("Qwen engine decode requires paged hybrid caches")
        self._linear_state_pool.capture_from_cache(
            int(batch_idx),
            cache_state.past_key_values,
        )
        self._mark_decode_state_coherent((int(batch_idx),))
        self._decode_rope_deltas[int(batch_idx)].zero_()
        if cache_state.rope_deltas is not None:
            rope_deltas = cache_state.rope_deltas.to(
                device=self.device,
                dtype=torch.long,
            )
            if rope_deltas.ndim == 1:
                rope_deltas = rope_deltas.view(-1, 1)
            if rope_deltas.shape != (1, 1):
                raise RuntimeError("Qwen M-RoPE delta must have shape [1, 1]")
            self._decode_rope_deltas[int(batch_idx) : int(batch_idx) + 1].copy_(
                rope_deltas
            )
        self._caches[int(batch_idx)] = cache_state

    def _store_packed_sequence_caches(
        self,
        batch_idx: torch.Tensor,
        cache: Any,
        *,
        host_batch_indices: Sequence[int],
    ) -> None:
        cache_state = self._as_forward_cache(cache)
        if not isinstance(cache_state.past_key_values, Qwen35PagedHybridCache):
            raise RuntimeError("Qwen engine decode requires paged hybrid caches")
        indices = batch_idx.to(device=self.device, dtype=torch.long).view(-1)
        batch_size = int(indices.shape[0])
        if len(host_batch_indices) != batch_size:
            raise ValueError("host batch indices must match packed batch size")
        self._linear_state_pool.capture_batch_from_cache(
            indices,
            cache_state.past_key_values,
            batch_size=batch_size,
            # Packed prefill starts from a fresh cache. Each GDN layer has just
            # checkpointed its final recurrent state and reset every replay
            # cursor, so K/U/G payload bytes are unreachable and need not be
            # copied into the persistent decode pool.
            copy_replay_payload=False,
        )
        self._mark_decode_state_coherent(host_batch_indices)
        self._decode_rope_deltas.index_fill_(0, indices, 0)
        rope_deltas = cache_state.rope_deltas
        if rope_deltas is not None:
            rope_deltas = rope_deltas.to(device=self.device, dtype=torch.long)
            if rope_deltas.ndim == 1:
                rope_deltas = rope_deltas.view(-1, 1)
            if rope_deltas.shape != (batch_size, 1):
                raise RuntimeError(
                    "Qwen packed M-RoPE deltas must have shape [batch, 1]"
                )
            self._decode_rope_deltas.index_copy_(0, indices, rope_deltas)

        linear_state_rows = torch.arange(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
        for row, value in enumerate(host_batch_indices):
            self._caches[int(value)] = _QwenForwardCache(
                cache_state.past_key_values,
                None if rope_deltas is None else rope_deltas[row : row + 1],
                linear_state_rows[row : row + 1],
            )

    def _clear_decode_state(self, batch_idx: int) -> None:
        if hasattr(self, "_decode_rope_deltas"):
            self._decode_rope_deltas[int(batch_idx)].zero_()
        if hasattr(self, "_linear_state_pool"):
            self._linear_state_pool.clear(int(batch_idx))
        self._mark_decode_state_coherent((int(batch_idx),))

    def _mark_decode_state_coherent(self, rows: Sequence[int]) -> None:
        coordinator = getattr(self, "_decode_state_coordinator", None)
        if coordinator is not None:
            coordinator.mark_coherent(rows)

    def _decode_cache_for_slot(self, slot: Qwen35DecodeSlot) -> Qwen35PagedHybridCache:
        return self._decode_caches[int(slot.slot_id)]

    def _as_forward_cache(self, cache: Any) -> _QwenForwardCache:
        return (
            cache
            if isinstance(cache, _QwenForwardCache)
            else _QwenForwardCache(cache)
        )

    def _cache_batch_idx_tensor(
        self, batch_idx: int | Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(batch_idx, torch.Tensor):
            return batch_idx.to(device=self.device, dtype=torch.int64).view(-1)
        if isinstance(batch_idx, int):
            values = [batch_idx]
        else:
            values = [int(idx) for idx in batch_idx]
        staging = getattr(self, "_cache_batch_idx_staging", None)
        if staging is None or len(values) > staging.cpu.shape[0]:
            staging = CpuGpuBuffer(
                max(len(values), int(getattr(self, "max_batch_slots", len(values)))),
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )
            self._cache_batch_idx_staging = staging
        if len(values) > staging.cpu.shape[0]:
            raise ValueError(
                f"Qwen batch index staging buffer holds "
                f"{staging.cpu.shape[0]} values, got {len(values)}"
            )
        for idx, value in enumerate(values):
            staging.cpu[idx] = int(value)
        return staging.copy_to_gpu(len(values)).view(-1)

    def _gdn_state_indices_for_cache(
        self,
        cache_state: _QwenForwardCache,
        *,
        cache_batch_idx: torch.Tensor,
        batch_count: int,
    ) -> torch.Tensor:
        if not isinstance(cache_state.past_key_values, Qwen35PagedHybridCache):
            return cache_batch_idx.to(dtype=torch.long)

        if cache_state.linear_state_row_indices is not None:
            indices = cache_state.linear_state_row_indices.to(
                device=self.device,
                dtype=torch.long,
            ).view(-1)
            if int(indices.shape[0]) != batch_count:
                raise ValueError(
                    "linear_state_row_indices batch dimension must match batch_idx"
                )
            return indices

        return torch.arange(batch_count, dtype=torch.long, device=self.device)

    def _logits_for_last(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(last_hidden.unsqueeze(0))[0]

    def _sample_next(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
    ) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits).item())
        logits = logits / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = int((cum <= top_p).sum().item()) + 1
            sorted_probs[keep:] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            pick = torch.multinomial(sorted_probs, 1)
            return int(sorted_idx[pick].item())
        return int(torch.multinomial(probs, 1).item())


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


__all__ = ["Qwen35Runtime", "QwenImageInputs"]
