"""Gemma 4 paged multimodal runtime."""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from kestrel.device import make_event, make_stream, materialize_blas_runtime
from kestrel.kv_cache import (
    KVMemoryPool,
    PageTable,
    PagedKVLayerSpec,
    PagedKVStorage,
    allocate_paged_kv_layers,
    fit_and_allocate_paged_kv_storage,
)
from kestrel.runtime import ExecutionShape, SequenceState, TextToken, Token
from kestrel.runtime.compilation import (
    canonicalize_immutable_scalar_buffers,
    materialize_dynamic_batch_domain,
)
from kestrel.runtime.decode_slot import create_decode_slot
from kestrel.runtime.paged_resources import (
    PrefillSlot,
    bound_kv_cache_pages,
    decode_slot_rows,
)
from kestrel.runtime.preprocessing import derive_image_insertion_offset
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.staging import AsyncPreprocessor, BatchedTensorStager
from kestrel.runtime.tokenizer import load_tokenizer
from kestrel.runtime.uncached_paged import UncachedPagedRuntime

from .image import (
    MAX_IMAGE_TOKENS,
    MAX_PATCHES,
    PATCH_SIZE,
    POOLING_KERNEL_SIZE,
    GemmaImageInputs,
    preprocess_image,
)
from .loader import load_model
from .paged_cache import paged_kv_specs
from .prompt_template import (
    END_OF_IMAGE_ID,
    Gemma4PromptTemplate,
    NEWLINE_ID,
    START_OF_IMAGE_ID,
    TURN_ID,
    USER_ROLE_ID,
)


@dataclass(frozen=True, slots=True)
class _PagedRuntimeResources:
    rope_inputs: dict[str, torch.Tensor] | None
    decode_page_tables: tuple[torch.Tensor, ...]
    page_table: PageTable


@dataclass(frozen=True, slots=True)
class _PageTableSnapshot:
    free_pages: tuple[int, ...]
    free_batch_idx: tuple[int, ...]
    page_table_cpu: tuple[tuple[int, ...], ...]
    capacity: tuple[int, ...]
    num_blocks_per_row: tuple[int, ...]
    cpu_mapping: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ImagePrefillProbeResult:
    logits: torch.Tensor
    image_features: tuple[torch.Tensor, ...]
    prepared_sequences: tuple[Any, ...]


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach cleanup context on every supported Python version."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        setattr(error, "__notes__", notes)
    notes.append(note)


def _generated_kv_binding_inputs(
    layer_specs: Sequence[Any],
    layer_kinds: Sequence[str],
) -> dict[str, tuple[Any | None, ...]]:
    """Describe the exact pre-allocation layer collections for estimation."""

    if len(layer_specs) != len(layer_kinds):
        raise ValueError("Gemma K/V specs and layer kinds must have equal length")
    marker = object()
    inputs = {}
    for suffix, kind in (
        ("local", "sliding_attention"),
        ("global", "full_attention"),
    ):
        layers = tuple(
            marker if spec is not None and actual_kind == kind else None
            for spec, actual_kind in zip(layer_specs, layer_kinds, strict=True)
        )
        inputs[f"mK_{suffix}"] = layers
        inputs[f"mV_{suffix}"] = layers
    return inputs


def _allocate_decode_page_tables(
    *,
    count: int,
    rows: int,
    pages: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Allocate exact per-slot page tables without publishing their pointers."""

    return tuple(
        torch.empty(
            (rows, pages),
            dtype=torch.int32,
            device=device,
        )
        for _index in range(int(count))
    )


def _install_decode_page_tables(
    slots: Sequence[Any], tables: Sequence[torch.Tensor]
) -> None:
    """Publish fitted page tables once, before native or generated binding."""

    if len(slots) != len(tables):
        raise ValueError("decode slots and fitted page tables must have equal length")
    if not tables:
        return
    expected = (int(tables[0].shape[0]), 1)
    if any(tuple(slot.paged_kv_page_table.shape) != expected for slot in slots):
        raise RuntimeError("decode page tables must be unbound one-column placeholders")
    for slot, table in zip(slots, tables, strict=True):
        slot.paged_kv_page_table = table


def _maximum_vision_grid(config: Any) -> tuple[int, int]:
    """Validate the fixed Gemma preprocessing contract and return its grid."""

    pooling = int(config.pooling_kernel_size)
    if int(config.patch_size) != PATCH_SIZE:
        raise ValueError(
            f"Gemma vision patch_size must be {PATCH_SIZE}, got {config.patch_size}"
        )
    if pooling != POOLING_KERNEL_SIZE:
        raise ValueError(
            f"Gemma vision pooling_kernel_size must be {POOLING_KERNEL_SIZE}, "
            f"got {pooling}"
        )
    if MAX_IMAGE_TOKENS != MAX_PATCHES // pooling**2:
        raise RuntimeError("Gemma maximum image-token contract is inconsistent")

    grid_height = 0
    grid_width = 0
    for candidate in range(math.isqrt(MAX_PATCHES), 0, -1):
        if MAX_PATCHES % candidate:
            continue
        other = MAX_PATCHES // candidate
        if candidate % pooling == 0 and other % pooling == 0:
            grid_height, grid_width = candidate, other
            break
    if not grid_height:
        raise RuntimeError(
            "maximum vision patch count has no pooling-aligned rectangular grid"
        )
    position_embedding_size = int(config.position_embedding_size)
    if position_embedding_size < max(grid_height, grid_width):
        raise ValueError(
            "Gemma vision position_embedding_size cannot represent the "
            f"maximum {grid_height}x{grid_width} patch grid"
        )
    return grid_height, grid_width


def _vision_probe_inputs(runtime: Any) -> tuple[GemmaImageInputs, ...]:
    """Build a distinct full-patch image input for every admitted batch row."""

    config = runtime._config.vision_config
    grid_height, grid_width = _maximum_vision_grid(config)

    pixel_values = torch.zeros(
        (MAX_PATCHES, 3 * PATCH_SIZE**2),
        dtype=runtime.dtype,
    )
    position_ids = torch.stack(
        (
            torch.arange(grid_width).repeat(grid_height),
            torch.arange(grid_height).repeat_interleave(grid_width),
        ),
        dim=-1,
    )
    return tuple(
        GemmaImageInputs(
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            num_image_tokens=MAX_IMAGE_TOKENS,
        )
        for _row in range(int(runtime.max_batch_size))
    )


def _run_vision_transient_probe(
    runtime: Any,
    inputs: Sequence[GemmaImageInputs],
) -> torch.Tensor:
    """Run the maximum admitted full vision path through its real stager."""

    records = tuple(
        {
            "pixel_values": image.pixel_values,
            "position_ids": image.image_position_ids,
        }
        for image in inputs
    )
    staged = runtime._vision_stager.stage(records)
    return runtime.model.model.get_image_features(
        staged["pixel_values"],
        staged["position_ids"],
    ).detach()


def _materialize_fixed_runtime_resources(
    runtime: Any,
    vision_inputs: Sequence[GemmaImageInputs],
) -> None:
    """Prime exact scheduler-thread BLAS and maximum vision resources."""

    slot = runtime.decode_slots[0]
    rows = min(int(runtime.max_batch_size), int(slot.hidden_last.shape[0]))

    def warm() -> torch.Tensor:
        torch.mm(
            slot.hidden_last[:rows],
            runtime.model.lm_head.weight.t(),
            out=slot.logits[:rows],
        )
        return _run_vision_transient_probe(runtime, vision_inputs)

    materialize_blas_runtime(
        runtime.device,
        runtime._compute_stream,
        warm,
    )


def _copy_image_features_into_embeddings(
    inputs_embeds: torch.Tensor,
    token_rows: Sequence[Sequence[int]],
    image_features: Sequence[torch.Tensor | None],
    *,
    image_token_id: int,
) -> None:
    """Copy each contiguous image feature block without a packed duplicate."""

    if len(token_rows) != len(image_features):
        raise ValueError("token rows and image features must have equal length")
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("prefill embeddings must have shape [1, tokens, hidden]")

    copies = []
    flat_offset = 0
    for row_index, (token_row, features) in enumerate(
        zip(token_rows, image_features, strict=True)
    ):
        positions = [
            index
            for index, token_id in enumerate(token_row)
            if int(token_id) == int(image_token_id)
        ]
        if features is None:
            if positions:
                raise RuntimeError(
                    f"prefill row {row_index} has image tokens without features"
                )
        else:
            if not isinstance(features, torch.Tensor):
                raise TypeError(f"image features for row {row_index} must be a tensor")
            if not positions:
                raise RuntimeError(
                    f"prefill row {row_index} has image features without tokens"
                )
            start = positions[0]
            if positions != list(range(start, start + len(positions))):
                raise RuntimeError(
                    f"prefill row {row_index} image tokens must be contiguous"
                )
            if features.ndim != 2:
                raise RuntimeError(
                    f"image features for row {row_index} must be 2D, got "
                    f"{features.ndim}D"
                )
            if int(features.shape[0]) != len(positions):
                raise RuntimeError(
                    f"encoded {features.shape[0]} image features for "
                    f"{len(positions)} image tokens in row {row_index}"
                )
            if int(features.shape[1]) != int(inputs_embeds.shape[2]):
                raise RuntimeError(
                    f"image features for row {row_index} have shape "
                    f"{tuple(features.shape)}, expected "
                    f"({len(positions)}, {inputs_embeds.shape[2]})"
                )
            if features.dtype is not inputs_embeds.dtype:
                raise RuntimeError(
                    f"image features for row {row_index} have dtype "
                    f"{features.dtype}, expected {inputs_embeds.dtype}"
                )
            if features.device != inputs_embeds.device:
                raise RuntimeError(
                    f"image features for row {row_index} are on "
                    f"{features.device}, expected {inputs_embeds.device}"
                )
            copies.append(
                (
                    flat_offset + start,
                    flat_offset + start + len(positions),
                    features,
                )
            )
        flat_offset += len(token_row)

    if flat_offset != int(inputs_embeds.shape[1]):
        raise ValueError("token rows do not cover the packed prefill embeddings")
    for start, end, features in copies:
        inputs_embeds[0, start:end].copy_(features)


def _snapshot_page_table(page_table: PageTable) -> _PageTableSnapshot:
    if page_table._dirty_rows:
        raise RuntimeError("transient probe requires a committed candidate page table")
    return _PageTableSnapshot(
        free_pages=tuple(page_table.free_pages),
        free_batch_idx=tuple(page_table.free_batch_idx),
        page_table_cpu=tuple(tuple(row) for row in page_table.page_table_cpu),
        capacity=tuple(int(value) for value in page_table.capacity),
        num_blocks_per_row=tuple(int(value) for value in page_table.num_blocks_per_row),
        cpu_mapping=page_table._page_table_cpu_tensor.clone(),
    )


def _restore_page_table(
    page_table: PageTable,
    snapshot: _PageTableSnapshot,
) -> None:
    page_table.free_pages[:] = snapshot.free_pages
    page_table.free_batch_idx[:] = snapshot.free_batch_idx
    page_table.page_table_cpu[:] = [list(row) for row in snapshot.page_table_cpu]
    page_table.capacity[:] = snapshot.capacity
    page_table.num_blocks_per_row[:] = snapshot.num_blocks_per_row
    page_table._page_table_cpu_tensor.copy_(snapshot.cpu_mapping)
    page_table._dirty_rows.clear()
    page_table._sync_full_page_table()


def _maximum_admitted_image_prefill_prompt(
    runtime: Any,
    page_table: PageTable,
    image_inputs: GemmaImageInputs,
) -> tuple[TextToken, ...]:
    """Build the longest image query admitted by the candidate page budget."""

    image_tokens = int(image_inputs.num_image_tokens) + 2
    if image_tokens != int(runtime.image_prefix_length):
        raise RuntimeError(
            "Gemma capacity probe image length does not match runtime admission"
        )
    target_length = min(
        int(runtime.max_seq_length),
        int(page_table.pages_available) * int(page_table.page_size),
    )
    # Engine generation requests reserve at least one token beyond prefill.
    text_length = target_length - 1 - image_tokens
    required_prefix = (TURN_ID, USER_ROLE_ID, NEWLINE_ID, NEWLINE_ID)
    if text_length < len(required_prefix):
        raise MemoryError(
            "candidate K/V cache cannot admit the minimum Gemma image query"
        )
    filler = (TextToken(token_id=NEWLINE_ID),) * (
        text_length - len(required_prefix)
    )
    return tuple(TextToken(token_id=token_id) for token_id in required_prefix) + filler


def _run_image_prefill_transient_probe(
    runtime: Any,
    vision_inputs: Sequence[GemmaImageInputs],
    storage: PagedKVStorage,
    resources: _PagedRuntimeResources | None,
    layer_specs: Sequence[PagedKVLayerSpec | None],
    *,
    prompt_tokens: Sequence[TextToken] | None = None,
) -> _ImagePrefillProbeResult:
    """Run a production image prefill while candidate K/V resources are live."""

    if resources is None:
        raise RuntimeError("Gemma transient probe requires paged resources")
    if not 0 < len(vision_inputs) <= int(runtime.max_batch_size):
        raise ValueError("Gemma transient probe batch is outside runtime capacity")

    page_table = resources.page_table
    page_table_snapshot = _snapshot_page_table(page_table)
    previous_page_table = getattr(runtime, "page_table", None)
    previous_kv_cache = getattr(runtime, "_kv_cache", None)
    had_page_table = hasattr(runtime, "page_table")
    had_kv_cache = hasattr(runtime, "_kv_cache")
    probe_kv_cache = allocate_paged_kv_layers(
        layer_specs=layer_specs,
        page_table=page_table,
        pool=runtime._kv_pool,
        dtype=runtime.dtype,
        storage=storage,
    )
    runtime.page_table = page_table
    runtime._kv_cache = probe_kv_cache

    prepared_sequences = []
    projected_features: list[torch.Tensor] = []
    touched_pages: set[int] = set()
    result = None

    def retain_projected_features(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        projected_features.append(output)

    hook = None
    try:
        if prompt_tokens is None:
            prompt_tokens = tuple(
                TextToken(token_id=token_id)
                for token_id in (TURN_ID, USER_ROLE_ID, NEWLINE_ID, NEWLINE_ID)
            )
        hook = runtime.model.model.embed_vision.register_forward_hook(
            retain_projected_features
        )
        for image_inputs in vision_inputs:
            prepared = runtime.prepare_sequence(
                prompt_tokens,
                image_crops=image_inputs,
                max_new_tokens=1,
            )
            prepared_sequences.append(prepared)
            touched_pages.update(
                page_table.page_table_cpu[int(prepared.state.batch_idx)]
            )
        logits = runtime.launch_prepared_batch(
            prepared_sequences,
            runtime._prefill_slot,
            image_crops_list=vision_inputs,
        )
        if len(projected_features) != 1:
            raise RuntimeError(
                "maximum-batch image prefill must produce one packed feature owner"
            )
        result = _ImagePrefillProbeResult(
            logits=logits,
            image_features=tuple(projected_features),
            prepared_sequences=tuple(prepared_sequences),
        )
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_failures = []

        def clean(label: str, operation: Any) -> None:
            try:
                operation()
            except BaseException as exc:
                cleanup_failures.append(f"{label}: {type(exc).__name__}: {exc}")

        def synchronize() -> None:
            if runtime.device.type == "cuda":
                runtime._compute_stream.synchronize()

        def abort_prepared() -> None:
            for prepared in reversed(prepared_sequences):
                if int(prepared.state.batch_idx) not in page_table.free_batch_idx:
                    runtime.abort_prepared_sequence(prepared)

        def assert_page_zero_unmapped() -> None:
            if 0 in touched_pages:
                raise RuntimeError(
                    "transient prefill probe mapped reserved physical page 0"
                )

        def restore_runtime_attributes() -> None:
            if had_page_table:
                runtime.page_table = previous_page_table
            else:
                del runtime.page_table
            if had_kv_cache:
                runtime._kv_cache = previous_kv_cache
            else:
                del runtime._kv_cache

        clean("pre-cleanup stream synchronization", synchronize)
        clean("reserved page-zero mapping validation", assert_page_zero_unmapped)
        # Probe pages remain unowned after restoring the table. A serving
        # prefill overwrites each mapped K/V slot before it can be read.
        clean("prepared-sequence abort", abort_prepared)
        clean(
            "accepted page-table restoration",
            lambda: _restore_page_table(page_table, page_table_snapshot),
        )
        clean("prefill-slot restoration", runtime._prefill_slot.batch_idx.zero_)
        clean("post-cleanup stream synchronization", synchronize)
        if hook is not None:
            clean("feature-owner hook removal", hook.remove)
        clean("runtime resource restoration", restore_runtime_attributes)

        if cleanup_failures:
            detail = "; ".join(cleanup_failures)
            if primary_error is None:
                raise RuntimeError(f"Gemma transient probe cleanup failed: {detail}")
            _add_exception_note(
                primary_error,
                f"Gemma transient probe cleanup failures: {detail}",
            )

    assert result is not None
    return result


def _run_steady_state_image_prefill_probe(
    runtime: Any,
    vision_inputs: Sequence[GemmaImageInputs],
    storage: PagedKVStorage,
    resources: _PagedRuntimeResources | None,
    layer_specs: Sequence[PagedKVLayerSpec | None],
) -> _ImagePrefillProbeResult:
    """Validate consecutive maximum-admitted prefills with resources live."""

    if resources is None:
        raise RuntimeError("Gemma transient probe requires paged resources")
    if len(vision_inputs) != int(runtime.max_batch_size):
        raise ValueError("Gemma steady-state probe must cover the maximum batch")
    probe_inputs = vision_inputs[:1]
    prompt_tokens = _maximum_admitted_image_prefill_prompt(
        runtime,
        resources.page_table,
        probe_inputs[0],
    )

    maximum_batch = _run_image_prefill_transient_probe(
        runtime,
        vision_inputs,
        storage,
        resources,
        layer_specs,
    )
    # The maximum-batch image+language path and the maximum-length language
    # path are independent serving peaks. Neither result may overlap the next.
    del maximum_batch
    warmup = _run_image_prefill_transient_probe(
        runtime,
        probe_inputs,
        storage,
        resources,
        layer_specs,
        prompt_tokens=prompt_tokens,
    )
    # Release every first-pass result owner while preserving allocator state.
    # The second maximum-length pass is the steady-state acceptance criterion.
    del warmup
    return _run_image_prefill_transient_probe(
        runtime,
        probe_inputs,
        storage,
        resources,
        layer_specs,
        prompt_tokens=prompt_tokens,
    )


class Gemma4Runtime(UncachedPagedRuntime):
    def __init__(
        self,
        cfg: Any,
        *,
        max_lora_rank: int | None = None,
        kv_pool: KVMemoryPool | None = None,
        compute_stream: Any = None,
    ) -> None:
        del max_lora_rank
        self.device = cfg.resolved_device()
        self.dtype = cfg.resolved_dtype()
        if self.device.type != "cuda" or self.dtype is not torch.bfloat16:
            raise ValueError("paged multimodal inference requires CUDA with bfloat16")
        self._kv_pool = (
            kv_pool if kv_pool is not None else KVMemoryPool(device=self.device)
        )
        if self._kv_pool.device != self.device:
            raise ValueError(
                f"kv_pool.device ({self._kv_pool.device}) must match runtime "
                f"device ({self.device})"
            )

        self._model_name = cfg.model
        self.decode_path = getattr(cfg, "decode_path", "auto")
        self.max_batch_size = cfg.max_batch_size
        from kestrel.models.registry import get_spec

        model_spec = get_spec(self._model_name)
        model_source = (
            cfg.model_path if cfg.model_path is not None else model_spec.repo_id
        )
        if model_source is None:
            raise ValueError("Gemma model spec must declare repo_id")
        self._generated_weight_storage = None
        prepare_model = None
        finalize_model = None
        if self.decode_path != "native":
            from kestrel.runtime.generated_decode import (
                finalize_generated_weight_storage_after_loading,
                prepare_generated_weight_storage_for_loading,
            )
            from .generated_decode import _WEIGHT_LAYER_PREFIX

            def prepare_model(model: torch.nn.Module) -> None:
                self._generated_weight_storage = (
                    prepare_generated_weight_storage_for_loading(
                        self,
                        model,
                        label="Gemma",
                        layer_prefix=_WEIGHT_LAYER_PREFIX,
                        required_batch_sizes=range(1, self.max_batch_size + 1),
                        required=self.decode_path == "generated",
                    )
                )

            def finalize_model(model: torch.nn.Module) -> None:
                if self._generated_weight_storage is not None:
                    self._generated_weight_storage = (
                        finalize_generated_weight_storage_after_loading(
                            self,
                            model,
                            self._generated_weight_storage,
                            label="Gemma",
                            layer_prefix=_WEIGHT_LAYER_PREFIX,
                            required_batch_sizes=range(1, self.max_batch_size + 1),
                        )
                    )

        self.model = load_model(
            model_source,
            device=self.device,
            dtype=self.dtype,
            revision=model_spec.revision,
            prepare_model=prepare_model,
            finalize_model=finalize_model,
        )
        self._config = self.model.config
        self._configure_model(cfg)
        self.tokenizer = load_tokenizer(
            model_spec.tokenizer_id,
            cfg.tokenizer_path,
            revision=(
                model_spec.revision
                if model_spec.tokenizer_id == model_spec.repo_id
                else None
            ),
        )
        self.tokenizer.post_processor = None
        self.prompt_template = Gemma4PromptTemplate(self._model_name)
        self.eos_token_ids = (self.prompt_template.eos_id,)

        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        self.spec = None
        self.page_size = cfg.page_size
        self.max_seq_length = int(self._config.text_config.max_position_embeddings)
        requested_kv_cache_pages = bound_kv_cache_pages(
            cfg.kv_cache_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_size,
            max_seq_length=self.max_seq_length,
        )
        self.image_prefix_length = MAX_IMAGE_TOKENS + 2
        self._vision_stager = BatchedTensorStager(
            capacity=self.max_batch_size,
            device=self.device,
            with_numpy={"pixel_values": False},
        )
        self._image_preprocessor = AsyncPreprocessor(
            preprocess_image,
            workers=derive_preprocessing_workers(self.max_batch_size),
        )
        vision_probe_inputs = _vision_probe_inputs(self)

        text_config = self._config.text_config
        self.vocab_size = int(text_config.vocab_size)
        self.max_batch_slots = self.max_batch_size + 2
        decode_rows = decode_slot_rows(self.max_batch_size)
        self._padding_batch_idx = self.max_batch_slots - 1
        self._compute_stream = (
            compute_stream if compute_stream is not None else make_stream(self.device)
        )
        self._copy_stream = make_stream(self.device)
        self.graph_capture_lock = threading.RLock()
        self._prefill_slot = PrefillSlot(
            slot_id=0,
            batch_idx=torch.zeros(
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            step_done_event=make_event(
                self.device, enable_timing=False, blocking=False
            ),
            commit_done_event=make_event(
                self.device, enable_timing=False, blocking=False
            ),
        )
        self._prefill_slot_in_use = False
        self.prefill_slots = (self._prefill_slot,)
        self.decode_slots = tuple(
            create_decode_slot(
                slot_id=index,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=decode_rows,
                kv_cache_pages=1,
                vocab_size=text_config.vocab_size,
                hidden_dim=text_config.hidden_size,
                position_shape=(decode_rows, 1),
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for index in range(2)
        )
        self.active_sequences: dict[int, SequenceState] = {}

        kv_layer_specs = paged_kv_specs(text_config)
        self._generated_rope_inputs = None
        has_generated_decode = (
            self.decode_path != "native" and self._generated_weight_storage is not None
        )
        if self.decode_path == "generated" and not has_generated_decode:
            raise RuntimeError(
                "generated Gemma decode has no finalized load-time weight storage"
            )
        self._generated_binding_reservation = None
        if has_generated_decode:
            from kestrel.runtime.generated_decode import (
                generated_weight_programs_for_loading,
                reserve_generated_binding_storage,
            )
            from .generated_decode import _WEIGHT_LAYER_PREFIX

            programs = generated_weight_programs_for_loading(
                self,
                self.model,
                label="Gemma",
                layer_prefix=_WEIGHT_LAYER_PREFIX,
                required_batch_sizes=range(1, self.max_batch_size + 1),
                required=self.decode_path == "generated",
            )
            if programs:
                for program in programs:
                    program.preload(self.device)
            sparse_inputs = _generated_kv_binding_inputs(
                kv_layer_specs,
                tuple(text_config.layer_types),
            )
            if programs:
                self._generated_binding_reservation = reserve_generated_binding_storage(
                    programs,
                    weight_storage=self._generated_weight_storage,
                    runtime_inputs_by_slot=tuple(
                        sparse_inputs for _slot in self.decode_slots
                    ),
                    device=self.device,
                    stream=self._compute_stream,
                    label="Gemma",
                    required=self.decode_path == "generated",
                )
            if self._generated_binding_reservation is None:
                has_generated_decode = False
                self._generated_weight_storage = None
        rope_builder = None
        if has_generated_decode:
            from .generated_decode import _rope_tables

            rope_builder = _rope_tables

        def allocate_paged_resources(pages: int) -> _PagedRuntimeResources:
            reachable_tokens = min(
                self.max_seq_length,
                (int(pages) - 2) * self.page_size,
            )
            rope_inputs = (
                None if rope_builder is None else rope_builder(self, reachable_tokens)
            )
            decode_page_tables = _allocate_decode_page_tables(
                count=len(self.decode_slots),
                rows=decode_rows,
                pages=pages,
                device=self.device,
            )
            page_table = PageTable(
                n_pages=pages,
                page_size=self.page_size,
                max_batch_size=self.max_batch_slots,
                device=str(self.device),
                prefix_cache=None,
                h2d_stream=self._compute_stream,
            )
            page_table.free_batch_idx.remove(self._padding_batch_idx)
            page_table.reserve(self._padding_batch_idx, 1)
            page_table.commit_block_table([self._padding_batch_idx])
            return _PagedRuntimeResources(
                rope_inputs=rope_inputs,
                decode_page_tables=decode_page_tables,
                page_table=page_table,
            )

        self._kv_storage, paged_resources = fit_and_allocate_paged_kv_storage(
            requested_kv_cache_pages,
            layer_specs=kv_layer_specs,
            page_size=self.page_size,
            dtype=self.dtype,
            pool=self._kv_pool,
            stream=self._compute_stream,
            materialize_fixed=lambda: _materialize_fixed_runtime_resources(
                self,
                vision_probe_inputs,
            ),
            allocate_additional=allocate_paged_resources,
            validate_transient=lambda storage, resources: (
                _run_steady_state_image_prefill_probe(
                    self,
                    vision_probe_inputs,
                    storage,
                    resources,
                    kv_layer_specs,
                )
            ),
        )
        if paged_resources is None:
            raise RuntimeError("Gemma paged resource allocation returned no resources")
        self._kv_cache_pages = self._kv_storage.n_pages
        self._generated_rope_inputs = paged_resources.rope_inputs
        _install_decode_page_tables(
            self.decode_slots,
            paged_resources.decode_page_tables,
        )
        self.page_table = paged_resources.page_table
        self._kv_cache = allocate_paged_kv_layers(
            layer_specs=kv_layer_specs,
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
            storage=self._kv_storage,
        )
        self._initialize_generated_decode()

    def _initialize_generated_decode(self) -> None:
        """Apply the runtime-wide decode policy after model state is ready."""

        # Generated bindings own exactly the tensors represented by this
        # reservation. Release it only after every other resident allocation,
        # so binding assembly can reuse the reserved allocator block.
        reservation = self._generated_binding_reservation
        self._generated_binding_reservation = None
        del reservation
        self.generated_decode = None
        if self.decode_path == "native":
            return
        if self._generated_weight_storage is None:
            if self.decode_path == "generated":
                raise RuntimeError(
                    "generated Gemma decode has no finalized load-time weight storage"
                )
            return

        from .generated_decode import create_generated_decode

        self.generated_decode = create_generated_decode(
            self,
            required=self.decode_path == "generated",
        )

    def _configure_model(self, cfg: Any) -> None:
        vision = self.model.model.vision_tower
        canonicalize_immutable_scalar_buffers(vision.encoder)
        vision.encoder.forward = torch.compile(
            vision.encoder.forward,
            dynamic=True,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )
        config = self._config.vision_config
        _maximum_vision_grid(config)

        def inputs(batch_size: int) -> tuple[torch.Tensor, ...]:
            return (
                torch.zeros(
                    (batch_size, MAX_PATCHES, config.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.ones(
                    (batch_size, MAX_PATCHES),
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.zeros(
                    (batch_size, MAX_PATCHES, 2),
                    dtype=torch.long,
                    device=self.device,
                ),
            )

        materialize_dynamic_batch_domain(
            vision.encoder,
            max_batch_size=cfg.max_batch_size,
            inputs_for_batch=inputs,
            synchronize=lambda: torch.cuda.synchronize(self.device),
        )
        torch.cuda.empty_cache()

    def _prepare_prompt(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image_crops: Any,
    ) -> tuple[list[Token], int, int]:
        tokens = list(prompt_tokens)
        text_length = len(tokens)
        if image_crops is None:
            return tokens, 0, text_length
        count = int(image_crops.num_image_tokens)
        image_block = (
            [TextToken(token_id=START_OF_IMAGE_ID)]
            + [TextToken(token_id=self._config.image_token_id)] * count
            + [TextToken(token_id=END_OF_IMAGE_ID)]
        )
        query = self.prompt_template.query()
        offset = derive_image_insertion_offset(
            tokens,
            user_turn_opener=(TURN_ID, USER_ROLE_ID, NEWLINE_ID),
            fallback_offset=1 + (len(query.prefix) if query else 0),
        )
        return (
            tokens[:offset] + image_block + tokens[offset:],
            count + 2,
            text_length,
        )

    def _prefill(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        last_token_offsets: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_config = self._config.text_config
        language_model = self.model.model.language_model
        per_layer_inputs = (
            language_model.get_per_layer_inputs(input_ids)
            if text_config.hidden_size_per_layer_input
            else None
        )
        hidden = language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            kv_cache=self._kv_cache,
            per_layer_inputs=per_layer_inputs,
            cache_position_ids=position_ids,
            slot_mapping=slot_mapping,
            cu_seqlens=cu_seqlens,
        )
        rows = (
            hidden[0, -1:]
            if last_token_offsets is None
            else hidden[0].index_select(0, last_token_offsets)
        )
        logits = self.model.lm_head(rows)
        cap = text_config.final_logit_softcapping
        return rows, torch.tanh(logits / cap) * cap

    def _image_features_for_batch(
        self,
        image_crops_list: Sequence[Any],
    ) -> list[torch.Tensor | None]:
        features: list[torch.Tensor | None] = [None] * len(image_crops_list)
        unique: dict[int, tuple[Any, list[int]]] = {}
        for row, crops in enumerate(image_crops_list):
            if crops is not None:
                unique.setdefault(id(crops), (crops, []))[1].append(row)
        if not unique:
            return features

        groups = list(unique.values())
        records = [
            {
                "pixel_values": item.pixel_values,
                "position_ids": item.image_position_ids,
            }
            for item, _ in groups
        ]
        staged = self._vision_stager.stage(records)
        packed = self.model.model.get_image_features(
            staged["pixel_values"],
            staged["position_ids"],
        ).detach()
        lengths = [int(item.num_image_tokens) for item, _ in groups]
        if packed.shape[0] != sum(lengths):
            raise RuntimeError(
                f"vision encoder returned {packed.shape[0]} tokens "
                f"for declared split {lengths}"
            )
        for (_, rows), encoded in zip(
            groups,
            packed.split(lengths, dim=0),
            strict=True,
        ):
            for row in rows:
                features[row] = encoded
        return features

    def acquire_prefill_slot(self, slot_id: int | None = None) -> Any:
        if self._prefill_slot_in_use:
            raise RuntimeError("Prefill slot pool exhausted")
        if slot_id is not None and slot_id != 0:
            raise ValueError(f"Invalid prefill_slot_id {slot_id}")
        self._prefill_slot_in_use = True
        return self._prefill_slot

    def release_prefill_slot(self, slot: Any) -> None:
        if slot is not self._prefill_slot:
            raise ValueError("cannot release a foreign prefill slot")
        if not self._prefill_slot_in_use:
            raise RuntimeError("prefill slot is not acquired")
        self._prefill_slot_in_use = False

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: Any = None,
        image_crops: Any = None,
        encoder_input: object | None = None,
        max_new_tokens: int | None = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> Any:
        del image
        if encoder_input is not None:
            raise ValueError("Gemma4Runtime does not support encoder inputs")
        tokens, image_tokens, text_length = self._prepare_prompt(
            prompt_tokens,
            image_crops=image_crops,
        )
        prompt_length = len(tokens)
        new_tokens = 128 if max_new_tokens is None else max_new_tokens
        target_length = max(
            text_length + self.image_prefix_length + new_tokens,
            prompt_length + new_tokens,
        )
        return self._prepare_uncached_sequence(
            tokens=tokens,
            target_length=target_length,
            image_length=image_tokens,
            lora_slot=lora_slot,
            adapter_id=adapter_id,
            image_hash=image_hash,
        )

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[Any],
        prefill_slot: Any,
        *,
        images: Sequence[Any] | None = None,
        image_crops_list: Sequence[Any] | None = None,
        encoder_inputs: Sequence[object | None] | None = None,
    ) -> torch.Tensor:
        del images
        if encoder_inputs is not None and any(
            item is not None for item in encoder_inputs
        ):
            raise ValueError("Gemma4Runtime does not support encoder inputs")
        batch_size = len(prepared_sequences)
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(f"prefill batch must lie in [1, {self.max_batch_size}]")
        if image_crops_list is None:
            image_crops_list = [None] * batch_size
        if len(image_crops_list) != batch_size:
            raise ValueError("image_crops_list must match prepared_sequences")
        batch_indices = [
            int(prepared.state.batch_idx) for prepared in prepared_sequences
        ]
        self.page_table.commit_block_table(batch_indices)

        token_rows = []
        lengths = []
        image_features = self._image_features_for_batch(image_crops_list)
        for prepared in prepared_sequences:
            tokens = prepared.tokens_list
            if not tokens or not all(isinstance(token, TextToken) for token in tokens):
                raise ValueError("prefill requires non-empty text-token rows")
            token_rows.append([int(token.token_id) for token in tokens])
            lengths.append(len(tokens))

        model_ids = torch.tensor(
            [
                [
                    0 if token_id == self._config.image_token_id else token_id
                    for row in token_rows
                    for token_id in row
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )
        inputs_embeds = self.model.model.language_model.embed(model_ids)
        _copy_image_features_into_embeddings(
            inputs_embeds,
            token_rows,
            image_features,
            image_token_id=self._config.image_token_id,
        )

        prefill_slot.batch_idx[:batch_size].copy_(
            torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        )
        position_ids = torch.tensor(
            [[position for length in lengths for position in range(length)]],
            dtype=torch.long,
            device=self.device,
        )
        token_batch_indices = torch.tensor(
            [
                [
                    batch_idx
                    for batch_idx, length in zip(batch_indices, lengths, strict=True)
                    for _ in range(length)
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=token_batch_indices,
            positions=position_ids,
        )
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        cu_seqlens = torch.tensor(
            cumulative,
            dtype=torch.int32,
            device=self.device,
        )
        last_token_offsets = (
            None
            if batch_size == 1
            else torch.tensor(
                [end - 1 for end in cumulative[1:]],
                dtype=torch.long,
                device=self.device,
            )
        )
        hidden_rows, logits = self._prefill(
            inputs_embeds,
            model_ids,
            position_ids,
            slot_mapping,
            last_token_offsets,
            cu_seqlens,
        )
        for row, prepared in enumerate(prepared_sequences):
            prepared.state.last_hidden = hidden_rows[row].detach()
        return logits

    def decode_with_slot(self, slot: Any, batch_size: int) -> None:
        if batch_size == 0:
            return
        use_generated = (
            self.generated_decode is not None
            and self.generated_decode.supports(batch_size)
        )
        if not use_generated and self.decode_path == "generated":
            raise RuntimeError(
                "required generated Gemma decode does not cover "
                f"active batch size {batch_size}"
            )
        with torch.cuda.stream(slot.compute_stream):
            if use_generated:
                assert self.generated_decode is not None
                self.generated_decode.run(slot, batch_size)
            else:
                self._run_native_decode(slot, batch_size)

    def _run_native_decode(self, slot: Any, batch_size: int) -> None:
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]
        input_pos = slot.meta.input_pos.gpu[:batch_size]
        positions = slot.cache_position_ids[:batch_size]
        positions[:, 0].copy_(input_pos)
        slot.position_ids[:batch_size].copy_(positions)
        self.page_table.populate_paged_kv_metadata(
            batch_idx=batch_idx,
            input_pos=input_pos,
            out_page_table=slot.paged_kv_page_table[:batch_size],
            out_seqused_k=slot.paged_kv_seqlens_k[:batch_size],
        )
        slot.slot_mapping[:batch_size].copy_(
            self.page_table.build_slot_mapping(
                batch_idx=batch_idx,
                positions=positions,
            )
        )

        input_ids = slot.decode_token_ids[:batch_size].view(batch_size, 1)
        language_model = self.model.model.language_model
        inputs_embeds = language_model.embed(input_ids)
        per_layer_inputs = (
            language_model.get_per_layer_inputs(input_ids)
            if self._config.text_config.hidden_size_per_layer_input
            else None
        )
        hidden = language_model(
            inputs_embeds=inputs_embeds,
            position_ids=slot.position_ids[:batch_size],
            kv_cache=self._kv_cache,
            per_layer_inputs=per_layer_inputs,
            cache_position_ids=positions,
            slot_mapping=slot.slot_mapping[:batch_size],
            cu_seqlens=None,
            page_table=slot.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
        )[:, 0]
        slot.hidden_last[:batch_size].copy_(hidden)
        torch.mm(
            hidden,
            self.model.lm_head.weight.t(),
            out=slot.logits[:batch_size],
        )
        cap = self._config.text_config.final_logit_softcapping
        if cap is not None:
            slot.logits[:batch_size].div_(cap).tanh_().mul_(cap)


__all__ = ["Gemma4Runtime"]
