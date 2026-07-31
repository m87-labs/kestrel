"""Gemma 4 runtime for the Kestrel inference engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch.nn import functional as F

from kestrel.kv_cache import KVMemoryPool, LayeredPagedKV, PageTable
from kestrel.device import make_event, make_stream
from kestrel.models.registry import get_spec
from kestrel.runtime import ExecutionShape, TextToken
from kestrel.utils import CpuGpuBuffer
from kestrel.runtime.compilation import (
    canonicalize_immutable_scalar_buffers,
    materialize_dynamic_batch_domain,
)
from kestrel.runtime.prefill import project_padded_last_rows
from kestrel.runtime.preprocessing import (
    derive_image_insertion_offset,
    derive_preprocessing_workers,
)

from .decode_slot import GemmaDecodeSlot, create_gemma_decode_slot
from .loader import load_model
from .image import (
    Gemma4ImagePreprocessor,
    IMAGE_SEQ_LENGTH,
    MAX_PATCHES,
)
from .prompt_template import Gemma4PromptTemplate
from .paged_cache import paged_kv_layout


_MAX_SHIPPED_KV_LEN = 2048


def _decode_slot_rows(max_batch_size: int) -> int:
    """Cover both logical scheduler slots and the next decode capacity."""

    max_batch_size = int(max_batch_size)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    scheduler_rows = max_batch_size + 2
    decode_capacity = 1 << (max_batch_size - 1).bit_length()
    return max(scheduler_rows, decode_capacity)


@dataclass
class _SimplePrefillSlot:
    """Scheduler-facing state for Gemma prefill."""

    slot_id: int
    batch_idx: torch.Tensor
    step_done_event: Any
    commit_done_event: Any
    scratch: Any = None


@dataclass
class _EncodeResult:
    """Match the surface kestrel skills expect: ``encode(s).ids``."""

    ids: list[int]


class _TokenizerShim:
    """Adapts ``tokenizers.Tokenizer`` to the encode/decode surface."""

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
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)


class Gemma4Runtime:
    """Runtime wrapping vendored Gemma 4 modeling for the kestrel engine."""

    def __init__(
        self,
        cfg: Any,
        *,
        max_lora_rank: Optional[int] = None,
        kv_pool: KVMemoryPool | None = None,
        compute_stream: Any = None,
    ) -> None:
        self._cfg = cfg
        self.device = cfg.resolved_device()
        self.dtype = cfg.resolved_dtype()
        if self.device.type != "cuda" or self.dtype is not torch.bfloat16:
            raise ValueError("Gemma 4 inference requires CUDA with bfloat16")
        self._kv_pool = kv_pool if kv_pool is not None else KVMemoryPool(device=self.device)
        if self._kv_pool.device != self.device:
            raise ValueError(
                f"kv_pool.device ({self._kv_pool.device}) must match runtime "
                f"device ({self.device})"
            )

        from tokenizers import Tokenizer

        repo_id = cfg.model
        self.model = load_model(
            repo_id,
            device=self.device,
            dtype=self.dtype,
        )
        self._config = self.model.config
        vision_tower = self.model.model.vision_tower
        if self.device.type == "cuda" and vision_tower is not None:
            canonicalize_immutable_scalar_buffers(vision_tower.encoder)
            vision_tower.encoder.forward = torch.compile(
                vision_tower.encoder.forward,
                dynamic=True,
                fullgraph=False,
                options={"triton.cudagraphs": False},
            )
            vision_cfg = self._config.vision_config

            def vision_inputs(batch_size: int) -> tuple[torch.Tensor, ...]:
                return (
                    torch.zeros(
                        (batch_size, MAX_PATCHES, int(vision_cfg.hidden_size)),
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
                vision_tower.encoder,
                max_batch_size=int(getattr(cfg, "max_batch_size", 1)),
                inputs_for_batch=vision_inputs,
                synchronize=lambda: torch.cuda.synchronize(self.device),
            )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.tokenizer = _TokenizerShim(Tokenizer.from_pretrained(repo_id))
        self.prompt_template = Gemma4PromptTemplate(repo_id)

        self._model_name = repo_id
        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        self.spec = None
        self.max_batch_size = getattr(cfg, "max_batch_size", 1)
        self.max_batch_slots = self.max_batch_size + 2
        self._vision_pixel_staging: CpuGpuBuffer | None = None
        self._vision_position_staging: CpuGpuBuffer | None = None
        self._decode_slot_rows = _decode_slot_rows(self.max_batch_size)
        self._padding_batch_idx = self.max_batch_slots - 1
        self.page_size = int(getattr(cfg, "page_size", 1))
        self._kv_cache_pages = int(getattr(cfg, "kv_cache_pages", 65536))
        self.max_seq_length = min(
            self._config.text_config.max_position_embeddings,
            _MAX_SHIPPED_KV_LEN,
        )
        self.image_prefix_length = IMAGE_SEQ_LENGTH + 2

        self._image_preprocessor = Gemma4ImagePreprocessor(
            num_workers=derive_preprocessing_workers(self.max_batch_size),
            dtype=self.dtype,
        )

        self._primary_stream = (
            compute_stream if compute_stream is not None else make_stream(self.device)
        )
        self._compute_stream = self._primary_stream
        self._copy_stream = make_stream(self.device)
        self.graph_capture_lock = threading.RLock()
        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=None,
            h2d_stream=self._primary_stream,
        )
        self.page_table.free_batch_idx.remove(self._padding_batch_idx)
        self.page_table.reserve(self._padding_batch_idx, 1)
        self.page_table.commit_block_table([self._padding_batch_idx])
        self._prefill_slot = _SimplePrefillSlot(
            slot_id=0,
            batch_idx=torch.zeros((self.max_batch_size,), dtype=torch.int64, device=self.device),
            step_done_event=make_event(self.device, enable_timing=False, blocking=False),
            commit_done_event=make_event(self.device, enable_timing=False, blocking=False),
        )
        self._prefill_slot_in_use = False
        self.prefill_slots: Sequence[Any] = (self._prefill_slot,)

        text_cfg = self._config.text_config
        self._decode_slots = tuple(
            create_gemma_decode_slot(
                slot_id=i,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=self._decode_slot_rows,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=text_cfg.vocab_size,
                hidden_dim=text_cfg.hidden_size,
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for i in range(2)
        )
        self.decode_slots: Sequence[Any] = self._decode_slots
        self.active_sequences: dict[int, Any] = {}

        layer_specs, source_layer_idx = paged_kv_layout(text_cfg)
        self._kv_cache = LayeredPagedKV.allocate(
            layer_specs=layer_specs,
            source_layer_idx=source_layer_idx,
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )

        from .generated_decode import Gemma4DecodeMegakernel

        self._decode_megakernel = Gemma4DecodeMegakernel.try_create(self)
        self.prefix_cache = None

    def _stage_vision_inputs(
        self,
        crops_list: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(crops_list)
        if batch_size < 1:
            raise ValueError("vision staging requires at least one input row")
        capacity = max(batch_size, int(getattr(self, "max_batch_size", batch_size)))
        first_crops = crops_list[0]
        pixel_shape = tuple(first_crops.pixel_values.shape)
        position_shape = tuple(first_crops.image_position_ids.shape)
        if len(pixel_shape) != 2 or len(position_shape) != 2:
            raise ValueError("vision inputs must be rank-2 tensors")

        pixel_staging = getattr(self, "_vision_pixel_staging", None)
        position_staging = getattr(self, "_vision_position_staging", None)
        expected_pixel_shape = (capacity, *pixel_shape)
        expected_position_shape = (capacity, *position_shape)
        if pixel_staging is None:
            pixel_staging = CpuGpuBuffer(
                *expected_pixel_shape,
                dtype=first_crops.pixel_values.dtype,
                device=self.device,
                pin_memory=self.device.type == "cuda",
                with_numpy=False,
                zero=False,
            )
            self._vision_pixel_staging = pixel_staging
        elif tuple(pixel_staging.cpu.shape) != expected_pixel_shape:
            raise RuntimeError(
                "vision pixel staging shape changed after allocation: "
                f"{tuple(pixel_staging.cpu.shape)} -> {expected_pixel_shape}"
            )
        if position_staging is None:
            position_staging = CpuGpuBuffer(
                *expected_position_shape,
                dtype=first_crops.image_position_ids.dtype,
                device=self.device,
                pin_memory=self.device.type == "cuda",
                zero=False,
            )
            self._vision_position_staging = position_staging
        elif tuple(position_staging.cpu.shape) != expected_position_shape:
            raise RuntimeError(
                "vision position staging shape changed after allocation: "
                f"{tuple(position_staging.cpu.shape)} -> {expected_position_shape}"
            )

        for row, crops in enumerate(crops_list):
            if (
                tuple(crops.pixel_values.shape) != pixel_shape
                or crops.pixel_values.dtype != pixel_staging.cpu.dtype
            ):
                raise ValueError("batched vision pixel tensors must share shape and dtype")
            if (
                tuple(crops.image_position_ids.shape) != position_shape
                or crops.image_position_ids.dtype != position_staging.cpu.dtype
            ):
                raise ValueError(
                    "batched vision position tensors must share shape and dtype"
                )
            pixel_staging.cpu[row].copy_(crops.pixel_values)
            position_staging.cpu[row].copy_(crops.image_position_ids)

        return (
            pixel_staging.copy_to_gpu(batch_size),
            position_staging.copy_to_gpu(batch_size),
        )

    def _image_features_for_batch(
        self,
        image_crops_list: Sequence[Any],
    ) -> list[torch.Tensor | None]:
        features: list[torch.Tensor | None] = [None] * len(image_crops_list)
        unique: dict[int, tuple[Any, list[int]]] = {}
        for row, crops in enumerate(image_crops_list):
            if crops is not None:
                unique.setdefault(id(crops), (crops, []))[1].append(row)
        if unique:
            groups = list(unique.values())
            crops = [item for item, _ in groups]
            pixel_values, position_ids = self._stage_vision_inputs(crops)
            packed = self.model.model.get_image_features(
                pixel_values,
                position_ids,
            ).detach()
            lengths = [int(item.num_image_tokens) for item in crops]
            if int(packed.shape[0]) != sum(lengths):
                raise RuntimeError(
                    "vision encoder returned "
                    f"{int(packed.shape[0])} tokens for declared split {lengths}"
                )
            for (_, rows), encoded in zip(groups, packed.split(lengths, dim=0)):
                for row in rows:
                    features[row] = encoded
        return features

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def compute_stream(self):
        return self._primary_stream

    @property
    def primary_stream(self):
        return self._primary_stream

    @property
    def copy_stream(self):
        return self._copy_stream

    @property
    def vocab_size(self) -> int:
        return int(self._config.text_config.vocab_size)

    @property
    def cuda_graphs_enabled(self) -> bool:
        return False

    def skills(self):
        return get_spec(self.model_name).skills()

    def tasks(self) -> tuple[str, ...]:
        return self.skills().names()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Greedy / nucleus single-stream generation through our model."""
        query = self.prompt_template.query()
        if query is None:  # pragma: no cover - all Gemma 4 variants expose query()
            raise RuntimeError("Gemma 4 prompt template missing query()")
        stop_token_ids = {self.prompt_template.eos_id, *query.stop_token_ids}
        user_ids = self.tokenizer.encode(prompt).ids
        ids = (
            [self.prompt_template.bos_id]
            + list(query.prefix)
            + list(user_ids)
            + list(query.answer_prefix)
        )
        input_ids = torch.tensor([ids], device=self.device)
        prompt_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            logits = self.model(input_ids=input_ids)[0, -1]
            if temperature > 0.0:
                logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cum > top_p).nonzero()
                    keep = cutoff[0, 0].item() + 1 if cutoff.numel() else sorted_probs.numel()
                    sorted_probs[keep:] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    pick = torch.multinomial(sorted_probs, 1)
                    next_id = sorted_idx[pick].item()
                else:
                    next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(logits.argmax().item())

            if next_id in stop_token_ids:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=self.device)], dim=1
            )

        new_tokens = input_ids[0, prompt_len:].tolist()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def preprocess_image_async(self, image):
        return self._image_preprocessor.submit(image)

    def shutdown(self) -> None:
        self._image_preprocessor.shutdown(wait=True)

    def shutdown_image_preprocessor(self) -> None:
        self.shutdown()

    def can_reserve(self, total_length: int) -> bool:
        return (
            total_length <= self.max_seq_length
            and self.page_table.can_reserve_with_eviction(total_length)
        )

    def prefill_budget(self) -> tuple[int, int]:
        return (self.page_table.pages_available, self._available_batch_slots())

    def _available_batch_slots(self) -> int:
        active_headroom = max(0, self.max_batch_size - len(self.active_sequences))
        return min(active_headroom, len(self.page_table.free_batch_idx))

    def acquire_prefill_slot(self, slot_id: int | None = None) -> Any:
        if self._prefill_slot_in_use:
            raise RuntimeError("Prefill slot pool exhausted (single-stream runtime)")
        if slot_id is not None and slot_id != 0:
            raise ValueError(f"Invalid prefill_slot_id {slot_id}")
        self._prefill_slot_in_use = True
        return self._prefill_slot

    def release_prefill_slot(self, slot: Any) -> None:
        self._prefill_slot_in_use = False

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int:
        raise NotImplementedError("LoRA adapters not supported on Gemma4Runtime yet")

    def release_adapter_slot(self, slot: int) -> None:
        raise NotImplementedError("LoRA adapters not supported on Gemma4Runtime yet")

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
        image_crops: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
        image_hash: Optional[bytes] = None,
        adapter_id: Optional[str] = None,
    ) -> Any:
        from kestrel.runtime.state import (
            PreparedSequence,
            SequenceState,
            _CacheLookupResult,
        )
        from .prompt_template import (
            END_OF_IMAGE_ID,
            NEWLINE_ID,
            START_OF_IMAGE_ID,
            TURN_ID,
            USER_ROLE_ID,
        )

        tokens_list = list(prompt_tokens)
        text_only_len = len(tokens_list)
        num_image_tokens = 0
        if image is not None:
            if image_crops is None:
                raise RuntimeError(
                    "Gemma4Runtime.prepare_sequence: image given but image_crops "
                    "is None — preprocess_image_async should have populated it."
                )
            num_image_tokens = int(image_crops.num_image_tokens)
            image_block = (
                [TextToken(token_id=START_OF_IMAGE_ID)]
                + [TextToken(token_id=self._config.image_token_id)] * num_image_tokens
                + [TextToken(token_id=END_OF_IMAGE_ID)]
            )
            query_template = self.prompt_template.query()
            fallback_offset = 1 + (
                len(query_template.prefix) if query_template else 0
            )
            offset = derive_image_insertion_offset(
                tokens_list,
                user_turn_opener=(TURN_ID, USER_ROLE_ID, NEWLINE_ID),
                fallback_offset=fallback_offset,
            )
            tokens_list = (
                tokens_list[:offset] + image_block + tokens_list[offset:]
            )

        prompt_len = len(tokens_list)
        budget_for_finalize = (
            text_only_len + self.image_prefix_length + (max_new_tokens or 128)
        )
        actual_kv_budget = prompt_len + (max_new_tokens or 128)
        target_length = max(budget_for_finalize, actual_kv_budget)
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds max_seq_length={self.max_seq_length}"
            )

        if self._available_batch_slots() <= 0:
            raise RuntimeError("Cannot reserve Gemma4 batch slot")
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
            image_length=(num_image_tokens + 2) if image is not None else 0,
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
        return PreparedSequence(
            state=state,
            tokens_list=tokens_list,
            cache_tokens=[],
            cache_result=cache_result,
            adapter_id=adapter_id,
            image_hash=image_hash,
        )

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[Any],
        prefill_slot: Any,
        *,
        images: Optional[Sequence[Any]] = None,
        image_crops_list: Optional[Sequence[Any]] = None,
    ) -> torch.Tensor:
        batch_size = len(prepared_sequences)
        if batch_size == 0:
            raise ValueError("prepared_sequences must be non-empty")
        if batch_size > self.max_batch_size:
            raise NotImplementedError(
                f"Gemma4Runtime prefill: batch_size={batch_size} > "
                f"max_batch_size={self.max_batch_size}"
            )
        if images is None:
            images = [None] * batch_size
        if image_crops_list is None:
            image_crops_list = [None] * batch_size

        batch_indices = [int(prepared.state.batch_idx) for prepared in prepared_sequences]
        self.page_table.commit_block_table(batch_indices)

        text_cfg = self._config.text_config
        softcap = text_cfg.final_logit_softcapping
        pad_id = 0

        seq_lengths: list[int] = []
        embeds_rows: list[torch.Tensor] = []
        llm_id_rows: list[torch.Tensor] = []
        image_features_by_row = self._image_features_for_batch(image_crops_list)
        for row, (prepared, image, crops) in enumerate(
            zip(prepared_sequences, images, image_crops_list)
        ):
            prefill_slot.batch_idx[row] = prepared.state.batch_idx
            tokens = prepared.tokens_list
            if not all(isinstance(t, TextToken) for t in tokens):
                raise ValueError(
                    "Gemma4Runtime prefill only supports TextToken (no spatial tokens)"
                )
            token_ids = [int(t.token_id) for t in tokens]
            if not token_ids:
                raise ValueError("Prefill prompt must contain at least one token")

            seq_lengths.append(len(token_ids))
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            if image is not None and crops is not None:
                image_features = image_features_by_row[row]
                if image_features is None:
                    raise RuntimeError("missing encoded features for image row")
                image_mask = self.model.model.image_placeholder_mask(input_ids)
                llm_ids = input_ids.clone()
                llm_ids[image_mask] = pad_id
                embeds = self.model.model.get_input_embeddings()(llm_ids)
                embeds = embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(embeds),
                    image_features.to(device=embeds.device, dtype=embeds.dtype),
                )
            else:
                llm_ids = input_ids
                embeds = self.model.model.get_input_embeddings()(input_ids)

            embeds_rows.append(embeds)
            llm_id_rows.append(llm_ids)

        max_len = max(seq_lengths)
        padded_embeds: list[torch.Tensor] = []
        padded_llm_ids: list[torch.Tensor] = []
        for embeds, llm_ids, seq_len in zip(embeds_rows, llm_id_rows, seq_lengths):
            pad = max_len - seq_len
            if pad:
                embeds = F.pad(embeds, (0, 0, 0, pad))
                llm_ids = F.pad(llm_ids, (0, pad), value=pad_id)
            padded_embeds.append(embeds)
            padded_llm_ids.append(llm_ids)

        inputs_embeds = torch.cat(padded_embeds, dim=0)
        llm_ids_batch = torch.cat(padded_llm_ids, dim=0)
        position_ids = torch.arange(
            max_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)

        per_layer_inputs = None
        if text_cfg.hidden_size_per_layer_input:
            per_layer_inputs = self.model.model.language_model.get_per_layer_inputs(
                llm_ids_batch
            )

        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=prefill_slot.batch_idx[:batch_size],
            positions=position_ids,
        )
        hidden_states = self.model.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            kv_cache=self._kv_cache,
            cache_position_ids=position_ids,
            slot_mapping=slot_mapping,
        )
        last_hidden_rows, logits = project_padded_last_rows(
            hidden_states,
            seq_lengths,
            self.model.lm_head,
        )
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        for row, (prepared, seq_len) in enumerate(zip(prepared_sequences, seq_lengths)):
            prepared.state.last_hidden = last_hidden_rows[row].detach()

        return logits

    def finalize_prepared_sequence_after_prefill(self, prepared: Any) -> None:
        self.active_sequences[prepared.state.batch_idx] = prepared.state
        return None

    def abort_prepared_sequence(self, prepared: Any) -> None:
        self.active_sequences.pop(prepared.state.batch_idx, None)
        self._release_batch_idx(prepared.state.batch_idx)

    def retain_sequence_prefix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def release_sequence(self, state: Any) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self._release_batch_idx(state.batch_idx)

    def _release_batch_idx(self, batch_idx: int) -> None:
        if batch_idx not in self.page_table.free_batch_idx:
            self.page_table.erase(batch_idx, 0)

    def _build_decode_metadata(self, slot: GemmaDecodeSlot, batch_size: int) -> None:
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]
        input_pos = slot.meta.input_pos.gpu[:batch_size]
        slot.cache_position_ids[:batch_size, 0].copy_(input_pos)
        slot.position_ids[:batch_size, 0].copy_(input_pos)
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

    def decode_with_slot(self, slot: GemmaDecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        if (
            self._decode_megakernel is not None
            and self._decode_megakernel.supports(batch_size)
        ):
            with torch.cuda.stream(slot.compute_stream):
                self._decode_megakernel.run(slot, batch_size)
            return
        self._run_decode_eager(slot, batch_size)

    def _run_decode_eager(self, slot: GemmaDecodeSlot, batch_size: int) -> None:
        text_cfg = self._config.text_config
        softcap = text_cfg.final_logit_softcapping
        token_ids_gpu = slot.decode_token_ids[:batch_size]
        self._build_decode_metadata(slot, batch_size)
        input_ids = token_ids_gpu.view(batch_size, 1)
        position_ids = slot.meta.input_pos.gpu[:batch_size].view(batch_size, 1)
        hidden_states = self.model.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=self._kv_cache,
            cache_position_ids=slot.cache_position_ids[:batch_size],
            slot_mapping=slot.slot_mapping[:batch_size],
            page_table=slot.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
            paged_kv_use_sliding_window=True,
        )
        last_hidden = hidden_states[:, 0]
        torch.mm(last_hidden, self.model.lm_head.weight.t(), out=slot.logits[:batch_size])
        if softcap is not None:
            slot.logits[:batch_size].div_(softcap).tanh_().mul_(softcap)
        slot.hidden_last[:batch_size].copy_(last_hidden)

__all__ = ["Gemma4Runtime"]
