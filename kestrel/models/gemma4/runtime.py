"""Gemma 4 paged multimodal runtime."""

from __future__ import annotations

import threading
from typing import Any, Sequence

import torch

from kestrel.device import make_event, make_stream
from kestrel.kv_cache import KVMemoryPool, PageTable, allocate_paged_kv_layers
from kestrel.runtime import ExecutionShape, SequenceState, TextToken, Token
from kestrel.runtime.compilation import (
    canonicalize_immutable_scalar_buffers,
    materialize_dynamic_batch_domain,
)
from kestrel.runtime.decode_slot import create_decode_slot
from kestrel.runtime.paged_resources import PrefillSlot, decode_slot_rows
from kestrel.runtime.preprocessing import derive_image_insertion_offset
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.staging import AsyncPreprocessor, BatchedTensorStager
from kestrel.runtime.tokenizer import load_tokenizer
from kestrel.runtime.uncached_paged import UncachedPagedRuntime

from .generated_decode import create_generated_decode
from .image import MAX_IMAGE_TOKENS, MAX_PATCHES, preprocess_image
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
        from kestrel.models.registry import get_spec

        model_spec = get_spec(self._model_name)
        model_source = cfg.model_path if cfg.model_path is not None else model_spec.repo_id
        if model_source is None:
            raise ValueError("Gemma model spec must declare repo_id")
        self.model = load_model(
            model_source,
            device=self.device,
            dtype=self.dtype,
        )
        self._config = self.model.config
        self._configure_model(cfg)
        self.tokenizer = load_tokenizer(model_spec.tokenizer_id, cfg.tokenizer_path)
        self.tokenizer.post_processor = None
        self.prompt_template = Gemma4PromptTemplate(self._model_name)

        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        self.spec = None
        self.max_batch_size = cfg.max_batch_size
        self.page_size = cfg.page_size
        self._kv_cache_pages = cfg.kv_cache_pages
        self.max_seq_length = min(
            self._config.text_config.max_position_embeddings,
            2048,
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
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=text_config.vocab_size,
                hidden_dim=text_config.hidden_size,
                position_shape=(decode_rows, 1),
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for index in range(2)
        )
        self.active_sequences: dict[int, SequenceState] = {}

        self._kv_cache = allocate_paged_kv_layers(
            layer_specs=paged_kv_specs(text_config),
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )
        self._decode_megakernel = create_generated_decode(self)

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
        image: Any,
        image_crops: Any,
    ) -> tuple[list[Token], int, int]:
        tokens = list(prompt_tokens)
        text_length = len(tokens)
        if image is None:
            return tokens, 0, text_length
        if image_crops is None:
            raise RuntimeError("image preprocessing did not produce crops")
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
            input_ids=None,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            kv_cache=self._kv_cache,
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

    def _eager_decode(self, slot: Any, batch_size: int) -> None:
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
        hidden = self.model.model.language_model(
            input_ids=slot.decode_token_ids[:batch_size].view(batch_size, 1),
            position_ids=input_pos.view(batch_size, 1),
            kv_cache=self._kv_cache,
            cache_position_ids=slot.cache_position_ids[:batch_size],
            slot_mapping=slot.slot_mapping[:batch_size],
            page_table=slot.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
        )[:, 0]
        torch.mm(hidden, self.model.lm_head.weight.t(), out=slot.logits[:batch_size])
        cap = self._config.text_config.final_logit_softcapping
        slot.logits[:batch_size].div_(cap).tanh_().mul_(cap)
        slot.hidden_last[:batch_size].copy_(hidden)

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
        max_new_tokens: int | None = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> Any:
        tokens, image_tokens, text_length = self._prepare_prompt(
            prompt_tokens,
            image=image,
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
    ) -> torch.Tensor:
        batch_size = len(prepared_sequences)
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(
                f"prefill batch must lie in [1, {self.max_batch_size}]"
            )
        if images is None:
            images = [None] * batch_size
        if image_crops_list is None:
            image_crops_list = [None] * batch_size
        if len(images) != batch_size or len(image_crops_list) != batch_size:
            raise ValueError(
                "images and image_crops_list must match prepared_sequences"
            )
        batch_indices = [
            int(prepared.state.batch_idx) for prepared in prepared_sequences
        ]
        self.page_table.commit_block_table(batch_indices)

        token_rows = []
        lengths = []
        image_features = self._image_features_for_batch(image_crops_list)
        for prepared, image, crops in zip(
            prepared_sequences, images, image_crops_list, strict=True
        ):
            if (image is None) != (crops is None):
                raise ValueError("each image row requires matching preprocessed crops")
            tokens = prepared.tokens_list
            if not tokens or not all(isinstance(token, TextToken) for token in tokens):
                raise ValueError("prefill requires non-empty text-token rows")
            token_rows.append([int(token.token_id) for token in tokens])
            lengths.append(len(tokens))

        flat_ids = [token_id for row in token_rows for token_id in row]
        input_ids = torch.tensor([flat_ids], dtype=torch.long, device=self.device)
        image_mask = input_ids == self._config.image_token_id
        model_ids = input_ids.masked_fill(image_mask, 0)
        inputs_embeds = self.model.model.language_model.embed(model_ids)
        packed_image_features = [
            features for features in image_features if features is not None
        ]
        feature_count = sum(int(features.shape[0]) for features in packed_image_features)
        image_token_count = sum(
            row.count(self._config.image_token_id) for row in token_rows
        )
        if feature_count != image_token_count:
            raise RuntimeError(
                f"encoded {feature_count} image features for "
                f"{image_token_count} image tokens"
            )
        if packed_image_features:
            inputs_embeds.masked_scatter_(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                torch.cat(packed_image_features),
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
                    for batch_idx, length in zip(
                        batch_indices, lengths, strict=True
                    )
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
        if bool((slot_mapping < 0).any()):
            raise RuntimeError("packed prefill resolved an unreserved KV slot")
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
        if (
            self._decode_megakernel is not None
            and self._decode_megakernel.supports(batch_size)
        ):
            with torch.cuda.stream(slot.compute_stream):
                self._decode_megakernel.run(slot, batch_size)
            return
        self._eager_decode(slot, batch_size)


__all__ = ["Gemma4Runtime"]
