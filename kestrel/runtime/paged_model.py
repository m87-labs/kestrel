"""Descriptor-driven paged multimodal autoregressive runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.nn import functional as F

from kestrel.kv_cache import KVMemoryPool, LayeredPagedKV
from kestrel.models.registry import get_spec
from kestrel.runtime import ExecutionShape, SequenceState, TextToken, Token
from kestrel.runtime.paged_resources import create_paged_runtime_resources
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.staging import AsyncPreprocessor, BatchedTensorStager


@dataclass(frozen=True)
class PagedModelOps:
    load_model: Callable[..., Any]
    configure_model: Callable[[Any, Any], None]
    prompt_template: Callable[[str], Any]
    preprocess_image: Callable[[torch.dtype], Callable[[Any], Any]]
    config: Callable[[Any], Any]
    text_config: Callable[[Any], Any]
    max_seq_length: Callable[[Any], int]
    image_prefix_length: int
    paged_kv_layout: Callable[[Any], tuple[Any, Any]]
    create_generated_decode: Callable[[Any], Any]
    image_record: Callable[[Any], Mapping[str, torch.Tensor]]
    encode_images: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]
    image_token_count: Callable[[Any], int]
    prepare_prompt: Callable[..., tuple[list[Token], int, int]]
    embed_row: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    prefill: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    eager_decode: Callable[[Any, Any, int], None]


class PagedMultimodalRuntime:
    def __init__(
        self,
        cfg: Any,
        *,
        ops: PagedModelOps,
        max_lora_rank: int | None = None,
        kv_pool: KVMemoryPool | None = None,
        compute_stream: Any = None,
    ) -> None:
        del max_lora_rank
        self._ops = ops
        self._cfg = cfg
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

        from tokenizers import Tokenizer

        self._model_name = cfg.model
        self.model = ops.load_model(
            self._model_name,
            device=self.device,
            dtype=self.dtype,
        )
        self._config = ops.config(self.model)
        ops.configure_model(self, cfg)
        self.tokenizer = Tokenizer.from_pretrained(self._model_name)
        self.tokenizer.post_processor = None
        self.prompt_template = ops.prompt_template(self._model_name)

        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        self.spec = None
        self.max_batch_size = cfg.max_batch_size
        self.page_size = cfg.page_size
        self._kv_cache_pages = cfg.kv_cache_pages
        self.max_seq_length = ops.max_seq_length(self._config)
        self.image_prefix_length = ops.image_prefix_length
        self._vision_stager = BatchedTensorStager(
            capacity=self.max_batch_size,
            device=self.device,
            with_numpy={"pixel_values": False},
        )
        self._image_preprocessor = AsyncPreprocessor(
            ops.preprocess_image(self.dtype),
            workers=derive_preprocessing_workers(self.max_batch_size),
        )

        text_config = ops.text_config(self._config)
        resources = create_paged_runtime_resources(
            device=self.device,
            dtype=self.dtype,
            max_batch_size=self.max_batch_size,
            page_size=self.page_size,
            kv_cache_pages=self._kv_cache_pages,
            vocab_size=text_config.vocab_size,
            hidden_dim=text_config.hidden_size,
            compute_stream=compute_stream,
        )
        self._compute_stream = resources.compute_stream
        self._copy_stream = resources.copy_stream
        self.graph_capture_lock = resources.graph_capture_lock
        self.page_table = resources.page_table
        self.max_batch_slots = resources.max_batch_slots
        self._decode_slot_rows = resources.decode_rows
        self._padding_batch_idx = resources.padding_batch_idx
        self._prefill_slot = resources.prefill_slot
        self._prefill_slot_in_use = False
        self.prefill_slots = (resources.prefill_slot,)
        self._decode_slots = resources.decode_slots
        self.decode_slots = resources.decode_slots
        self.active_sequences: dict[int, SequenceState] = {}

        layer_specs, source_layers = ops.paged_kv_layout(text_config)
        self._kv_cache = LayeredPagedKV.allocate(
            layer_specs=layer_specs,
            source_layer_idx=source_layers,
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )
        self._decode_megakernel = ops.create_generated_decode(self)
        self.prefix_cache = None

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
        records = [self._ops.image_record(item) for item, _ in groups]
        staged = self._vision_stager.stage(records)
        packed = self._ops.encode_images(self.model, staged).detach()
        lengths = [self._ops.image_token_count(item) for item, _ in groups]
        if packed.shape[0] != sum(lengths):
            raise RuntimeError(
                f"vision encoder returned {packed.shape[0]} tokens "
                f"for declared split {lengths}"
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
        return self._compute_stream

    @property
    def kv_pool(self) -> KVMemoryPool:
        return self._kv_pool

    @property
    def copy_stream(self):
        return self._copy_stream

    @property
    def vocab_size(self) -> int:
        return int(self._ops.text_config(self._config).vocab_size)

    def skills(self):
        return get_spec(self.model_name).skills()

    def tasks(self) -> tuple[str, ...]:
        return self.skills().names()

    def preprocess_image_async(self, image):
        return self._image_preprocessor.submit(image)

    def shutdown(self) -> None:
        self._image_preprocessor.shutdown()

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
            raise RuntimeError("Prefill slot pool exhausted")
        if slot_id is not None and slot_id != 0:
            raise ValueError(f"Invalid prefill_slot_id {slot_id}")
        self._prefill_slot_in_use = True
        return self._prefill_slot

    def release_prefill_slot(self, slot: Any) -> None:
        self._prefill_slot_in_use = False

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int:
        raise NotImplementedError("LoRA adapters are not supported")

    def release_adapter_slot(self, slot: int) -> None:
        raise NotImplementedError("LoRA adapters are not supported")

    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool = False,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
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
        prompt_tokens: Sequence[Token],
        *,
        image: Any = None,
        image_crops: Any = None,
        max_new_tokens: int | None = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> Any:
        from kestrel.runtime.state import PreparedSequence, _CacheLookupResult

        tokens, image_tokens, text_length = self._ops.prepare_prompt(
            prompt_tokens,
            image=image,
            image_crops=image_crops,
            config=self._config,
            prompt_template=self.prompt_template,
        )
        prompt_length = len(tokens)
        new_tokens = max_new_tokens or 128
        target_length = max(
            text_length + self.image_prefix_length + new_tokens,
            prompt_length + new_tokens,
        )
        if target_length > self.max_seq_length:
            raise ValueError(
                f"Requested length {target_length} exceeds "
                f"max_seq_length={self.max_seq_length}"
            )
        if self._available_batch_slots() <= 0:
            raise RuntimeError("Cannot reserve batch slot")
        batch_idx = self.page_table.allocate()
        try:
            self.page_table.reserve(batch_idx, target_length)
        except Exception:
            self.page_table.erase(batch_idx, 0)
            raise
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_length,
            max_length=target_length,
            prompt_length=prompt_length,
            image_length=image_tokens,
            last_hidden=None,
            lora_slot=lora_slot,
            cache_tokens=None,
            cache_lock_node=None,
            cache_owned_page_count=0,
            reused_page_count=0,
        )
        return PreparedSequence(
            state=state,
            tokens_list=tokens,
            cache_tokens=[],
            cache_result=_CacheLookupResult(
                match=None,
                skip_positions=0,
                temp_lock_node=None,
                can_reuse=False,
                namespace=None,
            ),
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
        batch_indices = [
            int(prepared.state.batch_idx) for prepared in prepared_sequences
        ]
        self.page_table.commit_block_table(batch_indices)

        lengths = []
        embed_rows = []
        id_rows = []
        image_features = self._image_features_for_batch(image_crops_list)
        for row, (prepared, image, crops) in enumerate(
            zip(prepared_sequences, images, image_crops_list)
        ):
            prefill_slot.batch_idx[row] = prepared.state.batch_idx
            tokens = prepared.tokens_list
            if not tokens or not all(isinstance(token, TextToken) for token in tokens):
                raise ValueError("prefill requires non-empty text-token rows")
            token_ids = torch.tensor(
                [[int(token.token_id) for token in tokens]],
                dtype=torch.long,
                device=self.device,
            )
            embeds, model_ids = self._ops.embed_row(
                self.model,
                self._config,
                token_ids,
                image=image,
                crops=crops,
                image_features=image_features[row],
            )
            lengths.append(token_ids.shape[1])
            embed_rows.append(embeds)
            id_rows.append(model_ids)

        max_length = max(lengths)
        for index, length in enumerate(lengths):
            pad = max_length - length
            if pad:
                embed_rows[index] = F.pad(embed_rows[index], (0, 0, 0, pad))
                id_rows[index] = F.pad(id_rows[index], (0, pad), value=0)
        inputs_embeds = torch.cat(embed_rows)
        input_ids = torch.cat(id_rows)
        position_ids = torch.arange(
            max_length,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0).expand(batch_size, -1)
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=prefill_slot.batch_idx[:batch_size],
            positions=position_ids,
        )
        hidden_rows, logits = self._ops.prefill(
            self,
            inputs_embeds,
            input_ids,
            position_ids,
            slot_mapping,
            lengths,
        )
        for row, prepared in enumerate(prepared_sequences):
            prepared.state.last_hidden = hidden_rows[row].detach()
        return logits

    def finalize_prepared_sequence_after_prefill(self, prepared: Any) -> None:
        self.active_sequences[prepared.state.batch_idx] = prepared.state

    def abort_prepared_sequence(self, prepared: Any) -> None:
        self.active_sequences.pop(prepared.state.batch_idx, None)
        self._release_batch_idx(prepared.state.batch_idx)

    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None:
        pass

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)
        self._release_batch_idx(state.batch_idx)

    def _release_batch_idx(self, batch_idx: int) -> None:
        if batch_idx not in self.page_table.free_batch_idx:
            self.page_table.erase(batch_idx, 0)

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
        self._ops.eager_decode(self, slot, batch_size)


__all__ = ["PagedModelOps", "PagedMultimodalRuntime"]
