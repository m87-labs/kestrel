"""Gemma descriptors for the shared paged multimodal runtime."""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import torch

from kestrel.kv_cache import KVMemoryPool
from kestrel.runtime import TextToken, Token
from kestrel.runtime.compilation import (
    canonicalize_immutable_scalar_buffers,
    materialize_dynamic_batch_domain,
)
from kestrel.runtime.paged_model import PagedModelOps, PagedMultimodalRuntime
from kestrel.runtime.prefill import project_padded_last_rows
from kestrel.runtime.preprocessing import derive_image_insertion_offset

from .generated_decode import create_generated_decode
from .image import IMAGE_SEQ_LENGTH, MAX_PATCHES, preprocess_image
from .loader import load_model
from .paged_cache import paged_kv_layout
from .prompt_template import (
    END_OF_IMAGE_ID,
    Gemma4PromptTemplate,
    NEWLINE_ID,
    START_OF_IMAGE_ID,
    TURN_ID,
    USER_ROLE_ID,
)


def _configure_model(runtime: Any, cfg: Any) -> None:
    vision = runtime.model.model.vision_tower
    canonicalize_immutable_scalar_buffers(vision.encoder)
    vision.encoder.forward = torch.compile(
        vision.encoder.forward,
        dynamic=True,
        fullgraph=False,
        options={"triton.cudagraphs": False},
    )
    config = runtime._config.vision_config

    def inputs(batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                (batch_size, MAX_PATCHES, config.hidden_size),
                dtype=runtime.dtype,
                device=runtime.device,
            ),
            torch.ones(
                (batch_size, MAX_PATCHES),
                dtype=torch.bool,
                device=runtime.device,
            ),
            torch.zeros(
                (batch_size, MAX_PATCHES, 2),
                dtype=torch.long,
                device=runtime.device,
            ),
        )

    materialize_dynamic_batch_domain(
        vision.encoder,
        max_batch_size=cfg.max_batch_size,
        inputs_for_batch=inputs,
        synchronize=lambda: torch.cuda.synchronize(runtime.device),
    )
    torch.cuda.empty_cache()


def _prepare_prompt(
    prompt_tokens: Sequence[Token],
    *,
    image: Any,
    image_crops: Any,
    config: Any,
    prompt_template: Any,
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
        + [TextToken(token_id=config.image_token_id)] * count
        + [TextToken(token_id=END_OF_IMAGE_ID)]
    )
    query = prompt_template.query()
    offset = derive_image_insertion_offset(
        tokens,
        user_turn_opener=(TURN_ID, USER_ROLE_ID, NEWLINE_ID),
        fallback_offset=1 + (len(query.prefix) if query else 0),
    )
    return tokens[:offset] + image_block + tokens[offset:], count + 2, text_length


def _embed_row(
    model: Any,
    config: Any,
    input_ids: torch.Tensor,
    *,
    image: Any,
    crops: Any,
    image_features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    language_model = model.model.language_model
    if image is None or crops is None:
        return language_model.embed_tokens(input_ids), input_ids
    if image_features is None:
        raise RuntimeError("missing encoded features for image row")
    image_mask = input_ids == config.image_token_id
    model_ids = input_ids.clone()
    model_ids[image_mask] = 0
    embeds = language_model.embed_tokens(model_ids)
    return (
        embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeds),
            image_features.to(device=embeds.device, dtype=embeds.dtype),
        ),
        model_ids,
    )


def _prefill(
    runtime: PagedMultimodalRuntime,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    slot_mapping: torch.Tensor,
    lengths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    text_config = runtime._config.text_config
    language_model = runtime.model.model.language_model
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
        kv_cache=runtime._kv_cache,
        cache_position_ids=position_ids,
        slot_mapping=slot_mapping,
    )
    rows, logits = project_padded_last_rows(hidden, lengths, runtime.model.lm_head)
    cap = text_config.final_logit_softcapping
    return rows, torch.tanh(logits / cap) * cap


def _eager_decode(
    runtime: PagedMultimodalRuntime,
    slot: Any,
    batch_size: int,
) -> None:
    batch_idx = slot.meta.batch_idx.gpu[:batch_size]
    input_pos = slot.meta.input_pos.gpu[:batch_size]
    slot.cache_position_ids[:batch_size, 0].copy_(input_pos)
    slot.position_ids[:batch_size, 0].copy_(input_pos)
    runtime.page_table.populate_paged_kv_metadata(
        batch_idx=batch_idx,
        input_pos=input_pos,
        out_page_table=slot.paged_kv_page_table[:batch_size],
        out_seqused_k=slot.paged_kv_seqlens_k[:batch_size],
    )
    slot.slot_mapping[:batch_size].copy_(
        runtime.page_table.build_slot_mapping(
            batch_idx=batch_idx,
            positions=slot.cache_position_ids[:batch_size],
        )
    )
    hidden = runtime.model.model.language_model(
        input_ids=slot.decode_token_ids[:batch_size].view(batch_size, 1),
        position_ids=input_pos.view(batch_size, 1),
        kv_cache=runtime._kv_cache,
        cache_position_ids=slot.cache_position_ids[:batch_size],
        slot_mapping=slot.slot_mapping[:batch_size],
        page_table=slot.paged_kv_page_table[:batch_size],
        paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
        paged_kv_use_sliding_window=True,
    )[:, 0]
    torch.mm(hidden, runtime.model.lm_head.weight.t(), out=slot.logits[:batch_size])
    cap = runtime._config.text_config.final_logit_softcapping
    slot.logits[:batch_size].div_(cap).tanh_().mul_(cap)
    slot.hidden_last[:batch_size].copy_(hidden)


def _encode_images(
    model: Any,
    staged: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model.model.get_image_features(
        staged["pixel_values"],
        staged["position_ids"],
    )


def _image_token_count(crops: Any) -> int:
    return int(crops.num_image_tokens)


_OPS = PagedModelOps(
    load_model=load_model,
    configure_model=_configure_model,
    prompt_template=Gemma4PromptTemplate,
    preprocess_image=lambda dtype: partial(preprocess_image, dtype=dtype),
    max_seq_length=lambda config: min(
        config.text_config.max_position_embeddings,
        2048,
    ),
    image_prefix_length=IMAGE_SEQ_LENGTH + 2,
    paged_kv_layout=paged_kv_layout,
    create_generated_decode=create_generated_decode,
    image_record=lambda crops: {
        "pixel_values": crops.pixel_values,
        "position_ids": crops.image_position_ids,
    },
    encode_images=_encode_images,
    image_token_count=_image_token_count,
    prepare_prompt=_prepare_prompt,
    embed_row=_embed_row,
    prefill=_prefill,
    eager_decode=_eager_decode,
)


def create_gemma4_runtime(
    cfg: Any,
    *,
    max_lora_rank: int | None = None,
    kv_pool: KVMemoryPool | None = None,
    compute_stream: Any = None,
) -> PagedMultimodalRuntime:
    return PagedMultimodalRuntime(
        cfg,
        ops=_OPS,
        max_lora_rank=max_lora_rank,
        kv_pool=kv_pool,
        compute_stream=compute_stream,
    )


__all__ = ["create_gemma4_runtime"]
