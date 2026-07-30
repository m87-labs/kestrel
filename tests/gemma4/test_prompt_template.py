"""Gemma4PromptTemplate produces the expected Gemma 4 chat tokens."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

from kestrel.models.protocols import PromptTemplate
from kestrel.models.gemma4.image import preprocess_image
from kestrel.runtime.preprocessing import derive_image_insertion_offset
from kestrel.models.gemma4.prompt_template import (
    BOS_ID,
    END_OF_CHANNEL_ID,
    END_OF_IMAGE_ID,
    END_OF_TURN_ID,
    EOS_ID,
    IMAGE_TOKEN_ID,
    MODEL_ROLE_ID,
    NEWLINE_ID,
    START_OF_IMAGE_ID,
    SYSTEM_ROLE_ID,
    THINK_ID,
    TOOL_RESPONSE_ID,
    TURN_ID,
    USER_ROLE_ID,
    Gemma4PromptTemplate,
)
from tokenizers import Tokenizer


_MODEL_ID = "google/gemma-4-E2B-it"
_BASE_MODEL_ID = "google/gemma-4-E2B"
_QUESTION = "What is 2+2?"


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_MODEL_ID)


def _encode(text: str) -> list[int]:
    return list(_tokenizer().encode(text, add_special_tokens=False).ids)


def test_satisfies_prompt_template_protocol():
    assert isinstance(Gemma4PromptTemplate(_MODEL_ID), PromptTemplate)


def test_query_template_shape_matches_gemma4_chat_template():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None
    user_tokens = _encode(_QUESTION)
    expected = [
        BOS_ID,
        TURN_ID,
        USER_ROLE_ID,
        NEWLINE_ID,
        *user_tokens,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]
    assembled = [pt.bos_id] + list(q.prefix) + user_tokens + list(q.answer_prefix)
    assert assembled == expected


def test_query_reasoning_prefix_matches_gemma4_thinking_template():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None and q.prefix_when_reasoning is not None
    user_tokens = _encode(_QUESTION)
    expected = [
        BOS_ID,
        TURN_ID,
        SYSTEM_ROLE_ID,
        NEWLINE_ID,
        THINK_ID,
        NEWLINE_ID,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        USER_ROLE_ID,
        NEWLINE_ID,
        *user_tokens,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]
    assembled = [
        pt.bos_id,
        *q.prefix_when_reasoning,
        *user_tokens,
        *q.reasoning_prefix,
    ]
    assert assembled == expected


def test_image_query_template_shape_matches_gemma4_chat_template():
    from PIL import Image

    image_inputs = preprocess_image(Image.new("RGB", (96, 64), "red"))
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None
    user_tokens = _encode(_QUESTION)
    image_token_count = int(image_inputs.num_image_tokens)
    assert image_token_count > 0
    expected = [
        BOS_ID,
        TURN_ID,
        USER_ROLE_ID,
        NEWLINE_ID,
        START_OF_IMAGE_ID,
        *([IMAGE_TOKEN_ID] * image_token_count),
        END_OF_IMAGE_ID,
        *user_tokens,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]
    assembled = [
        pt.bos_id,
        *q.prefix,
        START_OF_IMAGE_ID,
        *([IMAGE_TOKEN_ID] * image_token_count),
        END_OF_IMAGE_ID,
        *user_tokens,
        *q.answer_prefix,
    ]
    assert assembled == expected


def test_reasoning_image_follows_user_turn_opener():
    from PIL import Image

    image_inputs = preprocess_image(Image.new("RGB", (96, 64), "red"))
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None and q.prefix_when_reasoning is not None
    user_tokens = _encode(_QUESTION)
    prompt_ids = [
        pt.bos_id,
        *q.prefix_when_reasoning,
        *user_tokens,
        *q.reasoning_prefix,
    ]
    prompt_tokens = [SimpleNamespace(token_id=token_id) for token_id in prompt_ids]
    offset = derive_image_insertion_offset(
        prompt_tokens,
        user_turn_opener=(TURN_ID, USER_ROLE_ID, NEWLINE_ID),
        fallback_offset=1 + len(q.prefix),
    )
    image_block = [
        START_OF_IMAGE_ID,
        *([IMAGE_TOKEN_ID] * image_inputs.num_image_tokens),
        END_OF_IMAGE_ID,
    ]

    assembled = prompt_ids[:offset] + image_block + prompt_ids[offset:]

    assert assembled == [
        BOS_ID,
        TURN_ID,
        SYSTEM_ROLE_ID,
        NEWLINE_ID,
        THINK_ID,
        NEWLINE_ID,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        USER_ROLE_ID,
        NEWLINE_ID,
        *image_block,
        *user_tokens,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]


def test_prompt_template_magic_ids():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    assert pt.bos_id == BOS_ID
    assert pt.eos_id == EOS_ID
    assert pt.answer_id == END_OF_CHANNEL_ID
    assert pt.thinking_id == THINK_ID

    query = pt.query()
    assert query is not None
    assert query.stop_token_ids == [
        END_OF_TURN_ID,
        TOOL_RESPONSE_ID,
    ]


def test_base_query_uses_only_model_eos() -> None:
    pt = Gemma4PromptTemplate(_BASE_MODEL_ID)
    query = pt.query()

    assert pt.eos_id == EOS_ID
    assert query is not None
    assert query.stop_token_ids == []


def test_non_query_skills_return_none():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    assert pt.caption("short") is None
    assert pt.detect() is None
    assert pt.point() is None
    assert pt.segment() is None
