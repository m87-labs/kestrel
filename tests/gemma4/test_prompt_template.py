"""Gemma4PromptTemplate produces the expected Gemma 4 chat tokens."""

from __future__ import annotations

from types import SimpleNamespace

from kestrel.models.protocols import PromptTemplate
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
_MODEL_ID = "google/gemma-4-E2B-it"
_BASE_MODEL_ID = "google/gemma-4-E2B"
_QUESTION_TOKENS = [17, 23]


def test_satisfies_prompt_template_protocol():
    assert isinstance(Gemma4PromptTemplate(_MODEL_ID), PromptTemplate)


def test_query_template_shape_matches_gemma4_chat_template():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None
    expected = [
        BOS_ID,
        TURN_ID,
        USER_ROLE_ID,
        NEWLINE_ID,
        *_QUESTION_TOKENS,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]
    assembled = [
        pt.bos_id,
        *q.prefix,
        *_QUESTION_TOKENS,
        *q.answer_prefix,
    ]
    assert assembled == expected


def test_query_reasoning_prefix_matches_gemma4_thinking_template():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None and q.prefix_when_reasoning is not None
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
        *_QUESTION_TOKENS,
        END_OF_TURN_ID,
        NEWLINE_ID,
        TURN_ID,
        MODEL_ROLE_ID,
        NEWLINE_ID,
    ]
    assembled = [
        pt.bos_id,
        *q.prefix_when_reasoning,
        *_QUESTION_TOKENS,
        *q.reasoning_prefix,
    ]
    assert assembled == expected


def test_reasoning_image_follows_user_turn_opener():
    pt = Gemma4PromptTemplate(_MODEL_ID)
    q = pt.query()
    assert q is not None and q.prefix_when_reasoning is not None
    prompt_ids = [
        pt.bos_id,
        *q.prefix_when_reasoning,
        *_QUESTION_TOKENS,
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
        *([IMAGE_TOKEN_ID] * 3),
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
        *_QUESTION_TOKENS,
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
