"""Qwen35PromptTemplate produces the expected Qwen 3.5 chat tokens."""

from __future__ import annotations

from functools import lru_cache

from kestrel.models.protocols import PromptTemplate
from kestrel.models.qwen35.qwen_image import preprocess_image
from kestrel.models.qwen35.prompt_template import (
    END_OF_TEXT_ID,
    IMAGE_PAD_ID,
    IM_END_ID,
    IM_START_ID,
    Qwen35PromptTemplate,
    THINK_END_ID,
    THINK_START_ID,
    VISION_END_ID,
    VISION_START_ID,
)
from tokenizers import Tokenizer


_MODEL_ID = "Qwen/Qwen3.5-2B"
_QUESTION = "What is 2+2?"
_USER_ID = 846
_ASSISTANT_ID = 74_455
_NEWLINE_ID = 198
_DOUBLE_NEWLINE_ID = 271


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_MODEL_ID)


def _encode(text: str) -> list[int]:
    return list(_tokenizer().encode(text, add_special_tokens=False).ids)


def test_satisfies_prompt_template_protocol():
    assert isinstance(Qwen35PromptTemplate(), PromptTemplate)


def test_query_template_shape_matches_qwen_chat_template():
    pt = Qwen35PromptTemplate()
    q = pt.query()
    assert q is not None
    user_tokens = _encode(_QUESTION)
    expected = [
        IM_START_ID,
        _USER_ID,
        _NEWLINE_ID,
        *user_tokens,
        IM_END_ID,
        _NEWLINE_ID,
        IM_START_ID,
        _ASSISTANT_ID,
        _NEWLINE_ID,
        THINK_START_ID,
        _DOUBLE_NEWLINE_ID,
        THINK_END_ID,
        _DOUBLE_NEWLINE_ID,
    ]
    assembled = [pt.bos_id] + list(q.prefix) + user_tokens + list(q.answer_prefix)
    assert assembled == expected


def test_query_reasoning_prefix_matches_qwen_thinking_template():
    pt = Qwen35PromptTemplate()
    q = pt.query()
    assert q is not None
    user_tokens = _encode(_QUESTION)
    expected = [
        IM_START_ID,
        _USER_ID,
        _NEWLINE_ID,
        *user_tokens,
        IM_END_ID,
        _NEWLINE_ID,
        IM_START_ID,
        _ASSISTANT_ID,
        _NEWLINE_ID,
        THINK_START_ID,
        _NEWLINE_ID,
    ]
    assembled = [pt.bos_id] + list(q.prefix) + user_tokens + list(q.reasoning_prefix)
    assert assembled == expected


def test_image_query_template_shape_matches_qwen_chat_template():
    import numpy as np

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 255
    _, image_grid_thw = preprocess_image(image)

    pt = Qwen35PromptTemplate()
    q = pt.query()
    assert q is not None
    user_tokens = _encode(_QUESTION)
    image_token_count = int(image_grid_thw.prod(-1).sum().item()) // 4
    assert image_token_count > 0
    expected = [
        IM_START_ID,
        _USER_ID,
        _NEWLINE_ID,
        VISION_START_ID,
        *([IMAGE_PAD_ID] * image_token_count),
        VISION_END_ID,
        *user_tokens,
        IM_END_ID,
        _NEWLINE_ID,
        IM_START_ID,
        _ASSISTANT_ID,
        _NEWLINE_ID,
        THINK_START_ID,
        _DOUBLE_NEWLINE_ID,
        THINK_END_ID,
        _DOUBLE_NEWLINE_ID,
    ]
    assembled = [
        pt.bos_id,
        *q.prefix,
        VISION_START_ID,
        *([IMAGE_PAD_ID] * image_token_count),
        VISION_END_ID,
        *user_tokens,
        *q.answer_prefix,
    ]
    assert assembled == expected


def test_prompt_template_magic_ids():
    pt = Qwen35PromptTemplate()
    assert pt.bos_id == IM_START_ID
    assert pt.eos_id == END_OF_TEXT_ID
    assert pt.answer_id == THINK_END_ID
    assert pt.thinking_id == THINK_START_ID


def test_non_query_skills_return_none():
    pt = Qwen35PromptTemplate()
    assert pt.caption("short") is None
    assert pt.detect() is None
    assert pt.point() is None
    assert pt.segment() is None
