"""Qwen 3.5 implementation of ``kestrel.models.PromptTemplate``.

The tokenizer chat template for a single user turn expands to:

    <|im_start|>user\n
    <question>
    <|im_end|>\n
    <|im_start|>assistant\n
    <think>\n\n</think>\n\n

when ``enable_thinking=False``. With thinking enabled, the assistant
prefix ends at ``<think>\n`` and the model later emits ``</think>``
before the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from kestrel.models.protocols import ChatTemplate, PrefixSuffix, QueryTemplate


IM_START_ID = 248_045
IM_END_ID = 248_046
VISION_START_ID = 248_053
VISION_END_ID = 248_054
IMAGE_PAD_ID = 248_056
VIDEO_PAD_ID = 248_057
THINK_START_ID = 248_068
THINK_END_ID = 248_069
END_OF_TEXT_ID = 248_044

_USER_ID = 846
_ASSISTANT_ID = 74_455
_NEWLINE_ID = 198
_DOUBLE_NEWLINE_ID = 271

_USER_PREFIX_IDS = [_USER_ID, _NEWLINE_ID]
_ASSISTANT_EMPTY_THINK_IDS = [
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
_ASSISTANT_THINK_IDS = [
    IM_END_ID,
    _NEWLINE_ID,
    IM_START_ID,
    _ASSISTANT_ID,
    _NEWLINE_ID,
    THINK_START_ID,
    _NEWLINE_ID,
]


@dataclass(frozen=True)
class Qwen35PromptTemplate:
    """Prompt template for Qwen 3.5 chat models."""

    # ``QuerySkill`` always prepends a single ``bos_id``. Qwen's chat
    # template has no BOS token there, so this field carries the first
    # template token instead.
    bos_id: int = IM_START_ID
    eos_id: int = END_OF_TEXT_ID
    answer_id: int = THINK_END_ID
    thinking_id: int = THINK_START_ID
    coord_id: int = 0
    size_id: int = 0
    start_ground_points_id: int = 0
    end_ground_id: int = 0

    def caption(self, length: str) -> Optional[List[int]]:
        return None

    def query(self) -> Optional[QueryTemplate]:
        return QueryTemplate(
            prefix=list(_USER_PREFIX_IDS),
            answer_prefix=list(_ASSISTANT_EMPTY_THINK_IDS),
            reasoning_prefix=list(_ASSISTANT_THINK_IDS),
            post_reasoning_prefix=[_DOUBLE_NEWLINE_ID],
            prefix_when_reasoning=None,
            stop_token_ids=[IM_END_ID],
        )

    def chat(self) -> Optional[ChatTemplate]:
        # A turn renders as ``<|im_start|>{role}\n{content}<|im_end|>\n`` and
        # the assistant generation opener mirrors the single-turn ``query``
        # template's thinking block. The role word ("system"/"user"/
        # "assistant") is encoded by the chat skill via the tokenizer, so a
        # single user turn reproduces the proven ``query`` prompt exactly.
        return ChatTemplate(
            bos=[],
            turn_prefix=[IM_START_ID],
            role_suffix=[_NEWLINE_ID],
            turn_suffix=[IM_END_ID, _NEWLINE_ID],
            assistant_open=[
                THINK_START_ID,
                _DOUBLE_NEWLINE_ID,
                THINK_END_ID,
                _DOUBLE_NEWLINE_ID,
            ],
            assistant_open_reasoning=[THINK_START_ID, _NEWLINE_ID],
            # Empty: after </think> Qwen emits the "\n\n" + answer on its own, so
            # we don't force it. (Forcing would require constrained decoding,
            # which the Qwen decode slot doesn't support.) The "\n\n" is kept in
            # the answer verbatim — matching vLLM's Qwen3 reasoning parser, which
            # returns everything after </think> as content with no strip.
            post_reasoning_prefix=[],
            turn_end_ids=[IM_END_ID],
            roles={"system": "system", "user": "user", "assistant": "assistant"},
            supports_system=True,
        )

    def detect(self) -> Optional[PrefixSuffix]:
        return None

    def point(self) -> Optional[PrefixSuffix]:
        return None

    def segment(self) -> Optional[PrefixSuffix]:
        return None


__all__ = [
    "END_OF_TEXT_ID",
    "IMAGE_PAD_ID",
    "IM_END_ID",
    "IM_START_ID",
    "Qwen35PromptTemplate",
    "THINK_END_ID",
    "THINK_START_ID",
    "VIDEO_PAD_ID",
    "VISION_END_ID",
    "VISION_START_ID",
]
