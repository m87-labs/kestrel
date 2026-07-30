"""Gemma 4 implementation of ``kestrel.models.PromptTemplate``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from kestrel.models.protocols import ChatTemplate, PrefixSuffix, QueryTemplate


BOS_ID = 2
EOS_ID = 1
TOOL_RESPONSE_ID = 50
TURN_ID = 105
END_OF_TURN_ID = 106
NEWLINE_ID = 107
THINK_ID = 98
END_OF_CHANNEL_ID = 101
SYSTEM_ROLE_ID = 9731
USER_ROLE_ID = 2364
MODEL_ROLE_ID = 4368
START_OF_IMAGE_ID = 255_999
IMAGE_TOKEN_ID = 258_880
END_OF_IMAGE_ID = 258_882

_TURN_USER_PREFIX_IDS = [TURN_ID, USER_ROLE_ID, NEWLINE_ID]
_TURN_MODEL_PREFIX_IDS = [END_OF_TURN_ID, NEWLINE_ID, TURN_ID, MODEL_ROLE_ID, NEWLINE_ID]
_SYSTEM_TURN_THINK_IDS = [
    TURN_ID,
    SYSTEM_ROLE_ID,
    NEWLINE_ID,
    THINK_ID,
    NEWLINE_ID,
    END_OF_TURN_ID,
    NEWLINE_ID,
]
_TURN_USER_PREFIX_WITH_THINK_IDS = _SYSTEM_TURN_THINK_IDS + _TURN_USER_PREFIX_IDS


@dataclass(frozen=True)
class Gemma4PromptTemplate:
    model_name: str
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    answer_id: int = END_OF_CHANNEL_ID
    thinking_id: int = THINK_ID
    coord_id: int = 0
    size_id: int = 0
    start_ground_points_id: int = 0
    end_ground_id: int = 0

    def caption(self, length: str) -> Optional[List[int]]:
        return None

    def query(self) -> Optional[QueryTemplate]:
        stop_token_ids = []
        if self.model_name.rsplit("/", 1)[-1].endswith("-it"):
            stop_token_ids = [END_OF_TURN_ID, TOOL_RESPONSE_ID]
        return QueryTemplate(
            prefix=list(_TURN_USER_PREFIX_IDS),
            answer_prefix=list(_TURN_MODEL_PREFIX_IDS),
            reasoning_prefix=list(_TURN_MODEL_PREFIX_IDS),
            post_reasoning_prefix=[],
            prefix_when_reasoning=list(_TURN_USER_PREFIX_WITH_THINK_IDS),
            stop_token_ids=stop_token_ids,
        )

    def chat(self) -> Optional[ChatTemplate]:
        return None

    def detect(self) -> Optional[PrefixSuffix]:
        return None

    def point(self) -> Optional[PrefixSuffix]:
        return None

    def segment(self) -> Optional[PrefixSuffix]:
        return None


__all__ = [
    "BOS_ID",
    "END_OF_IMAGE_ID",
    "END_OF_TURN_ID",
    "EOS_ID",
    "Gemma4PromptTemplate",
    "IMAGE_TOKEN_ID",
    "START_OF_IMAGE_ID",
    "THINK_ID",
    "TOOL_RESPONSE_ID",
]
