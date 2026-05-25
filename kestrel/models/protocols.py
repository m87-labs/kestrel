"""Protocols for the prompt-template seam.

``runtime.prompt_template`` builds the per-skill prompt token sequences
and exposes the magic token IDs the skills key off (``answer_id``,
``coord_id``, etc.). Each model family supplies its own
``PromptTemplate`` implementation; skills consume it via this protocol
so a different model with a different prompt format can plug in without
touching skill code.

``runtime.tokenizer`` (free-form text ↔ token-id conversion) is typed
as the HuggingFace ``tokenizers.Tokenizer`` class today; no abstraction
exists for it yet because there is no concrete second implementation.
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class QueryTemplate:
    """Token sequences used by the question-answering skill."""

    prefix: List[int]
    answer_prefix: List[int]
    reasoning_prefix: List[int]
    post_reasoning_prefix: List[int]


@dataclass(frozen=True)
class PrefixSuffix:
    """Prefix/suffix token IDs framing a structured-output skill."""

    prefix: List[int]
    suffix: List[int]


@runtime_checkable
class PromptTemplate(Protocol):
    """Prompt builder + magic-token-ID surface used by skills.

    A model that does not support a given skill returns ``None`` from
    the corresponding builder; callers must handle that.
    """

    # --- Magic token IDs ---
    bos_id: int
    eos_id: int
    answer_id: int
    thinking_id: int
    coord_id: int
    size_id: int
    start_ground_points_id: int
    end_ground_id: int

    # --- Skill prompt builders ---
    def caption(self, length: str) -> Optional[List[int]]:
        """Token sequence for the caption skill at ``length``.

        Returns ``None`` if the model has no caption template, or raises
        ``ValueError`` if ``length`` is not a supported variant.
        """

    def query(self) -> Optional[QueryTemplate]:
        """Token sequences for the question-answering skill."""

    def detect(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the detect skill."""

    def point(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the point skill."""

    def segment(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the segment skill, or ``None``."""


__all__ = [
    "PrefixSuffix",
    "PromptTemplate",
    "QueryTemplate",
]
