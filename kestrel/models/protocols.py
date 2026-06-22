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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class QueryTemplate:
    """Token sequences used by the question-answering skill."""

    prefix: List[int]
    answer_prefix: List[int]
    reasoning_prefix: List[int]
    post_reasoning_prefix: List[int]
    # Optional override for ``prefix`` when ``reasoning=True``. Set this
    # for models whose CoT mode needs a different *pre-question* structure
    # (e.g. Gemma 4 emits an extra ``<|turn>system\n<|think|>`` block to
    # activate thinking). ``None`` (default) keeps ``prefix`` for both
    # modes — Moondream's pre-question layout is reasoning-independent.
    prefix_when_reasoning: Optional[List[int]] = None


@dataclass(frozen=True)
class PrefixSuffix:
    """Prefix/suffix token IDs framing a structured-output skill."""

    prefix: List[int]
    suffix: List[int]


@dataclass(frozen=True)
class ChatTemplate:
    """Token framing for rendering a multi-turn chat into a prompt.

    Each turn is rendered by the chat skill as::

        turn_prefix  <role-word>  role_suffix  <content>  turn_suffix

    The role word (``roles[role]``) and the message content are encoded by
    the skill via ``runtime.tokenizer``; only the framing/special tokens
    live here, so a model never has to hard-code the token id of "user" or
    "assistant". After the final turn the skill appends the assistant
    opener — ``turn_prefix <assistant-word> role_suffix`` followed by
    ``assistant_open`` (or ``assistant_open_reasoning`` when reasoning is
    enabled). ``post_reasoning_prefix`` is force-injected once the model
    emits ``answer_id`` (the reasoning→answer boundary). ``turn_end_ids``
    are the token ids that terminate an assistant turn; the scheduler stops
    decoding on them in addition to ``eos_id``.
    """

    turn_prefix: List[int]
    role_suffix: List[int]
    turn_suffix: List[int]
    assistant_open: List[int]
    assistant_open_reasoning: List[int]
    post_reasoning_prefix: List[int]
    turn_end_ids: List[int]
    roles: Dict[str, str]
    bos: List[int] = field(default_factory=list)
    supports_system: bool = True


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

    def chat(self) -> Optional["ChatTemplate"]:
        """Token framing for multi-turn chat, or ``None``.

        ``None`` means the model has no native multi-turn chat format; the
        chat skill then falls back to flattening the conversation into the
        single-turn ``query`` prompt.
        """

    def detect(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the detect skill."""

    def point(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the point skill."""

    def segment(self) -> Optional[PrefixSuffix]:
        """Prefix/suffix for the segment skill, or ``None``."""


__all__ = [
    "ChatTemplate",
    "PrefixSuffix",
    "PromptTemplate",
    "QueryTemplate",
]
