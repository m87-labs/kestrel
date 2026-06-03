"""QuerySkill.build_prompt_tokens behaviour around the new
``QueryTemplate.prefix_when_reasoning`` field."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

from kestrel.models.protocols import QueryTemplate
from kestrel.models.moondream.skills.query import QueryRequest, QuerySkill


@dataclass
class _PromptTemplate:
    bos_id: int = 2
    eos_id: int = 99
    answer_id: int = 0
    thinking_id: int = 0
    coord_id: int = 0
    size_id: int = 0
    start_ground_points_id: int = 0
    end_ground_id: int = 0
    _query_template: Optional[QueryTemplate] = None

    def query(self) -> Optional[QueryTemplate]:
        return self._query_template


class _Tokenizer:
    """``runtime.tokenizer.encode(prompt).ids`` — minimal shape."""

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[42])


class _Runtime:
    def __init__(self, *, prefix_when_reasoning: Optional[List[int]] = None) -> None:
        self.prompt_template = _PromptTemplate(
            _query_template=QueryTemplate(
                prefix=[10, 11],
                answer_prefix=[20],
                reasoning_prefix=[21],
                post_reasoning_prefix=[],
                prefix_when_reasoning=prefix_when_reasoning,
            )
        )
        self.tokenizer = _Tokenizer()


def _ids(tokens) -> List[int]:
    return [t.token_id for t in tokens]


def test_query_skill_uses_prefix_when_reasoning_override() -> None:
    runtime = _Runtime(prefix_when_reasoning=[30, 31, 32])
    req = QueryRequest(
        question="hi", image=None, reasoning=True, stream=False,
    )
    skill = QuerySkill()
    tokens = skill.build_prompt_tokens(runtime, req)
    # bos + prefix_when_reasoning + encoded question + reasoning_prefix
    assert _ids(tokens) == [2, 30, 31, 32, 42, 21]


def test_query_skill_falls_back_to_prefix_when_override_unset() -> None:
    runtime = _Runtime(prefix_when_reasoning=None)
    req = QueryRequest(
        question="hi", image=None, reasoning=True, stream=False,
    )
    skill = QuerySkill()
    tokens = skill.build_prompt_tokens(runtime, req)
    # bos + prefix + encoded question + reasoning_prefix
    assert _ids(tokens) == [2, 10, 11, 42, 21]


def test_query_skill_non_reasoning_ignores_override() -> None:
    runtime = _Runtime(prefix_when_reasoning=[30, 31, 32])
    req = QueryRequest(
        question="hi", image=None, reasoning=False, stream=False,
    )
    skill = QuerySkill()
    tokens = skill.build_prompt_tokens(runtime, req)
    # bos + prefix + encoded question + answer_prefix  (override is reasoning-only)
    assert _ids(tokens) == [2, 10, 11, 42, 20]
