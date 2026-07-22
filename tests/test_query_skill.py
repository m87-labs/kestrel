"""QuerySkill.build_prompt_tokens behaviour around the new
``QueryTemplate.prefix_when_reasoning`` field."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest

from kestrel.models.moondream.config import TokenizerConfig
from kestrel.models.protocols import QueryTemplate
from kestrel.models.moondream.skills.query import QueryRequest, QuerySkill
from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep


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
    def __init__(
        self,
        *,
        prefix_when_reasoning: Optional[List[int]] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> None:
        self.prompt_template = _PromptTemplate(
            _query_template=QueryTemplate(
                prefix=[10, 11],
                answer_prefix=[20],
                reasoning_prefix=[21],
                post_reasoning_prefix=[],
                prefix_when_reasoning=prefix_when_reasoning,
                stop_token_ids=stop_token_ids or [],
            )
        )
        self.tokenizer = _Tokenizer()


class _VisibleTokenizer(_Tokenizer):
    def decode(self, token_ids) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


def _ids(tokens) -> List[int]:
    return [t.token_id for t in tokens]


def test_query_template_stop_ids_default_empty() -> None:
    template = QueryTemplate(
        prefix=[],
        answer_prefix=[],
        reasoning_prefix=[],
        post_reasoning_prefix=[],
    )

    assert template.stop_token_ids == []


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


def test_tokenizer_config_preserves_query_stop_ids() -> None:
    config = TokenizerConfig(
        templates={
            "query": {
                "prefix": [10],
                "answer_prefix": [20],
                "reasoning_prefix": [21],
                "post_reasoning_prefix": [],
                "stop_token_ids": [88],
            }
        }
    )

    template = config.query()

    assert template is not None
    assert template.stop_token_ids == [88]


@pytest.mark.parametrize("stop_id", [88, 99])
def test_query_stop_tokens_are_not_emitted(stop_id: int) -> None:
    runtime = _Runtime(stop_token_ids=[88])
    runtime.tokenizer = _VisibleTokenizer()
    context = QueryRequest(
        question="hi",
        image=None,
        reasoning=False,
        stream=True,
    )
    skill = QuerySkill()
    state = skill.create_state(runtime, SimpleNamespace(), context)

    state.consume_step(runtime, DecodeStep(TextToken(token_id=42), position=0))
    state.consume_step(runtime, DecodeStep(TextToken(token_id=stop_id), position=1))

    assert state.pop_stream_delta(runtime) == "<42>"
    result = state.finalize(runtime, reason="stop")
    assert result.text == "<42>"
    assert result.output["answer"] == "<42>"
    assert _ids(result.tokens) == [42, stop_id]


def test_reasoning_stop_token_is_not_emitted() -> None:
    runtime = _Runtime(stop_token_ids=[88])
    runtime.tokenizer = _VisibleTokenizer()
    context = QueryRequest(
        question="hi",
        image=None,
        reasoning=True,
        stream=True,
    )
    state = QuerySkill().create_state(runtime, SimpleNamespace(), context)

    state.consume_step(runtime, DecodeStep(TextToken(token_id=42), position=0))
    state.consume_step(runtime, DecodeStep(TextToken(token_id=88), position=1))

    assert state.pop_stream_delta(runtime) is None
    result = state.finalize(runtime, reason="stop")
    assert result.output == {
        "answer": "",
        "reasoning": {"text": "<42>", "grounding": []},
    }
    assert _ids(result.tokens) == [42, 88]
