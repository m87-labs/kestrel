"""Qwen35ChatSkill renders multi-turn chat; a single user turn reproduces the
proven single-turn ``query`` prompt exactly, and image parts become markers."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

from kestrel.runtime.tokens import ImageMarker
from kestrel.models.qwen35.prompt_template import (
    Qwen35PromptTemplate,
    THINK_START_ID,
)
from tokenizers import Tokenizer


_MODEL_ID = "Qwen/Qwen3.5-2B"
_QUESTION = "What is 2+2?"
_NEWLINE_ID = 198
# bytes are irrelevant here — only the emitted ImageMarker / image count matter.
_IMG = "data:image/png;base64,aGk="


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_MODEL_ID)


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(prompt_template=Qwen35PromptTemplate(), tokenizer=_tokenizer())


def _ctx(messages, reasoning: bool = False):
    from kestrel.models.qwen35.skills import Qwen35ChatSkill

    return (
        Qwen35ChatSkill()
        .build_request(None, {"messages": messages, "reasoning": reasoning}, None)
        .request_context
    )


def _norm(tokens):
    return [("IMG", t.index) if isinstance(t, ImageMarker) else t.token_id for t in tokens]


def _render(messages, reasoning: bool = False):
    from kestrel.models.qwen35.skills import Qwen35ChatSkill

    return _norm(
        Qwen35ChatSkill().build_prompt_tokens(_runtime(), _ctx(messages, reasoning))
    )


def test_single_user_turn_reproduces_query_prompt() -> None:
    # The chat skeleton for one user turn must equal the single-turn query
    # prompt (BOS + query prefix + user text + answer_prefix), so chat inherits
    # the proven query behavior.
    chat = _render([{"role": "user", "content": _QUESTION}])
    pt = Qwen35PromptTemplate()
    q = pt.query()
    user = list(_tokenizer().encode(_QUESTION, add_special_tokens=False).ids)
    query_prompt = [pt.bos_id] + list(q.prefix) + user + list(q.answer_prefix)
    assert chat == query_prompt


def test_reasoning_opener_uses_thinking_prefix() -> None:
    # reasoning=True opens the thinking block (<think>\n) instead of the empty
    # think skeleton the non-reasoning opener uses.
    chat = _render([{"role": "user", "content": _QUESTION}], reasoning=True)
    assert chat[-2:] == [THINK_START_ID, _NEWLINE_ID]


def test_image_part_becomes_marker_at_content_position() -> None:
    ctx = _ctx([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _IMG}},
            {"type": "text", "text": "describe"},
        ]},
    ])
    assert len(ctx.images) == 1
    markers = [t for t in _norm_ctx(ctx) if isinstance(t, tuple)]
    assert markers == [("IMG", 0)]


def test_multiple_images_render_markers_in_order() -> None:
    ctx = _ctx([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _IMG}},
            {"type": "image_url", "image_url": {"url": _IMG}},
            {"type": "text", "text": "compare"},
        ]},
    ])
    assert len(ctx.images) == 2
    markers = [t for t in _norm_ctx(ctx) if isinstance(t, tuple)]
    assert markers == [("IMG", 0), ("IMG", 1)]


def _norm_ctx(ctx):
    from kestrel.models.qwen35.skills import Qwen35ChatSkill

    return _norm(Qwen35ChatSkill().build_prompt_tokens(_runtime(), ctx))
