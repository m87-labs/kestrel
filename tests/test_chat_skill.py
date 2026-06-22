"""ChatSkill: message validation, prompt-token rendering (native chat
template vs. flatten-into-query), and decode-state output shape."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest

from kestrel.models.protocols import ChatTemplate, QueryTemplate
from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep
from kestrel.skills.chat import (
    ChatRequest,
    ChatSkill,
    _ChatMessage,
    _flatten_messages,
)

# Synthetic token ids for a Qwen-like chat template.
IM_START, IM_END, NL, DNL, THINK_S, THINK_E = 900, 901, 198, 271, 800, 801


def _qwen_chat_template() -> ChatTemplate:
    return ChatTemplate(
        bos=[],
        turn_prefix=[IM_START],
        role_suffix=[NL],
        turn_suffix=[IM_END, NL],
        assistant_open=[THINK_S, DNL, THINK_E, DNL],
        assistant_open_reasoning=[THINK_S, NL],
        post_reasoning_prefix=[DNL],
        turn_end_ids=[IM_END],
        roles={"system": "system", "user": "user", "assistant": "assistant"},
        supports_system=True,
    )


def _md_query_template() -> QueryTemplate:
    return QueryTemplate(
        prefix=[10, 11],
        answer_prefix=[20],
        reasoning_prefix=[21],
        post_reasoning_prefix=[3],
    )


@dataclass
class _PromptTemplate:
    bos_id: int = 5
    eos_id: int = 99
    answer_id: int = THINK_E
    thinking_id: int = THINK_S
    coord_id: int = 0
    size_id: int = 0
    start_ground_points_id: int = 0
    end_ground_id: int = 0
    _chat: Optional[ChatTemplate] = None
    _query: Optional[QueryTemplate] = None

    def chat(self) -> Optional[ChatTemplate]:
        return self._chat

    def query(self) -> Optional[QueryTemplate]:
        return self._query


class _Tokenizer:
    """Reversible char-level fake: encode→code points, decode→chars."""

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def _chat_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_template=_PromptTemplate(_chat=_qwen_chat_template()),
        tokenizer=_Tokenizer(),
    )


def _flatten_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_template=_PromptTemplate(answer_id=3, _query=_md_query_template()),
        tokenizer=_Tokenizer(),
    )


def _ids(tokens) -> List[int]:
    return [t.token_id for t in tokens]


def _ords(text: str) -> List[int]:
    return [ord(c) for c in text]


# -- validation -----------------------------------------------------------


def test_build_request_accepts_minimal_conversation() -> None:
    built = ChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, None
    )
    ctx = built.request_context
    assert isinstance(ctx, ChatRequest)
    assert [(m.role, m.text) for m in ctx.messages] == [("user", "hi")]
    assert ctx.reasoning is False  # chat defaults to non-thinking


def test_build_request_requires_messages() -> None:
    with pytest.raises(ValueError, match="messages must be provided"):
        ChatSkill().build_request(None, {}, None)


def test_build_request_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ChatSkill().build_request(None, {"messages": []}, None)


def test_build_request_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        ChatSkill().build_request(
            None, {"messages": [{"role": "tool", "content": "x"}]}, None
        )


def test_build_request_system_must_be_first() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be brief"},
    ]
    with pytest.raises(ValueError, match="first message"):
        ChatSkill().build_request(None, {"messages": msgs}, None)


def test_build_request_last_must_be_user() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    with pytest.raises(ValueError, match="last message must be from 'user'"):
        ChatSkill().build_request(None, {"messages": msgs}, None)


def test_build_request_rejects_two_images() -> None:
    img = "data:image/png;base64,aGk="  # b"hi"
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "two?"},
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "image_url", "image_url": {"url": img}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="at most one image"):
        ChatSkill().build_request(None, {"messages": msgs}, None)


def test_build_request_extracts_image_from_data_url() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ],
        }
    ]
    built = ChatSkill().build_request(None, {"messages": msgs}, None)
    assert built.image == b"hi"
    assert built.request_context.image == b"hi"
    assert built.request_context.messages[-1].text == "what is this?"


def test_build_request_reasoning_opt_in_via_settings() -> None:
    built = ChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, {"reasoning": True}
    )
    assert built.request_context.reasoning is True


# -- rendering: native chat template --------------------------------------


def test_render_single_turn_chat() -> None:
    runtime = _chat_runtime()
    ctx = ChatRequest(
        messages=(_ChatMessage("user", "hi"),),
        image=None,
        reasoning=False,
        stream=False,
    )
    tokens = ChatSkill().build_prompt_tokens(runtime, ctx)
    expected = (
        [IM_START] + _ords("user") + [NL] + _ords("hi") + [IM_END, NL]
        + [IM_START] + _ords("assistant") + [NL] + [THINK_S, DNL, THINK_E, DNL]
    )
    assert _ids(tokens) == expected


def test_render_multi_turn_with_system_and_reasoning() -> None:
    runtime = _chat_runtime()
    ctx = ChatRequest(
        messages=(
            _ChatMessage("system", "be nice"),
            _ChatMessage("user", "a"),
            _ChatMessage("assistant", "b"),
            _ChatMessage("user", "c"),
        ),
        image=None,
        reasoning=True,
        stream=False,
    )
    tokens = ChatSkill().build_prompt_tokens(runtime, ctx)

    def turn(role: str, text: str) -> List[int]:
        return [IM_START] + _ords(role) + [NL] + _ords(text) + [IM_END, NL]

    expected = (
        turn("system", "be nice")
        + turn("user", "a")
        + turn("assistant", "b")
        + turn("user", "c")
        + [IM_START] + _ords("assistant") + [NL] + [THINK_S, NL]  # reasoning opener
    )
    assert _ids(tokens) == expected


# -- rendering: flatten into query ----------------------------------------


def test_render_flatten_single_turn_matches_query_shape() -> None:
    runtime = _flatten_runtime()
    ctx = ChatRequest(
        messages=(_ChatMessage("user", "hi"),),
        image=None,
        reasoning=False,
        stream=False,
    )
    tokens = ChatSkill().build_prompt_tokens(runtime, ctx)
    # bos + prefix + encoded(question) + answer_prefix
    assert _ids(tokens) == [5, 10, 11] + _ords("hi") + [20]


def test_flatten_messages_labels_history() -> None:
    msgs = [
        _ChatMessage("system", "sys"),
        _ChatMessage("user", "u1"),
        _ChatMessage("assistant", "a1"),
        _ChatMessage("user", "u2"),
    ]
    assert _flatten_messages(msgs) == "sys\n\nUser: u1\n\nAssistant: a1\n\nu2"


def test_flatten_messages_single_user_passthrough() -> None:
    assert _flatten_messages([_ChatMessage("user", "just this")]) == "just this"


# -- decode state ---------------------------------------------------------


def _drive(state, runtime, ids: List[int]) -> None:
    for pos, tid in enumerate(ids):
        state.consume_step(runtime, DecodeStep(token=TextToken(token_id=tid), position=pos))


def test_state_non_reasoning_strips_turn_end_token() -> None:
    runtime = _chat_runtime()
    skill = ChatSkill()
    ctx = ChatRequest(
        messages=(_ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    state = skill.create_state(runtime, SimpleNamespace(), ctx)
    assert list(state.stop_token_ids(runtime)) == [IM_END]
    _drive(state, runtime, _ords("ok") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    assert result.output == {
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }


def test_state_reasoning_splits_thinking_and_answer() -> None:
    runtime = _chat_runtime()
    skill = ChatSkill()
    ctx = ChatRequest(
        messages=(_ChatMessage("user", "hi"),), image=None, reasoning=True, stream=False
    )
    state = skill.create_state(runtime, SimpleNamespace(), ctx)
    state.on_prefill(runtime)
    # reasoning text, then the answer_id boundary
    _drive(state, runtime, _ords("think"))
    state.consume_step(
        runtime, DecodeStep(token=TextToken(token_id=THINK_E), position=99)
    )
    # the post-reasoning opener is now forced one token at a time
    assert list(state.allowed_token_ids(runtime)) == [DNL]
    _drive(state, runtime, [DNL])
    assert state.allowed_token_ids(runtime) is None
    _drive(state, runtime, _ords("ans") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    assert result.output["message"] == {"role": "assistant", "content": "ans"}
    assert result.output["reasoning"] == {"text": "think"}


def test_state_flatten_path_has_no_extra_stop_tokens() -> None:
    runtime = _flatten_runtime()
    skill = ChatSkill()
    ctx = ChatRequest(
        messages=(_ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    state = skill.create_state(runtime, SimpleNamespace(), ctx)
    assert state.stop_token_ids(runtime) is None
    _drive(state, runtime, _ords("ok") + [99])  # 99 == eos_id, stripped
    result = state.finalize(runtime, reason="stop")
    assert result.output["message"]["content"] == "ok"
