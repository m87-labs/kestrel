"""ChatSkill: message validation, native chat-template rendering, decode-state
output shape, and Moondream's flatten-into-query subclass."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest

from kestrel.models.protocols import ChatTemplate, QueryTemplate
from kestrel.runtime.tokens import TextToken
from kestrel.skills.base import DecodeStep
from kestrel.skills.chat import ChatMessage, ChatRequest, ChatSkill

# Synthetic token ids for a native (ChatML-like) chat template.
IM_START, IM_END, NL, DNL, THINK_S, THINK_E = 900, 901, 198, 271, 800, 801


def _native_chat_template() -> ChatTemplate:
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


@dataclass
class _NativePromptTemplate:
    bos_id: int = 5
    eos_id: int = 99
    answer_id: int = THINK_E
    thinking_id: int = THINK_S
    coord_id: int = 0
    size_id: int = 0
    start_ground_points_id: int = 0
    end_ground_id: int = 0
    _chat: Optional[ChatTemplate] = None

    def chat(self) -> Optional[ChatTemplate]:
        return self._chat

    def query(self) -> Optional[QueryTemplate]:
        return None


class _Tokenizer:
    """Reversible char-level fake: encode→code points, decode→chars."""

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def _chat_runtime(chat: bool = True) -> SimpleNamespace:
    tpl = _native_chat_template() if chat else None
    return SimpleNamespace(
        prompt_template=_NativePromptTemplate(_chat=tpl), tokenizer=_Tokenizer()
    )


def _ids(tokens) -> List[int]:
    return [t.token_id for t in tokens]


def _ords(text: str) -> List[int]:
    return [ord(c) for c in text]


# -- validation (generic, lives on the base) ------------------------------


def test_build_request_accepts_minimal_conversation() -> None:
    built = ChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, None
    )
    ctx = built.request_context
    assert isinstance(ctx, ChatRequest)
    assert [(m.role, m.text) for m in ctx.messages] == [("user", "hi")]
    assert ctx.reasoning is False


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


# -- native chat rendering ------------------------------------------------


def test_render_single_turn_chat() -> None:
    runtime = _chat_runtime()
    ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
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
            ChatMessage("system", "be nice"),
            ChatMessage("user", "a"),
            ChatMessage("assistant", "b"),
            ChatMessage("user", "c"),
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
        + [IM_START] + _ords("assistant") + [NL] + [THINK_S, NL]
    )
    assert _ids(tokens) == expected


def test_base_chat_skill_requires_chat_template() -> None:
    runtime = _chat_runtime(chat=False)  # chat() returns None
    ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    with pytest.raises(ValueError, match="no chat\\(\\) template"):
        ChatSkill().build_prompt_tokens(runtime, ctx)


# -- native decode state --------------------------------------------------


def _drive(state, runtime, ids: List[int]) -> None:
    for pos, tid in enumerate(ids):
        state.consume_step(runtime, DecodeStep(token=TextToken(token_id=tid), position=pos))


def test_state_non_reasoning_strips_turn_end_token() -> None:
    runtime = _chat_runtime()
    ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    state = ChatSkill().create_state(runtime, SimpleNamespace(), ctx)
    assert list(state.stop_token_ids(runtime)) == [IM_END]
    _drive(state, runtime, _ords("ok") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    assert result.output == {
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }


def test_state_reasoning_splits_thinking_and_answer() -> None:
    runtime = _chat_runtime()
    ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=True, stream=False
    )
    state = ChatSkill().create_state(runtime, SimpleNamespace(), ctx)
    state.on_prefill(runtime)
    _drive(state, runtime, _ords("think"))
    state.consume_step(
        runtime, DecodeStep(token=TextToken(token_id=THINK_E), position=99)
    )
    assert list(state.allowed_token_ids(runtime)) == [DNL]
    _drive(state, runtime, [DNL])
    assert state.allowed_token_ids(runtime) is None
    _drive(state, runtime, _ords("ans") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    assert result.output["message"] == {"role": "assistant", "content": "ans"}
    assert result.output["reasoning"] == {"text": "think"}


# -- Moondream flatten-into-query subclass --------------------------------


@dataclass
class _MdPromptTemplate:
    bos_id: int = 5
    eos_id: int = 0
    answer_id: int = 3
    thinking_id: int = 4
    coord_id: int = 5
    size_id: int = 6
    start_ground_points_id: int = 7
    end_ground_id: int = 9

    def chat(self) -> None:
        return None

    def query(self) -> QueryTemplate:
        return QueryTemplate(
            prefix=[10, 11], answer_prefix=[20], reasoning_prefix=[21],
            post_reasoning_prefix=[3],
        )


def _md_runtime(model_name: str = "moondream2") -> SimpleNamespace:
    return SimpleNamespace(
        prompt_template=_MdPromptTemplate(),
        tokenizer=_Tokenizer(),
        model_name=model_name,
    )


def test_moondream_chat_flattens_to_query_tokens() -> None:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill
    from kestrel.models.moondream.skills.query import QueryRequest, QuerySkill

    runtime = _md_runtime()
    chat_ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    chat_tokens = MoondreamChatSkill().build_prompt_tokens(runtime, chat_ctx)
    query_tokens = QuerySkill().build_prompt_tokens(
        runtime, QueryRequest(question="hi", image=None, reasoning=False, stream=False)
    )
    assert _ids(chat_tokens) == _ids(query_tokens)


def test_moondream_flatten_messages_labels_history() -> None:
    from kestrel.models.moondream.skills.chat import _flatten_messages

    msgs = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "u1"),
        ChatMessage("assistant", "a1"),
        ChatMessage("user", "u2"),
    ]
    assert _flatten_messages(msgs) == "sys\n\nUser: u1\n\nAssistant: a1\n\nu2"
    assert _flatten_messages([ChatMessage("user", "just this")]) == "just this"


def test_moondream_chat_finalizes_as_message() -> None:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill

    runtime = _md_runtime()
    ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    state = MoondreamChatSkill().create_state(runtime, SimpleNamespace(), ctx)
    _drive(state, runtime, _ords("Paris"))
    result = state.finalize(runtime, reason="stop")
    assert result.output == {
        "message": {"role": "assistant", "content": "Paris"},
        "finish_reason": "stop",
    }


def test_moondream_chat_inherits_query_suppression() -> None:
    """Codex P2: flattened chat must mirror the query skill's md2 masking."""
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill

    skill = MoondreamChatSkill()
    rt = _md_runtime("moondream2")

    answer_ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=False, stream=False
    )
    answer_state = skill.create_state(rt, SimpleNamespace(), answer_ctx)
    assert list(answer_state.suppressed_token_ids(rt)) == [3]  # answer_id

    think_ctx = ChatRequest(
        messages=(ChatMessage("user", "hi"),), image=None, reasoning=True, stream=False
    )
    think_state = skill.create_state(rt, SimpleNamespace(), think_ctx)
    assert list(think_state.suppressed_token_ids(rt)) == [0, 6]  # eos_id, size_id

    # Non-moondream2 models are unaffected.
    rt3 = _md_runtime("moondream3-preview")
    state3 = skill.create_state(rt3, SimpleNamespace(), answer_ctx)
    assert state3.suppressed_token_ids(rt3) is None
