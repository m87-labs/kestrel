"""ChatSkill: message validation, native chat-template rendering (incl.
multi-image at content positions), decode-state output, and Moondream's
flatten-into-query subclass."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest

from kestrel.models.protocols import ChatTemplate, QueryTemplate
from kestrel.runtime.tokens import ImageMarker, TextToken
from kestrel.skills.base import DecodeStep
from kestrel.skills.chat import ChatRequest, ChatSkill

# Synthetic token ids for a native (ChatML-like) chat template.
IM_START, IM_END, NL, DNL, THINK_S, THINK_E = 900, 901, 198, 271, 800, 801

_IMG = "data:image/png;base64,aGk="  # decodes to b"hi" (content irrelevant here)


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


class _VisionChatSkill(ChatSkill):
    """Test-only subclass that renders an image as an ImageMarker sentinel —
    standing in for a model's multimodal chat subclass (e.g. Qwen)."""

    def render_content(self, tokenizer, parts):
        tokens = []
        for part in parts:
            if part.image_index is not None:
                tokens.append(ImageMarker(index=part.image_index))
            elif part.text:
                tokens.extend(TextToken(token_id=t) for t in tokenizer.encode(part.text).ids)
        return tokens


class _Tokenizer:
    """Reversible char-level fake: encode→code points, decode→chars."""

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def _chat_runtime(chat: bool = True) -> SimpleNamespace:
    tpl = _native_chat_template() if chat else None
    return SimpleNamespace(
        prompt_template=_NativePromptTemplate(_chat=tpl),
        tokenizer=_Tokenizer(),
    )


def _ctx(messages, reasoning: bool = False) -> ChatRequest:
    built = ChatSkill().build_request(
        None, {"messages": messages, "reasoning": reasoning}, None
    )
    return built.request_context


def _ids(tokens) -> List[int]:
    return [t.token_id for t in tokens]


def _norm(tokens):
    """TextToken -> its id; ImageMarker -> ("IMG", index). For mixed streams."""
    return [
        ("IMG", t.index) if isinstance(t, ImageMarker) else t.token_id for t in tokens
    ]


def _ords(text: str) -> List[int]:
    return [ord(c) for c in text]


# -- validation -----------------------------------------------------------


def test_build_request_accepts_minimal_conversation() -> None:
    ctx = _ctx([{"role": "user", "content": "hi"}])
    assert isinstance(ctx, ChatRequest)
    assert [(m.role, m.text) for m in ctx.messages] == [("user", "hi")]
    assert ctx.images == ()
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


def test_build_request_collects_multiple_images_in_order() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMG}},
                {"type": "text", "text": "first?"},
            ],
        },
        {"role": "assistant", "content": "a"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "second?"},
                {"type": "image_url", "image_url": {"url": _IMG}},
            ],
        },
    ]
    built = ChatSkill().build_request(None, {"messages": msgs}, None)
    ctx = built.request_context
    assert len(ctx.images) == 2
    assert built.image == (b"hi", b"hi")
    # parts carry image indices in conversation order
    assert ctx.messages[0].parts[0].image_index == 0
    assert ctx.messages[2].parts[1].image_index == 1


def test_build_request_rejects_image_in_system_message() -> None:
    msgs = [
        {"role": "system", "content": [{"type": "image_url", "image_url": {"url": _IMG}}]},
        {"role": "user", "content": "hi"},
    ]
    with pytest.raises(ValueError, match="system message cannot contain images"):
        ChatSkill().build_request(None, {"messages": msgs}, None)


@pytest.mark.parametrize("url", ["/etc/passwd", "https://example.com/cat.png"])
def test_build_request_rejects_non_data_image_url(url: str) -> None:
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    with pytest.raises(ValueError, match="data: URL"):
        ChatSkill().build_request(None, {"messages": msgs}, None)


def test_build_request_reasoning_opt_in_via_settings() -> None:
    built = ChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, {"reasoning": True}
    )
    assert built.request_context.reasoning is True


def test_base_reasoning_default_off() -> None:
    built = ChatSkill().build_request(None, {"messages": [{"role": "user", "content": "hi"}]}, None)
    assert built.request_context.reasoning is False


def test_subclass_reasoning_default_on() -> None:
    class _ThinkingChatSkill(ChatSkill):
        default_reasoning = True

    built = _ThinkingChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}]}, None
    )
    assert built.request_context.reasoning is True
    # Explicit opt-out still wins.
    off = _ThinkingChatSkill().build_request(
        None, {"messages": [{"role": "user", "content": "hi"}], "reasoning": False}, None
    )
    assert off.request_context.reasoning is False


# -- native chat rendering ------------------------------------------------


def test_render_single_turn_chat() -> None:
    runtime = _chat_runtime()
    tokens = ChatSkill().build_prompt_tokens(runtime, _ctx([{"role": "user", "content": "hi"}]))
    expected = (
        [IM_START] + _ords("user") + [NL] + _ords("hi") + [IM_END, NL]
        + [IM_START] + _ords("assistant") + [NL] + [THINK_S, DNL, THINK_E, DNL]
    )
    assert _ids(tokens) == expected


def test_subclass_renders_multi_image_at_content_positions() -> None:
    runtime = _chat_runtime()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMG}},
                {"type": "text", "text": "a"},
            ],
        },
        {"role": "assistant", "content": "b"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "c"},
                {"type": "image_url", "image_url": {"url": _IMG}},
            ],
        },
    ]
    tokens = _VisionChatSkill().build_prompt_tokens(runtime, _ctx(msgs))
    expected = (
        # turn 1: image marker BEFORE text
        [IM_START] + _ords("user") + [NL] + [("IMG", 0)] + _ords("a") + [IM_END, NL]
        # turn 2: assistant text
        + [IM_START] + _ords("assistant") + [NL] + _ords("b") + [IM_END, NL]
        # turn 3: image marker AFTER text
        + [IM_START] + _ords("user") + [NL] + _ords("c") + [("IMG", 1)] + [IM_END, NL]
        # generation opener
        + [IM_START] + _ords("assistant") + [NL] + [THINK_S, DNL, THINK_E, DNL]
    )
    assert _norm(tokens) == expected


def test_render_reasoning_opener() -> None:
    runtime = _chat_runtime()
    tokens = ChatSkill().build_prompt_tokens(
        runtime, _ctx([{"role": "user", "content": "hi"}], reasoning=True)
    )
    assert _ids(tokens)[-2:] == [THINK_S, NL]  # assistant_open_reasoning


def test_base_chat_skill_rejects_images() -> None:
    runtime = _chat_runtime()
    ctx = _ctx([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _IMG}}]}])
    with pytest.raises(ValueError, match="does not render images"):
        ChatSkill().build_prompt_tokens(runtime, ctx)


def test_base_chat_skill_requires_chat_template() -> None:
    runtime = _chat_runtime(chat=False)
    ctx = _ctx([{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="no chat\\(\\) template"):
        ChatSkill().build_prompt_tokens(runtime, ctx)


# -- native decode state --------------------------------------------------


def _drive(state, runtime, ids: List[int]) -> None:
    for pos, tid in enumerate(ids):
        state.consume_step(runtime, DecodeStep(token=TextToken(token_id=tid), position=pos))


def test_state_non_reasoning_strips_turn_end_token() -> None:
    runtime = _chat_runtime()
    state = ChatSkill().create_state(runtime, SimpleNamespace(), _ctx([{"role": "user", "content": "hi"}]))
    assert list(state.stop_token_ids(runtime)) == [IM_END]
    _drive(state, runtime, _ords("ok") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    assert result.output == {
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }


def test_state_reasoning_splits_thinking_and_answer() -> None:
    runtime = _chat_runtime()
    state = ChatSkill().create_state(
        runtime, SimpleNamespace(), _ctx([{"role": "user", "content": "hi"}], reasoning=True)
    )
    state.on_prefill(runtime)
    _drive(state, runtime, _ords("think"))
    state.consume_step(runtime, DecodeStep(token=TextToken(token_id=THINK_E), position=99))
    assert list(state.allowed_token_ids(runtime)) == [DNL]
    _drive(state, runtime, [DNL])
    assert state.allowed_token_ids(runtime) is None
    _drive(state, runtime, _ords("ans") + [IM_END])
    result = state.finalize(runtime, reason="stop")
    # OpenRouter style: reasoning is a string on the message, beside content.
    assert result.output["message"] == {
        "role": "assistant",
        "content": "ans",
        "reasoning": "think",
    }


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


def _md_ctx(messages, reasoning: bool = False) -> ChatRequest:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill

    return MoondreamChatSkill().build_request(
        None, {"messages": messages, "reasoning": reasoning}, None
    ).request_context


def test_moondream_chat_flattens_to_query_tokens() -> None:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill
    from kestrel.models.moondream.skills.query import QueryRequest, QuerySkill

    runtime = _md_runtime()
    ctx = _md_ctx([{"role": "user", "content": "hi"}])
    chat_tokens = MoondreamChatSkill().build_prompt_tokens(runtime, ctx)
    query_tokens = QuerySkill().build_prompt_tokens(
        runtime, QueryRequest(question="hi", image=None, reasoning=False, stream=False)
    )
    assert _ids(chat_tokens) == _ids(query_tokens)


def test_moondream_flatten_messages_labels_history() -> None:
    from kestrel.models.moondream.skills.chat import _flatten_messages

    ctx = _md_ctx([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    assert _flatten_messages(ctx.messages) == "sys\n\nUser: u1\n\nAssistant: a1\n\nu2"


def test_moondream_rejects_multiple_images() -> None:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill

    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _IMG}},
        {"type": "image_url", "image_url": {"url": _IMG}},
    ]}]
    with pytest.raises(ValueError, match="at most one image"):
        MoondreamChatSkill().build_request(None, {"messages": msgs}, None)


def test_moondream_chat_finalizes_as_message() -> None:
    from kestrel.models.moondream.skills.chat import MoondreamChatSkill

    runtime = _md_runtime()
    state = MoondreamChatSkill().create_state(
        runtime, SimpleNamespace(), _md_ctx([{"role": "user", "content": "hi"}])
    )
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

    answer_state = skill.create_state(rt, SimpleNamespace(), _md_ctx([{"role": "user", "content": "hi"}]))
    assert list(answer_state.suppressed_token_ids(rt)) == [3]  # answer_id

    think_state = skill.create_state(rt, SimpleNamespace(), _md_ctx([{"role": "user", "content": "hi"}], reasoning=True))
    assert list(think_state.suppressed_token_ids(rt)) == [0, 6]  # eos_id, size_id

    rt3 = _md_runtime("moondream3-preview")
    state3 = skill.create_state(rt3, SimpleNamespace(), _md_ctx([{"role": "user", "content": "hi"}]))
    assert state3.suppressed_token_ids(rt3) is None
