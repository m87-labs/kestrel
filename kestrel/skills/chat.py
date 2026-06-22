"""Shared, model-agnostic chat skill (OpenAI-compatible message shape).

``ChatSkill`` is a *base* capability: a model "implements" it by exposing a
``chat()`` template from its ``prompt_template`` (see
:class:`kestrel.models.protocols.ChatTemplate`). The skill itself owns the
generic parts — validating OpenAI-style ``messages``, extracting at most one
image, rendering the conversation into prompt tokens, interpreting the decode
stream, and materialising an assistant ``message`` — so the same instance
serves any model.

Two rendering modes, selected per model:

* ``prompt_template.chat()`` returns a template → native multi-turn rendering
  with role markers (ChatML-style ``<|im_start|>role ... <|im_end|>`` turns).
* ``prompt_template.chat()`` returns ``None`` → the conversation is flattened
  into the single-turn ``query`` prompt (Moondream, which has no trained
  multi-turn chat format).

Validation is plain Python (raises ``ValueError``); pydantic stays at the HTTP
boundary. The ``output`` shape (``{"message": {...}, "finish_reason": ...}``)
maps directly onto an OpenAI chat-completion *choice*, so a future
``/v1/chat/completions`` route is a thin wrapper.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from kestrel.runtime.tokens import TextToken, Token
from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    BuiltRequest,
    DecodeStep,
    SkillFinalizeResult,
    SkillSpec,
    SkillState,
    parse_settings,
)
from kestrel.utils.image import load_image_bytes_from_base64

if False:  # pragma: no cover - type-checking imports
    from kestrel.models.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest

_ROLES = ("system", "user", "assistant")


@dataclass(slots=True)
class _ChatMessage:
    """One validated turn: role + flattened text (image handled separately)."""

    role: str
    text: str
    has_image: bool = False


@dataclass(slots=True)
class ChatRequest:
    """Validated chat payload carried to decode."""

    messages: Tuple[_ChatMessage, ...]
    image: "Optional[np.ndarray | bytes]"
    reasoning: bool
    stream: bool


class ChatSkill(SkillSpec):
    """Multi-turn chat capability shared across models."""

    def __init__(self) -> None:
        super().__init__(name="chat")

    # -- request building --------------------------------------------------

    def build_request(
        self,
        image: "Optional[np.ndarray | bytes]",
        prompt: "_Mapping",
        settings: "Optional[_Mapping]",
    ) -> BuiltRequest:
        raw = prompt.get("messages")
        if raw is None:
            raise ValueError("messages must be provided")
        if not isinstance(raw, _Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError("messages must be a list of message objects")
        if len(raw) == 0:
            raise ValueError("messages must be a non-empty list")

        messages: List[_ChatMessage] = []
        found_image: "Optional[np.ndarray | bytes]" = None
        for idx, m in enumerate(raw):
            if not isinstance(m, _Mapping):
                raise ValueError(
                    "each message must be a mapping with 'role' and 'content'"
                )
            role = m.get("role")
            if role not in _ROLES:
                raise ValueError(
                    f"message role must be one of {_ROLES}, got {role!r}"
                )
            if role == "system" and idx != 0:
                raise ValueError("a 'system' message must be the first message")
            text, img = _parse_content(m.get("content"))
            if img is not None:
                if found_image is not None:
                    raise ValueError(
                        "at most one image is supported across chat messages"
                    )
                found_image = img
            messages.append(_ChatMessage(role=role, text=text, has_image=img is not None))

        if messages[-1].role != "user":
            raise ValueError("the last message must be from 'user'")
        if not messages[-1].text and not messages[-1].has_image:
            raise ValueError("the last user message must have text or an image")

        # Image precedence: one carried inside the messages, else the
        # top-level ``image=`` argument (a convenience for non-OpenAI callers).
        resolved_image = found_image if found_image is not None else image

        reasoning = bool(prompt.get("reasoning", False))
        if settings is not None and "reasoning" in settings:
            reasoning = bool(settings["reasoning"])

        s = parse_settings(
            settings,
            temperature=AR_DEFAULT_TEMPERATURE,
            top_p=AR_DEFAULT_TOP_P,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        request = ChatRequest(
            messages=tuple(messages),
            image=resolved_image,
            reasoning=reasoning,
            stream=bool(prompt.get("stream", False)),
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            image=resolved_image,
        )

    def prompt_text(self, request_context: object) -> str:
        if isinstance(request_context, ChatRequest) and request_context.messages:
            return request_context.messages[-1].text
        return ""

    # -- prompt tokens -----------------------------------------------------

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Sequence[Token]:
        if not isinstance(request_context, ChatRequest):
            raise ValueError("ChatSkill.build_prompt_tokens requires a ChatRequest")
        chat_tpl = runtime.prompt_template.chat()
        if chat_tpl is not None:
            return self._render_chat(runtime, chat_tpl, request_context)
        return self._render_flattened(runtime, request_context)

    def _render_chat(
        self, runtime: "MoondreamRuntime", ct, ctx: ChatRequest
    ) -> List[Token]:
        tokenizer = runtime.tokenizer
        tokens: List[Token] = [TextToken(token_id=int(t)) for t in ct.bos]
        for msg in ctx.messages:
            if msg.role == "system" and not ct.supports_system:
                raise ValueError(
                    "this model's chat template does not support a system message"
                )
            role_word = ct.roles.get(msg.role)
            if role_word is None:
                raise ValueError(f"chat template has no role word for {msg.role!r}")
            tokens.extend(TextToken(token_id=int(t)) for t in ct.turn_prefix)
            tokens.extend(
                TextToken(token_id=int(t)) for t in tokenizer.encode(role_word).ids
            )
            tokens.extend(TextToken(token_id=int(t)) for t in ct.role_suffix)
            if msg.text:
                tokens.extend(
                    TextToken(token_id=int(t)) for t in tokenizer.encode(msg.text).ids
                )
            tokens.extend(TextToken(token_id=int(t)) for t in ct.turn_suffix)
        # Open the assistant turn for the model to complete.
        tokens.extend(TextToken(token_id=int(t)) for t in ct.turn_prefix)
        tokens.extend(
            TextToken(token_id=int(t)) for t in tokenizer.encode(ct.roles["assistant"]).ids
        )
        tokens.extend(TextToken(token_id=int(t)) for t in ct.role_suffix)
        opener = ct.assistant_open_reasoning if ctx.reasoning else ct.assistant_open
        tokens.extend(TextToken(token_id=int(t)) for t in opener)
        return tokens

    def _render_flattened(
        self, runtime: "MoondreamRuntime", ctx: ChatRequest
    ) -> List[Token]:
        pt = runtime.prompt_template
        template = pt.query()
        if template is None:
            raise ValueError("model supports neither a chat nor a query template")
        question = _flatten_messages(ctx.messages)
        pre = (
            template.prefix_when_reasoning
            if ctx.reasoning and template.prefix_when_reasoning is not None
            else template.prefix
        )
        opener = template.reasoning_prefix if ctx.reasoning else template.answer_prefix
        encoded = runtime.tokenizer.encode(question).ids if question else []
        tokens: List[Token] = [TextToken(token_id=int(pt.bos_id))]
        tokens.extend(TextToken(token_id=int(t)) for t in pre)
        tokens.extend(TextToken(token_id=int(t)) for t in encoded)
        tokens.extend(TextToken(token_id=int(t)) for t in opener)
        return tokens

    # -- decode state ------------------------------------------------------

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: object,
    ) -> "ChatSkillState":
        if not isinstance(request_context, ChatRequest):
            raise ValueError("ChatSkill.create_state requires a ChatRequest context")
        chat_tpl = runtime.prompt_template.chat()
        if chat_tpl is not None:
            post_reasoning = list(chat_tpl.post_reasoning_prefix)
            turn_end = list(chat_tpl.turn_end_ids)
        else:
            template = runtime.prompt_template.query()
            post_reasoning = list(template.post_reasoning_prefix) if template else []
            turn_end = []
        return ChatSkillState(
            self,
            request,
            request_context,
            post_reasoning_prefix=post_reasoning,
            turn_end_ids=turn_end,
        )


class ChatSkillState(SkillState):
    """Buffers an assistant turn (optional thinking block) into a chat message."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        chat_request: ChatRequest,
        *,
        post_reasoning_prefix: Sequence[int],
        turn_end_ids: Sequence[int],
    ) -> None:
        super().__init__(spec, request)
        self._request = chat_request
        self._reasoning_enabled = bool(chat_request.reasoning)
        self._collecting_reasoning = self._reasoning_enabled
        self._reasoning_tokens: List[int] = []
        self._answer_tokens: List[int] = []
        self._streaming = bool(chat_request.stream)
        self._answer_stream_offset = 0
        self._answer_id: Optional[int] = None
        self._stop_ids: Optional[Set[int]] = None
        self._post_reasoning_prefix = list(post_reasoning_prefix)
        # After the reasoning→answer boundary, the model's post-reasoning
        # opener is replayed one token at a time via allowed_token_ids.
        self._post_reasoning_tokens: Optional[List[int]] = None
        self._post_reasoning_idx = 0
        self._turn_end_ids = list(turn_end_ids)

    def _ensure_ids(self, runtime: "MoondreamRuntime") -> None:
        if self._answer_id is None:
            self._answer_id = runtime.prompt_template.answer_id
        if self._stop_ids is None:
            self._stop_ids = set(self._turn_end_ids) | {runtime.prompt_template.eos_id}

    def on_prefill(self, runtime: "MoondreamRuntime") -> None:
        self._ensure_ids(runtime)
        return None

    def stop_token_ids(self, runtime: "MoondreamRuntime") -> Optional[Sequence[int]]:
        return self._turn_end_ids or None

    def allowed_token_ids(
        self, runtime: "MoondreamRuntime"
    ) -> Optional[Sequence[int]]:
        if (
            self._post_reasoning_tokens is not None
            and self._post_reasoning_idx < len(self._post_reasoning_tokens)
        ):
            return [self._post_reasoning_tokens[self._post_reasoning_idx]]
        return None

    def consume_step(self, runtime: "MoondreamRuntime", step: DecodeStep) -> None:
        self._ensure_ids(runtime)
        self.append_token(step.token)

        # Replaying the forced post-reasoning opener: don't collect these.
        if (
            self._post_reasoning_tokens is not None
            and self._post_reasoning_idx < len(self._post_reasoning_tokens)
        ):
            self._post_reasoning_idx += 1
            return None

        token = step.token
        if not self._reasoning_enabled:
            self._collect_answer(token)
            return None

        if self._collecting_reasoning:
            if isinstance(token, TextToken):
                if token.token_id == self._answer_id:
                    # reasoning → answer boundary
                    self._collecting_reasoning = False
                    self._post_reasoning_tokens = list(self._post_reasoning_prefix)
                    self._post_reasoning_idx = 0
                    self._answer_stream_offset = 0
                    return None
                self._reasoning_tokens.append(token.token_id)
            return None

        self._collect_answer(token)
        return None

    def _collect_answer(self, token: Token) -> None:
        assert self._stop_ids is not None
        if isinstance(token, TextToken) and token.token_id not in self._stop_ids:
            self._answer_tokens.append(token.token_id)

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        if not self._streaming or self._collecting_reasoning:
            return None
        if not self._answer_tokens:
            return None
        text = runtime.tokenizer.decode(self._answer_tokens)
        if len(text) <= self._answer_stream_offset:
            return None
        chunk = text[self._answer_stream_offset :]
        self._answer_stream_offset = len(text)
        return chunk or None

    def finalize(
        self, runtime: "MoondreamRuntime", *, reason: str
    ) -> SkillFinalizeResult:
        tokenizer = runtime.tokenizer
        answer_text = tokenizer.decode(self._answer_tokens) if self._answer_tokens else ""
        message: Dict[str, object] = {"role": "assistant", "content": answer_text}
        output: Dict[str, object] = {"message": message, "finish_reason": reason}
        if self._reasoning_enabled and self._reasoning_tokens:
            output["reasoning"] = {"text": tokenizer.decode(self._reasoning_tokens)}
        return SkillFinalizeResult(
            text=answer_text,
            tokens=list(self.tokens),
            output=output,
        )


# -- helpers --------------------------------------------------------------


def _parse_content(content: object) -> Tuple[str, "Optional[np.ndarray | bytes]"]:
    """Return (text, image) from a message ``content`` (str or parts list)."""
    if content is None:
        return "", None
    if isinstance(content, str):
        return content, None
    if isinstance(content, _Sequence) and not isinstance(content, (str, bytes, bytearray)):
        texts: List[str] = []
        image: "Optional[np.ndarray | bytes]" = None
        for part in content:
            if not isinstance(part, _Mapping):
                raise ValueError("each content part must be a mapping")
            ptype = part.get("type")
            if ptype == "text":
                value = part.get("text")
                if value is None:
                    raise ValueError("a text content part is missing 'text'")
                texts.append(str(value))
            elif ptype in ("image_url", "image"):
                if image is not None:
                    raise ValueError("a message may contain at most one image")
                image = _load_image_part(part)
            else:
                raise ValueError(f"unsupported content part type: {ptype!r}")
        return "\n".join(texts), image
    raise ValueError("message content must be a string or a list of content parts")


def _load_image_part(part: "_Mapping") -> "np.ndarray | bytes":
    if part.get("type") == "image":
        # Convenience for Python callers: a native image object passed directly.
        obj = part.get("image")
        if obj is None:
            raise ValueError("an image content part is missing 'image'")
        return _coerce_image(obj)
    spec = part.get("image_url")
    if isinstance(spec, _Mapping):
        spec = spec.get("url")
    if spec is None:
        raise ValueError("an image_url content part is missing 'url'")
    return _coerce_image(spec)


def _coerce_image(obj: object) -> "np.ndarray | bytes":
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, str):
        value = obj.strip()
        if value.startswith("data:"):
            return load_image_bytes_from_base64(value)
        if value.startswith(("http://", "https://")):
            raise ValueError(
                "remote image URLs are not supported yet; pass a data: URL, a "
                "local file path, raw bytes, or a numpy array"
            )
        if os.path.exists(value):
            with open(value, "rb") as handle:
                return handle.read()
        # Last resort: treat the string as a raw base64 payload.
        return load_image_bytes_from_base64(value)
    raise ValueError(f"unsupported image value of type {type(obj).__name__}")


def _flatten_messages(messages: Sequence[_ChatMessage]) -> str:
    """Render a conversation as a single question for models without a chat
    template. A lone user turn (optionally with a system preamble) passes
    through cleanly; longer histories are labelled best-effort."""
    if len(messages) == 1:
        return messages[0].text
    system = ""
    convo: List[_ChatMessage] = []
    for msg in messages:
        if msg.role == "system":
            system = msg.text
        else:
            convo.append(msg)
    if len(convo) == 1 and not system:
        return convo[0].text
    lines: List[str] = []
    if system:
        lines.append(system)
    for i, msg in enumerate(convo):
        is_last = i == len(convo) - 1
        if is_last and msg.role == "user":
            lines.append(msg.text)
        else:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {msg.text}")
    return "\n\n".join(line for line in lines if line)


__all__ = ["ChatRequest", "ChatSkill", "ChatSkillState"]
