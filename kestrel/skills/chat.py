"""Shared, model-agnostic chat skill (OpenAI-compatible message shape).

``ChatSkill`` is a *base* capability for models with a native multi-turn
chat format: a model opts in by exposing a ``chat()`` prompt template (see
:class:`kestrel.models.protocols.ChatTemplate`). The skill owns the generic
parts — validating OpenAI-style ``messages``, parsing each message's content
into an ordered list of text/image parts, rendering the conversation into
prompt tokens (emitting one ``image_placeholder`` per image **at its content
position**), interpreting the decode stream, and materialising an assistant
``message``. The runtime expands each image placeholder to that image's token
count.

A model that needs different behaviour *subclasses* this skill rather than
having the kernel branch on model specifics — e.g. a model with no trained
chat format (Moondream) flattens the conversation into its single-turn
``query`` prompt (``kestrel.models.moondream.skills.chat``).

Validation is plain Python (raises ``ValueError``); pydantic stays at the
HTTP boundary. The ``output`` shape (``{"message": {...}, "finish_reason":
...}``) maps directly onto an OpenAI chat-completion *choice*.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from dataclasses import dataclass, field
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
class ChatContentPart:
    """One ordered piece of a message's content: text or an image reference.

    Exactly one of ``text`` / ``image_index`` is set. ``image_index`` points
    into :attr:`ChatRequest.images`.
    """

    text: Optional[str] = None
    image_index: Optional[int] = None


@dataclass(slots=True)
class ChatMessage:
    """One validated turn: role + ordered content parts.

    Part of the subclass contract: a ``ChatSkill`` subclass receives these
    via ``ChatRequest.messages``.
    """

    role: str
    parts: Tuple[ChatContentPart, ...]

    @property
    def text(self) -> str:
        """Concatenated text of this turn (image parts omitted)."""
        return "".join(p.text for p in self.parts if p.text is not None)

    @property
    def has_image(self) -> bool:
        return any(p.image_index is not None for p in self.parts)


@dataclass(slots=True)
class ChatRequest:
    """Validated chat payload carried to decode."""

    messages: Tuple[ChatMessage, ...]
    images: Tuple[bytes, ...]
    reasoning: bool
    stream: bool


class ChatSkill(SkillSpec):
    """Multi-turn chat for models with a native ``chat()`` template.

    Subclass to customise rendering or decode for a model without one
    (override ``build_prompt_tokens`` / ``create_state``); ``build_request``
    — OpenAI message validation and image extraction — is model-agnostic and
    inherited.
    """

    # Per-model reasoning default when the caller doesn't pass ``reasoning``.
    # The base is off; a model whose chat format defaults to thinking sets
    # this True in its subclass (Qwen, Moondream).
    default_reasoning: bool = False

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

        images: List[bytes] = []
        messages: List[ChatMessage] = []
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
            parts = _parse_content(m.get("content"), images, role)
            messages.append(ChatMessage(role=role, parts=tuple(parts)))

        if messages[-1].role != "user":
            raise ValueError("the last message must be from 'user'")
        if not messages[-1].text and not messages[-1].has_image:
            raise ValueError("the last user message must have text or an image")

        reasoning = bool(prompt.get("reasoning", self.default_reasoning))
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
            images=tuple(images),
            reasoning=reasoning,
            stream=bool(prompt.get("stream", False)),
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            image=tuple(images) if images else None,
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
        if chat_tpl is None:
            raise ValueError(
                "model has no chat() template; a model without a native chat "
                "format should register a ChatSkill subclass that renders chat "
                "its own way (e.g. Moondream flattens into the query prompt)"
            )
        return self._render_chat(runtime, chat_tpl, request_context)

    def _render_chat(
        self, runtime: "MoondreamRuntime", ct, ctx: ChatRequest
    ) -> List[Token]:
        tokenizer = runtime.tokenizer

        def emit(ids) -> List[Token]:
            return [TextToken(token_id=int(t)) for t in ids]

        tokens: List[Token] = emit(ct.bos)
        for msg in ctx.messages:
            role_word = ct.roles.get(msg.role)
            if role_word is None:
                raise ValueError(f"chat template has no role word for {msg.role!r}")
            tokens += emit(ct.turn_prefix)
            tokens += emit(tokenizer.encode(role_word).ids)
            tokens += emit(ct.role_suffix)
            tokens += list(self.render_content(tokenizer, msg.parts))
            tokens += emit(ct.turn_suffix)
        # Open the assistant turn for the model to complete.
        tokens += emit(ct.turn_prefix)
        tokens += emit(tokenizer.encode(ct.roles["assistant"]).ids)
        tokens += emit(ct.role_suffix)
        tokens += emit(ct.assistant_open_reasoning if ctx.reasoning else ct.assistant_open)
        return tokens

    def render_content(
        self, tokenizer, parts: "Sequence[ChatContentPart]"
    ) -> List[Token]:
        """Render one message's content parts to tokens.

        The base handles text only and rejects images. A multimodal model
        *subclasses* and overrides this to emit its own image tokens (e.g.
        ``<|vision_start|><|image_pad|><|vision_end|>``) at each image's
        position — so the kernel and the shared chat template carry no image
        concept.
        """
        tokens: List[Token] = []
        for part in parts:
            if part.image_index is not None:
                raise ValueError(
                    "this chat skill does not render images; subclass ChatSkill "
                    "and override render_content to add image support"
                )
            if part.text:
                tokens.extend(
                    TextToken(token_id=int(t)) for t in tokenizer.encode(part.text).ids
                )
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
        if chat_tpl is None:
            raise ValueError("model has no chat() template for ChatSkill")
        return ChatSkillState(
            self,
            request,
            request_context,
            post_reasoning_prefix=list(chat_tpl.post_reasoning_prefix),
            turn_end_ids=list(chat_tpl.turn_end_ids),
        )


class ChatSkillState(SkillState):
    """Buffers an assistant turn (optional thinking block) into a chat message."""

    # The chat mask is genuinely stateful: it transitions INACTIVE -> ACTIVE
    # mid-run. While collecting reasoning, ``allowed_token_ids`` is ``None`` and
    # this state overrides no ``suppressed_token_ids`` (base returns ``None``) --
    # no active constraint. Once the model emits ``answer_id``, the state flips
    # to forcing ``post_reasoning_prefix`` one id at a time via
    # ``allowed_token_ids``. A single spec macro-step commits a variable run
    # under ONE mask, so a run that begins in reasoning (mask = None) and crosses
    # ``answer_id`` would verify the post-boundary tokens under the stale,
    # unconstrained first-position mask -- accepting them WITHOUT the required
    # prefix constraint and corrupting the output. The scheduler's behavioural
    # fallback ("any ACTIVE constraint is stateful") cannot see this transition
    # because the mask is ``None`` at the run's first position, so declare the
    # mask stateful explicitly: the scheduler then caps such a row to one
    # committed token per macro-step, forcing exactly one constraint transition
    # per run (the regime where the per-step mask is exact). Always-cap is the
    # correctness-first verdict; a transition-only cap is not expressible here
    # because the boundary (when the model emits ``answer_id``) is model-decided
    # and unknowable before the run commits. Tradeoff (follow-up): this also caps
    # the long unconstrained answer/reasoning phases, costing intra-step
    # speculation on those rows.
    mask_is_stateful = True

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
        # Decode the answer tokens as-is — matching the query skill. The
        # reasoning→answer boundary is the answer_id token plus the (force-fed,
        # uncollected) post_reasoning_prefix, so _answer_tokens is already just
        # the answer; stripping here would mutate legitimate leading whitespace
        # and desync streamed deltas (decoded un-stripped) from this content.
        answer_text = tokenizer.decode(self._answer_tokens) if self._answer_tokens else ""
        message: Dict[str, object] = {"role": "assistant", "content": answer_text}
        # OpenRouter-style: reasoning is a string field on the message, separate
        # from content (omitted when there's no reasoning).
        if self._reasoning_enabled and self._reasoning_tokens:
            message["reasoning"] = tokenizer.decode(self._reasoning_tokens)
        return SkillFinalizeResult(
            text=answer_text,
            tokens=list(self.tokens),
            output={"message": message, "finish_reason": reason},
        )


# -- helpers --------------------------------------------------------------


def _parse_content(
    content: object,
    images: List[bytes],
    role: str,
) -> List[ChatContentPart]:
    """Parse a message ``content`` into ordered parts, appending any images
    (in order) to ``images`` and referencing them by index."""
    if content is None:
        return []
    if isinstance(content, str):
        return [ChatContentPart(text=content)]
    if isinstance(content, _Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: List[ChatContentPart] = []
        for part in content:
            if not isinstance(part, _Mapping):
                raise ValueError("each content part must be a mapping")
            ptype = part.get("type")
            if ptype == "text":
                value = part.get("text")
                if value is None:
                    raise ValueError("a text content part is missing 'text'")
                parts.append(ChatContentPart(text=str(value)))
            elif ptype == "image_url":
                if role == "system":
                    raise ValueError("a system message cannot contain images")
                images.append(_load_image_part(part))
                parts.append(ChatContentPart(image_index=len(images) - 1))
            else:
                raise ValueError(f"unsupported content part type: {ptype!r}")
        return parts
    raise ValueError("message content must be a string or a list of content parts")


def _load_image_part(part: "_Mapping") -> bytes:
    spec = part.get("image_url")
    if isinstance(spec, _Mapping):
        spec = spec.get("url")
    if spec is None:
        raise ValueError("an image_url content part is missing 'url'")
    # The payload maps onto an untrusted OpenAI HTTP request: accept only a
    # base64 data: URL — no remote (http) fetches and no server filesystem reads.
    if not isinstance(spec, str):
        raise ValueError(
            f"an image_url 'url' must be a string, got {type(spec).__name__}"
        )
    value = spec.strip()
    if not value.startswith("data:"):
        raise ValueError("an image_url 'url' must be a base64 data: URL")
    return load_image_bytes_from_base64(value)


__all__ = [
    "ChatContentPart",
    "ChatMessage",
    "ChatRequest",
    "ChatSkill",
    "ChatSkillState",
]
