"""Moondream's chat capability.

Moondream has no trained multi-turn chat format, so chat is implemented as a
*subclass* of the shared :class:`~kestrel.skills.chat.ChatSkill` rather than
by the kernel branching on model specifics. It flattens the conversation into
the single-turn ``query`` prompt and reuses the query decode path wholesale —
including moondream2's HF-parity token masking and grounded reasoning — so it
differs from ``query`` only in the OpenAI-style message I/O.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from kestrel.skills.base import BuiltRequest, SkillFinalizeResult, SkillSpec
from kestrel.skills.chat import ChatMessage, ChatRequest, ChatSkill

from .query import QueryRequest, QuerySkill, QuerySkillState

if False:  # pragma: no cover - type-checking imports
    from typing import Mapping, Optional

    import numpy as np

    from ..runtime import MoondreamRuntime, Token
    from kestrel.scheduler.types import GenerationRequest


def _single_image(ctx: ChatRequest):
    """Moondream is single-image; reject more and return the one (or None)."""
    if len(ctx.images) > 1:
        raise ValueError("Moondream supports at most one image per request")
    return ctx.images[0] if ctx.images else None


class MoondreamChatSkill(ChatSkill):
    """Chat over Moondream by flattening the conversation into ``query``."""

    # Moondream's query path defaults to reasoning on; honor that for chat.
    default_reasoning = True

    def build_request(
        self,
        image: "Optional[np.ndarray | bytes]",
        prompt: "Mapping",
        settings: "Optional[Mapping]",
    ) -> BuiltRequest:
        built = super().build_request(image, prompt, settings)
        # The base may carry an images tuple; Moondream's pipeline takes one.
        return BuiltRequest(
            request_context=built.request_context,
            max_new_tokens=built.max_new_tokens,
            temperature=built.temperature,
            top_p=built.top_p,
            image=_single_image(built.request_context),
        )

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> "Sequence[Token]":
        if not isinstance(request_context, ChatRequest):
            raise ValueError(
                "MoondreamChatSkill.build_prompt_tokens requires a ChatRequest"
            )
        query_request = QueryRequest(
            question=_flatten_messages(request_context.messages),
            image=_single_image(request_context),
            reasoning=request_context.reasoning,
            stream=request_context.stream,
            spatial_refs=None,
        )
        return QuerySkill().build_prompt_tokens(runtime, query_request)

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: object,
    ) -> "MoondreamChatSkillState":
        if not isinstance(request_context, ChatRequest):
            raise ValueError(
                "MoondreamChatSkill.create_state requires a ChatRequest context"
            )
        return MoondreamChatSkillState(self, request, request_context)


class MoondreamChatSkillState(QuerySkillState):
    """Query decode (incl. md2 masking + grounding), emitted as a chat message."""

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        chat_request: ChatRequest,
    ) -> None:
        query_request = QueryRequest(
            question="",
            image=_single_image(chat_request),
            reasoning=chat_request.reasoning,
            stream=chat_request.stream,
            spatial_refs=None,
        )
        super().__init__(spec, request, query_request)

    def finalize(
        self, runtime: "MoondreamRuntime", *, reason: str
    ) -> SkillFinalizeResult:
        result = super().finalize(runtime, reason=reason)
        message: Dict[str, object] = {
            "role": "assistant",
            "content": result.output.get("answer", ""),
        }
        # OpenRouter-style: reasoning as a string on the message; Moondream's
        # spatial grounding (if any) is preserved under reasoning_details.
        reasoning = result.output.get("reasoning")
        if reasoning:
            message["reasoning"] = reasoning.get("text", "")
            grounding = reasoning.get("grounding")
            if grounding:
                message["reasoning_details"] = {"grounding": grounding}
        return SkillFinalizeResult(
            text=result.text,
            tokens=result.tokens,
            output={"message": message, "finish_reason": reason},
        )


def _flatten_messages(messages: Sequence[ChatMessage]) -> str:
    """Render a conversation as a single question. A lone user turn (optionally
    with a system preamble) passes through cleanly; longer histories are
    labelled best-effort, since Moondream isn't trained on multi-turn chat."""
    if len(messages) == 1:
        return messages[0].text
    system = ""
    convo: List[ChatMessage] = []
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


__all__ = ["MoondreamChatSkill", "MoondreamChatSkillState"]
