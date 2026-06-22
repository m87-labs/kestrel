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

from kestrel.skills.base import SkillFinalizeResult, SkillSpec
from kestrel.skills.chat import ChatMessage, ChatRequest, ChatSkill

from .query import QueryRequest, QuerySkill, QuerySkillState

if False:  # pragma: no cover - type-checking imports
    from ..runtime import MoondreamRuntime, Token
    from kestrel.scheduler.types import GenerationRequest


class MoondreamChatSkill(ChatSkill):
    """Chat over Moondream by flattening the conversation into ``query``."""

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
            image=request_context.image,
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
            image=chat_request.image,
            reasoning=chat_request.reasoning,
            stream=chat_request.stream,
            spatial_refs=None,
        )
        super().__init__(spec, request, query_request)

    def finalize(
        self, runtime: "MoondreamRuntime", *, reason: str
    ) -> SkillFinalizeResult:
        result = super().finalize(runtime, reason=reason)
        answer = result.output.get("answer", "")
        output: Dict[str, object] = {
            "message": {"role": "assistant", "content": answer},
            "finish_reason": reason,
        }
        if "reasoning" in result.output:
            output["reasoning"] = result.output["reasoning"]
        return SkillFinalizeResult(
            text=result.text, tokens=result.tokens, output=output
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
