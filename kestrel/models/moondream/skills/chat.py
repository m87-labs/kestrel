"""Moondream's chat capability.

Moondream has no trained multi-turn chat format, so chat is implemented as a
*subclass* of the shared :class:`~kestrel.skills.chat.ChatSkill`. It renders
the conversation as a sequence of Moondream ``query`` blocks — one per user
turn — reusing the query template's tokens (so MD2's doubled-id quirk is
honored) and the query decode path (incl. md2 token masking and grounded
reasoning). Each turn's images sit at the start of the turn as ``ImageMarker``
sentinels, which the runtime replaces with image-embedding blocks.

Per-turn layout (the system message is folded into the first user turn):

    [ImageMarker x K]  <query.prefix>  <user text>  <query.answer_prefix>  <assistant text>

The final user turn ends with the generation opener — ``answer_prefix``, or
``reasoning_prefix`` when reasoning is on — for the model to complete.
"""

from __future__ import annotations

from typing import Dict, List

from kestrel.runtime.tokens import ImageMarker, TextToken
from kestrel.skills.base import SkillFinalizeResult, SkillSpec
from kestrel.skills.chat import ChatRequest, ChatSkill

from .query import QueryRequest, QuerySkillState

if False:  # pragma: no cover - type-checking imports
    from ..runtime import MoondreamRuntime, Token
    from kestrel.scheduler.types import GenerationRequest


class MoondreamChatSkill(ChatSkill):
    """Multi-turn, multi-image chat rendered as a sequence of query blocks."""

    # Moondream's query path defaults to reasoning on; honor that for chat.
    default_reasoning = True

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> "List[Token]":
        if not isinstance(request_context, ChatRequest):
            raise ValueError(
                "MoondreamChatSkill.build_prompt_tokens requires a ChatRequest"
            )
        pt = runtime.prompt_template
        template = pt.query()
        if template is None:
            raise ValueError("Moondream prompt template has no query()")
        tokenizer = runtime.tokenizer

        def ids(seq) -> "List[Token]":
            return [TextToken(token_id=int(t)) for t in seq]

        def text(value: str) -> "List[Token]":
            if not value:
                return []
            return [TextToken(token_id=int(t)) for t in tokenizer.encode(value).ids]

        messages = list(request_context.messages)
        # Fold a leading system message into the first user turn's text.
        pending_system = ""
        if messages and messages[0].role == "system":
            pending_system = messages[0].text
            messages = messages[1:]

        out: "List[Token]" = [TextToken(token_id=int(pt.bos_id))]
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            if msg.role == "user":
                # Images go at the start of the turn.
                for part in msg.parts:
                    if part.image_index is not None:
                        out.append(ImageMarker(index=part.image_index))
                out += ids(template.prefix)
                user_text = msg.text
                if pending_system:
                    user_text = (
                        f"{pending_system}\n\n{user_text}" if user_text else pending_system
                    )
                    pending_system = ""
                out += text(user_text)
                if i + 1 < n and messages[i + 1].role == "assistant":
                    out += ids(template.answer_prefix)
                    out += text(messages[i + 1].text)
                    i += 2
                    continue
                i += 1
            else:  # a stray assistant turn (no preceding user) — emit its answer
                out += ids(template.answer_prefix)
                out += text(msg.text)
                i += 1

        # Generation opener for the final (user) turn.
        opener = (
            template.reasoning_prefix
            if request_context.reasoning and template.reasoning_prefix
            else template.answer_prefix
        )
        out += ids(opener)
        return out

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
        # The decode state only needs reasoning/stream; images are handled by
        # the runtime via the prompt's ImageMarkers, not here.
        query_request = QueryRequest(
            question="",
            image=None,
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


__all__ = ["MoondreamChatSkill", "MoondreamChatSkillState"]
