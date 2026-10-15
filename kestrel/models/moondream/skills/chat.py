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

import json
import re
from dataclasses import dataclass, replace
from typing import Dict, List, cast

from kestrel.models.protocols import QueryPromptTemplate
from kestrel.runtime.tokens import ImageMarker, TextToken
from kestrel.skills.base import BuiltRequest, SkillFinalizeResult, SkillSpec, SkillState
from kestrel.skills.chat import ChatRequest, ChatSkill
from kestrel.skills.query import QueryRequest, QuerySkillState

from .caption import CaptionSkill, CaptionSkillState
from .detect import DetectSkill, DetectSkillState
from .point import PointSkill, PointSkillState

if False:  # pragma: no cover - type-checking imports
    from ..runtime import MoondreamRuntime, Token
    from kestrel.scheduler.types import GenerationRequest


_NATIVE_SKILLS = {
    "detect": DetectSkill(),
    "point": PointSkill(),
    "caption": CaptionSkill(),
}
_COMMAND = re.compile(r"^/(detect|point|caption)(?=\s|\{|$)")


@dataclass(frozen=True)
class _NativeChatRequest:
    operation: str
    context: object
    stream: bool


def _parse_command(text: str):
    match = _COMMAND.match(text)
    if match is None:
        return None
    operation = match[1]
    try:
        arguments = json.loads(text[match.end() :])
    except ValueError as exc:
        raise ValueError(
            f"/{operation} arguments must be a JSON object: {exc}"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"/{operation} arguments must be a JSON object")
    supported = {
        "detect": {"object", "max_objects"},
        "point": {"object", "spatial_refs", "max_objects"},
        "caption": {"length"},
    }[operation]
    unsupported = arguments.keys() - supported
    if unsupported:
        raise ValueError(
            f"/{operation} unsupported fields: {', '.join(sorted(unsupported))}"
        )
    if operation != "caption":
        obj = arguments.get("object")
        if not isinstance(obj, str) or not obj.strip():
            raise ValueError(f"/{operation} object must be a non-empty string")
    elif "length" in arguments and not isinstance(arguments["length"], str):
        raise ValueError("/caption length must be a string")
    if "max_objects" in arguments:
        cap = arguments["max_objects"]
        if type(cap) is not int or cap <= 0:
            raise ValueError(f"/{operation} max_objects must be a positive integer")
    if "spatial_refs" in arguments:
        refs = arguments["spatial_refs"]
        if refs is not None and (
            not isinstance(refs, list)
            or any(
                not isinstance(ref, list)
                or any(type(v) not in (int, float) for v in ref)
                for ref in refs
            )
        ):
            raise ValueError(
                "/point spatial_refs must be an array of coordinate arrays"
            )
    return operation, arguments


class MoondreamChatSkill(ChatSkill):
    """Multi-turn, multi-image chat rendered as a sequence of query blocks."""

    # Moondream's query path defaults to reasoning on; honor that for chat.
    default_reasoning = True

    def build_request(self, image, prompt, settings) -> BuiltRequest:
        built = super().build_request(image, prompt, settings)
        chat = cast(ChatRequest, built.request_context)
        command = _parse_command(chat.messages[-1].text)
        if command is None:
            return built
        operation, arguments = command
        if len(chat.images) != 1 or image is not None:
            raise ValueError(
                f"/{operation} requires exactly one image in user messages"
            )
        if any(m.role != "user" and m.has_image for m in chat.messages):
            raise ValueError(f"/{operation} supports images only in user messages")
        native_settings = dict(settings or {})
        if "max_objects" in arguments:
            native_settings["max_objects"] = arguments.pop("max_objects")
        native = _NATIVE_SKILLS[operation].build_request(
            chat.images[0], arguments, native_settings
        )
        return replace(
            native,
            request_context=_NativeChatRequest(
                operation, native.request_context, chat.stream
            ),
            image=chat.images[0],
        )

    def prompt_text(self, request_context: object) -> str:
        if isinstance(request_context, _NativeChatRequest):
            return _NATIVE_SKILLS[request_context.operation].prompt_text(
                request_context.context
            )
        return super().prompt_text(request_context)

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> "List[Token]":
        if isinstance(request_context, _NativeChatRequest):
            return list(
                _NATIVE_SKILLS[request_context.operation].build_prompt_tokens(
                    runtime, request_context.context
                )
            )
        if not isinstance(request_context, ChatRequest):
            raise ValueError(
                "MoondreamChatSkill.build_prompt_tokens requires a ChatRequest"
            )
        # Moondream renders images only at user turns (one ImageMarker per user
        # image part). A system/assistant image part is parsed into the image
        # tuple but never gets a marker here, desyncing the tuple from the
        # prompt and the length accounting — reject it rather than silently
        # dropping the image.
        for m in request_context.messages:
            if m.role != "user" and any(p.image_index is not None for p in m.parts):
                raise ValueError(
                    "Moondream chat supports images only in user messages; "
                    f"found image content in a {m.role!r} message"
                )
        pt = runtime.prompt_template
        template = cast(QueryPromptTemplate, pt).query()
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
                        f"{pending_system}\n\n{user_text}"
                        if user_text
                        else pending_system
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
    ) -> SkillState:
        if isinstance(request_context, _NativeChatRequest):
            state = _NATIVE_STATES[request_context.operation](
                _NATIVE_SKILLS[request_context.operation],
                request,
                request_context.context,
            )
            state._chat_stream = request_context.stream
            return state
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


class _NativeChatState:
    """Keep native decoding and counters; publish only the completed JSON."""

    _chat_stream = False
    _pending_json = None

    def pop_stream_output(self, runtime):
        if self._pending_json is None:
            # Retain token-only updates for engine consumers tracking usage.
            return {}
        text = self._pending_json
        self._pending_json = None
        return {"text": text}

    def finalize(self, runtime, *, reason: str) -> SkillFinalizeResult:
        result = super().finalize(runtime, reason=reason)
        text = json.dumps(result.output, allow_nan=False)
        if self._chat_stream and reason != "cancelled":
            self._pending_json = text
        return SkillFinalizeResult(
            text=text,
            tokens=result.tokens,
            output={
                "message": {"role": "assistant", "content": text},
                "finish_reason": reason,
            },
        )


class _DetectChatState(_NativeChatState, DetectSkillState):
    pass


class _PointChatState(_NativeChatState, PointSkillState):
    pass


class _CaptionChatState(_NativeChatState, CaptionSkillState):
    pass


_NATIVE_STATES = {
    "detect": _DetectChatState,
    "point": _PointChatState,
    "caption": _CaptionChatState,
}


__all__ = ["MoondreamChatSkill", "MoondreamChatSkillState"]
