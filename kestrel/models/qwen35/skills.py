"""Qwen 3.5 skill registry."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from kestrel.runtime.tokens import ImageMarker, TextToken, Token
from kestrel.skills import ChatSkill, QueryRequest, QuerySkill, SkillRegistry
from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    BuiltRequest,
    parse_settings,
)
from kestrel.utils.spatial_refs import normalize_spatial_refs

class Qwen35QuerySkill(QuerySkill):
    """Query skill with Qwen's non-reasoning chat path as the default."""

    def build_request(
        self,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        question = prompt.get("question")
        if question is None:
            raise ValueError("question must be provided")
        question = str(question).strip()
        if not question:
            raise ValueError("question must be a non-empty string")
        refs = normalize_spatial_refs(prompt.get("spatial_refs"))
        if refs is not None:
            raise ValueError("Qwen 3.5 query does not support spatial_refs")
        s = parse_settings(
            settings,
            temperature=0.0,
            top_p=1.0,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        # ``InferenceEngine.query`` carries request semantics in ``prompt``;
        # ``settings`` is reserved for sampling controls such as temperature.
        reasoning = bool(prompt.get("reasoning", False))
        request = QueryRequest(
            question=question,
            image=image,
            reasoning=reasoning,
            stream=bool(prompt.get("stream", False)),
            spatial_refs=None,
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
        )

class Qwen35ChatSkill(ChatSkill):
    """Qwen chat: emit a vision placeholder per image at its content position.

    The shared ``ChatSkill`` renders the ChatML turn skeleton; this subclass
    owns the model-specific bit — an image part becomes
    ``<|vision_start|><|image_pad|><|vision_end|>`` (one image-pad placeholder),
    which the runtime expands to that image's token count. Keeps all image
    knowledge out of the kernel and the shared chat template.
    """

    # Qwen's chat template defaults to thinking on; honor that.
    default_reasoning = True

    def render_content(self, tokenizer, parts) -> list[Token]:
        tokens: list[Token] = []
        for part in parts:
            if part.image_index is not None:
                # Sentinel; the runtime expands it to
                # <|vision_start|><|image_pad|>×N<|vision_end|>.
                tokens.append(ImageMarker(index=part.image_index))
            elif part.text:
                tokens.extend(
                    TextToken(token_id=int(t)) for t in tokenizer.encode(part.text).ids
                )
        return tokens


def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([Qwen35QuerySkill(), Qwen35ChatSkill()])


__all__ = [
    "Qwen35ChatSkill",
    "Qwen35QuerySkill",
    "build_skill_registry",
]
