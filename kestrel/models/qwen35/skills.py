"""Qwen 3.5 skill registry."""

from __future__ import annotations

from kestrel.runtime.tokens import ImageMarker, TextToken, Token
from kestrel.skills import ChatSkill, QueryPolicy, QuerySkill, SkillRegistry


_QUERY_POLICY = QueryPolicy(
    temperature=0.0,
    top_p=1.0,
    default_reasoning=False,
    supports_spatial_refs=False,
    strip_client_sampling_defaults=True,
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
    return SkillRegistry([QuerySkill(_QUERY_POLICY), Qwen35ChatSkill()])


__all__ = [
    "Qwen35ChatSkill",
    "build_skill_registry",
]
