"""Gemma 4 skills registry."""

from functools import cache
from typing import Mapping, Optional

import numpy as np

from kestrel.skills.query import QueryRequest, QuerySkill
from kestrel.skills import SkillRegistry
from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    BuiltRequest,
    parse_settings,
)
from kestrel.utils.spatial_refs import normalize_spatial_refs


class Gemma4QuerySkill(QuerySkill):
    """Query skill with Gemma 4's direct-answer behavior as default."""

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
            raise ValueError("Gemma 4 query does not support spatial_refs")
        sampling_settings = _without_generic_sampling_defaults(settings)
        s = parse_settings(
            sampling_settings,
            temperature=0.0,
            top_p=1.0,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        # Gemma 4 uses its direct-answer template by default; callers opt in
        # to its reasoning template through settings={"reasoning": True}.
        reasoning = bool(settings.get("reasoning", False)) if settings else False
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


def _without_generic_sampling_defaults(
    settings: Optional[Mapping[str, object]],
) -> Optional[Mapping[str, object]]:
    if settings is None:
        return None
    if (
        settings.get("temperature") == AR_DEFAULT_TEMPERATURE
        and settings.get("top_p") == AR_DEFAULT_TOP_P
    ):
        stripped = dict(settings)
        stripped.pop("temperature", None)
        stripped.pop("top_p", None)
        return stripped
    return settings


@cache
def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([Gemma4QuerySkill()])


__all__ = ["Gemma4QuerySkill", "build_skill_registry"]
