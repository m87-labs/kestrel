"""Gemma 4 skills registry."""

from functools import cache

from kestrel.skills import QueryPolicy, QuerySkill, SkillRegistry


_QUERY_POLICY = QueryPolicy(
    temperature=0.0,
    top_p=1.0,
    default_reasoning=False,
    reasoning_in_settings=True,
    supports_spatial_refs=False,
)


@cache
def build_skill_registry() -> SkillRegistry:
    return SkillRegistry([QuerySkill(_QUERY_POLICY)])


__all__ = ["build_skill_registry"]
