"""Skill abstractions for the Kestrel inference engine."""

from .base import SkillRegistry, SkillSpec
from .query import QuerySkill

__all__ = [
    "SkillRegistry",
    "SkillSpec",
    "QuerySkill",
]
