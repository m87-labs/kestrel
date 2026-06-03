"""Moondream's autoregressive skills — its capability implementations.

The kernel defines the *generic* skill contract (``SkillSpec`` /
``SkillState`` / ``SkillRegistry`` in :mod:`kestrel.skills`); this package
holds Moondream's *concrete* capabilities (query, caption, detect, point,
segment) and assembles them into the registry the model advertises.

Skills are model-owned: each validates the raw request and builds the
model's own request type, so they live with the model rather than in the
model-agnostic kernel.
"""

from functools import cache

from kestrel.skills import SkillRegistry

from .caption import CaptionRequest, CaptionSkill
from .detect import DetectRequest, DetectSkill
from .point import PointRequest, PointSkill
from .query import QueryRequest, QuerySkill
from .segment import SegmentRequest, SegmentSkill


@cache
def build_skill_registry() -> SkillRegistry:
    """The skill set Moondream serves.

    Registered on the model's :class:`~kestrel.models.registry.ModelSpec`
    (``skills=``) so capabilities are known from static metadata — input
    validation and ``tasks`` introspection work without building the GPU
    runtime. Memoized: skills are stateless spec objects, so one shared
    registry is safe and lets engine, scheduler, and runtime agree on the
    same instance.
    """
    return SkillRegistry(
        [
            QuerySkill(),
            PointSkill(),
            DetectSkill(),
            CaptionSkill(),
            SegmentSkill(),
        ]
    )


__all__ = [
    "build_skill_registry",
    "CaptionRequest",
    "CaptionSkill",
    "DetectRequest",
    "DetectSkill",
    "PointRequest",
    "PointSkill",
    "QueryRequest",
    "QuerySkill",
    "SegmentRequest",
    "SegmentSkill",
]
