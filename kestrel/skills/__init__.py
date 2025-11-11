"""Skill abstractions for the Kestrel inference engine."""

from .base import (
    DecodeStep,
    SkillFinalizeResult,
    SkillRegistry,
    SkillSpec,
    SkillState,
)
from .detect import DetectRequest, DetectSettings, DetectSkill
from .caption import CaptionRequest, CaptionSettings, CaptionSkill
from .point import PointRequest, PointSettings, PointSkill
from .query import QueryRequest, QuerySettings, QuerySkill
from .segment import SegmentRequest, SegmentSettings, SegmentSkill

__all__ = [
    "DecodeStep",
    "SkillFinalizeResult",
    "SkillRegistry",
    "SkillSpec",
    "SkillState",
    "DetectRequest",
    "DetectSettings",
    "DetectSkill",
    "CaptionRequest",
    "CaptionSettings",
    "CaptionSkill",
    "PointRequest",
    "PointSettings",
    "PointSkill",
    "QueryRequest",
    "QuerySettings",
    "QuerySkill",
    "SegmentRequest",
    "SegmentSettings",
    "SegmentSkill",
]
