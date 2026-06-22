"""The generic skill contract used by the Kestrel kernel.

This package is model-agnostic: it defines what a *skill* is — the
``SkillSpec`` behavior protocol, the per-request ``SkillState``, the
``SkillRegistry`` mapping capability names to specs, and the small value
types they exchange with the kernel (``DecodeStep``, ``BuiltRequest``,
``SkillFinalizeResult``, autoregressive sampling defaults).

Most concrete capabilities live with their model (e.g.
``kestrel.models.moondream.skills``), since validation and request-building
are model-specific. The exception is :class:`~kestrel.skills.chat.ChatSkill`,
a model-agnostic chat capability that any model can serve by exposing a
``chat()`` prompt template — it lives here so models share one instance.
"""

from .base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    BuiltRequest,
    DecodeStep,
    SkillFinalizeResult,
    SkillRegistry,
    SkillSettings,
    SkillSpec,
    SkillState,
    parse_settings,
)
from .chat import ChatRequest, ChatSkill, ChatSkillState

__all__ = [
    "AR_DEFAULT_MAX_NEW_TOKENS",
    "AR_DEFAULT_TEMPERATURE",
    "AR_DEFAULT_TOP_P",
    "BuiltRequest",
    "ChatRequest",
    "ChatSkill",
    "ChatSkillState",
    "DecodeStep",
    "SkillFinalizeResult",
    "SkillRegistry",
    "SkillSettings",
    "SkillSpec",
    "SkillState",
    "parse_settings",
]
