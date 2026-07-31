"""Model registry + per-family runtime packages."""

from .protocols import (
    CaptionPromptTemplate,
    ChatPromptTemplate,
    PrefixSuffix,
    PromptTemplate,
    QueryPromptTemplate,
    QueryTemplate,
    SpatialPromptTemplate,
)
from .registry import ModelSpec, get_spec, known_models, register

# Model packages register their specs at import time.
from . import gemma4, moondream, qwen35  # noqa: F401

__all__ = [
    "CaptionPromptTemplate",
    "ChatPromptTemplate",
    "ModelSpec",
    "PrefixSuffix",
    "PromptTemplate",
    "QueryPromptTemplate",
    "QueryTemplate",
    "SpatialPromptTemplate",
    "get_spec",
    "known_models",
    "register",
]
