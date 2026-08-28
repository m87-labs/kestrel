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
from . import (  # noqa: F401
    gemma4,
    moondream,
    parakeet_tdt,
    qwen35,
    qwen3_asr,
    whisper,
)

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
