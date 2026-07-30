"""Model registry + per-family runtime packages."""

from .protocols import PrefixSuffix, PromptTemplate, QueryTemplate
from .registry import ModelSpec, get_spec, known_models, register

# Model packages register their specs at import time.
from . import moondream, qwen35  # noqa: F401

__all__ = [
    "ModelSpec",
    "PrefixSuffix",
    "PromptTemplate",
    "QueryTemplate",
    "get_spec",
    "known_models",
    "register",
]
