"""Model registry + per-family runtime packages."""

from .protocols import PrefixSuffix, PromptTemplate, QueryTemplate
from .registry import ModelSpec, get_spec, known_models, register

# Importing the moondream package registers its specs as a side effect.
from . import moondream  # noqa: F401

__all__ = [
    "ModelSpec",
    "PrefixSuffix",
    "PromptTemplate",
    "QueryTemplate",
    "get_spec",
    "known_models",
    "register",
]
