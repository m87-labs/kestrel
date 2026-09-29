"""Moondream registration metadata and lazy public model exports."""

from importlib import import_module
from kestrel.models.registry import register_lazy

from .config import (
    DEFAULT_MOONDREAM2_CONFIG,
    DEFAULT_MOONDREAM3_CONFIG,
    MoondreamConfig,
    MoondreamTextConfig,
    TextConfig,
    TextMoeConfig,
    TokenizerConfig,
    VisionConfig,
)

_MODEL_NAMES = ["moondream2", "moondream3-preview", "moondream3.1-9B-A2B"]
register_lazy(_MODEL_NAMES, __name__ + ".registration")

__all__ = [
    "DEFAULT_MOONDREAM2_CONFIG",
    "DEFAULT_MOONDREAM3_CONFIG",
    "MoondreamTextConfig",
    "MoondreamConfig",
    "TextConfig",
    "TextMoeConfig",
    "TokenizerConfig",
    "VisionConfig",
    "MoondreamModel",
    "MoondreamTextModel",
    "MoondreamRuntime",
    "SequenceState",
    "DEFAULT_MAX_TOKENS",
    "load_moondream_weights",
    "load_text_weights",
]

_LAZY_EXPORTS = {
    "build_skill_registry": ".skills",
    "MoondreamModel": ".model", "MoondreamTextModel": ".model",
    "MoondreamRuntime": ".runtime", "SequenceState": ".runtime",
    "DEFAULT_MAX_TOKENS": ".runtime",
    "load_moondream_weights": ".weights", "load_text_weights": ".weights",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
