"""Lightweight qwen3_asr registration and lazy public exports."""

from importlib import import_module
from kestrel.models.registry import register_lazy
from .metadata import QWEN3_ASR_MODELS, ALIGNER_MODEL_ID

register_lazy([*QWEN3_ASR_MODELS, ALIGNER_MODEL_ID], __name__ + ".registration")

__all__ = [
    "QWEN3_ASR_MODELS",
    "Qwen3AsrRuntime",
    "Qwen3ForcedAlignerRuntime",
    "load_qwen3_asr",
]

_LAZY_EXPORTS = {
    "Qwen3AsrRuntime": ".runtime",
    "Qwen3ForcedAlignerRuntime": ".alignment",
    "load_qwen3_asr": ".weights"
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
