"""Qwen3-TTS CustomVoice support for Kestrel."""

from importlib import import_module

from kestrel.models.registry import register_lazy

from .config import SUPPORTED_CHECKPOINTS

register_lazy(list(SUPPORTED_CHECKPOINTS), __name__ + ".registration")


__all__ = ["SUPPORTED_CHECKPOINTS", "Qwen3TTSRuntime"]


def __getattr__(name):
    if name != "Qwen3TTSRuntime":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".runtime", __name__), name)
    globals()[name] = value
    return value
