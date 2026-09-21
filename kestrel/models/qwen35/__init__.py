"""Lightweight qwen35 registration; implementation imports are demand-driven."""

from importlib import import_module

from kestrel.models.registry import register_lazy

_VARIANTS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-27B-FP8",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-27B-FP8",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-35B-A3B-FP8",
]

register_lazy(_VARIANTS, __name__ + ".registration")

__all__ = ["Qwen35PromptTemplate", "Qwen35Runtime"]


def __getattr__(name):
    if name == "Qwen35PromptTemplate":
        module = ".prompt_template"
    elif name == "Qwen35Runtime":
        module = ".runtime"
    elif name == "build_skill_registry":
        module = ".skills"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
