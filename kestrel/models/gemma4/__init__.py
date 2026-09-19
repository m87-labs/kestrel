"""Lightweight gemma4 registration; implementation imports are demand-driven."""

from importlib import import_module

from kestrel.models.registry import register_lazy

_VARIANTS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E2B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-31B-it",
    "google/gemma-4-31B",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-26B-A4B",
]

register_lazy(_VARIANTS, __name__ + ".registration")

__all__ = ["Gemma4PromptTemplate", "Gemma4Runtime"]


def __getattr__(name):
    if name == "Gemma4PromptTemplate":
        module = ".prompt_template"
    elif name == "Gemma4Runtime":
        module = ".runtime"
    elif name == "build_skill_registry":
        module = ".skills"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
