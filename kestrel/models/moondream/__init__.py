"""Self-contained Moondream model components used by Kestrel."""

from kestrel.models.registry import ModelSpec, register

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
from .model import MoondreamModel, MoondreamTextModel
from .runtime import MoondreamRuntime, SequenceState, DEFAULT_MAX_TOKENS
from .weights import load_moondream_weights, load_text_weights

# Imported after ``.runtime``: the skill modules pull Moondream's token
# types from it, so the runtime module must be fully initialized first.
from .skills import build_skill_registry

# Both MD2 and MD3 share the Starmie tokenizer; the checkpoint_format tag
# is what the weight loader keys off to pick the right key-name layout.
register(
    ModelSpec(
        name="moondream2",
        repo_id="vikhyatk/moondream2",
        filename="model.safetensors",
        checkpoint_format="md2",
        default_config=DEFAULT_MOONDREAM2_CONFIG,
        tokenizer_id="moondream/starmie-v1",
        runtime=MoondreamRuntime,
        skills=build_skill_registry,
    )
)
register(
    ModelSpec(
        name="moondream3-preview",
        repo_id="moondream/moondream3-preview",
        filename="model_fp8.pt",
        checkpoint_format="md3",
        default_config=DEFAULT_MOONDREAM3_CONFIG,
        tokenizer_id="moondream/starmie-v1",
        runtime=MoondreamRuntime,
        skills=build_skill_registry,
    )
)

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
