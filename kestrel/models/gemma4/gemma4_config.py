"""Gemma 4 configuration dataclasses.

Vendored from ``transformers.models.gemma4.configuration_gemma4`` —
fields are kept name-compatible with HF so we can load HF safetensors
and instantiate from HF's published ``config.json`` files unchanged.

Notable Gemma-4 specifics that show up here:

* ``layer_types`` — per-layer attention kind (``"sliding_attention"``
  vs ``"full_attention"``); defaults to a 5:1 SWA:global pattern.
* ``hidden_size_per_layer_input`` / ``vocab_size_per_layer_input`` —
  Per-Layer Embeddings (PLE): a second embedding table fed in as a
  residual at every decoder layer.
* ``num_kv_shared_layers`` — last N layers reuse the previous global /
  sliding layer's K/V, skipping their own K/V projections + cache
  writes.
* ``rope_parameters`` — dual RoPE: distinct config per layer type.
* ``attention_k_eq_v`` — global layers can reuse K as V (no v_proj).
* ``enable_moe_block`` / ``num_experts`` — MoE for the 26B-A4B variant
  (E2B / E4B / 31B are dense).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# Gemma 4's ``rope_parameters`` for the text decoder are a dict-of-dicts
# keyed by layer type. Defaults match HF's published E2B-it config.
def _default_text_rope_parameters() -> dict[str, dict[str, Any]]:
    return {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
    }


def attention_kv_heads(config: Any, *, is_sliding: bool) -> int:
    """Return the K/V head count used by one attention layer."""
    if not is_sliding and bool(config.attention_k_eq_v):
        return int(config.num_global_key_value_heads)
    return int(config.num_key_value_heads)


@dataclass
class Gemma4TextConfig:
    """Text decoder configuration.

    Field names + defaults match HF's ``Gemma4TextConfig`` so HF
    ``config.json`` payloads round-trip without translation.
    """

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = 0
    eos_token_id: Any = 1  # int | list[int]
    bos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = True
    rope_parameters: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 512
    layer_types: Optional[list[str]] = None
    final_logit_softcapping: Optional[float] = None
    use_bidirectional_attention: Optional[Literal["all", "vision"]] = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: Optional[int] = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    # Backend selector — ``"eager"`` is the only path the vendored
    # decoder currently exercises. Surface kept so configs from HF
    # round-trip cleanly.
    _attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        if self.use_bidirectional_attention == "all":
            self.sliding_window = (self.sliding_window // 2) + 1
        if self.layer_types is None:
            # Default 5:1 SWA:global pattern (every 6th layer is full).
            pattern = 6
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = _default_text_rope_parameters()
        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads

    # --- HF-config compatibility shims ---
    # ``Gemma4PreTrainedModel.config`` is sometimes accessed via these
    # convenience methods. Implementing them keeps weight-loading helpers
    # and HF config-json deserialization simple.
    def get_text_config(self) -> "Gemma4TextConfig":
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4TextConfig":
        """Construct from an HF-style config dict, ignoring unknown keys.

        HF's ``config.json`` carries a number of fields the modeling
        code doesn't read (``architectures``, ``transformers_version``,
        ``torch_dtype``, etc.). We tolerate them so the same payload
        loads here unchanged.
        """
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class Gemma4VisionConfig:
    """Vision tower configuration.

    Field names + defaults match HF's ``Gemma4VisionConfig`` so HF
    ``config.json`` payloads round-trip without translation.
    """

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131_072
    attention_bias: Optional[bool] = False
    attention_dropout: Optional[float] = 0.0
    rope_parameters: Optional[dict] = None
    pooling_kernel_size: int = 3
    patch_size: int = 16
    position_embedding_size: int = 10 * 1024
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02
    # Backend selector — vendored vision attention always uses eager.
    _attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4VisionConfig":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class Gemma4Config:
    """Top-level Gemma 4 config for the supported text-and-vision runtime.

    Field names + nested-config types match the checkpoint where they affect
    the supported runtime. Set ``vision_config`` to ``None`` to skip the vision
    tower and its embedder.
    """

    text_config: Optional[Gemma4TextConfig] = None
    vision_config: Optional[Gemma4VisionConfig] = None
    image_token_id: Optional[int] = 258_880
    initializer_range: Optional[float] = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = Gemma4TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4VisionConfig.from_dict(self.vision_config)

    def get_text_config(self) -> "Gemma4TextConfig":
        return self.text_config

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4Config":
        # HF nests configs under ``text_config`` / ``vision_config``. Top-level
        # metadata and unsupported modality configs are dropped.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
]
