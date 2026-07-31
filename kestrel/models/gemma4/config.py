"""Strict inference architecture descriptors for supported Gemma 4 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


LayerType = Literal["sliding_attention", "full_attention"]


@dataclass(frozen=True)
class RopeSpec:
    kind: Literal["default", "proportional"]
    theta: float
    partial_rotary_factor: float = 1.0
    factor: float = 1.0


@dataclass(frozen=True)
class Gemma4TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope: Mapping[LayerType, RopeSpec]
    sliding_window: int
    layer_types: tuple[LayerType, ...]
    final_logit_softcapping: float
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int
    num_global_key_value_heads: int
    global_head_dim: int
    attention_k_eq_v: bool
    num_kv_shared_layers: int
    use_double_wide_mlp: bool


@dataclass(frozen=True)
class Gemma4VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope: RopeSpec
    pooling_kernel_size: int
    patch_size: int
    position_embedding_size: int
    use_clipped_linears: bool
    standardize: bool


@dataclass(frozen=True)
class Gemma4Config:
    text_config: Gemma4TextConfig
    vision_config: Gemma4VisionConfig
    image_token_id: int


def _required(data: Mapping[str, Any], key: str, owner: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ValueError(f"{owner} is missing required field {key!r}") from exc


def _rope(raw: Mapping[str, Any], owner: str) -> RopeSpec:
    kind = str(_required(raw, "rope_type", owner))
    if kind not in ("default", "proportional"):
        raise ValueError(f"{owner} has unsupported rope_type {kind!r}")
    partial = float(raw.get("partial_rotary_factor", 1.0))
    if kind == "default" and partial != 1.0:
        raise ValueError(f"{owner} default RoPE cannot be partial")
    return RopeSpec(
        kind=kind,
        theta=float(_required(raw, "rope_theta", owner)),
        partial_rotary_factor=partial,
        factor=float(raw.get("factor", 1.0)),
    )


def _validate_fixed_inference_contract(
    data: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    if data.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError(f"{owner} requires hidden_activation='gelu_pytorch_tanh'")
    if bool(data.get("attention_bias", False)):
        raise ValueError(f"{owner} requires bias-free attention")
    if float(data.get("attention_dropout", 0.0)) != 0.0:
        raise ValueError(f"{owner} does not support attention dropout")


def parse_gemma4_config(raw: Mapping[str, Any]) -> Gemma4Config:
    """Parse the architecture fields consumed by inference from a checkpoint config."""

    text = _required(raw, "text_config", "Gemma 4 config")
    vision = _required(raw, "vision_config", "Gemma 4 config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("Gemma 4 text_config and vision_config must be mappings")
    _validate_fixed_inference_contract(text, owner="Gemma 4 text config")
    _validate_fixed_inference_contract(vision, owner="Gemma 4 vision config")
    if bool(text.get("enable_moe_block", False)):
        raise ValueError("Gemma 4 MoE checkpoints are not supported")
    if not bool(raw.get("tie_word_embeddings", False)):
        raise ValueError("Gemma 4 inference requires tied token embeddings")

    layer_types = tuple(map(str, _required(text, "layer_types", "Gemma 4 text config")))
    allowed = {"sliding_attention", "full_attention"}
    if len(layer_types) != int(_required(text, "num_hidden_layers", "Gemma 4 text config")):
        raise ValueError("Gemma 4 layer_types must cover every decoder layer")
    if set(layer_types) - allowed:
        raise ValueError(f"unsupported Gemma 4 layer types {sorted(set(layer_types) - allowed)}")
    num_kv_shared = int(_required(text, "num_kv_shared_layers", "Gemma 4 text config"))
    if not 0 <= num_kv_shared < len(layer_types):
        raise ValueError("num_kv_shared_layers must leave at least one K/V producer")

    rope_raw = _required(text, "rope_parameters", "Gemma 4 text config")
    rope = {
        kind: _rope(
            _required(rope_raw, kind, "Gemma 4 text RoPE"),
            f"Gemma 4 {kind} RoPE",
        )
        for kind in ("sliding_attention", "full_attention")
    }
    num_kv_heads = int(_required(text, "num_key_value_heads", "Gemma 4 text config"))
    text_config = Gemma4TextConfig(
        vocab_size=int(_required(text, "vocab_size", "Gemma 4 text config")),
        hidden_size=int(_required(text, "hidden_size", "Gemma 4 text config")),
        intermediate_size=int(_required(text, "intermediate_size", "Gemma 4 text config")),
        num_hidden_layers=len(layer_types),
        num_attention_heads=int(_required(text, "num_attention_heads", "Gemma 4 text config")),
        num_key_value_heads=num_kv_heads,
        head_dim=int(_required(text, "head_dim", "Gemma 4 text config")),
        max_position_embeddings=int(
            _required(text, "max_position_embeddings", "Gemma 4 text config")
        ),
        rms_norm_eps=float(_required(text, "rms_norm_eps", "Gemma 4 text config")),
        rope=rope,
        sliding_window=int(_required(text, "sliding_window", "Gemma 4 text config")),
        layer_types=layer_types,
        final_logit_softcapping=float(
            _required(text, "final_logit_softcapping", "Gemma 4 text config")
        ),
        vocab_size_per_layer_input=int(
            _required(text, "vocab_size_per_layer_input", "Gemma 4 text config")
        ),
        hidden_size_per_layer_input=int(
            _required(text, "hidden_size_per_layer_input", "Gemma 4 text config")
        ),
        num_global_key_value_heads=int(text.get("num_global_key_value_heads") or num_kv_heads),
        global_head_dim=int(_required(text, "global_head_dim", "Gemma 4 text config")),
        attention_k_eq_v=bool(_required(text, "attention_k_eq_v", "Gemma 4 text config")),
        num_kv_shared_layers=num_kv_shared,
        use_double_wide_mlp=bool(
            _required(text, "use_double_wide_mlp", "Gemma 4 text config")
        ),
    )
    if text_config.num_attention_heads % text_config.num_key_value_heads:
        raise ValueError("text attention heads must be divisible by local K/V heads")
    if text_config.num_attention_heads % text_config.num_global_key_value_heads:
        raise ValueError("text attention heads must be divisible by global K/V heads")

    vision_rope = _rope(
        _required(vision, "rope_parameters", "Gemma 4 vision config"),
        "Gemma 4 vision RoPE",
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=int(_required(vision, "hidden_size", "Gemma 4 vision config")),
        intermediate_size=int(
            _required(vision, "intermediate_size", "Gemma 4 vision config")
        ),
        num_hidden_layers=int(
            _required(vision, "num_hidden_layers", "Gemma 4 vision config")
        ),
        num_attention_heads=int(
            _required(vision, "num_attention_heads", "Gemma 4 vision config")
        ),
        num_key_value_heads=int(
            _required(vision, "num_key_value_heads", "Gemma 4 vision config")
        ),
        head_dim=int(_required(vision, "head_dim", "Gemma 4 vision config")),
        rms_norm_eps=float(_required(vision, "rms_norm_eps", "Gemma 4 vision config")),
        rope=vision_rope,
        pooling_kernel_size=int(
            _required(vision, "pooling_kernel_size", "Gemma 4 vision config")
        ),
        patch_size=int(_required(vision, "patch_size", "Gemma 4 vision config")),
        position_embedding_size=int(
            _required(vision, "position_embedding_size", "Gemma 4 vision config")
        ),
        use_clipped_linears=bool(
            _required(vision, "use_clipped_linears", "Gemma 4 vision config")
        ),
        standardize=bool(_required(vision, "standardize", "Gemma 4 vision config")),
    )
    if vision_config.num_attention_heads % vision_config.num_key_value_heads:
        raise ValueError("vision attention heads must be divisible by K/V heads")

    return Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=int(_required(raw, "image_token_id", "Gemma 4 config")),
    )


def attention_kv_heads(config: Gemma4TextConfig, *, is_sliding: bool) -> int:
    if not is_sliding and config.attention_k_eq_v:
        return config.num_global_key_value_heads
    return config.num_key_value_heads


__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "LayerType",
    "RopeSpec",
    "attention_kv_heads",
    "parse_gemma4_config",
]
