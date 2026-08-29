"""Strict inference architecture descriptors for supported Gemma 4 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from kestrel.models.config import required_config, required_config_kwargs


def _validate_dense_inference(data: Mapping[str, Any], scope: str) -> None:
    if data.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError(f"{scope} requires hidden_activation='gelu_pytorch_tanh'")
    if bool(data.get("attention_bias", False)):
        raise ValueError(f"{scope} requires bias-free attention")
    if float(data.get("attention_dropout", 0.0)) != 0.0:
        raise ValueError(f"{scope} does not support attention dropout")


@dataclass(frozen=True, slots=True)
class RopeSpec:
    kind: Literal["default", "proportional"]
    theta: float
    partial_rotary_factor: float = 1.0
    factor: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], scope: str) -> "RopeSpec":
        kind = str(required_config(data, "rope_type", scope))
        if kind not in ("default", "proportional"):
            raise ValueError(f"{scope} has unsupported rope_type {kind!r}")
        partial = float(data.get("partial_rotary_factor", 1.0))
        if kind == "default" and partial != 1.0:
            raise ValueError(f"{scope} default RoPE cannot be partial")
        return cls(
            kind=kind,
            theta=float(required_config(data, "rope_theta", scope)),
            partial_rotary_factor=partial,
            factor=float(data.get("factor", 1.0)),
        )


@dataclass(frozen=True, slots=True)
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
    rope: Mapping[str, RopeSpec]
    sliding_window: int
    layer_types: tuple[str, ...]
    final_logit_softcapping: float
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int
    num_global_key_value_heads: int
    global_head_dim: int
    attention_k_eq_v: bool
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    enable_moe_block: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Gemma 4 layer_types must cover every decoder layer")
        unsupported = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if unsupported:
            raise ValueError(
                f"unsupported Gemma 4 layer types {sorted(unsupported)}"
            )
        if not 0 <= self.num_kv_shared_layers < self.num_hidden_layers:
            raise ValueError(
                "num_kv_shared_layers must leave at least one K/V producer"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "text attention heads must be divisible by local K/V heads"
            )
        if self.num_attention_heads % self.num_global_key_value_heads:
            raise ValueError(
                "text attention heads must be divisible by global K/V heads"
            )
        moe_values = (
            self.num_experts,
            self.top_k_experts,
            self.moe_intermediate_size,
        )
        if self.enable_moe_block:
            if any(value is None for value in moe_values):
                raise ValueError(
                    "Gemma 4 MoE requires num_experts, top_k_experts, and "
                    "moe_intermediate_size"
                )
            assert self.num_experts is not None
            assert self.top_k_experts is not None
            assert self.moe_intermediate_size is not None
            if min(
                self.num_experts,
                self.top_k_experts,
                self.moe_intermediate_size,
            ) <= 0:
                raise ValueError("Gemma 4 MoE dimensions must be positive")
            if self.top_k_experts > self.num_experts:
                raise ValueError("top_k_experts cannot exceed num_experts")
        elif any(value is not None for value in moe_values):
            raise ValueError(
                "dense Gemma 4 config cannot define MoE dimensions"
            )

    @property
    def is_moe(self) -> bool:
        return self.enable_moe_block

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gemma4TextConfig":
        _validate_dense_inference(data, "Gemma 4 text config")
        rope_data = required_config(data, "rope_parameters", "Gemma 4 text")
        num_kv_heads = int(
            required_config(data, "num_key_value_heads", "Gemma 4 text")
        )
        kwargs = required_config_kwargs(
            cls,
            data,
            scope="Gemma 4 text",
            transformed=frozenset({"layer_types", "rope", "num_global_key_value_heads"}),
        )
        kwargs.update(
            layer_types=tuple(
                map(str, required_config(data, "layer_types", "Gemma 4 text"))
            ),
            rope={
                kind: RopeSpec.from_dict(
                    required_config(rope_data, kind, "Gemma 4 text RoPE"),
                    f"Gemma 4 {kind} RoPE",
                )
                for kind in ("sliding_attention", "full_attention")
            },
            num_global_key_value_heads=int(
                data.get("num_global_key_value_heads") or num_kv_heads
            ),
            enable_moe_block=bool(data.get("enable_moe_block", False)),
            num_experts=(
                int(data["num_experts"])
                if data.get("num_experts") is not None
                else None
            ),
            top_k_experts=(
                int(data["top_k_experts"])
                if data.get("top_k_experts") is not None
                else None
            ),
            moe_intermediate_size=(
                int(data["moe_intermediate_size"])
                if data.get("moe_intermediate_size") is not None
                else None
            ),
        )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("vision attention heads must be divisible by K/V heads")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gemma4VisionConfig":
        _validate_dense_inference(data, "Gemma 4 vision config")
        kwargs = required_config_kwargs(
            cls,
            data,
            scope="Gemma 4 vision",
            transformed=frozenset({"rope"}),
        )
        kwargs["rope"] = RopeSpec.from_dict(
            required_config(data, "rope_parameters", "Gemma 4 vision"),
            "Gemma 4 vision RoPE",
        )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    text_config: Gemma4TextConfig
    vision_config: Gemma4VisionConfig
    image_token_id: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gemma4Config":
        if not bool(data.get("tie_word_embeddings", False)):
            raise ValueError("Gemma 4 inference requires tied token embeddings")
        text = required_config(data, "text_config", "Gemma 4")
        vision = required_config(data, "vision_config", "Gemma 4")
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ValueError("Gemma 4 text_config and vision_config must be mappings")
        return cls(
            text_config=Gemma4TextConfig.from_dict(text),
            vision_config=Gemma4VisionConfig.from_dict(vision),
            image_token_id=int(required_config(data, "image_token_id", "Gemma 4")),
        )


def attention_kv_heads(config: Gemma4TextConfig, *, is_sliding: bool) -> int:
    if not is_sliding and config.attention_k_eq_v:
        return config.num_global_key_value_heads
    return config.num_key_value_heads


__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "RopeSpec",
    "attention_kv_heads",
]
