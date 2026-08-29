"""Validated inference shape descriptors for Qwen 3.5/3.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel.models.config import required_config, required_config_kwargs


@dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    head_dim: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    layer_types: tuple[str, ...] | list[str]
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    num_experts_per_tok: int | None = None
    num_experts: int | None = None
    expert_weight_format: str = "bf16"

    def __post_init__(self) -> None:
        layer_types = tuple(str(kind) for kind in self.layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError("Qwen layer_types length must equal num_hidden_layers")
        unsupported = set(layer_types) - {"linear_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"unsupported Qwen layer types: {sorted(unsupported)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "Qwen num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "Qwen linear value heads must be divisible by linear key heads"
            )
        if len(self.mrope_section) != 3:
            raise ValueError("Qwen M-RoPE requires exactly three sections")
        if self.expert_weight_format not in {"bf16", "fp8_e4m3"}:
            raise ValueError(
                f"unsupported Qwen expert weight format {self.expert_weight_format!r}"
            )
        moe_fields = (
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.num_experts_per_tok,
            self.num_experts,
        )
        if any(value is not None for value in moe_fields) and not all(
            value is not None for value in moe_fields
        ):
            raise ValueError("Qwen MoE dimensions must be provided together")
        object.__setattr__(self, "layer_types", layer_types)
        object.__setattr__(
            self,
            "mrope_section",
            tuple(int(value) for value in self.mrope_section),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        is_moe: bool,
        tie_word_embeddings: bool,
        expert_weight_format: str,
    ) -> "Qwen3_5TextConfig":
        if required_config(data, "hidden_act", "Qwen text") != "silu":
            raise ValueError("Qwen text inference requires hidden_act='silu'")
        if required_config(data, "mamba_ssm_dtype", "Qwen text") != "float32":
            raise ValueError("Qwen GDN inference requires mamba_ssm_dtype='float32'")
        if bool(required_config(data, "attention_bias", "Qwen text")):
            raise ValueError("Qwen inference requires attention_bias=false")

        rope = dict(required_config(data, "rope_parameters", "Qwen text"))
        if required_config(rope, "rope_type", "Qwen RoPE") != "default":
            raise ValueError("Qwen inference only supports default RoPE")
        if not bool(required_config(rope, "mrope_interleaved", "Qwen RoPE")):
            raise ValueError("Qwen inference requires interleaved M-RoPE")

        if "layer_types" in data:
            layer_types = tuple(data["layer_types"])
        else:
            interval = int(
                required_config(data, "full_attention_interval", "Qwen text")
            )
            if interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            layer_types = tuple(
                "linear_attention" if (idx + 1) % interval else "full_attention"
                for idx in range(int(data["num_hidden_layers"]))
            )

        transformed = frozenset(
            {
                "intermediate_size",
                "tie_word_embeddings",
                "rope_theta",
                "partial_rotary_factor",
                "mrope_section",
                "layer_types",
            }
        )
        kwargs = required_config_kwargs(
            cls,
            data,
            scope="Qwen text",
            transformed=transformed,
        )
        kwargs.update(
            intermediate_size=(
                data.get("intermediate_size")
                if is_moe
                else required_config(data, "intermediate_size", "Qwen dense text")
            ),
            tie_word_embeddings=bool(tie_word_embeddings),
            rope_theta=float(required_config(rope, "rope_theta", "Qwen RoPE")),
            partial_rotary_factor=float(
                required_config(rope, "partial_rotary_factor", "Qwen RoPE")
            ),
            mrope_section=tuple(
                int(value)
                for value in required_config(rope, "mrope_section", "Qwen RoPE")
            ),
            layer_types=layer_types,
            expert_weight_format=expert_weight_format,
        )
        if is_moe:
            for name in (
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "num_experts_per_tok",
                "num_experts",
            ):
                kwargs[name] = required_config(data, name, "Qwen MoE")
        return cls(**kwargs)

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


@dataclass(frozen=True, slots=True)
class Qwen3_5VisionConfig:
    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int

    def __post_init__(self) -> None:
        if self.hidden_act not in {"gelu", "gelu_pytorch_tanh"}:
            raise ValueError(
                f"unsupported Qwen vision activation {self.hidden_act!r}"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError(
                "Qwen vision hidden_size must be divisible by num_heads"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5VisionConfig":
        if data.get("deepstack_visual_indexes"):
            raise ValueError("Qwen inference does not support deepstack vision")
        return cls(**required_config_kwargs(cls, data, scope="Qwen vision"))


@dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig
    image_token_id: int
    video_token_id: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5Config":
        model_type = str(required_config(data, "model_type", "Qwen"))
        text_data = dict(required_config(data, "text_config", "Qwen"))
        quantization = data.get("quantization_config") or {}
        expert_weight_format = (
            "fp8_e4m3"
            if quantization.get("quant_method") == "fp8"
            and quantization.get("fmt") == "e4m3"
            else "bf16"
        )
        text = Qwen3_5TextConfig.from_dict(
            text_data,
            is_moe=model_type == "qwen3_5_moe"
            or str(text_data.get("model_type", "")).startswith("qwen3_5_moe"),
            tie_word_embeddings=bool(
                required_config(data, "tie_word_embeddings", "Qwen")
            ),
            expert_weight_format=expert_weight_format,
        )
        return cls(
            text_config=text,
            vision_config=Qwen3_5VisionConfig.from_dict(
                dict(required_config(data, "vision_config", "Qwen"))
            ),
            image_token_id=int(required_config(data, "image_token_id", "Qwen")),
            video_token_id=int(required_config(data, "video_token_id", "Qwen")),
        )


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
