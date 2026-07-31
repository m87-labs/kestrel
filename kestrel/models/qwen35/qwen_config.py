"""Validated inference shape descriptors for Qwen 3.5/3.6."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def _known_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def _default(cls: type, name: str) -> Any:
    return cls.__dataclass_fields__[name].default


def _require(data: dict[str, Any], names: tuple[str, ...], scope: str) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{scope} config is missing required fields: {missing}")


_TEXT_SHAPE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_parameters",
    "attention_bias",
    "head_dim",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
)

_VISION_SHAPE_FIELDS = (
    "depth",
    "hidden_size",
    "hidden_act",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "out_hidden_size",
    "num_position_embeddings",
)


@dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: tuple[str, ...] | list[str] | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    num_experts_per_tok: int | None = None
    num_experts: int | None = None
    expert_weight_format: str = "bf16"

    def __post_init__(self) -> None:
        layer_types = self.layer_types
        if layer_types is None:
            layer_types = tuple(
                "linear_attention" if (idx + 1) % 4 else "full_attention"
                for idx in range(self.num_hidden_layers)
            )
        else:
            layer_types = tuple(str(kind) for kind in layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError(
                "Qwen layer_types length must equal num_hidden_layers"
            )
        unsupported = set(layer_types) - {"linear_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"unsupported Qwen layer types: {sorted(unsupported)}")
        if len(self.mrope_section) != 3:
            raise ValueError("Qwen M-RoPE requires exactly three sections")
        if self.expert_weight_format not in {"bf16", "fp8_e4m3"}:
            raise ValueError(
                f"unsupported Qwen expert weight format {self.expert_weight_format!r}"
            )
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
        is_moe: bool = False,
        tie_word_embeddings: bool | None = None,
        expert_weight_format: str = "bf16",
    ) -> "Qwen3_5TextConfig":
        _require(data, _TEXT_SHAPE_FIELDS, "Qwen text")
        if "layer_types" not in data and "full_attention_interval" not in data:
            raise ValueError(
                "Qwen text config requires layer_types or full_attention_interval"
            )
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("Qwen text inference requires hidden_act='silu'")
        if data.get("mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("Qwen GDN inference requires mamba_ssm_dtype='float32'")
        if bool(data.get("attention_bias", False)):
            raise ValueError("Qwen inference requires attention_bias=false")

        rope = dict(data["rope_parameters"])
        _require(
            rope,
            (
                "rope_type",
                "rope_theta",
                "partial_rotary_factor",
                "mrope_section",
                "mrope_interleaved",
            ),
            "Qwen RoPE",
        )
        if rope["rope_type"] != "default":
            raise ValueError("Qwen inference only supports default RoPE")
        if not bool(rope["mrope_interleaved"]):
            raise ValueError("Qwen inference requires interleaved M-RoPE")

        kwargs = _known_kwargs(cls, data)
        if "layer_types" not in data:
            interval = int(data["full_attention_interval"])
            if interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            kwargs["layer_types"] = tuple(
                "linear_attention" if (idx + 1) % interval else "full_attention"
                for idx in range(int(data["num_hidden_layers"]))
            )
        kwargs.update(
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope["partial_rotary_factor"]),
            mrope_section=tuple(
                int(value) for value in rope["mrope_section"]
            ),
            expert_weight_format=expert_weight_format,
        )
        if tie_word_embeddings is not None:
            kwargs["tie_word_embeddings"] = bool(tie_word_embeddings)
        if is_moe:
            _require(
                data,
                (
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                    "num_experts_per_tok",
                    "num_experts",
                ),
                "Qwen MoE",
            )
        config = cls(**kwargs)
        config._validate_inference()
        return config

    def _validate_inference(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "Qwen hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "Qwen num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "Qwen linear value heads must be divisible by linear key heads"
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

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


@dataclass(frozen=True, slots=True)
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304

    def _validate_inference(self) -> None:
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
        _require(data, _VISION_SHAPE_FIELDS, "Qwen vision")
        deepstack = data.get("deepstack_visual_indexes")
        if deepstack:
            raise ValueError("Qwen inference does not support deepstack vision")
        config = cls(**_known_kwargs(cls, data))
        config._validate_inference()
        return config


@dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5Config":
        _require(
            data,
            (
                "text_config",
                "vision_config",
                "image_token_id",
                "video_token_id",
                "tie_word_embeddings",
            ),
            "Qwen",
        )
        model_type = str(data.get("model_type", "qwen3_5"))
        text_data = dict(data.get("text_config") or {})
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
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            expert_weight_format=expert_weight_format,
        )
        return cls(
            text_config=text,
            vision_config=Qwen3_5VisionConfig.from_dict(
                data.get("vision_config") or {}
            ),
            image_token_id=int(
                data.get("image_token_id", _default(cls, "image_token_id"))
            ),
            video_token_id=int(
                data.get("video_token_id", _default(cls, "video_token_id"))
            ),
        )


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
